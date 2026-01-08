"""
TTS Service 核心业务逻辑（切片+平滑拼接+RTF诊断增强版）

主要改进：
1) 更细的标点切片（默认 target=18，首段更短 first_target=14）
2) 列表/编号合并（避免第一段韵律解析/前处理异常导致偶发极慢）
3) WAV 拼接：段间静音 + crossfade（减少割裂感，允许更细切片）
4) 记录每段/整体 duration 与 RTF（实时系数），便于判断“慢到底是不是异常”
5) 同一 job 内 segments 串行处理（ModelScope pipeline 不是线程安全的，并行会导致性能下降）
"""

import os
import logging
import base64
import uuid
import asyncio
import concurrent.futures
import time
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

import re
import io
import wave

logger = logging.getLogger(__name__)

# 模型配置：优先使用本地模型，否则从 ModelScope 下载
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_DIR = os.path.join(
    BASE_DIR, "models", "damo", "speech_sambert-hifigan_tts_zh-cn_16k"
)
MODEL_ID = "damo/speech_sambert-hifigan_tts_zh-cn_16k"


class TTSManager:
    """TTS 任务管理器"""
    
    def __init__(self):
        self._tts_pipeline = None
        # 并发：用于并发处理不同 job（吞吐）；同一 job 的分段推理可并行（利用空闲显存）
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        # 并行 segment 处理配置（可通过环境变量调整）
        # 注意：ModelScope pipeline 不是线程安全的，并行处理会导致性能下降
        # 默认禁用并行，保持串行处理（更稳定、更快）
        self._parallel_segments = os.getenv("TTS_PARALLEL_SEGMENTS", "false").lower() == "true"
        self._max_parallel_segments = int(os.getenv("TTS_MAX_PARALLEL_SEGMENTS", "4"))  # 最多并行处理几个 segment
        
        # 批处理配置（利用 Pipeline 的批处理能力）
        self._use_batch_processing = os.getenv("TTS_USE_BATCH", "true").lower() == "true"
        self._batch_size = int(os.getenv("TTS_BATCH_SIZE", "2"))  # 批处理大小
        self.jobs: Dict[str, Dict] = {}
    
        # 模型加载状态管理（用于后台预热）
        import threading
        self._loading_lock = threading.Lock()  # 线程锁，防止并发加载
        self._loading_event = None  # 用于等待加载完成
        self._is_loading = False  # 是否正在加载

        # 允许你用环境变量调参（上线后不改代码也能调）
        self._seg_target = int(os.getenv("TTS_SEG_TARGET", "18"))          # 常规段目标长度
        self._seg_first_target = int(os.getenv("TTS_SEG_FIRST", "14"))     # 首段更短，降低“首段极慢”概率
        self._seg_hard_max = int(os.getenv("TTS_SEG_HARD_MAX", "22"))      # 硬上限：超过必须继续拆
        self._crossfade_ms = int(os.getenv("TTS_CROSSFADE_MS", "60"))      # 拼接 crossfade（ms）
        self._pause_soft_ms = int(os.getenv("TTS_PAUSE_SOFT_MS", "120"))   # 逗号/弱停顿
        self._pause_hard_ms = int(os.getenv("TTS_PAUSE_HARD_MS", "200"))   # 句号/问号/叹号/换行

        # 推理配置（可按需再调）
        self._beam_size = int(os.getenv("TTS_BEAM_SIZE", "1"))
        # 采样率：模型是 16k，使用 8000 会导致语速变慢和音质下降
        self._sampling_rate = int(os.getenv("TTS_SAMPLING_RATE", "16000"))

    # ----------------------------- Pipeline 管理 -----------------------------

    def _ensure_pipeline(self, wait_if_loading: bool = True):
        """
        确保 TTS pipeline 已加载（GPU 证据闭环版）
        
        Args:
            wait_if_loading: 如果正在加载，是否等待加载完成（默认 True）
        
        Returns:
            pipeline 实例
        """
        # 如果已加载，直接返回
        if self._tts_pipeline is not None:
            logger.debug("TTS 模型已加载，跳过加载步骤")
            return self._tts_pipeline
        
        # 如果正在加载，等待完成
        if self._is_loading and wait_if_loading:
            if self._loading_event is not None:
                logger.info("[TTS] 模型正在后台加载中，等待加载完成...")
                # 使用 threading.Event 等待（因为这是同步方法）
                import threading
                if isinstance(self._loading_event, threading.Event):
                    self._loading_event.wait(timeout=60)  # 最多等待60秒
                    if self._tts_pipeline is not None:
                        logger.info("[TTS] 模型加载完成（等待结束）")
                        return self._tts_pipeline
                    else:
                        logger.warning("[TTS] 等待超时或加载失败，尝试重新加载")
                else:
                    # 如果是 asyncio.Event，需要特殊处理
                    logger.warning("[TTS] 加载事件类型不匹配，尝试重新加载")
        
        # 开始加载（线程安全）
        import threading
        self._loading_lock.acquire()
        try:
            # 双重检查：加载过程中可能已经被其他线程加载完成
            if self._tts_pipeline is not None:
                return self._tts_pipeline
            
            # 如果已经在加载，直接返回或等待
            if self._is_loading:
                if wait_if_loading and self._loading_event:
                    if isinstance(self._loading_event, threading.Event):
                        # 释放锁，等待加载完成
                        self._loading_lock.release()
                        try:
                            self._loading_event.wait(timeout=60)
                            if self._tts_pipeline is not None:
                                return self._tts_pipeline
                        finally:
                            self._loading_lock.acquire()
                # 如果不等或等待失败，继续执行加载逻辑
            
            # 标记为正在加载
            self._is_loading = True
            self._loading_event = threading.Event()
        finally:
            self._loading_lock.release()
        
        # 执行实际加载（在锁外执行，避免长时间持有锁）
        try:
            model_load_start = time.time()
            model_or_path = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else MODEL_ID
            
            # 设备选择
            device_check_start = time.time()
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                device = "cuda:0"
                prefer_dtype = torch.float16
                logger.info(f"[TTS] CUDA available: True, device={device}, name={torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                prefer_dtype = torch.float32
                logger.warning("[TTS] CUDA available: False, using CPU (slow).")
            device_check_elapsed = (time.time() - device_check_start) * 1000

            src = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else MODEL_ID
            logger.info(f"[TTS] Initializing pipeline: model={src}, device={device}, prefer_dtype={prefer_dtype}")

            # 创建 pipeline（dtype 兼容兜底）
            pipeline_create_start = time.time()
            try:
                self._tts_pipeline = pipeline(
                    task=Tasks.text_to_speech,
                    model=model_or_path,
                    device=device,
                    dtype=prefer_dtype,
                )
                logger.info("[TTS] pipeline created with dtype param.")
            except TypeError as e:
                logger.warning(f"[TTS] pipeline() does not accept dtype, fallback without dtype. err={e}")
            self._tts_pipeline = pipeline(
                task=Tasks.text_to_speech,
                    model=model_or_path,
                    device=device,
                )
            pipeline_create_elapsed = (time.time() - pipeline_create_start) * 1000

            pid = os.getpid()
            logger.info(f"[TTS] service pid={pid} (container pid; nvidia-smi PID alignment may differ without host pid namespace)")

            # 尽最大可能检查参数所在 device（如果 pipeline 暴露 torch.nn.Module）
            try:
                p = self._tts_pipeline
                m = getattr(p, "model", None) or getattr(p, "_model", None)
                candidates = []
                if m is not None:
                    candidates.append(("pipeline.model", m))
                    for name in ["am", "acoustic_model", "tts_model", "net", "network",
                                 "generator", "vocoder", "hifigan", "net_g"]:
                        sub = getattr(m, name, None)
                        if sub is not None and hasattr(sub, "parameters"):
                            candidates.append((f"pipeline.model.{name}", sub))

                found_any = False
                for name, obj in candidates:
                    try:
                        dev = next(obj.parameters()).device
                        found_any = True
                        logger.info(f"[TTS] {name} param device = {dev}")
                    except Exception:
                        pass

                if not found_any:
                    logger.warning("[TTS] Cannot inspect model parameter device (pipeline may not expose torch.nn.Module).")
            except Exception as e:
                logger.warning(f"[TTS] Device inspection failed: {e}")

            model_load_total = (time.time() - model_load_start) * 1000
            logger.info(
                f"[TTS] Pipeline ready. total_load={model_load_total:.2f}ms "
                f"(device_check={device_check_elapsed:.2f}ms, pipeline_create={pipeline_create_elapsed:.2f}ms)"
            )
        except Exception as e:
            logger.error(f"[TTS] 模型加载失败: {e}", exc_info=True)
            raise
        finally:
            # 加载完成，通知等待的线程
            self._loading_lock.acquire()
            try:
                self._is_loading = False
                if self._loading_event is not None:
                    self._loading_event.set()
            finally:
                self._loading_lock.release()

    # ----------------------------- 文本预处理 & 切片 -----------------------------

    def _normalize_text(self, text: str) -> str:
        """
        规范化文本，降低前端/韵律失败概率：
        - 收敛空白字符
        - 收敛多余空行
        - 统一列表编号形态（避免"1.""1、""1:"等混乱）
        
        优化：去掉编号（"1. "、"2. "等），将空行替换为逗号，缩短文本以提升处理速度。
        """
        t = (text or "").strip()
        if not t:
            return ""

        # 把 \r\n 统一为 \n
        t = t.replace("\r\n", "\n").replace("\r", "\n")

        # 收敛空格：保留换行
        t = re.sub(r"[ \t]+", " ", t)

        # 将多个空行替换为逗号（用于 TTS，缩短文本）
        # 先处理连续的空行，替换为逗号+空格
        t = re.sub(r"\n\s*\n+", "，", t)
        
        # 将单个换行也替换为逗号（如果前后都有内容，且不是标点）
        t = re.sub(r"([^\n，。！？；\s])\s*\n\s*([^\n，。！？；\s])", r"\1，\2", t)

        # 去掉编号：移除 "1. "、"2. " 等格式（用于 TTS，缩短文本）
        # 匹配：行首或换行后的 "数字. " 或 "数字、" 或 "数字：" 等
        t = re.sub(r"(^|\n)\s*\d{1,2}\s*[\.、:：\)]\s*", r"\1", t)
        
        # 清理多余的逗号（避免连续逗号）
        t = re.sub(r"，+", "，", t)
        
        # 清理开头和结尾的逗号
        t = re.sub(r"^，+|，+$", "", t)

        return t.strip()

    def _merge_list_items(self, parts: List[str]) -> List[str]:
        """
        合并列表项：把 "1. xxx" "2. yyy" 视为较自然的分段边界，
        但避免出现“第一段过长+多换行+冒号”的结构导致偶发极慢。
        做法：若某段以编号开头，则尽量与其后短语保持在同段。
        """
        out: List[str] = []
        buf = ""

        def flush():
            nonlocal buf
            if buf.strip():
                out.append(buf.strip())
            buf = ""

        for p in parts:
            s = p.strip()
            if not s:
                continue

            # 如果是编号开头，优先切断（作为新段的开始）
            is_list = bool(re.match(r"^\d{1,2}\.\s+", s))
            if is_list:
                flush()
                out.append(s)
            else:
                # 常规文本，累积到 buf
                if not buf:
                    buf = s
                else:
                    # 连接时保留原句间隔
                    buf += " " + s

        flush()
        return out

    def _split_text_for_tts(self, text: str,
                            target: int,
                            first_target: int,
                            hard_max: int) -> List[str]:
        """
        更细切片策略（比“>30硬切”聪明）：
        1) 先 normalize
        2) 先按强边界切：换行/句末标点
        3) 合并/规整列表项
        4) 再按弱边界切：逗号/顿号/冒号
        5) 仍超 hard_max 才硬切
        """
        t = self._normalize_text(text)
        if not t:
            return []

        # 先按强标点/换行切
        strong_parts = re.split(r"(?<=[。！？；\n])", t)
        strong_parts = [p.strip() for p in strong_parts if p and p.strip()]

        # 规整列表结构（避免第一段结构过于复杂）
        strong_parts = self._merge_list_items(strong_parts)

        out: List[str] = []
        buf = ""

        def emit(s: str):
            if s.strip():
                out.append(s.strip())

        def buf_limit(is_first: bool) -> int:
            return first_target if is_first else target

        for p in strong_parts:
            p = p.strip()
            if not p:
                continue

            # 当前段目标长度（首段更短）
            limit = buf_limit(is_first=(len(out) == 0 and not buf))

            # 能放进 buf 就先放（尽量在自然边界拼）
            if buf and (len(buf) + len(p) <= limit):
                buf += p
                continue
            if (not buf) and (len(p) <= limit):
                buf = p
                continue

            # buf 先落地
            if buf:
                emit(buf)
                buf = ""

            # 若单段仍过长，按弱标点进一步切
            if len(p) > hard_max:
                subs = re.split(r"(?<=[，、：])", p)
                subs = [s.strip() for s in subs if s and s.strip()]
                tmp = ""
                for s in subs:
                    limit2 = buf_limit(is_first=(len(out) == 0 and not tmp))
                    if tmp and len(tmp) + len(s) <= limit2:
                        tmp += s
                    elif not tmp and len(s) <= limit2:
                        tmp = s
                    else:
                        if tmp:
                            emit(tmp)
                            tmp = ""
                        if len(s) <= hard_max:
                            emit(s)
                        else:
                            # 最终兜底：硬切
                            for i in range(0, len(s), hard_max):
                                emit(s[i:i + hard_max])
                if tmp:
                    emit(tmp)
            else:
                emit(p)

        if buf:
            emit(buf)

        # 听感优化：非最后一段末尾若无标点，补“，”避免割裂
        for i in range(len(out) - 1):
            if out[i] and out[i][-1] not in "。！？；，、：\n":
                out[i] += "，"

        return out

    # ----------------------------- WAV 工具：读取、拼接、RTF -----------------------------

    def _wav_read_all_pcm(self, wav_bytes: bytes) -> Tuple[int, int, int, bytes]:
        """
        读取 WAV 的 sr/nch/sampwidth 以及 PCM frames（不含 header）。
        """
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        return sr, nch, sw, frames

    def _wav_duration(self, wav_bytes: bytes) -> float:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
        return frames / float(sr) if sr else 0.0

    def _make_silence_frames(self, sr: int, nch: int, sw: int, ms: int) -> bytes:
        nframes = int(sr * ms / 1000.0)
        frame = (b"\x00" * sw) * nch
        return frame * nframes

    def _crossfade_frames(self, a: bytes, b: bytes, sw: int, nch: int, fade_ms: int, sr: int) -> Tuple[bytes, bytes]:
        """
        简化 crossfade：在拼接边界做线性淡出/淡入。
        仅支持 16-bit PCM（sw=2）；若不是，自动退化为不 crossfade。
        返回：处理后的 a_tail 淡出、b_head 淡入，并用于拼接替换原边界。
        """
        if fade_ms <= 0 or sw != 2:
            return a, b

        fade_frames = int(sr * fade_ms / 1000.0)
        if fade_frames <= 0:
            return a, b

        bytes_per_frame = sw * nch
        fade_bytes = fade_frames * bytes_per_frame

        if len(a) < fade_bytes or len(b) < fade_bytes:
            return a, b

        a_head = a[:-fade_bytes]
        a_tail = a[-fade_bytes:]
        b_head = b[:fade_bytes]
        b_tail = b[fade_bytes:]

        # 16-bit little-endian samples
        import struct
        fmt = "<" + "h" * (fade_bytes // 2)
        a_samples = list(struct.unpack(fmt, a_tail))
        b_samples = list(struct.unpack(fmt, b_head))

        mixed = []
        # 每个 sample 线性权重（注意：这里对“样本序列”做淡入淡出，足够消除咔哒感）
        n = len(a_samples)
        for i in range(n):
            wa = (n - 1 - i) / (n - 1) if n > 1 else 0.0   # a 从 1→0
            wb = i / (n - 1) if n > 1 else 1.0            # b 从 0→1
            v = int(a_samples[i] * wa + b_samples[i] * wb)
            # clamp
            if v > 32767:
                v = 32767
            if v < -32768:
                v = -32768
            mixed.append(v)

        mixed_bytes = struct.pack(fmt, *mixed)

        # 拼接：a_head + mixed + b_tail
        return (a_head + mixed_bytes), b_tail

    def _concat_wavs_smooth(self, wav_list: List[bytes], pauses_ms: List[int], crossfade_ms: int) -> bytes:
        """
        拼接多个 wav（PCM），段间插入静音，并做 crossfade 平滑边界。
        pauses_ms 长度 = len(wav_list) - 1
        """
        if not wav_list:
            return b""

        sr0, nch0, sw0, pcm0 = self._wav_read_all_pcm(wav_list[0])

        pcms: List[bytes] = [pcm0]
        for w in wav_list[1:]:
            sr, nch, sw, pcm = self._wav_read_all_pcm(w)
            # 基本一致性检查：不一致则直接放弃 crossfade（但仍拼接）
            if sr != sr0 or nch != nch0 or sw != sw0:
                logger.warning(f"[TTS] wav meta mismatch: first=({sr0},{nch0},{sw0}) vs seg=({sr},{nch},{sw})")
            pcms.append(pcm)

        out_pcm = pcms[0]
        for i in range(1, len(pcms)):
            # 段间静音（先插入静音再拼接下一段更自然）
            pause = pauses_ms[i - 1] if i - 1 < len(pauses_ms) else 0
            if pause > 0:
                out_pcm += self._make_silence_frames(sr0, nch0, sw0, pause)

            # crossfade：对 out_pcm 尾部与下一段头部
            a_new, b_tail = self._crossfade_frames(out_pcm, pcms[i], sw0, nch0, crossfade_ms, sr0)
            out_pcm = a_new + b_tail

        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as out_wf:
            out_wf.setnchannels(nch0)
            out_wf.setsampwidth(sw0)
            out_wf.setframerate(sr0)
            out_wf.writeframes(out_pcm)

        return out_buf.getvalue()

    # ----------------------------- TTS 任务执行 -----------------------------
    
    def _process_tts_task_sync(self, job_id: str, text: str, voice: str):
        """同步处理 TTS 任务（在后台线程执行）"""
        try:
            start_time = self.jobs[job_id].get("start_time", time.time())
            thread_start_time = time.time()
            thread_wait_time = (thread_start_time - start_time) * 1000

            self.jobs[job_id]["status"] = "processing"
            logger.info(
                f"[TTS] 开始处理 TTS 任务 - job_id: {job_id}, "
                f"text: '{text[:50]}{'...' if len(text) > 50 else ''}', "
                f"线程等待时间: {thread_wait_time:.2f}ms"
            )
            
            # pipeline
            pipeline_check_start = time.time()
            pipeline_instance = self._ensure_pipeline()
            if pipeline_instance is None:
                raise RuntimeError("TTS pipeline not initialized")
            pipeline_check_elapsed = (time.time() - pipeline_check_start) * 1000

            # 推理参数
            forward_params = {
                "beam_size": self._beam_size,
                "sampling_rate": self._sampling_rate,
            }

            # 切片（更细）
            segments = self._split_text_for_tts(
                text,
                target=self._seg_target,
                first_target=self._seg_first_target,
                hard_max=self._seg_hard_max,
            )
            if not segments:
                raise ValueError("empty text after normalization")

            logger.info(
                f"[TTS] split into {len(segments)} segments: lens={[len(s) for s in segments]} "
                f"(target={self._seg_target}, first={self._seg_first_target}, hard_max={self._seg_hard_max})"
            )

            pid = os.getpid()
            logger.info(f"[TTS] infer START | pid={pid} | voice={voice} | text_len={len(text)}")

            if torch.cuda.is_available():
                logger.info(
                    f"[TTS] before infer | cuda_alloc={torch.cuda.memory_allocated()//1024**2}MB "
                    f"cuda_reserved={torch.cuda.memory_reserved()//1024**2}MB"
                )

            # 分段推理（支持并行处理以利用空闲显存）
            # 说明：如果显存充足，可以并行处理多个 segment 来加速（特别是显存使用率低时）
            tts_start_time = time.time()
            
            # 检查是否启用并行处理
            use_parallel = (
                self._parallel_segments and 
                torch.cuda.is_available() and 
                len(segments) > 1
            )
            
            if use_parallel:
                # 检查显存是否充足（当前使用 < 50% 时启用并行）
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
                gpu_used = torch.cuda.memory_allocated() / 1024**2  # MB
                gpu_usage_ratio = gpu_used / gpu_total if gpu_total > 0 else 1.0
                
                if gpu_usage_ratio > 0.5:
                    use_parallel = False
                    logger.info(
                        f"[TTS] GPU 显存使用率 {gpu_usage_ratio*100:.1f}% > 50%，禁用并行处理 "
                        f"(used={gpu_used:.0f}MB / total={gpu_total:.0f}MB)"
                    )
                else:
                    logger.info(
                        f"[TTS] 启用并行处理 segments (显存使用率 {gpu_usage_ratio*100:.1f}%, "
                        f"最多并行 {self._max_parallel_segments} 个 segment)"
                    )
            
            # 处理单个 segment 的函数（用于并行）
            def process_segment(idx: int, seg: str) -> Tuple[int, bytes, float, float, bool]:
                """处理单个 segment，返回 (索引, wav数据, 耗时ms, 时长s, cpu_fallback)"""
                if self.jobs[job_id]["status"] == "cancelled":
                    return (idx, b"", 0.0, 0.0, False)
                
                seg_t0 = time.time()
                
                # 检查模型参数所在设备（推理前）
                model_device_before = None
                if torch.cuda.is_available():
                    try:
                        p = pipeline_instance
                        m = getattr(p, "model", None) or getattr(p, "_model", None)
                        if m is not None:
                            for name in ["am", "acoustic_model", "tts_model", "net", "network",
                                         "generator", "vocoder", "hifigan", "net_g"]:
                                sub = getattr(m, name, None)
                                if sub is not None and hasattr(sub, "parameters"):
                                    try:
                                        model_device_before = next(sub.parameters()).device
                                        break
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                
                try:
                    # 确保 voice 参数正确传递（用于调试）
                    logger.debug(f"[TTS] seg#{idx+1} 调用 pipeline - voice={voice}, sampling_rate={forward_params.get('sampling_rate')}")
                    out = pipeline_instance(
                        input=seg,
                        voice=voice,
                        forward_params=forward_params
                    )
                    # 类型检查：确保 out 是字典类型
                    if not isinstance(out, dict):
                        raise TypeError(f"Pipeline returned unexpected type: {type(out)}")
                    seg_wav = out.get(OutputKeys.OUTPUT_WAV, b"")
                    if not isinstance(seg_wav, bytes):
                        raise TypeError(f"Output WAV is not bytes: {type(seg_wav)}")
                except Exception as e:
                    logger.error(f"[TTS] seg#{idx+1} 推理失败: {e}", exc_info=True)
                    return (idx, b"", 0.0, 0.0, False)
                
                seg_ms = (time.time() - seg_t0) * 1000
                dur = self._wav_duration(seg_wav)
                rtf = (seg_ms / 1000.0) / dur if dur > 1e-6 else 0.0
                
                # 检测 CPU fallback
                cpu_fallback_detected = False
                if torch.cuda.is_available() and model_device_before is not None:
                    if model_device_before.type == "cpu":
                        cpu_fallback_detected = True
                        logger.warning(
                            f"[TTS] ⚠️ CPU FALLBACK 检测到！seg#{idx+1} 的模型参数在 CPU 上 "
                            f"(device={model_device_before})，推理可能很慢！RTF={rtf:.3f}"
                        )
                    elif model_device_before.type == "cuda" and rtf > 1.0:
                        logger.warning(
                            f"[TTS] ⚠️ seg#{idx+1} RTF={rtf:.3f} > 1.0，虽然模型在GPU上，但性能异常慢！"
                        )
                
                logger.info(
                    f"[TTS] seg#{idx+1}/{len(segments)} {'[并行]' if use_parallel else ''} END | "
                    f"elapsed={seg_ms:.2f}ms | wav_bytes={len(seg_wav)} | duration={dur:.2f}s | rtf={rtf:.3f}"
                    f"{' | ⚠️ CPU FALLBACK' if cpu_fallback_detected else ''}"
                )
                
                return (idx, seg_wav, seg_ms, dur, cpu_fallback_detected)
            
            # 执行推理（串行或并行）
            wav_chunks: List[bytes] = [b""] * len(segments)  # 预分配，保持顺序
            seg_times_ms: List[float] = [0.0] * len(segments)
            seg_durations_s: List[float] = [0.0] * len(segments)
            pauses_ms: List[int] = []
            
            # 预先计算停顿时间（并行和串行都需要）
            for i in range(len(segments) - 1):
                seg = segments[i]
                last = seg[-1] if seg else ""
                if last in "。！？；\n":
                    pauses_ms.append(self._pause_hard_ms)
                else:
                    pauses_ms.append(self._pause_soft_ms)
            
            if use_parallel:
                # 并行处理：使用 ThreadPoolExecutor（PyTorch 支持多线程 GPU 推理）
                logger.info(f"[TTS] 开始并行处理 {len(segments)} 个 segments（最多 {self._max_parallel_segments} 个并发）")
                with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_parallel_segments) as executor:
                    # 提交所有任务
                    future_to_idx = {
                        executor.submit(process_segment, i, seg): i 
                        for i, seg in enumerate(segments)
                    }
                    
                    # 收集结果（按完成顺序，但保持原始索引）
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx, seg_wav, seg_ms, dur, cpu_fallback = future.result()
                        wav_chunks[idx] = seg_wav
                        seg_times_ms[idx] = seg_ms
                        seg_durations_s[idx] = dur
                        completed += 1
                        
                        if self.jobs[job_id]["status"] == "cancelled":
                            logger.info(f"[TTS] 任务 {job_id} 已被取消，停止并行推理")
                            break
                    
                    logger.info(f"[TTS] 并行处理完成：{completed}/{len(segments)} 个 segments")
            else:
                # 串行处理：支持批处理（利用 Pipeline 的批处理能力）
                if self._use_batch_processing and len(segments) > 1:
                    # 批处理模式：将多个 segment 合并成 batch 输入
                    logger.info(
                        f"[TTS] 使用批处理模式：{len(segments)} 个 segments，batch_size={self._batch_size}"
                    )
                    
                    batch_start_idx = 0
                    while batch_start_idx < len(segments):
                        if self.jobs[job_id]["status"] == "cancelled":
                            logger.info(f"[TTS] 任务 {job_id} 已被取消，停止批处理推理")
                            return
                        
                        # 准备当前 batch
                        batch_end_idx = min(batch_start_idx + self._batch_size, len(segments))
                        batch_segments = segments[batch_start_idx:batch_end_idx]
                        batch_indices = list(range(batch_start_idx, batch_end_idx))
                        
                        logger.info(
                            f"[TTS] 批处理 batch {batch_start_idx//self._batch_size + 1}: "
                            f"segments {batch_start_idx+1}-{batch_end_idx} (共 {len(batch_segments)} 个)"
                        )
                        
                        batch_t0 = time.time()
                        
                        try:
                            # 尝试批处理：将多个文本作为列表输入
                            # 注意：需要检查 pipeline 是否支持批处理
                            batch_inputs = batch_segments
                            
                            # 检查模型参数所在设备（推理前）
                            model_device_before = None
                            if torch.cuda.is_available():
                                try:
                                    p = pipeline_instance
                                    m = getattr(p, "model", None) or getattr(p, "_model", None)
                                    if m is not None:
                                        for name in ["am", "acoustic_model", "tts_model", "net", "network",
                                                     "generator", "vocoder", "hifigan", "net_g"]:
                                            sub = getattr(m, name, None)
                                            if sub is not None and hasattr(sub, "parameters"):
                                                try:
                                                    model_device_before = next(sub.parameters()).device
                                                    break
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                            
                            # 尝试批处理调用
                            try:
                                # 确保 pipeline_instance 不为 None
                                if pipeline_instance is None:
                                    raise RuntimeError("TTS pipeline not initialized")
                                
                                # 调试日志：记录批处理输入
                                logger.info(
                                    f"[TTS] 批处理调用 - 输入类型: {type(batch_inputs)}, "
                                    f"输入长度: {len(batch_inputs) if isinstance(batch_inputs, list) else 'N/A'}, "
                                    f"输入内容: {[s[:20] + '...' if len(s) > 20 else s for s in batch_inputs[:2]]}"
                                )
                                
                                # 如果 pipeline 支持批处理，会返回列表
                                logger.debug(f"[TTS] 批处理调用 pipeline - voice={voice}, sampling_rate={forward_params.get('sampling_rate')}, batch_size={self._batch_size}")
                                batch_outputs = pipeline_instance(
                                    input=batch_inputs,
                                    voice=voice,
                                    forward_params=forward_params,
                                    batch_size=self._batch_size
                                )
                                
                                # 调试日志：记录批处理输出
                                logger.info(
                                    f"[TTS] 批处理返回 - 类型: {type(batch_outputs)}, "
                                    f"是否为列表: {isinstance(batch_outputs, list)}, "
                                    f"长度: {len(batch_outputs) if isinstance(batch_outputs, list) else 'N/A'}"
                                )
                                
                                # 检查返回类型：如果是列表，说明批处理成功
                                if isinstance(batch_outputs, list):
                                    # 批处理成功
                                    batch_elapsed = (time.time() - batch_t0) * 1000
                                    batch_total_dur = 0.0
                                    
                                    # 批处理是顺序执行的，总耗时是累加的
                                    # 无法精确测量每个 segment 的耗时，只能记录总耗时
                                    for batch_idx, (seg_idx, output) in enumerate(zip(batch_indices, batch_outputs)):
                                        if isinstance(output, dict):
                                            seg_wav = output.get(OutputKeys.OUTPUT_WAV, b"")
                                        elif isinstance(output, bytes):
                                            seg_wav = output
                                        else:
                                            # 如果输出格式不符合预期，回退到单个处理
                                            logger.warning(
                                                f"[TTS] 批处理输出格式异常，回退到单个处理 (seg#{seg_idx+1})"
                                            )
                                            idx, seg_wav, seg_ms, dur, cpu_fallback = process_segment(seg_idx, segments[seg_idx])
                                            wav_chunks[seg_idx] = seg_wav
                                            seg_times_ms[seg_idx] = seg_ms
                                            seg_durations_s[seg_idx] = dur
                                            continue
                                        
                                        dur = self._wav_duration(seg_wav)
                                        batch_total_dur += dur
                                        
                                        # 批处理是顺序执行的，无法精确测量单个 segment 的耗时
                                        # 使用总耗时记录（后续计算整体 RTF 时使用）
                                        # 注意：这不是并行处理，所以总耗时是累加的，不是单个的时间
                                        seg_ms = batch_elapsed  # 记录总耗时，不是单个耗时
                                        rtf = (batch_elapsed / 1000.0) / batch_total_dur if batch_total_dur > 1e-6 else 0.0
                                        
                                        wav_chunks[seg_idx] = seg_wav
                                        seg_times_ms[seg_idx] = batch_elapsed  # 记录总耗时
                                        seg_durations_s[seg_idx] = dur
                                        
                                        # 计算整体 RTF（基于总耗时和总时长）
                                        batch_rtf = (batch_elapsed / 1000.0) / batch_total_dur if batch_total_dur > 1e-6 else 0.0
                                        
                                        logger.info(
                                            f"[TTS] seg#{seg_idx+1}/{len(segments)} [批处理] END | "
                                            f"batch_elapsed={batch_elapsed:.2f}ms | wav_bytes={len(seg_wav)} | "
                                            f"duration={dur:.2f}s | batch_rtf={batch_rtf:.3f} "
                                            f"(批处理总耗时，非单个segment耗时)"
                                        )
                                    
                                    # 计算整体 RTF
                                    batch_rtf = (batch_elapsed / 1000.0) / batch_total_dur if batch_total_dur > 1e-6 else 0.0
                                    logger.info(
                                        f"[TTS] 批处理完成：{len(batch_segments)} 个 segments，"
                                        f"总耗时={batch_elapsed:.2f}ms，总时长={batch_total_dur:.2f}s，"
                                        f"整体RTF={batch_rtf:.3f} "
                                        f"(批处理是顺序执行，总耗时是累加的，不是并行)"
                                    )
                                else:
                                    # 批处理失败，回退到单个处理
                                    raise ValueError("Pipeline 不支持批处理，返回类型不是列表")
                                    
                            except (TypeError, ValueError, AttributeError) as e:
                                # 批处理失败，回退到单个处理
                                logger.warning(
                                    f"[TTS] 批处理失败（Pipeline 可能不支持批处理），回退到单个处理: {e}"
                                )
                                # 回退到单个处理
                                for seg_idx in batch_indices:
                                    if self.jobs[job_id]["status"] == "cancelled":
                                        return
                                    logger.info(
                                        f"[TTS] seg#{seg_idx+1}/{len(segments)} START | "
                                        f"len={len(segments[seg_idx])} | "
                                        f"text='{segments[seg_idx][:30]}{'...' if len(segments[seg_idx]) > 30 else ''}'"
                                    )
                                    idx, seg_wav, seg_ms, dur, cpu_fallback = process_segment(seg_idx, segments[seg_idx])
                                    wav_chunks[seg_idx] = seg_wav
                                    seg_times_ms[seg_idx] = seg_ms
                                    seg_durations_s[seg_idx] = dur
                                
                        except Exception as e:
                            logger.error(f"[TTS] 批处理异常，回退到单个处理: {e}", exc_info=True)
                            # 回退到单个处理
                            for seg_idx in batch_indices:
                                if self.jobs[job_id]["status"] == "cancelled":
                                    return
                                idx, seg_wav, seg_ms, dur, cpu_fallback = process_segment(seg_idx, segments[seg_idx])
                                wav_chunks[seg_idx] = seg_wav
                                seg_times_ms[seg_idx] = seg_ms
                                seg_durations_s[seg_idx] = dur
                        
                        batch_start_idx = batch_end_idx
                else:
                    # 单个处理模式（原有逻辑）
                    for i, seg in enumerate(segments):
                        if self.jobs[job_id]["status"] == "cancelled":
                            logger.info(f"[TTS] 任务 {job_id} 已被取消，停止分段推理")
                            return
                        
                        logger.info(
                            f"[TTS] seg#{i+1}/{len(segments)} START | len={len(seg)} | text='{seg[:30]}{'...' if len(seg) > 30 else ''}'"
                        )
                        
                        idx, seg_wav, seg_ms, dur, cpu_fallback = process_segment(i, seg)
                        wav_chunks[i] = seg_wav
                        seg_times_ms[i] = seg_ms
                        seg_durations_s[i] = dur

            # 拼接（带 crossfade）
            wav_data = self._concat_wavs_smooth(
                wav_chunks,
                pauses_ms=pauses_ms,
                crossfade_ms=self._crossfade_ms
            )

            tts_elapsed = (time.time() - tts_start_time) * 1000
            total_dur = self._wav_duration(wav_data)
            total_rtf = (tts_elapsed / 1000.0) / total_dur if total_dur > 1e-6 else 0.0

            if torch.cuda.is_available():
                logger.info(
                    f"[TTS] after infer | cuda_alloc={torch.cuda.memory_allocated()//1024**2}MB "
                    f"cuda_reserved={torch.cuda.memory_reserved()//1024**2}MB"
                )

            logger.info(
                f"[TTS] infer END | pid={pid} | elapsed={tts_elapsed:.2f}ms | "
                f"audio_duration={total_dur:.2f}s | rtf={total_rtf:.3f} | "
                f"segments={len(segments)}"
            )

            # base64
            base64_start_time = time.time()
            wav_base64 = base64.b64encode(wav_data).decode("utf-8")
            base64_elapsed = (time.time() - base64_start_time) * 1000

            # cancel check
            if self.jobs[job_id]["status"] == "cancelled":
                logger.info(f"[TTS] 任务 {job_id} 已被取消，丢弃结果")
                return
            
            total_elapsed = (time.time() - start_time) * 1000

            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["result"] = {
                "audio_base64": wav_base64,
                "text": text,
                "audio_size": len(wav_data)
            }
            self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
            self.jobs[job_id]["_elapsed_time_ms"] = total_elapsed
            self.jobs[job_id]["_detailed_timing"] = {
                "thread_wait_ms": thread_wait_time,
                "pipeline_check_ms": pipeline_check_elapsed,
                "tts_generation_ms": tts_elapsed,
                "base64_encode_ms": base64_elapsed,
                "total_ms": total_elapsed,
                "segments": len(segments),
                "audio_duration_s": total_dur,
                "rtf": total_rtf,
                "segment_times_ms": seg_times_ms,
                "segment_durations_s": seg_durations_s,
            }

            logger.info(
                f"[TTS] 任务 {job_id} 完成 - "
                f"text: '{text[:50]}{'...' if len(text) > 50 else ''}' (长度: {len(text)} 字符), "
                f"总耗时: {total_elapsed:.2f}ms | "
                f"线程等待: {thread_wait_time:.2f}ms | "
                f"Pipeline检查: {pipeline_check_elapsed:.2f}ms | "
                f"TTS生成(含分段+拼接): {tts_elapsed:.2f}ms | "
                f"Base64编码: {base64_elapsed:.2f}ms | "
                f"音频大小: {len(wav_data)} 字节 | duration={total_dur:.2f}s | rtf={total_rtf:.3f} | "
                f"voice: '{voice}'"
            )
            
        except Exception as e:
            if job_id in self.jobs and self.jobs[job_id]["status"] != "cancelled":
                self.jobs[job_id]["status"] = "error"
                self.jobs[job_id]["error"] = str(e)
                start_time = self.jobs[job_id].get("start_time", time.time())
                elapsed = (time.time() - start_time) * 1000
                logger.error(
                    f"[TTS] 任务 {job_id} 失败 (耗时: {elapsed:.2f}ms) - "
                    f"text: '{text[:50]}{'...' if len(text) > 50 else ''}', error: {e}",
                    exc_info=True
                )

    # ----------------------------- 对外 API：start/cancel/result/cleanup -----------------------------
    
    async def start_task(self, text: str, voice: str = "zhitian_emo") -> str:
        """启动 TTS 任务"""
        task_creation_start = time.time()
        job_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.jobs[job_id] = {
            "status": "pending",
            "text": text,
            "voice": voice,
            "created_at": datetime.now().isoformat(),
            "start_time": start_time
        }
        
        executor_submit_start = time.time()
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            self._executor,
            self._process_tts_task_sync,
            job_id,
            text,
            voice
        )
        executor_submit_elapsed = (time.time() - executor_submit_start) * 1000

        task_creation_elapsed = (time.time() - task_creation_start) * 1000
        logger.debug(
            f"[TTS Manager] 任务创建完成 - job_id: {job_id}, "
            f"总耗时: {task_creation_elapsed:.2f}ms "
            f"(提交到线程池: {executor_submit_elapsed:.2f}ms)"
        )
        
        return job_id
    
    async def cancel_task(self, job_id: str) -> Dict:
        """取消 TTS 任务"""
        if job_id not in self.jobs:
            return {"status": "not_found", "message": f"任务 {job_id} 不存在"}
        
        job = self.jobs[job_id]
        if job["status"] == "completed":
            return {"status": "already_completed", "message": "任务已完成，无法取消"}
        if job["status"] == "cancelled":
            return {"status": "already_cancelled", "message": "任务已被取消"}
        
        job["status"] = "cancelled"
        job["cancelled_at"] = datetime.now().isoformat()
        logger.info(f"[TTS] 任务 {job_id} 已取消")
        return {"status": "cancelled", "message": "任务已取消"}
    
    async def get_result(self, job_id: str) -> Dict:
        """获取 TTS 结果"""
        if job_id not in self.jobs:
            return {"status": "not_found", "job_id": job_id}
        
        job = self.jobs[job_id]
        if job["status"] in ("pending", "processing"):
            return {"status": "processing", "job_id": job_id, "message": "任务处理中，请稍后重试"}
        if job["status"] == "cancelled":
            return {"status": "cancelled", "job_id": job_id, "message": "任务已取消"}
        if job["status"] == "error":
            return {"status": "error", "job_id": job_id, "error": job.get("error", "未知错误")}
        if job["status"] == "completed":
            return {"status": "completed", "job_id": job_id, **job["result"]}
        
        return {"status": "unknown", "job_id": job_id, "message": "未知的任务状态"}
    
    async def cleanup_job(self, job_id: str) -> Dict:
        """清理已完成的任务"""
        if job_id not in self.jobs:
            return {"status": "not_found", "message": f"任务 {job_id} 不存在"}
        
        job = self.jobs[job_id]
        if job["status"] not in ["completed", "cancelled", "error"]:
            return {"status": "cannot_cleanup", "message": "只能清理已完成/已取消/失败的任务"}
        
        del self.jobs[job_id]
        return {"status": "deleted"}
