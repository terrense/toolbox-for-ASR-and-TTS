"""
ASR 接口：recognize_voice
KWS 唤醒接口：kws_wakeup
ASR & KWS 接口：asr_wake
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Union, Any, Tuple, Dict

# ASR使用FunASR WebSocket客户端
import asyncio
import websockets
import json
import base64
# KWS继续使用本地AutoModel
from funasr import AutoModel
from app.services.full_hotwords import SYMS
from app.services.hg_deepseek import process_speech_result

# 流式处理相关导入
import numpy as np
import wave
from io import BytesIO

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1

# FunASR WebSocket配置
FUNASR_WS_URL = "ws://localhost:10095"

def load_hotwords_from_file() -> str:
    """
    从热词文件加载热词和权重，返回FunASR所需的JSON字符串格式
    格式："{\"小云\":80,\"小云小云\":85}"
    
    支持两种格式：
    1. 带权重：word weight（例如："胸闷 80"）
    2. 不带权重：word（例如："胸闷"），使用默认权重 20
    """
    # 注意：热词加载通常只在首次调用时耗时，后续会使用缓存
    hotwords = {}
    try:
        hotwords_file = "app/services/hotwords.txt"
        with open(hotwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 检查是否有权重（包含空格且最后一部分是数字）
                if ' ' in line:
                    parts = line.rsplit(' ', 1)
                    word = parts[0].strip()
                    weight_str = parts[1].strip()
                    try:
                        weight = int(weight_str)
                        hotwords[word] = weight
                    except ValueError:
                        # 如果最后一部分不是数字，整行作为热词，使用默认权重
                        logger.debug("热词行无有效权重，使用默认权重: %s", line)
                        hotwords[line] = 20
                else:
                    # 没有空格，整行作为热词，使用默认权重
                    hotwords[line] = 20
        logger.info("已加载 %s 个热词", len(hotwords))
    except Exception as e:
        logger.error("加载热词文件失败: %s", e)
        # 使用默认热词
        hotwords = {"小云": 80, "小云小云": 85}
    
    # 返回JSON字符串格式（符合FunASR官方要求）
    if hotwords:
        return json.dumps(hotwords, ensure_ascii=False)
    return ""


# ---------------- ffmpeg-only 音频转换工具 ----------------

def _find_ffmpeg() -> Optional[str]:
    """返回系统中 ffmpeg 的路径（如果存在），否则 None。"""
    return shutil.which("ffmpeg")


def _convert_with_ffmpeg(ffmpeg_path: str, input_path: str, out_path: str,
                         target_sr: int = SAMPLE_RATE, channels: int = CHANNELS) -> bool:
    """
    使用 ffmpeg 进行转换为 16k 单声道 PCM WAV。
    返回 True 表示成功，False 表示失败。
    """
    ffmpeg_start = time.perf_counter()
    logger.info("_convert_with_ffmpeg开始: input_path=%s, out_path=%s", input_path, out_path)
    cmd = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ac", str(channels),
        "-ar", str(target_sr),
        "-acodec", "pcm_s16le",
        out_path
    ]
    try:
        if os.path.exists(input_path):
            logger.info("输入文件存在: %s", input_path)
        else:
            logger.error("输入文件不存在: %s", input_path)
            return False

        # capture_output 避免在控制台打印 ffmpeg 信息；若需要调试可去掉 capture_output 并观察 stderr
        logger.info("执行ffmpeg命令: %s", ' '.join(cmd))
        _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
        ffmpeg_time = (time.perf_counter() - ffmpeg_start) * 1000
        logger.info("ffmpeg转换成功")
        logger.info("耗时统计 - ffmpeg转换: %.2f ms", ffmpeg_time)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg转换失败: %s", e)
        logger.error("ffmpeg stderr: %s", e.stderr)
        logger.error("ffmpeg stdout: %s", e.stdout)
        return False
    except Exception as e:
        logger.error("ffmpeg转换异常: %s", e)
        return False


def ensure_wav_mono_16k(input_path: str, max_tmp: Optional[str] = None) -> str:
    """
    将输入文件使用 ffmpeg 转为临时 16k 单声道 PCM WAV，返回路径（以 .wav 结尾）。
    仅使用 ffmpeg；若系统找不到 ffmpeg 或转换失败则抛出 RuntimeError。
    """
    convert_start = time.perf_counter()
    logger.info("ensure_wav_mono_16k开始: input_path=%s", input_path)

    if not os.path.isfile(input_path):
        logger.error("音频文件不存在: %s", input_path)
        raise FileNotFoundError(f"audio file not found: {input_path}")

    ffmpeg_check_start = time.perf_counter()
    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        logger.error("ffmpeg未找到")
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg and make it available in PATH, "
            "or add its directory to PATH. Check with `ffmpeg -version`."
        )
    ffmpeg_check_time = (time.perf_counter() - ffmpeg_check_start) * 1000
    logger.info("耗时统计 - 查找ffmpeg: %.2f ms", ffmpeg_check_time)

    tmp_create_start = time.perf_counter()
    tmp_dir = max_tmp or tempfile.gettempdir()
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
    os.close(fd)
    tmp_create_time = (time.perf_counter() - tmp_create_start) * 1000
    logger.info("创建临时文件: %s", tmp_wav)
    logger.info("耗时统计 - 创建转换临时文件: %.2f ms", tmp_create_time)

    ffmpeg_path = "ffmpeg"
    ok = _convert_with_ffmpeg(ffmpeg_path, input_path, tmp_wav, target_sr=SAMPLE_RATE, channels=CHANNELS)
    if ok:
        total_convert_time = (time.perf_counter() - convert_start) * 1000
        logger.info("音频转换成功")
        logger.info("耗时统计 - ensure_wav_mono_16k总耗时: %.2f ms", total_convert_time)
        return tmp_wav

    # 转换失败，清理并抛错
    logger.error("音频转换失败")
    try:
        os.remove(tmp_wav)
    except Exception:
        pass

    raise RuntimeError(
        f"ffmpeg conversion failed for file: {input_path}\n"
        "Ensure the input file is valid and ffmpeg supports its format."
    )


# ---------------- 识别接口（hotwords 可选） ----------------
def _normalize_hotwords(hotwords: Optional[Union[List[str], str]]) -> str:
    """
    将热词列表或单字符串转为 funasr 所需的热词字符串（用空格分隔）。
    如果 hotwords 为 None 或空，则返回空字符串。
    """
    if not hotwords:
        return ""
    if isinstance(hotwords, str):
        return hotwords
    return " ".join([w.strip() for w in hotwords if w and w.strip()])


# ========== 旧的 FunASR WebSocket 接口（已废弃，保留作为参考） ==========
# async def recognize_voice_websocket(audio_path: str, hotwords: Optional[List[str]] = None) -> str:
#     """
#     WebSocket ASR接口（官方流式协议）：
#       - audio_path: 本地音频文件路径（支持多种格式）
#       - hotwords: 可选的热词列表（List[str]），传 None 或不传时不使用热词
#     返回：
#       - 识别结果字符串
#     """
#     websocket_start = time.perf_counter()
    
    # # ⚠️ 读取并记录FunASR LM配置状态（用于确认LM是否被禁用）
    # try:
    #     from app.config import config
    #     voice_config = getattr(config, "voice_service", None)
    #     if voice_config:
    #         funasr_disable_lm = voice_config.funasr_disable_lm
    #         lm_status = "已禁用" if funasr_disable_lm else "已启用"
    #         logger.info("🔧 [FunASR LM配置] funasr_disable_lm=%s (%s)", funasr_disable_lm, lm_status)
    #     else:
    #         logger.warning("⚠️ [FunASR LM配置] 无法读取voice_service配置，使用默认值")
    # except Exception as e:
    #     logger.warning("⚠️ [FunASR LM配置] 读取配置异常: %s", e)
    
    # logger.info("recognize_voice_websocket开始: audio_path=%s", audio_path)
    # tmp_wav = None
    # try:
    #     # 1. 音频格式转换
    #     tmp_wav = ensure_wav_mono_16k(audio_path)
    #     logger.info("音频转换完成: %s", tmp_wav)
        
    #     # 2. 读取音频文件（二进制，不用base64）
    #     read_start = time.perf_counter()
    #     with open(tmp_wav, 'rb') as f:
    #         audio_data = f.read()
    #     read_time = (time.perf_counter() - read_start) * 1000
    #     logger.info("耗时统计 - 读取音频文件: %.2f ms", read_time)
        
    #     # 3. 准备热词配置（JSON字符串格式）
    #     hotword_start = time.perf_counter()
    #     hotword_str = load_hotwords_from_file()
    #     if hotwords:
    #         # 如果传入额外热词，需要合并
    #         hotword_dict = json.loads(hotword_str) if hotword_str else {}
    #         for word in hotwords:
    #             if word not in hotword_dict:
    #                 hotword_dict[word] = 20  # 默认权重
    #         hotword_str = json.dumps(hotword_dict, ensure_ascii=False)
    #     hotword_time = (time.perf_counter() - hotword_start) * 1000
    #     logger.info("热词配置: %s...", hotword_str[:100])  # 只打印前100字符
    #     logger.info("耗时统计 - 加载热词配置: %.2f ms", hotword_time)
        
    #     # 4. WebSocket连接和识别（官方流式协议）
    #     ws_connect_start = time.perf_counter()
    #     async with websockets.connect(FUNASR_WS_URL) as websocket:
    #         ws_connect_time = (time.perf_counter() - ws_connect_start) * 1000
    #         logger.info("耗时统计 - WebSocket连接: %.2f ms", ws_connect_time)
            
    #         # 4.1 发送初始化JSON（mode: offline, wav_format: wav）
    #         init_start = time.perf_counter()
    #         init_request = {
    #             "mode": "offline",
    #             "wav_name": os.path.basename(tmp_wav),
    #             "wav_format": "wav",
    #             "is_speaking": True,
    #             "hotwords": hotword_str  # JSON字符串
    #         }
            
    #         await websocket.send(json.dumps(init_request))
    #         init_time = (time.perf_counter() - init_start) * 1000
    #         logger.info("已发送WebSocket初始化请求")
    #         logger.info("耗时统计 - 发送初始化请求: %.2f ms", init_time)
            
    #         # 4.2 分块发送二进制音频数据（参考官方示例，offline模式快速发送）
    #         send_start = time.perf_counter()
    #         chunk_size = 8192  # 8KB每块
    #         total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
    #         logger.info("准备分块发送音频: 总大小=%s字节, 共%s块", len(audio_data), total_chunks)
            
    #         for i in range(total_chunks):
    #             beg = i * chunk_size
    #             end = min(beg + chunk_size, len(audio_data))
    #             chunk = audio_data[beg:end]
    #             await websocket.send(chunk)  # 直接发送二进制
    #             # offline模式几乎不需要延迟（参考官方脚本0.001秒）
    #             if i < total_chunks - 1:  # 最后一块发送后立即发结束信号，不延迟
    #                 await asyncio.sleep(0.001)
    #             logger.debug("已发送音频块 %s/%s (%s字节)", i + 1, total_chunks, len(chunk))
            
    #         # 4.3 发送结束信号
    #         send_time = (time.perf_counter() - send_start) * 1000
    #         logger.info("耗时统计 - 发送音频数据: %.2f ms", send_time)
            
    #         end_start = time.perf_counter()
    #         end_request = {"is_speaking": False}
    #         await websocket.send(json.dumps(end_request))
    #         end_time = (time.perf_counter() - end_start) * 1000
    #         logger.info("已发送结束信号")
    #         logger.info("耗时统计 - 发送结束信号: %.2f ms", end_time)
            
    #         # 4.4 接收结果（offline模式：参照官方脚本，收到第一条有效结果即完成）
    #         receive_start = time.perf_counter()
    #         result_text = ""
    #         first_receive_timeout = 3.0  # 第一次接收超时：3秒（服务端处理需要时间）
    #         subsequent_timeout = 0.5  # 后续接收超时：0.5秒（快速检查是否还有更多结果）
    #         received_count = 0
            
    #         while received_count < 3:  # 最多接收3条消息（通常1条就够了）
    #             try:
    #                 # 第一次等待稍长（服务端需要处理），后续快速检查
    #                 timeout = first_receive_timeout if received_count == 0 else subsequent_timeout
    #                 response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    
    #                 # 尝试解析为JSON（可能是文本消息）
    #                 try:
    #                     result = json.loads(response)
    #                 except (json.JSONDecodeError, TypeError):
    #                     # 如果不是JSON，可能是二进制数据，继续等待
    #                     logger.warning("收到非JSON消息，继续等待")
    #                     continue
                    
    #                 received_count += 1
    #                 logger.info("WebSocket识别结果 #%s: %s...", received_count, result.get('text', '')[:100])
                    
    #                 # 提取文本（参照官方脚本：offline模式收到结果就完成）
    #                 if isinstance(result, dict):
    #                     if "text" in result:
    #                         current_text = result.get("text", "")
    #                         if current_text:
    #                             result_text = current_text  # offline模式覆盖之前的结果
    #                             logger.info("✅ 收到识别结果: %s", result_text)
                                
    #                             # offline模式：收到有效结果就退出（参照官方脚本第286行）
    #                             if result.get("mode") == "offline":
    #                                 logger.info("offline模式：收到结果，立即返回")
    #                                 break
                                
    #                             # 其他模式：等待is_final标志
    #                             if result.get("is_final", False):
    #                                 logger.info("收到最终结果 (is_final=True)")
    #                                 break
                
    #             except asyncio.TimeoutError:
    #                 if result_text:
    #                     # 已有结果，可能服务端处理完成，退出
    #                     logger.info("已收到结果且等待超时，使用当前结果")
    #                     break
    #                 else:
    #                     # 没有结果但超时，可能需要更多时间
    #                     if received_count == 0:
    #                         logger.warning("首次接收超时（%s秒），继续等待...", first_receive_timeout)
    #                         continue  # 第一次超时，再试一次
    #                     else:
    #                         logger.info("后续接收超时，使用已有结果")
    #                         break
    #             except Exception as e:
    #                 logger.error("接收结果异常: %s", e)
    #                 break
            
    #         receive_time = (time.perf_counter() - receive_start) * 1000
    #         logger.info("耗时统计 - 接收识别结果: %.2f ms", receive_time)
            
    #         if not result_text:
    #             logger.warning("未收到有效识别结果")
            
    #         total_websocket_time = (time.perf_counter() - websocket_start) * 1000
    #         logger.info("最终识别结果: %s", result_text)
    #         logger.info("耗时统计 - recognize_voice_websocket总耗时: %.2f ms", total_websocket_time)
    #         return result_text if result_text else ""
        
    # except Exception as e:
    #     logger.error("recognize_voice_websocket异常: %s", e, exc_info=True)
    #     raise
    # finally:
    #     # 清理临时文件（如果是临时生成的）
    #     if tmp_wav and os.path.abspath(tmp_wav) != os.path.abspath(audio_path):
    #         try:
    #             os.remove(tmp_wav)
    #         except Exception:
    #             logger.warning("清理临时文件失败")


# def recognize_voice(audio_path: str, hotwords: Optional[List[str]] = None) -> str:
#     """
#     同步包装器（仅用于非事件循环环境）。
#     在FastAPI协程环境中请直接调用 recognize_voice_websocket。
#     """
#     return asyncio.run(recognize_voice_websocket(audio_path, hotwords))


# ---------------- 唤醒接口（唤醒词"小云小云"）----------------
# KWS模型初始化
kws_model = None

def get_models():
    """
    获取模型实例（延迟加载）
    ASR已迁移到Docker容器，只返回KWS模型
    """
    global kws_model
    
    if kws_model is None:
        logger.info("KWS模型未初始化，开始初始化模型")
        
        # 使用本地模型路径，避免从ModelScope下载
        # Docker容器中模型路径：/workspace/models/damo/speech_charctc_kws_phone-xiaoyun/
        # 代码运行在 /workspace/voice-service，所以使用相对路径或绝对路径
        model_dir = "/workspace/models/damo/speech_charctc_kws_phone-xiaohu"
        
        # 如果绝对路径不存在，尝试相对路径（本地开发环境）
        if not os.path.exists(model_dir):
            # 尝试从当前文件位置计算相对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            relative_model_dir = os.path.join(current_dir, "models", "damo", "speech_charctc_kws_phone-xiaohu")
            if os.path.exists(relative_model_dir):
                model_dir = relative_model_dir
                logger.info("使用相对路径模型目录: %s", model_dir)
            else:
                logger.warning("模型目录不存在: %s 和 %s，将尝试从ModelScope下载", model_dir, relative_model_dir)
                # 如果本地路径都不存在，fallback到ModelScope（但应该不会发生）
                model_dir = "iic/speech_charctc_kws_phone-xiaoyun"
        else:
            logger.info("使用本地模型目录: %s", model_dir)
        
        # 初始化KWS模型 - 使用本地模型路径
        kws_model = AutoModel(
            model=model_dir,
            keywords="小护",
            output_dir="./outputs/debug",
            device='cpu',
            disable_update=True,
            disable_pbar=True
        )
        logger.info("KWS模型初始化完成")
    else:
        logger.info("使用已初始化的KWS模型")
    
    # 返回 (asr_model=None因为用HTTP, kws_model)
    return None, kws_model


def kws_wakeup(audio_input: Any = None) -> bool:
    """
    使用 model.generate(input=audio_input, cache={}) 解析关键词唤醒结果。
    返回:
      True  - 唤醒成功（检测到的文本不等于 'rejected' 且非空）
      False - 唤醒失败、解析错误或异常

    注意: audio_input 是 model.generate 能接受的音频输入（例如 wav 路径）。
    """
    kws_wakeup_start = time.perf_counter()
    logger.info("kws_wakeup开始: audio_input=%s", audio_input)
    try:
        # 获取KWS模型实例
        model_get_start = time.perf_counter()
        _, kws_model_instance = get_models()
        model_get_time = (time.perf_counter() - model_get_start) * 1000
        logger.info("耗时统计 - 获取KWS模型: %.2f ms", model_get_time)
        
        generate_start = time.perf_counter()
        res = kws_model_instance.generate(input=audio_input, cache={})
        generate_time = (time.perf_counter() - generate_start) * 1000
        logger.info("KWS模型返回结果: %s", res)
        logger.info("耗时统计 - KWS模型推理: %.2f ms", generate_time)
    except Exception as e:
        # 调用失败视为唤醒失败
        logger.error("KWS模型调用异常: %s", e, exc_info=True)
        return False

    # 基本安全检查：期待 res 是 list/tuple 且第 0 项为 dict，且包含 "text"
    if not (isinstance(res, (list, tuple)) and len(res) > 0 and isinstance(res[0], dict)):
        logger.error("KWS结果格式异常: %s %s", type(res), res)
        return False

    # 提取 text 字段
    wake_field = res[0].get("text", None)
    if wake_field is None:
        logger.info("KWS结果中无'text'字段 -> 唤醒失败")
        return False

    # 兼容 text 可能为字符串或列表的情况
    wake_text = None
    if isinstance(wake_field, str):
        wake_text = wake_field
    elif isinstance(wake_field, (list, tuple)) and len(wake_field) > 0:
        first = wake_field[0]
        if isinstance(first, dict):
            wake_text = first.get("text")
        else:
            wake_text = str(first)
    else:
        # 其它类型一律转字符串处理
        wake_text = str(wake_field)

    logger.info("KWS唤醒文本: %s", wake_text)

    # 最终判断：非空且不等于 'rejected' 则认为唤醒成功
    total_kws_time = (time.perf_counter() - kws_wakeup_start) * 1000
    if wake_text and wake_text != "rejected":
        logger.info("KWS唤醒成功: %s", wake_text)
        logger.info("耗时统计 - kws_wakeup总耗时: %.2f ms", total_kws_time)
        return True
    else:
        logger.info("KWS唤醒失败: %s", wake_text)
        logger.info("耗时统计 - kws_wakeup总耗时: %.2f ms", total_kws_time)
        return False

# 剔除唤醒词（"小云小云"或"小云"）


def remove_xiaoyun(s: str, *, collapse_spaces: bool = True) -> Tuple[str, int]:
    """
    从字符串 s 中移除所有 "小云" 或 "小云小云" 的出现（优先匹配 "小云小云"）。
    返回 (cleaned_string, removed_count)

    参数:
      s: 原始字符串
      collapse_spaces: 是否把移除后的连续空白收缩为单个空格并去除首尾空白（默认 True）
    """
    if not s:
        return s, 0

    # 匹配 "小云" 或连续两次 "小云小云"，优先匹配两次（模式按最长优先）
    pattern = re.compile(r'(?:小云){1,2}')
    cleaned, n = pattern.subn('', s)

    if collapse_spaces:
        # 收缩连续空白为单空格并去掉首尾空白
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned, n


# ---------------- 唤醒&识别 ----------------
async def asr_wake(audio_file: str, hotwords: Optional[List[str]] = SYMS, use_wake: bool = True, use_LLM: bool = True) -> str:
    """
    语音识别主接口 - 恢复KWS唤醒功能
    """
    asr_wake_start = time.perf_counter()
    # 从配置读取全局强制唤醒开关
    from app.config import config
    voice_config = getattr(config, "voice_service", None)
    require_wake = voice_config.voice_require_wake if voice_config else False
    eff_use_wake = require_wake or use_wake
    logger.info("asr_wake开始: audio_file=%s, use_wake=%s, VOICE_REQUIRE_WAKE=%s, eff_use_wake=%s", audio_file, use_wake, require_wake, eff_use_wake)
    
    try:
        # 1. KWS唤醒检测（如果启用）
        if use_wake:
            logger.info("KWS唤醒检测")
            kws_start = time.perf_counter()
            wake_result = kws_wakeup(audio_file)
            kws_time = (time.perf_counter() - kws_start) * 1000
            logger.info("KWS唤醒结果: %s", wake_result)
            logger.info("耗时统计 - KWS唤醒检测: %.2f ms", kws_time)
            if not wake_result:
                logger.info("唤醒失败，返回空结果")
                return ""
            logger.info("唤醒成功，继续ASR识别")
        
        # 2. 语音识别（已废弃：使用说话人分离模型）
        # out = await recognize_voice_websocket(audio_file, hotwords)
        # logger.info("识别结果: %s", out)
        logger.warning("asr_wake 函数中的 recognize_voice_websocket 调用已废弃，请使用新的说话人分离模型")
        out = ""  # 临时返回空，需要重构此函数
        
        # 3. LLM修正（可选）
        if use_LLM and out:
            llm_start = time.perf_counter()
            _, out = process_speech_result(latest_options=hotwords, text=out)
            llm_time = (time.perf_counter() - llm_start) * 1000
            logger.info("LLM修正后的识别结果: %s", out)
            logger.info("耗时统计 - LLM修正: %.2f ms", llm_time)
        
        total_asr_wake_time = (time.perf_counter() - asr_wake_start) * 1000
        logger.info("耗时统计 - asr_wake总耗时: %.2f ms", total_asr_wake_time)
        return out
        
    except Exception as e:
        total_asr_wake_time = (time.perf_counter() - asr_wake_start) * 1000
        logger.error("识别出错: %s (耗时: %.2f ms)", e, total_asr_wake_time, exc_info=True)
        return ""


# ========== 流式处理相关（新增，与旧逻辑隔离） ==========

# 流式处理模型配置
# 使用注册的模型ID + model_path 参数来指定本地模型路径
# 这样可以避免 "is not registered" 错误，同时使用本地已下载的模型
import os

# 容器内模型路径
_MODELS_BASE_DIR = "/workspace/models/damo"
# 本地开发环境路径（fallback）
_LOCAL_MODELS_BASE_DIR = os.path.join(os.path.dirname(__file__), "models", "damo")

def _get_model_path_and_id(local_dir_name: str, registered_model_id: str) -> Tuple[str, str]:
    """
    获取模型路径和注册的模型ID
    
    Args:
        local_dir_name: 本地目录名称（在 damo 目录下）
        registered_model_id: 注册表中的模型ID（用于 AutoModel 的 model 参数）
    
    Returns:
        (model_path, model_id): 模型路径（如果存在）和注册的模型ID
    """
    # 优先尝试容器内路径
    container_path = os.path.join(_MODELS_BASE_DIR, local_dir_name)
    if os.path.exists(container_path):
        logger.info("使用容器内模型路径: %s (注册ID: %s)", container_path, registered_model_id)
        return container_path, registered_model_id
    
    # 尝试本地开发环境路径
    local_path = os.path.join(_LOCAL_MODELS_BASE_DIR, local_dir_name)
    if os.path.exists(local_path):
        logger.info("使用本地模型路径: %s (注册ID: %s)", local_path, registered_model_id)
        return local_path, registered_model_id
    
    # 如果都不存在，使用模型ID（会从ModelScope下载）
    logger.warning("本地模型路径不存在，将使用模型ID（可能从ModelScope下载）: %s", registered_model_id)
    return None, registered_model_id

# VAD模型：使用注册ID + 本地路径（pytorch 版本）
_vad_path, _vad_id = _get_model_path_and_id(
    "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "fsmn-vad"
)
STREAMING_VAD_MODEL = _vad_id
STREAMING_VAD_MODEL_PATH = _vad_path  # 如果为 None，则不使用本地路径

# ASR模型：使用注册ID + 本地路径（pytorch 流式版本）
_asr_path, _asr_id = _get_model_path_and_id(
    "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "paraformer-zh-streaming"  # 保持原注册ID
)
STREAMING_ASR_MODEL = _asr_id
STREAMING_ASR_MODEL_PATH = _asr_path  # 如果为 None，则不使用本地路径

# PUNC模型：使用注册ID + 本地路径（pytorch 版本）
_punc_path, _punc_id = _get_model_path_and_id(
    "punc_ct-transformer_cn-en-common-vocab471067-large",
    "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
)
STREAMING_PUNC_MODEL = _punc_id
STREAMING_PUNC_MODEL_PATH = _punc_path  # 如果为 None，则不使用本地路径

STREAMING_DEVICE = "cuda:0"  # GPU设备

# 流式处理参数
STREAMING_TARGET_SAMPLE_RATE = 16000
STREAMING_FRONTEND_CHUNK_DURATION = 240  # 前端发送片段时长（ms）
STREAMING_SILENCE_THRESHOLD = 2.0  # 静默2秒触发结束
STREAMING_TAIL_PROTECTION_DURATION = 0.5  # 尾音保护时长（秒）：检测到静音后，如果之前有语音，继续累积0.5秒音频
STREAMING_CHUNK_SIZE = [0, 4, 5]  # ASR chunk配置
STREAMING_ENCODER_CHUNK_LOOK_BACK = 4
STREAMING_DECODER_CHUNK_LOOK_BACK = 1

# VAD 能量检测阈值（提高阈值以排除更多噪声）
STREAMING_VAD_ENERGY_THRESHOLD = 0.03  # 从0.03提高到0.05
STREAMING_VAD_MAX_THRESHOLD = 0.17     # 从0.15提高到0.20
STREAMING_VAD_USE_AND_LOGIC = True

# 全局模型实例（延迟加载）
_streaming_vad_model = None
_streaming_asr_model = None
_streaming_punc_model = None
_streaming_models_initialized = False

# 说话人分离模型（全局单例）
_speaker_diarization_pipeline = None
_speaker_diarization_initialized = False


def _load_model_with_local_path(model_id: str, model_path: Optional[str], device: str) -> AutoModel:
    """
    加载模型，优先使用本地路径
    
    Args:
        model_id: 注册表中的模型ID（fallback，当本地路径不存在时使用）
        model_path: 本地模型路径（如果存在）
        device: 设备（cuda:0 或 cpu）
    
    Returns:
        AutoModel 实例
    """
    if model_path and os.path.exists(model_path):
        # 本地路径存在，直接使用路径作为 model 参数（参考 KWS 模型的加载方式）
        logger.info("✅ 使用本地路径加载模型: %s", model_path)
        return AutoModel(
            model=model_path,  # 直接使用本地路径，FunASR 会自动读取 config.yaml
            device=device,
            disable_update=True,
            disable_pbar=True
        )
    else:
        # 本地路径不存在，使用注册ID（会从 ModelScope 下载或使用缓存）
        logger.warning("⚠️ 本地模型路径不存在，使用注册ID加载（可能从 ModelScope 下载或使用缓存）: %s", model_id)
        return AutoModel(
            model=model_id,
            device=device,
            disable_update=True,
            disable_pbar=True
        )


def init_streaming_models():
    """初始化流式处理模型（启动时调用一次，延迟加载）
    
    优先使用本地模型路径（/workspace/models/damo/...），避免重复下载。
    如果本地路径不存在，则使用模型ID（会从 ModelScope 下载或使用缓存）。
    """
    global _streaming_vad_model, _streaming_asr_model, _streaming_punc_model, _streaming_models_initialized
    
    if _streaming_models_initialized:
        logger.info("流式处理模型已初始化，跳过重复加载")
        return
    
    try:
        logger.info("正在加载流式处理模型...")
        logger.info("VAD模型ID: %s, 本地路径: %s", STREAMING_VAD_MODEL, STREAMING_VAD_MODEL_PATH)
        logger.info("ASR模型ID: %s, 本地路径: %s", STREAMING_ASR_MODEL, STREAMING_ASR_MODEL_PATH)
        logger.info("PUNC模型ID: %s, 本地路径: %s", STREAMING_PUNC_MODEL, STREAMING_PUNC_MODEL_PATH)
        logger.info("设备: %s", STREAMING_DEVICE)
        
        # 使用 _load_model_with_local_path 优先使用本地路径，避免重复下载
        _streaming_vad_model = _load_model_with_local_path(
            model_id=STREAMING_VAD_MODEL,
            model_path=STREAMING_VAD_MODEL_PATH,
            device=STREAMING_DEVICE
        )
        logger.info("✅ VAD模型加载完成")
        
        _streaming_asr_model = _load_model_with_local_path(
            model_id=STREAMING_ASR_MODEL,
            model_path=STREAMING_ASR_MODEL_PATH,
            device=STREAMING_DEVICE
        )
        logger.info("✅ ASR模型加载完成")
        
        _streaming_punc_model = _load_model_with_local_path(
            model_id=STREAMING_PUNC_MODEL,
            model_path=STREAMING_PUNC_MODEL_PATH,
            device=STREAMING_DEVICE
        )
        logger.info("✅ PUNC模型加载完成")
        
        _streaming_models_initialized = True
        logger.info("✅ 所有流式处理模型加载完成")
        
    except Exception as e:
        logger.error("❌ 流式处理模型加载失败: %s", e, exc_info=True)
        raise


def get_streaming_models():
    """获取流式处理模型实例（如果未初始化则先初始化）"""
    if not _streaming_models_initialized:
        init_streaming_models()
    return _streaming_vad_model, _streaming_asr_model, _streaming_punc_model


def _init_speaker_diarization_pipeline_global():
    """初始化说话人分离模型（全局单例，优先使用本地路径）"""
    global _speaker_diarization_pipeline, _speaker_diarization_initialized
    
    if _speaker_diarization_initialized:
        logger.debug("说话人分离模型已初始化，跳过重复加载")
        return _speaker_diarization_pipeline
    
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        logger.info("正在初始化说话人分离模型...")
        
        # 优先使用本地模型路径，避免从ModelScope下载
        # 容器内路径：/workspace/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn
        # 本地开发路径：app/services/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn
        diarization_model_id = 'iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn'
        diarization_model_revision = 'v2.0.4'
        
        # 尝试容器内路径
        container_path = "/workspace/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn"
        # 尝试本地开发路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(current_dir, "models", "damo", "speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn")
        
        diarization_model_path = None
        if os.path.exists(container_path):
            diarization_model_path = container_path
            logger.info("✅ 使用容器内说话人分离模型路径: %s", container_path)
        elif os.path.exists(local_path):
            diarization_model_path = local_path
            logger.info("✅ 使用本地说话人分离模型路径: %s", local_path)
        else:
            logger.warning("⚠️ 本地说话人分离模型路径不存在，将使用模型ID（可能从ModelScope下载）: %s", diarization_model_id)
        
        # VAD 模型路径（复用已有的逻辑）
        vad_model_id = 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'
        vad_model_revision = "v2.0.4"
        vad_container_path = "/workspace/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        vad_local_path = os.path.join(current_dir, "models", "damo", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
        
        vad_model_path = None
        if os.path.exists(vad_container_path):
            vad_model_path = vad_container_path
            logger.info("✅ 使用容器内VAD模型路径: %s", vad_container_path)
        elif os.path.exists(vad_local_path):
            vad_model_path = vad_local_path
            logger.info("✅ 使用本地VAD模型路径: %s", vad_local_path)
        else:
            logger.warning("⚠️ 本地VAD模型路径不存在，将使用模型ID: %s", vad_model_id)
        
        # PUNC 模型路径（复用已有的逻辑）
        punc_model_id = 'iic/punc_ct-transformer_cn-en-common-vocab471067-large'
        punc_model_revision = "v2.0.4"
        punc_container_path = "/workspace/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large"
        punc_local_path = os.path.join(current_dir, "models", "damo", "punc_ct-transformer_cn-en-common-vocab471067-large")
        
        punc_model_path = None
        if os.path.exists(punc_container_path):
            punc_model_path = punc_container_path
            logger.info("✅ 使用容器内PUNC模型路径: %s", punc_container_path)
        elif os.path.exists(punc_local_path):
            punc_model_path = punc_local_path
            logger.info("✅ 使用本地PUNC模型路径: %s", punc_local_path)
        else:
            logger.warning("⚠️ 本地PUNC模型路径不存在，将使用模型ID: %s", punc_model_id)
        
        # ModelScope pipeline 支持直接传递本地路径作为 model 参数
        model_param = diarization_model_path if diarization_model_path else diarization_model_id
        vad_model_param = vad_model_path if vad_model_path else vad_model_id
        punc_model_param = punc_model_path if punc_model_path else punc_model_id
        
        _speaker_diarization_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=model_param,
            model_revision=diarization_model_revision if not diarization_model_path else None,  # 本地路径不需要revision
            vad_model=vad_model_param,
            vad_model_revision=vad_model_revision if not vad_model_path else None,  # 本地路径不需要revision
            punc_model=punc_model_param,
            punc_model_revision=punc_model_revision if not punc_model_path else None,  # 本地路径不需要revision
            output_dir="./results",
        )
        _speaker_diarization_initialized = True
        logger.info("✅ 说话人分离模型已加载 (model=%s, vad_model=%s, punc_model=%s)", 
                   model_param, vad_model_param, punc_model_param)
        return _speaker_diarization_pipeline
    except Exception as e:
        logger.error(f"❌ 说话人分离模型加载失败：{e}", exc_info=True)
        raise


def get_speaker_diarization_pipeline():
    """获取说话人分离模型实例（如果未初始化则先初始化）"""
    if not _speaker_diarization_initialized:
        _init_speaker_diarization_pipeline_global()
    return _speaker_diarization_pipeline


def init_speaker_diarization_model():
    """初始化说话人分离模型（启动时调用一次，延迟加载）
    
    优先使用本地模型路径（/workspace/models/damo/...），避免重复下载。
    如果本地路径不存在，则使用模型ID（会从 ModelScope 下载或使用缓存）。
    """
    try:
        logger.info("正在预加载说话人分离模型...")
        _init_speaker_diarization_pipeline_global()
        logger.info("✅ 说话人分离模型预加载完成")
    except Exception as e:
        logger.error("❌ 说话人分离模型预加载失败: %s", e, exc_info=True)
        logger.warning("⚠️ 服务将继续启动，但首次使用时可能需要等待模型加载")


def _log_audio_statistics(audio_np: np.ndarray, sample_rate: int, context: str = ""):
    """
    打印音频统计信息，用于诊断动态范围和饱和问题
    
    Args:
        audio_np: 音频数组（float32，范围通常为 [-1, 1]）
        sample_rate: 采样率
        context: 上下文描述（例如："base64解码后"、"模型输入前"）
    """
    if len(audio_np) == 0:
        logger.warning(f"[音频统计{context}] 音频数组为空")
        return
    
    # 基本属性
    dtype = audio_np.dtype
    shape = audio_np.shape
    duration_s = len(audio_np) / sample_rate if sample_rate > 0 else 0.0
    
    # 统计值
    audio_max = float(np.max(audio_np))
    audio_min = float(np.min(audio_np))
    audio_abs_max = float(np.max(np.abs(audio_np)))
    
    # RMS（均方根）
    rms = float(np.sqrt(np.mean(audio_np ** 2)) + 1e-10)
    
    # Clipping ratio：|x| >= 0.999 的比例（接近饱和的比例）
    clipping_mask = np.abs(audio_np) >= 0.999
    clipping_ratio = float(np.sum(clipping_mask) / len(audio_np))
    
    # Dynamic range：max / (rms + 1e-10)
    dynamic_range = audio_abs_max / (rms + 1e-10) if rms > 0 else 0.0
    
    # 正负峰值对称性（正峰值 - |负峰值|）
    peak_symmetry = audio_max - abs(audio_min)
    
    # 打印统计信息
    logger.info(
        f"📊 [音频统计{context}] "
        f"dtype={dtype}, shape={shape}, sample_rate={sample_rate}Hz, "
        f"duration={duration_s:.3f}s, "
        f"max={audio_max:.6f}, min={audio_min:.6f}, "
        f"RMS={rms:.6f}, "
        f"clipping_ratio={clipping_ratio:.2%} (|x|>=0.999), "
        f"dynamic_range={dynamic_range:.2f}, "
        f"peak_symmetry={peak_symmetry:.6f} (max-|min|)"
    )
    
    # 如果 clipping 比例较高，记录警告
    if clipping_ratio > 0.01:  # 1%
        logger.warning(
            f"⚠️ [音频统计{context}] 检测到高clipping比例: {clipping_ratio:.2%} "
            f"(max={audio_max:.6f}, abs_max={audio_abs_max:.6f})"
        )
    
    return {
        "dtype": dtype,
        "shape": shape,
        "sample_rate": sample_rate,
        "duration_s": duration_s,
        "max": audio_max,
        "min": audio_min,
        "rms": rms,
        "clipping_ratio": clipping_ratio,
        "dynamic_range": dynamic_range,
        "peak_symmetry": peak_symmetry
    }


def _dump_clipped_audio(audio_np: np.ndarray, sample_rate: int, context: str = ""):
    """
    当检测到高clipping比例时，dump音频为WAV文件到临时目录
    
    Args:
        audio_np: 音频数组（float32，范围 [-1, 1]）
        sample_rate: 采样率
        context: 上下文描述
    """
    try:
        import datetime
        from pathlib import Path
        
        # 创建临时目录
        temp_dir = Path("/tmp") / "voice_service_debug_audio"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"clipped_audio_{context}_{timestamp}.wav"
        wav_path = temp_dir / filename
        
        # 转换为int16并保存（只做必要的clamp，不做归一化）
        # 确保在 [-1, 1] 范围内，然后转换为 [-32768, 32767]
        audio_clamped = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_clamped * 32767.0).astype(np.int16)
        
        # 保存WAV文件
        with wave.open(str(wav_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16-bit (2 bytes)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        logger.info(f"💾 [音频dump{context}] 已保存clipping音频到: {wav_path}")
        return str(wav_path)
    except Exception as e:
        logger.error(f"❌ [音频dump{context}] 保存音频文件失败: {e}", exc_info=True)
        return None


def base64_to_audio_np(base64_str: str) -> Tuple[np.ndarray, int]:
    """
    输入：前端传来的 WAV base64 字符串（前端已转换为WAV格式）
    输出：(模型可识别的 float32 音频数组, 16kHz采样率)
    优先使用 wave 模块（无需外部依赖），如果失败则尝试 torchaudio
    """
    try:
        # 步骤1：base64 解码 → 二进制音频数据
        audio_bytes = base64.b64decode(base64_str.strip())
        if not audio_bytes:
            raise ValueError("base64 解码后为空")

        # 步骤2：优先尝试使用 wave 模块解析 WAV（无需外部依赖）
        try:
            with wave.open(BytesIO(audio_bytes), "rb") as wf:
                orig_sr = wf.getframerate()
                orig_ch = wf.getnchannels()
                orig_sw = wf.getsampwidth()
                n_frames = wf.getnframes()
                wav_data = wf.readframes(n_frames)

            # 步骤3：二进制 → numpy 数组（按位深归一化到 [-1, 1]）
            if orig_sw == 1:
                audio_np = np.frombuffer(wav_data, dtype=np.uint8)
                audio_np = (audio_np - 128) / 128.0
            elif orig_sw == 2:
                audio_np = np.frombuffer(wav_data, dtype=np.int16)
                audio_np = audio_np / 32768.0
            elif orig_sw == 4:
                audio_np = np.frombuffer(wav_data, dtype=np.int32)
                audio_np = audio_np / 2147483648.0
            else:
                raise ValueError(f"不支持的位深：{orig_sw} 字节")

            # 步骤4：多通道 → 单通道
            if orig_ch > 1:
                audio_np = np.mean(audio_np.reshape(-1, orig_ch), axis=1)

            # 步骤5：重采样 → 16kHz（如果需要）
            if orig_sr != STREAMING_TARGET_SAMPLE_RATE:
                # 使用 scipy.signal.resample 进行重采样（如果可用）
                try:
                    from scipy import signal
                    num_samples = int(len(audio_np) * STREAMING_TARGET_SAMPLE_RATE / orig_sr)
                    audio_np = signal.resample(audio_np, num_samples)
                except ImportError:
                    # 如果没有 scipy，使用简单的线性插值（numpy实现）
                    old_length = len(audio_np)
                    new_length = int(old_length * STREAMING_TARGET_SAMPLE_RATE / orig_sr)
                    old_indices = np.linspace(0, old_length - 1, old_length)
                    new_indices = np.linspace(0, old_length - 1, new_length)
                    audio_np = np.interp(new_indices, old_indices, audio_np)
                except Exception as resample_error:
                    # 如果重采样失败，直接使用原采样率（模型可能也能处理）
                    logger.warning("无法重采样，使用原始采样率 %sHz: %s", orig_sr, resample_error)
                    audio_np = audio_np.astype(np.float32)
                    # 在返回前添加统计日志
                    stats = _log_audio_statistics(audio_np, orig_sr, "base64解码后")
                    if stats and stats.get("clipping_ratio", 0) > 0.01:
                        _dump_clipped_audio(audio_np, orig_sr, "base64_decode")
                    return audio_np, orig_sr

            # 步骤6：确保数据类型
            audio_np = audio_np.astype(np.float32)
            # 在返回前添加统计日志
            stats = _log_audio_statistics(audio_np, STREAMING_TARGET_SAMPLE_RATE, "base64解码后")
            if stats and stats.get("clipping_ratio", 0) > 0.01:
                _dump_clipped_audio(audio_np, STREAMING_TARGET_SAMPLE_RATE, "base64_decode")
            return audio_np, STREAMING_TARGET_SAMPLE_RATE

        except Exception as wave_error:
            # 如果 wave 模块解析失败，尝试使用 torchaudio（需要 torchaudio 库）
            logger.warning("wave 模块解析失败，尝试使用 torchaudio: %s", wave_error)
            try:
                import torch
                import torchaudio
                audio_stream = BytesIO(audio_bytes)
                waveform, sample_rate = torchaudio.load(
                    audio_stream,
                    format=None,  # 自动识别格式
                    backend="soundfile"  # 用 soundfile 后端
                )
                
                # 转单通道
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # 重采样
                if sample_rate != STREAMING_TARGET_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=STREAMING_TARGET_SAMPLE_RATE,
                        dtype=waveform.dtype
                    )
                    waveform = resampler(waveform)
                    sample_rate = STREAMING_TARGET_SAMPLE_RATE
                
                audio_np = waveform.squeeze().numpy().astype(np.float32)
                # 在返回前添加统计日志
                stats = _log_audio_statistics(audio_np, STREAMING_TARGET_SAMPLE_RATE, "base64解码后(torchaudio)")
                if stats and stats.get("clipping_ratio", 0) > 0.01:
                    _dump_clipped_audio(audio_np, STREAMING_TARGET_SAMPLE_RATE, "base64_decode_torchaudio")
                return audio_np, STREAMING_TARGET_SAMPLE_RATE
            except Exception as torch_error:
                raise RuntimeError(
                    f"音频处理失败：wave模块错误={wave_error}, torchaudio错误={torch_error}。"
                    f"请确保前端发送标准WAV格式（16kHz单声道16bit），或安装scipy/torchaudio。"
                )

    except Exception as e:
        raise RuntimeError(f"音频处理失败：{str(e)}")


class StreamingASRSession:
    """流式ASR会话状态管理（每个WebSocket连接一个实例）"""
    
    def __init__(self):
        # ASR 相关状态
        self.vad_cache = {}
        self.asr_cache = {}
        self.audio_buffer = np.array([], dtype=np.float32)
        self.accumulated_intermediate_text = ""
        self.silence_timer = 0.0
        self.last_voice_time = time.time()
        self.is_completed = False
        self.has_detected_speech = False  # 标记是否曾经检测到过语音（用于防止纯静音触发finalize）
        self.silence_chunk_count = 0  # 已累积的静音chunk数量（最多保留2个静音chunk）
        
        # 尾音保护机制相关状态
        self.tail_protection_start_time = None  # 尾音保护期开始时间（None表示未进入保护期）
        
        # 前向保护机制相关状态（防止丢失语音开头）
        self.pre_speech_buffer = np.array([], dtype=np.float32)  # 前向保护缓冲区（累积检测到语音之前的chunk）
        self.pre_speech_max_duration = 0.4  # 前向保护最大时长（400ms，保留1个chunk）
        
        # KWS 唤醒相关状态
        self.mode = "WAITING_FOR_WAKEUP"  # "WAITING_FOR_WAKEUP" 或 "WAITING_FOR_ENROLLMENT" 或 "WAITING_FOR_ENROLLMENT_CONFIRM" 或 "ASR_ACTIVE"
        self.kws_cache = {}  # KWS 模型的 cache（用于流式检测）
        self.kws_vad_cache = {}  # KWS 模式下 VAD 的 cache（与 ASR 模式的 VAD cache 分开）
        self.use_wake = True  # 是否启用唤醒模式（默认启用）
        self.is_activated = False  # 是否已通过 KWS 激活（用于控制是否发送 ASR 结果）
        # KWS 音频累积相关（滑动窗口，固定1600ms = 4个400ms chunk）
        self.kws_audio_buffer = np.array([], dtype=np.float32)  # KWS 音频累积缓冲区（滑动窗口）
        self.kws_min_duration = 1.6  # KWS 检测所需的最小音频长度（秒），1600ms = 4个400ms chunk
        
        # 热词配置（使用 SYMS 作为默认热词列表，与 asr_wake 保持一致）
        self.hotwords: Optional[List[str]] = SYMS  # 会话级别的热词列表
        
        # SV 声纹识别相关状态
        self.sv_pipeline = None  # 声纹识别管道（延迟加载）
        # 说话人分离模型使用全局单例，不需要实例变量
        self.enroll_audio_path: Optional[str] = None  # 注册样本文件路径
        self.enroll_audio_buffer = np.array([], dtype=np.float32)  # 注册音频缓冲区
        self.is_enrolled = False  # 是否已注册
        self.min_enroll_seconds = 5.0  # 注册要求的最短时长（5秒）
        self.enroll_has_detected_speech = False  # 标记是否在enrollment模式下检测到过语音（用于控制何时开始累积）
        self.enroll_first_speech_time = None  # 第一次检测到语音的时间（用于计算从语音开始累积的时长）
        self.sv_threshold = 0.40  # 声纹判定阈值（可调整：值越小越宽松，值越大越严格）
        self.use_speaker_verification = True  # 是否启用声纹验证（默认启用，用于测试）
        
        # 实验性：chunk级别的声纹验证缓冲区（用于实时验证实验）
        self.experimental_sv_buffer = np.array([], dtype=np.float32)  # 实验性验证缓冲区（当前chunk）
        self.experimental_sv_accumulated_buffer = np.array([], dtype=np.float32)  # 累积验证缓冲区（1+2+3+...）
        self.experimental_sv_min_duration = 1.0  # 实验性验证所需的最小音频长度（秒），累积到这么长才验证
        self.experimental_sv_last_verify_time = 0.0  # 上次验证的时间戳（用于控制验证频率）
        self.experimental_sv_verify_interval = 0.4  # 验证间隔（秒），避免过于频繁验证
        
    def reset(self):
        """重置会话状态（准备下一轮识别）
        
        注意：根据 use_wake 决定重置后的模式：
        - use_wake=True: 重置到等待唤醒模式
        - use_wake=False: 重置到 ASR 激活模式（不需要唤醒）
        """
        self.vad_cache = {}
        self.asr_cache = {}
        self.audio_buffer = np.array([], dtype=np.float32)
        self.accumulated_intermediate_text = ""
        self.silence_timer = 0.0
        self.last_voice_time = time.time()
        self.is_completed = False
        self.has_detected_speech = False  # 标记是否曾经检测到过语音（用于防止纯静音触发finalize）
        self.silence_chunk_count = 0  # 重置静音chunk计数器
        self.pre_speech_buffer = np.array([], dtype=np.float32)  # 重置前向保护缓冲区
        
        # 重置尾音保护状态
        self.tail_protection_start_time = None
        
        # 重置激活状态
        old_activated = self.is_activated
        self.is_activated = False  # 重置激活标记
        if old_activated:
            logger.info("🔄 [状态重置] KWS 激活状态已清除: True -> False")
        
        # 完全重置 SV 声纹识别状态（包括注册状态）
        # 注意：对话结束后，用户离开，下一次新用户来需要重新注册声纹
        old_enrolled = self.is_enrolled
        old_enroll_path = self.enroll_audio_path
        old_enroll_buffer_len = len(self.enroll_audio_buffer)
        self.is_enrolled = False  # 清空注册状态
        self.enroll_audio_path = None  # 清空注册样本路径
        self.enroll_audio_buffer = np.array([], dtype=np.float32)  # 清空注册缓冲区
        self.enroll_has_detected_speech = False  # 重置enrollment语音检测标记
        self.enroll_first_speech_time = None  # 重置enrollment首次语音时间
        if old_enrolled or old_enroll_path or old_enroll_buffer_len > 0:
            logger.info("🔄 [状态重置] SV 声纹注册状态已完全清除: is_enrolled=%s->False, enroll_audio_path=%s, buffer=%d样本 (%.2fs)", 
                       old_enrolled, old_enroll_path, old_enroll_buffer_len, 
                       old_enroll_buffer_len / STREAMING_TARGET_SAMPLE_RATE if old_enroll_buffer_len > 0 else 0.0)
        
        # 重置实验性SV验证缓冲区
        old_experimental_sv_buffer_len = len(self.experimental_sv_buffer)
        old_accumulated_buffer_len = len(self.experimental_sv_accumulated_buffer)
        self.experimental_sv_buffer = np.array([], dtype=np.float32)
        self.experimental_sv_accumulated_buffer = np.array([], dtype=np.float32)  # 清空累积缓冲区
        self.experimental_sv_last_verify_time = 0.0
        if old_experimental_sv_buffer_len > 0 or old_accumulated_buffer_len > 0:
            logger.info("🔄 [状态重置] 实验性SV验证缓冲区已清空: chunk=%d样本 (%.2fs), 累积=%d样本 (%.2fs)", 
                       old_experimental_sv_buffer_len, old_experimental_sv_buffer_len / STREAMING_TARGET_SAMPLE_RATE,
                       old_accumulated_buffer_len, old_accumulated_buffer_len / STREAMING_TARGET_SAMPLE_RATE)
        
        # 根据 use_wake 决定重置后的模式
        if self.use_wake:
            self.mode = "WAITING_FOR_WAKEUP"
            old_kws_buffer_len = len(self.kws_audio_buffer)
            self.kws_cache = {}
            self.kws_vad_cache = {}  # 清空 KWS 模式的 VAD cache（虽然不再使用，但保留定义）
            # 重置 KWS 音频累积相关状态
            self.kws_audio_buffer = np.array([], dtype=np.float32)
            if old_kws_buffer_len > 0:
                logger.info("🔄 [状态重置] KWS 音频缓冲区已清空: %d 样本 (%.2fs), cache已清除", 
                           old_kws_buffer_len, old_kws_buffer_len / STREAMING_TARGET_SAMPLE_RATE)
        else:
            self.mode = "ASR_ACTIVE"  # 不启用唤醒，直接进入 ASR 模式
            old_kws_buffer_len = len(self.kws_audio_buffer)
            self.kws_cache = {}
            self.kws_vad_cache = {}  # 清空 KWS 模式的 VAD cache（虽然不再使用，但保留定义）
            # 清空 KWS 音频累积相关状态
            self.kws_audio_buffer = np.array([], dtype=np.float32)
            if old_kws_buffer_len > 0:
                logger.info("🔄 [状态重置] KWS 音频缓冲区已清空: %d 样本 (%.2fs), cache已清除", 
                           old_kws_buffer_len, old_kws_buffer_len / STREAMING_TARGET_SAMPLE_RATE)
    
    def set_use_wake(self, use_wake: bool):
        """
        设置是否启用唤醒模式
        
        Args:
            use_wake: True 表示启用唤醒模式，False 表示直接进入 ASR 模式
        """
        self.use_wake = use_wake
        if not use_wake:
            # 如果禁用唤醒，直接切换到 ASR 模式
            if self.mode == "WAITING_FOR_WAKEUP":
                self.mode = "ASR_ACTIVE"
                self.kws_cache = {}
                logger.info("已禁用唤醒模式，切换到 ASR 模式")
        else:
            # 如果启用唤醒，且当前在 ASR 模式，切换到等待唤醒模式（取消激活状态）
            # 这是前端主动要求取消激活状态，无论 is_completed 状态如何
            if self.mode == "ASR_ACTIVE":
                old_activated = self.is_activated
                old_kws_buffer_len = len(self.kws_audio_buffer)
                self.mode = "WAITING_FOR_WAKEUP"
                self.kws_cache = {}
                self.kws_vad_cache = {}
                self.kws_audio_buffer = np.array([], dtype=np.float32)
                self.is_activated = False  # 取消激活状态
                logger.info("🔄 [状态切换] 已启用唤醒模式，取消激活状态，切换到等待唤醒模式")
                if old_activated:
                    logger.info("🔄 [状态清除] KWS 激活状态已清除: True -> False")
                if old_kws_buffer_len > 0:
                    logger.info("🔄 [状态清除] KWS 音频缓冲区已清空: %d 样本 (%.2fs), cache已清除", 
                               old_kws_buffer_len, old_kws_buffer_len / STREAMING_TARGET_SAMPLE_RATE)
    
    def reset_asr_state(self):
        """
        只重置 ASR 相关状态，不改变模式和激活状态
        用于 finalize() 后准备下一句话的识别
        """
        self.vad_cache = {}
        self.asr_cache = {}
        self.audio_buffer = np.array([], dtype=np.float32)
        self.accumulated_intermediate_text = ""
        self.silence_timer = 0.0
        self.last_voice_time = time.time()
        self.is_completed = False
        self.has_detected_speech = False  # 标记是否曾经检测到过语音（用于防止纯静音触发finalize）
        self.silence_chunk_count = 0  # 重置静音chunk计数器
        self.pre_speech_buffer = np.array([], dtype=np.float32)  # 重置前向保护缓冲区
        self.tail_protection_start_time = None  # 重置尾音保护状态
        # 注意：不重置 is_activated，保持激活状态
        logger.debug("已重置 ASR 状态，准备下一句话识别（模式: %s, 激活: %s）", 
                    self.mode, self.is_activated)
    
    async def process_wakeup_chunk(self, audio_np: np.ndarray) -> bool:
        """
        简化的KWS唤醒检测（滑动窗口，固定1600ms = 4个400ms chunk）
        
        流程：
        1. 直接累积所有chunk到kws_audio_buffer（不依赖VAD判断是否有声音）
        2. 滑动窗口：如果超过1600ms（4个chunk），只保留最新的1600ms（FIFO队列）
        3. 如果达到1600ms（4个chunk），立即触发KWS检测
        4. KWS模型自己会判断音频中是否有唤醒词
        
        Args:
            audio_np: numpy数组，float32，16kHz，单声道（前端发送的400ms chunk）
            
        Returns:
            bool: True 表示检测到唤醒词，False 表示未检测到
        """
        try:
            # 1. 直接累积所有chunk（不依赖VAD判断是否有声音）
            # KWS模型自己会判断音频中是否有唤醒词
            
            # ⚠️ 音量检测和日志（用于调试音量不一致问题）
            audio_energy = np.mean(np.abs(audio_np))
            audio_max = np.max(np.abs(audio_np))
            audio_rms = np.sqrt(np.mean(audio_np ** 2))
            logger.debug("KWS chunk音量检测: energy=%.6f, max=%.6f, rms=%.6f, len=%d样本 (%.2fms)", 
                       audio_energy, audio_max, audio_rms, len(audio_np), len(audio_np) / STREAMING_TARGET_SAMPLE_RATE * 1000)
            
            self.kws_audio_buffer = np.concatenate([self.kws_audio_buffer, audio_np])
            
            # 2. 滑动窗口：如果超过1600ms（4个chunk），只保留最新的1600ms（FIFO队列）
            # 新来一个chunk，如果超过1600ms，丢掉最旧的chunk，保留最新的1600ms
            target_samples = int(self.kws_min_duration * STREAMING_TARGET_SAMPLE_RATE)  # 1600ms = 25600 samples
            if len(self.kws_audio_buffer) > target_samples:
                old_buffer_len = len(self.kws_audio_buffer)
                self.kws_audio_buffer = self.kws_audio_buffer[-target_samples:]  # 只保留最新的1600ms
                logger.debug("KWS滑动窗口 - 超过1600ms，保留最新1600ms: %d样本 -> %d样本 (%.2fs -> %.2fs)", 
                           old_buffer_len, len(self.kws_audio_buffer),
                           old_buffer_len / STREAMING_TARGET_SAMPLE_RATE,
                           len(self.kws_audio_buffer) / STREAMING_TARGET_SAMPLE_RATE)
            
            # 3. 检查是否达到1600ms（4个chunk，触发检测）
            buffer_duration = len(self.kws_audio_buffer) / STREAMING_TARGET_SAMPLE_RATE
            if buffer_duration >= self.kws_min_duration:  # 1600ms
                logger.info("KWS 音频累积达到 1600ms (%.2fs, 4个chunk)，触发检测", buffer_duration)
                return await self._perform_kws_detection()
            
            # 4. 如果还没有达到1600ms，继续等待
            logger.debug("KWS模式 - 累积音频: 当前长度=%.2fs (需要≥%.2fs才检测，当前chunk数: %.1f)", 
                       buffer_duration, self.kws_min_duration, buffer_duration / 0.4)
            return False
                
        except Exception as e:
            logger.error("KWS 唤醒检测异常: %s", e, exc_info=True)
            # 异常时清空缓冲区，避免无限累积
            old_buffer_len = len(self.kws_audio_buffer)
            self.kws_audio_buffer = np.array([], dtype=np.float32)
            if old_buffer_len > 0:
                logger.info("🔄 [KWS清除] 异常时 KWS 音频缓冲区已清空: %d 样本 (%.2fs)", 
                           old_buffer_len, old_buffer_len / STREAMING_TARGET_SAMPLE_RATE)
            return False
    
    async def _perform_kws_detection(self) -> bool:
        """
        执行 KWS 检测（使用累积的音频）
        
        Returns:
            bool: True 表示检测到唤醒词，False 表示未检测到
        """
        if len(self.kws_audio_buffer) == 0:
            return False
        
        try:
            # 获取 KWS 模型实例
            _, kws_model_instance = get_models()
            
            if kws_model_instance is None:
                logger.error("KWS 模型实例为 None，无法进行唤醒检测")
                return False
            
            buffer_duration = len(self.kws_audio_buffer) / STREAMING_TARGET_SAMPLE_RATE
            
            # ⚠️ 音量检测和日志（用于调试KWS检测不稳定问题）
            buffer_energy = np.mean(np.abs(self.kws_audio_buffer))
            buffer_max = np.max(np.abs(self.kws_audio_buffer))
            buffer_rms = np.sqrt(np.mean(self.kws_audio_buffer ** 2))
            buffer_peak_db = 20 * np.log10(buffer_max + 1e-10)  # 避免log(0)
            buffer_rms_db = 20 * np.log10(buffer_rms + 1e-10)
            
            logger.info("KWS 检测 - 输入音频: shape=%s, 时长=%.2fs, energy=%.6f, max=%.6f (%.2f dB), rms=%.6f (%.2f dB)", 
                       self.kws_audio_buffer.shape, buffer_duration,
                       buffer_energy, buffer_max, buffer_peak_db, buffer_rms, buffer_rms_db)
            
            # 调用 KWS 模型（使用累积的完整音频，不使用 cache，因为这是完整的一段）
            res = kws_model_instance.generate(
                input=self.kws_audio_buffer,
                cache={},  # 每次检测使用新的 cache，因为这是完整的一段音频
                is_final=True  # 使用 is_final=True 表示这是完整的一段
            )
            
            logger.debug("KWS 检测 - 返回结果类型: %s, 内容: %s", type(res), res)
            
            # 解析 KWS 结果
            if not (isinstance(res, (list, tuple)) and len(res) > 0):
                logger.debug("KWS 结果格式异常: 不是列表或元组，或为空")
                # 清空 buffer，准备下一轮
                old_buffer_len = len(self.kws_audio_buffer)
                self.kws_audio_buffer = np.array([], dtype=np.float32)
                self.kws_cache = {}  # 清空 cache
                if old_buffer_len > 0:
                    logger.info("🔄 [KWS清除] KWS 结果格式异常，已清空缓冲区: %d 样本 (%.2fs), cache已清除", 
                               old_buffer_len, old_buffer_len / STREAMING_TARGET_SAMPLE_RATE)
                return False
            
            if not isinstance(res[0], dict):
                logger.debug("KWS 结果格式异常: 第一个元素不是字典，类型: %s", type(res[0]))
                old_buffer_len = len(self.kws_audio_buffer)
                self.kws_audio_buffer = np.array([], dtype=np.float32)
                self.kws_cache = {}
                if old_buffer_len > 0:
                    logger.info("🔄 [KWS清除] KWS 结果格式异常，已清空缓冲区: %d 样本 (%.2fs), cache已清除", 
                               old_buffer_len, old_buffer_len / STREAMING_TARGET_SAMPLE_RATE)
                return False
            
            # 提取 text 字段
            wake_field = res[0].get("text", None)
            if wake_field is None:
                logger.debug("KWS 结果中无 'text' 字段，keys: %s", list(res[0].keys()) if isinstance(res[0], dict) else "N/A")
                old_buffer_len = len(self.kws_audio_buffer)
                self.kws_audio_buffer = np.array([], dtype=np.float32)
                self.kws_cache = {}
                if old_buffer_len > 0:
                    logger.info("🔄 [KWS清除] KWS 结果中无 'text' 字段，已清空缓冲区: %d 样本 (%.2fs), cache已清除", 
                               old_buffer_len, old_buffer_len / STREAMING_TARGET_SAMPLE_RATE)
                return False
            
            # 兼容 text 可能为字符串或列表的情况
            wake_text = None
            if isinstance(wake_field, str):
                wake_text = wake_field
            elif isinstance(wake_field, (list, tuple)) and len(wake_field) > 0:
                first = wake_field[0]
                if isinstance(first, dict):
                    wake_text = first.get("text")
                else:
                    wake_text = str(first)
            else:
                wake_text = str(wake_field)
            
            logger.info("KWS 检测 - 提取的唤醒文本: '%s' (音频长度: %.2fs)", wake_text, buffer_duration)
            
            # 判断是否唤醒成功：非空且不等于 'rejected'
            if wake_text and wake_text != "rejected":
                logger.info("✅ KWS 唤醒成功: '%s' (音频长度: %.2fs) - 将切换到ASR_ACTIVE模式，当前chunk将被跳过", wake_text, buffer_duration)
                
                # 唤醒成功，先保存音频（在清空 buffer 之前）
                await self._save_kws_audio()
                
                # ✅ 修改：KWS唤醒后，不再转移buffer到enroll_audio_buffer，直接清空KWS buffer
                # enroll_audio_buffer将从WAITING_FOR_ENROLLMENT模式开始，通过VAD检测到声音后才开始累积
                old_kws_buffer_len = len(self.kws_audio_buffer)
                if old_kws_buffer_len > 0:
                    logger.info("🔄 [KWS清除] KWS唤醒成功，清空KWS音频缓冲区: %d样本 (%.2fs)", 
                               old_kws_buffer_len, buffer_duration)
                
                # 保存后清空 KWS buffer 和 cache，准备下一轮
                self.kws_audio_buffer = np.array([], dtype=np.float32)
                self.kws_cache = {}  # 清空 KWS 模型 cache
                self.kws_vad_cache = {}  # 清空 KWS 模式的 VAD cache
                logger.info("🔄 [KWS清除] KWS 唤醒成功，已保存音频并清空KWS缓冲区: %d 样本 (%.2fs), cache已清除", 
                           old_kws_buffer_len, buffer_duration)
                
                # ✅ 关键修复：KWS成功瞬间，清空所有ASR相关状态
                # 确保第一句话（包含唤醒词）不参与ASR识别
                old_audio_buffer_len = len(self.audio_buffer)
                old_accumulated_text = self.accumulated_intermediate_text
                self.audio_buffer = np.array([], dtype=np.float32)  # 清空ASR音频缓冲区
                self.vad_cache = {}  # 清空ASR依赖的VAD cache
                self.asr_cache = {}  # 清空ASR cache
                self.accumulated_intermediate_text = ""  # 清空累积的中间文本
                self.silence_timer = 0.0  # 重置静默计时器
                self.last_voice_time = time.time()  # 重置最后语音时间
                self.tail_protection_start_time = None  # 清空尾音保护状态
                self.is_completed = False  # 重置完成标记
                
                if old_audio_buffer_len > 0 or old_accumulated_text:
                    logger.info("🔄 [KWS激活] 已清空所有ASR相关状态: audio_buffer=%d样本 (%.2fs), accumulated_text='%s', vad_cache已清空, asr_cache已清空", 
                               old_audio_buffer_len, old_audio_buffer_len / STREAMING_TARGET_SAMPLE_RATE if old_audio_buffer_len > 0 else 0.0,
                               old_accumulated_text)
                else:
                    logger.info("🔄 [KWS激活] ASR相关状态为空（正常，WAITING_FOR_WAKEUP模式下不累积ASR）")
                
                return True
            else:
                # 唤醒失败，清空 buffer 和 cache，准备下一轮
                old_buffer_len = len(self.kws_audio_buffer)
                self.kws_audio_buffer = np.array([], dtype=np.float32)
                self.kws_cache = {}  # 清空 KWS 模型 cache
                self.kws_vad_cache = {}  # 清空 KWS 模式的 VAD cache
                logger.info("🔄 [KWS清除] KWS 唤醒失败，已清空缓冲区: %d 样本 (%.2fs), cache已清除", 
                           old_buffer_len, buffer_duration)
                logger.debug("KWS 唤醒失败: 文本='%s' (空或 rejected)", wake_text)
                return False
                
        except Exception as e:
            logger.error("KWS 检测执行异常: %s", e, exc_info=True)
            # 清空 buffer 和 cache
            self.kws_audio_buffer = np.array([], dtype=np.float32)
            self.kws_cache = {}  # 清空 KWS 模型 cache
            self.kws_vad_cache = {}  # 清空 KWS 模式的 VAD cache
            return False
    
    async def _save_kws_audio(self):
        """
        保存 KWS 检测音频到本地文件（用于调试和验证）
        
        保存路径：/workspace/voice-service/generated/kws_detection_audio/
        宿主机路径：./generated/kws_detection_audio/
        """
        if len(self.kws_audio_buffer) == 0:
            logger.warning("KWS 音频缓冲区为空，跳过保存")
            return
        
        try:
            from datetime import datetime
            from pathlib import Path
            import wave
            
            # 1. 创建保存目录
            save_dir = Path("/workspace/voice-service/generated/kws_detection_audio")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. 生成文件名（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            wav_filename = f"kws_detection_{timestamp}.wav"
            wav_file_path = save_dir / wav_filename
            
            # 3. 音量检测和日志（用于调试音量不一致问题）
            buffer_duration = len(self.kws_audio_buffer) / STREAMING_TARGET_SAMPLE_RATE
            buffer_energy = np.mean(np.abs(self.kws_audio_buffer))
            buffer_max = np.max(np.abs(self.kws_audio_buffer))
            buffer_rms = np.sqrt(np.mean(self.kws_audio_buffer ** 2))
            buffer_peak_db = 20 * np.log10(buffer_max + 1e-10)  # 避免log(0)
            buffer_rms_db = 20 * np.log10(buffer_rms + 1e-10)
            
            logger.info("📊 [KWS音频] 保存前音量检测: energy=%.6f, max=%.6f (%.2f dB), rms=%.6f (%.2f dB), len=%d样本 (%.2fs)", 
                       buffer_energy, buffer_max, buffer_peak_db, buffer_rms, buffer_rms_db,
                       len(self.kws_audio_buffer), buffer_duration)
            
            # 4. 将 numpy float32 数组转换为 int16 并保存为 WAV
            # 只做必要的 clamp 到 [-1, 1]，不做归一化，确保动态范围不被压缩
            audio_clamped = np.clip(self.kws_audio_buffer, -1.0, 1.0)
            audio_int16 = (audio_clamped * 32767.0).astype(np.int16)
            
            # 5. 使用 wave 模块保存 WAV 文件
            with wave.open(str(wav_file_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16-bit (2 bytes)
                wav_file.setframerate(STREAMING_TARGET_SAMPLE_RATE)  # 16kHz
                wav_file.writeframes(audio_int16.tobytes())
            
            buffer_duration = len(self.kws_audio_buffer) / STREAMING_TARGET_SAMPLE_RATE
            file_size = os.path.getsize(wav_file_path)
            logger.info("✅ 已保存 KWS 检测音频: %s (时长: %.2fs, 大小: %d 字节, %.2f KB)", 
                       wav_file_path, buffer_duration, file_size, file_size / 1024)
            logger.info("📁 宿主机路径: ./generated/kws_detection_audio/%s", wav_filename)
            
        except Exception as e:
            logger.error("保存 KWS 检测音频失败: %s", e, exc_info=True)
    
    def process_chunk(self, audio_np: np.ndarray) -> Dict[str, Any]:
        """
        处理一个音频片段，返回中间结果
        
        Args:
            audio_np: numpy数组，float32，16kHz，单声道
            
        Returns:
            dict: {
                "is_speech": bool,  # 是否检测到语音
                "intermediate_text": str,  # 累积的中间识别文本
                "should_finalize": bool  # 是否应该触发最终识别（静默≥1秒）
            }
        """
        vad_model, asr_model, _ = get_streaming_models()
        current_time = time.time()
        
        # 在进入模型前添加音频统计日志
        stats = _log_audio_statistics(audio_np, STREAMING_TARGET_SAMPLE_RATE, "模型输入前(process_chunk)")
        if stats and stats.get("clipping_ratio", 0) > 0.01:
            _dump_clipped_audio(audio_np, STREAMING_TARGET_SAMPLE_RATE, "process_chunk")
        
        # 1. VAD检测（双重标准：能量检测 + VAD模型）
        audio_energy = np.mean(np.abs(audio_np))
        audio_max = np.max(np.abs(audio_np))
        
        # 能量检测
        if STREAMING_VAD_USE_AND_LOGIC:
            # "与"逻辑：同时满足平均能量和最大值阈值才认为是语音（更严格）
            is_speech_energy = audio_energy > STREAMING_VAD_ENERGY_THRESHOLD and audio_max > STREAMING_VAD_MAX_THRESHOLD
        else:
            # "或"逻辑：满足任一条件就认为是语音（较宽松）
            is_speech_energy = audio_energy > STREAMING_VAD_ENERGY_THRESHOLD or audio_max > STREAMING_VAD_MAX_THRESHOLD
        
        # VAD模型检测
        is_speech_vad = False
        try:
            # 动态计算 chunk_size（毫秒），匹配实际音频长度
            chunk_duration_ms = len(audio_np) / STREAMING_TARGET_SAMPLE_RATE * 1000
            vad_res = vad_model.generate(
                input=audio_np,
                cache=self.vad_cache,
                is_final=False,
                chunk_size=int(chunk_duration_ms)
            )
            
            # 检查VAD返回格式
            if isinstance(vad_res, list) and len(vad_res) > 0:
                vad_item = vad_res[0]
                if isinstance(vad_item, dict):
                    value = vad_item.get("value", [])
                    if isinstance(value, list):
                        is_speech_vad = len(value) > 0
                    elif isinstance(value, str):
                        is_speech_vad = value.lower() == "speech"
        except Exception as vad_error:
            logger.warning("VAD检测异常（使用能量检测）: %s", vad_error)
        
        # 综合判断
        # is_speech = is_speech_energy or is_speech_vad
        is_speech = is_speech_energy
        
        
        # ⚠️ 在ASR_ACTIVE模式下输出每个chunk的VAD检测结果日志
        if self.mode == "ASR_ACTIVE":
            chunk_duration = len(audio_np) / STREAMING_TARGET_SAMPLE_RATE
            chunk_duration_ms = chunk_duration * 1000
            logger.info(
                "📊 [VAD检测] chunk检测结果: "
                "is_speech=%s (energy=%s, vad=%s), "
                "energy=%.6f (阈值=%.2f), max=%.6f (阈值=%.2f), "
                "chunk_len=%d样本 (%.2fms, %.3fs), "
                "能量检测逻辑=%s, 最终判断=OR",
                is_speech,
                is_speech_energy,
                is_speech_vad,
                audio_energy,
                STREAMING_VAD_ENERGY_THRESHOLD,
                audio_max,
                STREAMING_VAD_MAX_THRESHOLD,
                len(audio_np),
                chunk_duration_ms,
                chunk_duration,
                "AND" if STREAMING_VAD_USE_AND_LOGIC else "OR"
            )
        
        
        # 2. SV 声纹注册逻辑（如果启用且未注册，且已通过 KWS 激活）
        # 安全要求：声纹注册必须在 KWS 激活后才允许，防止未授权注册
        if self.use_speaker_verification and not self.is_enrolled and is_speech and self.is_activated:
            # 累积注册音频（只累积有语音的片段）
            self.enroll_audio_buffer = np.concatenate([self.enroll_audio_buffer, audio_np])
            enroll_duration = len(self.enroll_audio_buffer) / STREAMING_TARGET_SAMPLE_RATE
            
            # 检查是否达到注册时长要求（4秒）
            if enroll_duration >= self.min_enroll_seconds:
                enroll_path = self._save_enroll_sample()
                if enroll_path:
                    self.enroll_audio_path = enroll_path
                    old_enrolled = self.is_enrolled
                    self.is_enrolled = True
                    old_buffer_len = len(self.enroll_audio_buffer)
                    logger.info(f"✅ 声纹注册完成：{enroll_path} ({enroll_duration:.2f}s)")
                    # 清空注册缓冲区
                    self.enroll_audio_buffer = np.array([], dtype=np.float32)
                    logger.info("🔄 [SV清除] 声纹注册缓冲区已清空: %d 样本 (%.2fs), 注册状态: %s -> True", 
                               old_buffer_len, enroll_duration, old_enrolled)
        elif self.use_speaker_verification and not self.is_enrolled and is_speech and not self.is_activated:
            # 未激活时，清空任何累积的注册音频，防止未授权注册
            if len(self.enroll_audio_buffer) > 0:
                old_buffer_len = len(self.enroll_audio_buffer)
                logger.warning("⚠️ 检测到未激活状态下的声纹注册尝试，已清空注册缓冲区（安全保护）")
                self.enroll_audio_buffer = np.array([], dtype=np.float32)
                logger.info("🔄 [SV清除] 未激活状态下的注册缓冲区已清空: %d 样本 (%.2fs)", 
                           old_buffer_len, old_buffer_len / STREAMING_TARGET_SAMPLE_RATE)
        
        # 3. 更新静默计时器和音频累积（优化：累积所有chunk，保持音频连续性）
        # ✅ 改进逻辑：
        # 1. 只有检测到语音后，才开始累积buffer（防止纯静音进入buffer）
        # 2. 检测到语音后，累积所有chunk（包括静音），保持音频连续性
        # 3. 只有长时间静音（2秒）才触发finalize
        # 4. ⚠️ 前向保护机制：检测到语音时，将前向保护缓冲区的内容也累积到audio_buffer（防止丢失语音开头）
        # 这样可以保持音频完全连续，提高ASR识别效果
        # 短时间的停顿（如800ms）会被保留在音频中，有助于识别准确性
        
        # 更新静默计时器和语音检测标记
        if is_speech:
            # 检测到语音：重置静默计时器，标记已检测到语音
            self.silence_timer = 0.0
            self.last_voice_time = current_time
            self.has_detected_speech = True  # 标记已检测到语音
            self.silence_chunk_count = 0  # 重置静音chunk计数器（新的语音开始）
            
            # ⚠️ 前向保护机制：如果前向保护缓冲区有内容，先累积到audio_buffer（防止丢失语音开头）
            if len(self.pre_speech_buffer) > 0:
                old_pre_buffer_len = len(self.pre_speech_buffer)
                old_audio_buffer_len = len(self.audio_buffer)
                # 记录拼接前的统计
                if old_audio_buffer_len > 0:
                    stats_before = _log_audio_statistics(self.audio_buffer, STREAMING_TARGET_SAMPLE_RATE, "拼接前(audio_buffer)")
                stats_pre = _log_audio_statistics(self.pre_speech_buffer, STREAMING_TARGET_SAMPLE_RATE, "拼接前(pre_speech_buffer)")
                
                self.audio_buffer = np.concatenate([self.audio_buffer, self.pre_speech_buffer])
                
                # 记录拼接后的统计
                stats_after = _log_audio_statistics(self.audio_buffer, STREAMING_TARGET_SAMPLE_RATE, "拼接后(audio_buffer+pre_speech)")
                
                logger.info("🔧 [前向保护] 检测到语音，将前向保护缓冲区累积到audio_buffer: %d样本 (%.2fs) + %d样本 (%.2fs) -> %d样本 (%.2fs)", 
                           old_audio_buffer_len, old_audio_buffer_len / STREAMING_TARGET_SAMPLE_RATE,
                           old_pre_buffer_len, old_pre_buffer_len / STREAMING_TARGET_SAMPLE_RATE,
                           len(self.audio_buffer), len(self.audio_buffer) / STREAMING_TARGET_SAMPLE_RATE)
                # 清空前向保护缓冲区
                self.pre_speech_buffer = np.array([], dtype=np.float32)
            
            # 累积当前语音chunk
            old_audio_buffer_len = len(self.audio_buffer)
            # 记录拼接前的统计
            if old_audio_buffer_len > 0:
                stats_before = _log_audio_statistics(self.audio_buffer, STREAMING_TARGET_SAMPLE_RATE, "拼接前(audio_buffer)")
            stats_chunk = _log_audio_statistics(audio_np, STREAMING_TARGET_SAMPLE_RATE, "拼接前(当前chunk)")
            
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_np])
            
            # 记录拼接后的统计
            stats_after = _log_audio_statistics(self.audio_buffer, STREAMING_TARGET_SAMPLE_RATE, "拼接后(audio_buffer+chunk)")
            
            # 验证拼接是否正确：检查拼接后的前部分和原audio_buffer是否一致
            if old_audio_buffer_len > 0:
                original_part = self.audio_buffer[:old_audio_buffer_len]
                new_part = self.audio_buffer[old_audio_buffer_len:]
                if not np.array_equal(original_part, self.audio_buffer[:old_audio_buffer_len]):
                    logger.error("❌ [音频损坏检测] 拼接后audio_buffer的前部分与原始不一致！")
                elif not np.array_equal(new_part, audio_np):
                    logger.error("❌ [音频损坏检测] 拼接后audio_buffer的后部分与新chunk不一致！")
                else:
                    logger.debug("✅ [音频损坏检测] 拼接验证通过，音频数据保持一致")
        else:
            # 检测到静音：
            if self.has_detected_speech:
                # 已经检测到过语音，累积前2个静音chunk，后续静音chunk不再累积
                if self.silence_chunk_count < 2:
                    # 前2个静音chunk：累积到audio_buffer
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_np])
                    self.silence_chunk_count += 1
                    logger.debug("🔇 [静音处理] 累积第%d个静音chunk: %d样本 (%.2fs)", 
                               self.silence_chunk_count, len(audio_np), len(audio_np) / STREAMING_TARGET_SAMPLE_RATE)
                else:
                    # 第3个及以后的静音chunk：不再累积，只更新静默计时器
                    logger.debug("🔇 [静音处理] 跳过后续静音chunk（已累积2个）: %d样本 (%.2fs)", 
                               len(audio_np), len(audio_np) / STREAMING_TARGET_SAMPLE_RATE)
            
                # 更新静默计时器（从最后一次检测到语音的时间开始计算）
                self.silence_timer = current_time - self.last_voice_time
            else:
                # 从未检测到过语音，累积到前向保护缓冲区（防止丢失语音开头）
                # 前向保护缓冲区使用滑动窗口，只保留最新的400ms（1个chunk）
                self.pre_speech_buffer = np.concatenate([self.pre_speech_buffer, audio_np])
                target_samples = int(self.pre_speech_max_duration * STREAMING_TARGET_SAMPLE_RATE)  # 400ms
                if len(self.pre_speech_buffer) > target_samples:
                    # 只保留最新的400ms（FIFO队列）
                    self.pre_speech_buffer = self.pre_speech_buffer[-target_samples:]
                self.silence_timer = 0.0
        
        # 4. 流式ASR（仅处理语音片段）
        # 注释：不再进行流式ASR中间识别，chunk只做VAD检测和音频累积
        # 最终识别在 finalize() 中进行（使用完整的 audio_buffer）
        intermediate_text = ""  # 不再累积中间结果，始终返回空字符串
        # if is_speech:
        #     try:
        #         asr_res = asr_model.generate(
        #             input=audio_np,
        #             cache=self.asr_cache,
        #             is_final=False,
        #             chunk_size=STREAMING_CHUNK_SIZE,
        #             encoder_chunk_look_back=STREAMING_ENCODER_CHUNK_LOOK_BACK,
        #             decoder_chunk_look_back=STREAMING_DECODER_CHUNK_LOOK_BACK
        #         )
        #         
        #         # 提取文本
        #         new_text = ""
        #         if isinstance(asr_res, list) and len(asr_res) > 0:
        #             asr_item = asr_res[0]
        #             if isinstance(asr_item, dict):
        #                 new_text = asr_item.get("text", "")
        #                 if not new_text and "value" in asr_item:
        #                     value = asr_item.get("value", "")
        #                     if isinstance(value, str):
        #                         new_text = value
        #                     elif isinstance(value, list) and len(value) > 0:
        #                         first_val = value[0]
        #                         if isinstance(first_val, dict):
        #                             new_text = first_val.get("text", "")
        #             elif isinstance(asr_item, str):
        #                 new_text = asr_item
        #         elif isinstance(asr_res, dict):
        #             new_text = asr_res.get("text", "")
        #             if not new_text and "value" in asr_res:
        #                 value = asr_res.get("value", "")
        #                 if isinstance(value, str):
        #                     new_text = value
        #         
        #         new_text = new_text.strip() if new_text else ""
        #         
        #         # 智能合并累积文本
        #         if new_text:
        #             if not self.accumulated_intermediate_text:
        #                 self.accumulated_intermediate_text = new_text
        #             elif new_text == self.accumulated_intermediate_text:
        #                 pass  # 文本未变化
        #             elif new_text.startswith(self.accumulated_intermediate_text):
        #                 # 新文本是累积文本的扩展
        #                 self.accumulated_intermediate_text = new_text
        #             elif self.accumulated_intermediate_text.startswith(new_text):
        #                 # 新文本是累积文本的前缀（模型修正）
        #                 self.accumulated_intermediate_text = new_text
        #             else:
        #                 # 新文本与累积文本没有包含关系，可能是模型重新识别
        #                 if len(new_text) > len(self.accumulated_intermediate_text) * 0.5:
        #                     self.accumulated_intermediate_text = new_text
        #         
        #         intermediate_text = self.accumulated_intermediate_text
        #         
        #     except Exception as asr_error:
        #         logger.error("ASR中间识别异常: %s", asr_error, exc_info=True)
        
        # 5. 检查是否应该触发最终识别
        # ✅ 改进：累积所有chunk后，只需要检查静默时间是否达到阈值
        # 不需要尾音保护期的判断，因为所有chunk都已累积
        # ⚠️ 关键修复：只有在曾经检测到过语音的情况下，才允许触发finalize
        # 这样可以防止纯静音（从未有语音）触发finalize
        should_finalize = (self.silence_timer >= STREAMING_SILENCE_THRESHOLD and 
                          len(self.audio_buffer) > 0 and
                          self.has_detected_speech)  # 必须曾经检测到过语音
        
        # 调试日志：打印 should_finalize 的三个条件值（使用 INFO 级别确保输出）
        # 注释：暂时关闭 should_finalize 条件检查日志，减少日志输出
        # logger.info(
        #     "[should_finalize] 条件检查: tail_protection=None(%s), silence_timer>=%.1f(%s, 实际=%.3fs), audio_buffer>0(%s, 实际=%d样本, %.2fs), should_finalize=%s",
        #     condition1,
        #     STREAMING_SILENCE_THRESHOLD,
        #     condition2,
        #     self.silence_timer,
        #     condition3,
        #     len(self.audio_buffer),
        #     len(self.audio_buffer) / STREAMING_TARGET_SAMPLE_RATE if len(self.audio_buffer) > 0 else 0.0,
        #     should_finalize
        # )
        
        # 6. 实验性：chunk级别的声纹验证（仅在有语音且已注册且已激活时）
        if (self.use_speaker_verification and self.is_enrolled and self.is_activated and 
            self.enroll_audio_path and is_speech):
            # 累积音频到实验性验证缓冲区（当前chunk）
            self.experimental_sv_buffer = np.concatenate([self.experimental_sv_buffer, audio_np])
            buffer_duration = len(self.experimental_sv_buffer) / STREAMING_TARGET_SAMPLE_RATE
            
            # 同时累积到累积缓冲区（1+2+3+...）
            self.experimental_sv_accumulated_buffer = np.concatenate([self.experimental_sv_accumulated_buffer, audio_np])
            accumulated_duration = len(self.experimental_sv_accumulated_buffer) / STREAMING_TARGET_SAMPLE_RATE
            
            # 检查是否达到最小验证时长，且距离上次验证已过足够时间
            time_since_last_verify = current_time - self.experimental_sv_last_verify_time
            if (buffer_duration >= self.experimental_sv_min_duration and 
                time_since_last_verify >= self.experimental_sv_verify_interval):
                # 执行同步验证：验证当前chunk和累积chunks
                try:
                    # 1. 验证当前chunk
                    is_verified_chunk = self._verify_speaker_sync(self.experimental_sv_buffer, "chunk")
                    
                    # 2. 验证累积chunks（如果累积缓冲区足够长）
                    is_verified_accumulated = None
                    if accumulated_duration >= self.experimental_sv_min_duration:
                        is_verified_accumulated = self._verify_speaker_sync(self.experimental_sv_accumulated_buffer, "accumulated")
                    
                    # 清空当前chunk缓冲区，准备下一轮验证（累积缓冲区不清空）
                    self.experimental_sv_buffer = np.array([], dtype=np.float32)
                    self.experimental_sv_last_verify_time = current_time
                except Exception as e:
                    logger.error(f"❌ [实验性SV验证] 验证异常: {e}", exc_info=True)
                    # 验证失败时也清空当前chunk缓冲区，避免累积过多
                    self.experimental_sv_buffer = np.array([], dtype=np.float32)
        elif not is_speech:
            # 静音时，如果当前chunk缓冲区有内容但不够长，也清空（避免累积无效音频）
            if len(self.experimental_sv_buffer) > 0:
                buffer_duration = len(self.experimental_sv_buffer) / STREAMING_TARGET_SAMPLE_RATE
                if buffer_duration < self.experimental_sv_min_duration:
                    self.experimental_sv_buffer = np.array([], dtype=np.float32)
        
        return {
            "is_speech": is_speech,
            "intermediate_text": intermediate_text,
            "should_finalize": should_finalize
        }




    async def finalize(self) -> str:
        """
        最终识别：使用说话人分离模型进行 ASR 识别和说话人分离
        
        流程：
        1. 将 audio_buffer (numpy float32) 保存为 WAV 文件到挂载目录
           保存路径：/workspace/voice-service/generated/asr_final_audio/
           宿主机路径：./generated/asr_final_audio/
        2. 调用说话人分离模型（ModelScope pipeline）进行 ASR 识别和说话人分离
        3. 按 speaker ID 分组，同一 speaker 的句子按时间戳排序拼接
        4. 对每个 speaker 进行 SV 声纹验证（如果启用）
        5. 选择策略：
           - 单个 speaker：直接验证，通过返回文本，失败返回 __SV_VERIFICATION_FAILED__
           - 多个 speaker：选择分数最高的，如果都低于阈值则返回 __SV_VERIFICATION_FAILED__
           - 未启用 SV：返回所有 speaker 的文本拼接
        
        注意：
        - 如果 audio_buffer 为空，使用累积的中间结果作为后备
        - WAV 文件会保留在挂载目录中，方便在宿主机查看和调试
        - 在finalize前清空实验性SV验证的累积缓冲区
        - 临时 speaker 音频文件会在验证后自动清理
        
        Returns:
            str: 最终识别文本（带标点）或特殊标识（__SV_VERIFICATION_FAILED__ 等）
        """
        # ⚠️ 在finalize前清空累积缓冲区（每次finalize后重新开始累积）
        if len(self.experimental_sv_accumulated_buffer) > 0:
            old_accumulated_len = len(self.experimental_sv_accumulated_buffer)
            self.experimental_sv_accumulated_buffer = np.array([], dtype=np.float32)
            logger.info("🔄 [finalize前清空] 实验性SV累积缓冲区已清空: %d样本 (%.2fs)", 
                       old_accumulated_len, old_accumulated_len / STREAMING_TARGET_SAMPLE_RATE)
        
        if len(self.audio_buffer) == 0:
            # 如果audio_buffer为空，使用累积的中间结果
            final_text = self.accumulated_intermediate_text
            logger.warning("audio_buffer为空，使用累积中间结果: '%s'", final_text)
            if final_text and final_text.strip():
                return final_text.strip()
            else:
                # audio_buffer为空且中间结果也为空，返回特殊标识
                return "__ASR_RESULT_EMPTY__"
        
        
        # 1) 拿到最终要识别的音频
        audio = self.audio_buffer  # 或者 np.concatenate(self.audio_chunks)

        # 2) 在进入模型前添加详细的音频统计日志和损坏检测
        sr = 16000  # 项目里最终写 wav 的采样率
        
        # 详细分析 audio_buffer：检查是否有损坏、溢出、NaN等
        if len(audio) > 0:
            # 检查是否有 NaN 或 Inf
            has_nan = np.isnan(audio).any()
            has_inf = np.isinf(audio).any()
            if has_nan or has_inf:
                logger.error(f"❌ [音频损坏检测] audio_buffer包含异常值: NaN={has_nan}, Inf={has_inf}")
            
            # 检查是否超出 [-1, 1] 范围
            max_val = np.max(audio)
            min_val = np.min(audio)
            if max_val > 1.0 or min_val < -1.0:
                out_of_range_count = np.sum((audio > 1.0) | (audio < -1.0))
                logger.warning(f"⚠️ [音频损坏检测] audio_buffer超出[-1,1]范围: max={max_val:.6f}, min={min_val:.6f}, 超出范围样本数={out_of_range_count} (占比={out_of_range_count/len(audio)*100:.2f}%)")
            
            # 检查数据类型
            if audio.dtype != np.float32:
                logger.warning(f"⚠️ [音频损坏检测] audio_buffer数据类型异常: {audio.dtype}, 期望: float32")
        
        stats = _log_audio_statistics(audio, sr, "模型输入前(finalize)")
        if stats and stats.get("clipping_ratio", 0) > 0.01:
            _dump_clipped_audio(audio, sr, "finalize")
        
        # 保留原有的简单日志（用于兼容）
        duration_s = len(audio) / float(sr) if sr else 0.0
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        
        def _rms(x: np.ndarray) -> float:
            x = x.astype(np.float32)
            return float(np.sqrt(np.mean(x * x)) + 1e-12)
        
        rms = _rms(audio) if len(audio) else 0.0

        logger.info(
            "[FINALIZE][AUDIO] dur=%.3fs, len=%d, sr=%d, peak=%.6f, rms=%.6f",
            duration_s, len(audio), sr, peak, rms
        )
            
        
        # 保存 WAV 文件到挂载目录（方便在宿主机查看）
        wav_file_path = None
        try:
            # 1. 创建保存目录（使用 Docker 挂载的 generated 目录）
            from datetime import datetime
            from pathlib import Path
            
            # 使用挂载的 generated 目录：/workspace/voice-service/generated
            # 对应宿主机的 ./generated 目录
            save_dir = Path("/workspace/voice-service/generated/asr_final_audio")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. 生成文件名（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            wav_filename = f"asr_final_{timestamp}.wav"
            wav_file_path = save_dir / wav_filename
            
            logger.info("保存 WAV 文件到: %s (音频长度: %.2fs)", 
                       wav_file_path, len(self.audio_buffer) / STREAMING_TARGET_SAMPLE_RATE)
            
            # 3. 将 numpy float32 数组转换为 int16 并保存为 WAV
            # audio_buffer 是 float32，范围 [-1, 1]，需要转换为 int16
            # 只做必要的 clamp 到 [-1, 1]，不做归一化，确保动态范围不被压缩
            
            # 记录写入前的统计
            stats_before_write = _log_audio_statistics(self.audio_buffer, STREAMING_TARGET_SAMPLE_RATE, "写入WAV前")
            
            audio_clamped = np.clip(self.audio_buffer, -1.0, 1.0)
            
            # 检查 clamp 是否有影响（如果原始数据超出范围，clamp 会改变数据）
            clamped_count = np.sum((self.audio_buffer != audio_clamped))
            if clamped_count > 0:
                logger.warning(f"⚠️ [WAV写入] clamp改变了{clamped_count}个样本 (占比={clamped_count/len(self.audio_buffer)*100:.2f}%)")
                # 找出被 clamp 的值
                out_of_range = (self.audio_buffer > 1.0) | (self.audio_buffer < -1.0)
                if np.any(out_of_range):
                    out_max = np.max(self.audio_buffer[out_of_range])
                    out_min = np.min(self.audio_buffer[out_of_range])
                    logger.warning(f"  超出范围的值: max={out_max:.6f}, min={out_min:.6f}")
            
            audio_int16 = (audio_clamped * 32767.0).astype(np.int16)
            
            # 检查转换后的 int16 是否溢出
            int16_max = np.max(audio_int16)
            int16_min = np.min(audio_int16)
            if int16_max > 32767 or int16_min < -32768:
                logger.error(f"❌ [WAV写入] int16转换后溢出: max={int16_max}, min={int16_min}")
            
            # 使用 wave 模块保存 WAV 文件
            with wave.open(str(wav_file_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16-bit (2 bytes)
                wav_file.setframerate(STREAMING_TARGET_SAMPLE_RATE)  # 16kHz
                wav_file.writeframes(audio_int16.tobytes())
            
            file_size = os.path.getsize(wav_file_path)
            logger.info("✅ 已保存音频文件: %s (大小: %d 字节, %.2f KB)", 
                       wav_file_path, file_size, file_size / 1024)
            logger.info("📁 宿主机路径: ./generated/asr_final_audio/%s", wav_filename)
            
            # 4. 调用说话人分离模型进行 ASR 识别和说话人分离（使用全局单例）
            speaker_diarization_pipeline = get_speaker_diarization_pipeline()
            
            # 分析音频长度，调整 batch_size_s 参数以优化说话人分离
            audio_duration = len(self.audio_buffer) / STREAMING_TARGET_SAMPLE_RATE
            logger.info(f"🔍 [说话人分离] 音频时长: {audio_duration:.2f}s, 准备调用模型")
            
            # 对于较短的音频，使用更小的 batch_size_s 可以提高说话人分离的精度
            # 原始参数 batch_size_s=300 可能太大，导致所有片段被合并到同一个 batch
            # 尝试使用更小的值，让模型能够更细致地分析每个片段
            if audio_duration < 30:
                batch_size_s = 60  # 短音频使用更小的 batch
            elif audio_duration < 60:
                batch_size_s = 120
            else:
                batch_size_s = 300  # 长音频使用原始值
            
            logger.info(f"🔍 [说话人分离] 使用参数: batch_size_s={batch_size_s}, batch_size_token_threshold_s=40")
            
            rec_result = speaker_diarization_pipeline(
                str(wav_file_path), 
                batch_size_s=batch_size_s, 
                batch_size_token_threshold_s=40
            )
            
            # 5. 解析说话人分离结果
            if not rec_result or not isinstance(rec_result, list) or len(rec_result) == 0:
                logger.error("说话人分离模型返回空结果")
                final_text = "__ASR_RESULT_EMPTY__"
            else:
                # 详细打印原始返回结果结构（用于诊断）
                logger.info(f"🔍 [说话人分离] 原始返回结果类型: {type(rec_result)}, 长度: {len(rec_result)}")
                logger.info(f"🔍 [说话人分离] 返回结果结构: {rec_result}")
                
                result_dict = rec_result[0]
                logger.info(f"🔍 [说话人分离] result_dict 类型: {type(result_dict)}, 键: {result_dict.keys() if isinstance(result_dict, dict) else 'N/A'}")
                
                sentence_info_list = result_dict.get('sentence_info', [])
                
                if not sentence_info_list:
                    logger.warning("sentence_info 为空，无识别结果")
                    logger.info(f"🔍 [说话人分离] result_dict 完整内容: {result_dict}")
                    final_text = "__ASR_RESULT_EMPTY__"
                else:
                    logger.info(f"说话人分离结果: 共 {len(sentence_info_list)} 个句子")
                    
                    # 详细打印每个句子的完整信息（用于诊断）
                    # 同时分析时间戳间隔，识别可能的说话人切换点
                    logger.info(f"🔍 [说话人分离] 句子时间戳分析（用于识别说话人切换）:")
                    prev_end = None
                    for idx, sentence in enumerate(sentence_info_list):
                        logger.info(f"🔍 [说话人分离] 句子 {idx}: {sentence}")
                        logger.info(f"    - 类型: {type(sentence)}")
                        if isinstance(sentence, dict):
                            logger.info(f"    - 键: {sentence.keys()}")
                            # 尝试所有可能的 speaker ID 字段名
                            spk_fields = ['spk', 'speaker', 'speaker_id', 'spk_id', 'spkid']
                            for field in spk_fields:
                                if field in sentence:
                                    logger.info(f"    - {field} = {sentence[field]} (类型: {type(sentence[field])})")
                            
                            # 分析时间戳间隔
                            start_ms = sentence.get('start', 0)
                            end_ms = sentence.get('end', 0)
                            start_s = start_ms / 1000.0
                            end_s = end_ms / 1000.0
                            
                            if prev_end is not None:
                                gap_ms = start_ms - prev_end
                                gap_s = gap_ms / 1000.0
                                logger.info(f"    - 与前一句的间隔: {gap_ms}ms ({gap_s:.2f}s)")
                                # 如果间隔较大（>500ms），可能是说话人切换
                                if gap_ms > 500:
                                    logger.warning(f"    ⚠️ 检测到大间隔 ({gap_ms}ms)，可能是说话人切换点，但模型仍识别为同一speaker")
                            
                            logger.info(f"    - 时间范围: {start_ms}ms ({start_s:.2f}s) - {end_ms}ms ({end_s:.2f}s),  duration={end_s-start_s:.2f}s")
                            prev_end = end_ms
                    
                    # 6. 按 speaker ID 分组（尝试多种可能的字段名）
                    speaker_groups = {}
                    prev_end = None
                    current_speaker_id = 0
                    
                    # 分析时间戳间隔，如果所有句子都是同一个 speaker，尝试基于时间间隔推断不同的说话人
                    all_same_speaker = True
                    unique_speakers = set()
                    
                    for sentence in sentence_info_list:
                        if not isinstance(sentence, dict):
                            logger.warning(f"⚠️ 句子不是字典类型: {type(sentence)}, 值: {sentence}")
                            continue
                        
                        # 尝试多种可能的字段名获取 speaker ID
                        spk_id = None
                        for field in ['spk', 'speaker', 'speaker_id', 'spk_id', 'spkid']:
                            if field in sentence:
                                spk_id = sentence[field]
                                logger.debug(f"从字段 '{field}' 获取 speaker ID: {spk_id}")
                                break
                        
                        # 如果都没找到，使用默认值 0
                        if spk_id is None:
                            logger.warning(f"⚠️ 未找到 speaker ID 字段，句子内容: {sentence}")
                            spk_id = 0
                        
                        unique_speakers.add(spk_id)
                        
                        # 检查是否需要基于时间间隔重新分配 speaker ID
                        start_ms = sentence.get('start', 0)
                        gap_ms = (start_ms - prev_end) if prev_end is not None else 0
                        
                        # 如果所有句子都是同一个 speaker（模型识别失败），且间隔较大（>800ms），尝试分配新的 speaker ID
                        # 这是一个启发式方法，用于弥补模型说话人分离的不足
                        if len(unique_speakers) == 1 and gap_ms > 800:
                            # 间隔较大，可能是不同的说话人，分配新的 speaker ID
                            new_spk_id = current_speaker_id + 1
                            logger.warning(f"⚠️ [说话人分离启发式] 检测到大间隔 ({gap_ms}ms)，原speaker={spk_id}，尝试分配新speaker={new_spk_id}")
                            # 注意：这里不修改原始 sentence，而是使用新的 ID 进行分组
                            spk_id = new_spk_id
                            current_speaker_id = new_spk_id
                        else:
                            current_speaker_id = max(current_speaker_id, spk_id)
                        
                        prev_end = sentence.get('end', 0)
                        
                        # 确保 spk_id 是可哈希的类型（转换为 int 或 str）
                        if isinstance(spk_id, (int, str)):
                            spk_id_key = spk_id
                        else:
                            logger.warning(f"⚠️ Speaker ID 类型异常: {type(spk_id)}, 值: {spk_id}, 转换为字符串")
                            spk_id_key = str(spk_id)
                        
                        if spk_id_key not in speaker_groups:
                            speaker_groups[spk_id_key] = []
                        speaker_groups[spk_id_key].append(sentence)
                        
                        logger.debug(f"将句子添加到 Speaker {spk_id_key}: 文本='{sentence.get('text', 'N/A')}', 开始={sentence.get('start', 'N/A')}, 结束={sentence.get('end', 'N/A')}")
                    
                    # 如果应用了启发式规则，记录警告
                    if len(unique_speakers) == 1 and len(speaker_groups) > 1:
                        logger.warning(f"⚠️ [说话人分离启发式] 模型识别到 {len(unique_speakers)} 个唯一speaker，但基于时间间隔推测为 {len(speaker_groups)} 个不同说话人")
                    
                    # 对每个 speaker 的分段按时间戳排序
                    logger.info(f"🔍 [说话人分离] 识别到的 Speaker 数量: {len(speaker_groups)}")
                    for spk_id in speaker_groups:
                        speaker_groups[spk_id].sort(key=lambda x: x.get('start', 0))
                        logger.info(f"Speaker {spk_id}: {len(speaker_groups[spk_id])} 个句子")
                        # 打印每个句子的详细信息
                        for idx, sent in enumerate(speaker_groups[spk_id]):
                            logger.info(f"  - 句子 {idx}: 文本='{sent.get('text', 'N/A')}', "
                                      f"时间=[{sent.get('start', 'N/A'):.2f}s, {sent.get('end', 'N/A'):.2f}s], "
                                      f"speaker字段={sent.get('spk', 'N/A')}/{sent.get('speaker', 'N/A')}/{sent.get('speaker_id', 'N/A')}")
                    
                    # 7. SV 声纹验证（如果启用且已注册，且已通过 KWS 激活）
                    if self.use_speaker_verification and self.is_enrolled and self.enroll_audio_path:
                        if not self.is_activated:
                            logger.warning("⚠️ 未激活状态下不允许声纹验证（安全保护），跳过 ASR 识别")
                            final_text = "__SV_NOT_ACTIVATED__"
                        else:
                            speaker_scores = {}
                            temp_files_to_cleanup = []  # 记录需要清理的临时文件
                            
                            try:
                                for spk_id, sentences in speaker_groups.items():
                                    # 提取该 speaker 的音频
                                    speaker_audio = self._extract_speaker_audio(self.audio_buffer, sentences)
                                    
                                    if len(speaker_audio) == 0:
                                        logger.warning(f"⚠️ Speaker {spk_id} 音频为空，跳过验证")
                                        continue
                                    
                                    # 保存为临时文件
                                    temp_audio_path = self._save_temp_speaker_audio(speaker_audio, spk_id)
                                    temp_files_to_cleanup.append(temp_audio_path)
                                    
                                    # SV 验证（返回分数）
                                    is_verified, score = await self._verify_speaker_with_score(temp_audio_path)
                                    speaker_scores[spk_id] = {
                                        'score': score if score is not None else -1.0,
                                        'is_verified': is_verified,
                                        'sentences': sentences
                                    }
                                    logger.info(f"🔍 Speaker {spk_id} SV验证: is_verified={is_verified}, score={score if score is not None else 'N/A'}")
                                
                                # 8. 选择策略
                                if len(speaker_scores) == 0:
                                    logger.warning("所有 speaker 验证失败或音频为空")
                                    final_text = "__SV_VERIFICATION_FAILED__"
                                elif len(speaker_scores) == 1:
                                    # 单个 speaker
                                    spk_id = list(speaker_scores.keys())[0]
                                    if speaker_scores[spk_id]['is_verified']:
                                        # 拼接文本
                                        final_text = ''.join([s['text'] for s in speaker_scores[spk_id]['sentences']])
                                        logger.info(f"✅ 单个 Speaker {spk_id} 验证通过，返回文本")
                                    else:
                                        logger.warning(f"❌ 单个 Speaker {spk_id} 验证失败 (score={speaker_scores[spk_id]['score']})")
                                        final_text = "__SV_VERIFICATION_FAILED__"
                                else:
                                    # 多个 speaker：选择分数最高的
                                    best_spk_id = max(speaker_scores.keys(), key=lambda k: speaker_scores[k]['score'])
                                    best_score = speaker_scores[best_spk_id]['score']
                                    
                                    if best_score >= self.sv_threshold:
                                        # 分数最高的通过阈值，返回该 speaker 的文本
                                        final_text = ''.join([s['text'] for s in speaker_scores[best_spk_id]['sentences']])
                                        logger.info(f"✅ 选择 Speaker {best_spk_id} (分数: {best_score:.4f}, 阈值: {self.sv_threshold})")
                                        
                                        # 记录所有 speaker 的分数（用于调试）
                                        for spk_id, info in speaker_scores.items():
                                            logger.debug(f"  Speaker {spk_id}: score={info['score']:.4f}, is_verified={info['is_verified']}")
                                    else:
                                        # 所有 speaker 都低于阈值
                                        logger.warning(f"⚠️ 所有 speaker 分数都低于阈值 (最高: {best_score:.4f} < {self.sv_threshold})")
                                        final_text = "__SV_VERIFICATION_FAILED__"
                                
                            finally:
                                # 清理临时文件
                                for temp_file in temp_files_to_cleanup:
                                    try:
                                        if os.path.exists(temp_file):
                                            os.remove(temp_file)
                                            logger.debug(f"清理临时文件: {temp_file}")
                                    except Exception as e:
                                        logger.warning(f"清理临时文件失败: {temp_file}, {e}")
                    else:
                        # 未启用 SV，拼接所有 speaker 的文本（按 speaker ID 排序）
                        all_texts = []
                        for spk_id in sorted(speaker_groups.keys()):
                            sentences = speaker_groups[spk_id]
                            text = ''.join([s['text'] for s in sentences])
                            all_texts.append(text)
                        final_text = ''.join(all_texts)
                        logger.info(f"未启用SV，返回所有 speaker 的文本: {len(speaker_groups)} 个 speaker")
            
        except Exception as e:
            logger.error("最终识别异常: %s", e, exc_info=True)
            # 如果最终识别失败，返回空结果标识
            final_text = "__ASR_RESULT_EMPTY__"
        
        finally:
            # 注意：WAV 文件保留在挂载目录中，不删除，方便在宿主机查看
            # 如果需要清理旧文件，可以定期清理 ./generated/asr_final_audio/ 目录
            pass
        
        # 最后检查：如果final_text为空，返回特殊标识，避免返回空字符串
        if not final_text or not final_text.strip():
            return "__ASR_RESULT_EMPTY__"
        
        # ⚠️ 检查：如果结果只包含标点符号和语气词，视为无效结果
        cleaned_text = final_text.strip()
        
        # 定义语气词集合（包括单个和重复的语气词）
        interjections = {"嗯", "哈", "哼", "噗", "砰", "呀", "嗷", "啊", "哦", "额", "呃", "诶", "唉", "哎"}
        
        # 移除所有标点符号和空白字符，只保留汉字和字母数字
        # 移除标点符号（保留中文字符、字母、数字）
        text_without_punct = re.sub(r'[，。！？、；：""''（）【】《》〈〉「」『』〔〕〖〗…—～·\s]', '', cleaned_text)
        
        # 检查是否只包含语气词
        if text_without_punct:
            # 如果去除标点后还有内容，检查是否全是语气词
            # 将文本按字符分割，检查每个字符是否都是语气词
            # 这样可以过滤"嗯嗯"、"哈哈"、"嗯嗯嗯"等重复语气词
            chars = list(text_without_punct)
            if all(char in interjections for char in chars):
                logger.info("🔧 [无效结果过滤] 识别结果只包含语气词和标点: '%s' -> 视为空结果", cleaned_text)
                return "__ASR_RESULT_EMPTY__"
        else:
            # 去除标点后为空，说明只有标点符号
            logger.info("🔧 [无效结果过滤] 识别结果只包含标点符号: '%s' -> 视为空结果", cleaned_text)
            return "__ASR_RESULT_EMPTY__"
        
        return cleaned_text
    
    def _init_sv_pipeline(self):
        """延迟初始化声纹识别模型（优先使用本地路径）"""
        if self.sv_pipeline is None:
            try:
                from modelscope.pipelines import pipeline
                
                # 优先使用本地模型路径，避免从ModelScope下载
                # 容器内路径：/workspace/models/damo/speech_campplus_sv_zh-cn_16k-common
                # 本地开发路径：app/services/models/damo/speech_campplus_sv_zh-cn_16k-common
                sv_model_id = 'iic/speech_campplus_sv_zh-cn_16k-common'
                sv_model_revision = 'v1.0.0'
                
                # 尝试容器内路径
                container_path = "/workspace/models/damo/speech_campplus_sv_zh-cn_16k-common"
                # 尝试本地开发路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                local_path = os.path.join(current_dir, "models", "damo", "speech_campplus_sv_zh-cn_16k-common")
                
                sv_model_path = None
                if os.path.exists(container_path):
                    sv_model_path = container_path
                    logger.info("✅ 使用容器内SV模型路径: %s", container_path)
                elif os.path.exists(local_path):
                    sv_model_path = local_path
                    logger.info("✅ 使用本地SV模型路径: %s", local_path)
                else:
                    logger.warning("⚠️ 本地SV模型路径不存在，将使用模型ID（可能从ModelScope下载）: %s", sv_model_id)
                
                # ModelScope pipeline 支持直接传递本地路径作为 model 参数
                model_param = sv_model_path if sv_model_path else sv_model_id
                self.sv_pipeline = pipeline(
                    task='speaker-verification',
                    model=model_param,
                    model_revision=sv_model_revision if not sv_model_path else None  # 本地路径不需要revision
                )
                logger.info("✅ 声纹识别模型已加载 (model=%s)", model_param)
            except Exception as e:
                logger.error(f"❌ 声纹识别模型加载失败：{e}", exc_info=True)
                raise
        return self.sv_pipeline
    
    
    def _extract_speaker_audio(self, audio_np: np.ndarray, sentence_list: List[Dict], sample_rate: int = 16000) -> np.ndarray:
        """
        从完整音频中提取并拼接某个 speaker 的所有分段
        
        Args:
            audio_np: 完整音频（numpy float32数组）
            sentence_list: 该 speaker 的所有句子（已按时间戳排序）
            sample_rate: 采样率（默认16000）
        
        Returns:
            拼接后的音频片段（numpy float32数组）
        """
        segments = []
        for sentence in sentence_list:
            start_ms = sentence.get('start', 0)
            end_ms = sentence.get('end', 0)
            
            # 边界检查
            if start_ms < 0 or end_ms <= start_ms:
                logger.warning(f"⚠️ 无效时间戳: start={start_ms}ms, end={end_ms}ms")
                continue
            
            # 转换为采样点
            start_sample = int(start_ms * sample_rate / 1000)
            end_sample = int(end_ms * sample_rate / 1000)
            
            # 边界检查（避免越界）
            start_sample = max(0, min(start_sample, len(audio_np)))
            end_sample = max(start_sample, min(end_sample, len(audio_np)))
            
            if start_sample < end_sample:
                segment = audio_np[start_sample:end_sample]
                segments.append(segment)
                logger.debug(f"提取分段: {start_ms}ms-{end_ms}ms ({start_sample}-{end_sample}样本, {len(segment)/sample_rate:.2f}s)")
        
        if not segments:
            return np.array([], dtype=np.float32)
        
        # 拼接所有分段
        concatenated = np.concatenate(segments)
        logger.debug(f"拼接完成: {len(segments)}个分段, 总长度={len(concatenated)/sample_rate:.2f}s")
        return concatenated
    
    def _save_temp_speaker_audio(self, audio_np: np.ndarray, spk_id: int) -> str:
        """
        保存 speaker 的临时音频文件用于 SV 验证
        
        Args:
            audio_np: 音频数据（numpy float32数组）
            spk_id: speaker ID
        
        Returns:
            临时文件路径
        """
        from datetime import datetime
        from pathlib import Path
        
        save_dir = Path("/workspace/voice-service/generated/sv_speaker_segments")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        temp_filename = f"speaker_{spk_id}_{timestamp}.wav"
        temp_path = save_dir / temp_filename
        
        # 转换为 int16 并保存
        # 只做必要的 clamp 到 [-1, 1]，不做归一化，确保动态范围不被压缩
        audio_clamped = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_clamped * 32767.0).astype(np.int16)
        with wave.open(str(temp_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(STREAMING_TARGET_SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        
        logger.debug(f"保存 Speaker {spk_id} 临时音频: {temp_path} ({len(audio_np)/STREAMING_TARGET_SAMPLE_RATE:.2f}s)")
        return str(temp_path)
    
    async def _verify_speaker_with_score(self, current_audio_path: str) -> Tuple[bool, Optional[float]]:
        """声纹验证：返回验证结果和分数"""
        try:
            sv_pipeline = self._init_sv_pipeline()
            
            # 调用声纹验证
            sv_res = sv_pipeline([self.enroll_audio_path, current_audio_path])
            
            # 解析验证结果
            verdict_text, score = self._parse_sv_result(sv_res)
            
            # 判定是否通过
            is_verified = self._is_sv_verified(verdict_text, score)
            
            if is_verified:
                logger.info(f"✅ 声纹验证通过 (text={verdict_text}, score={score})")
            else:
                logger.warning(f"❌ 声纹验证失败 (text={verdict_text}, score={score})")
            
            return is_verified, score
            
        except Exception as e:
            logger.error(f"❌ 声纹验证异常：{e}", exc_info=True)
            return False, None
    
    def _save_enroll_sample(self) -> Optional[str]:
        """保存注册样本为 WAV 文件"""
        try:
            from datetime import datetime
            from pathlib import Path
            
            save_dir = Path("/workspace/voice-service/generated/sv_enroll")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            enroll_path = save_dir / f"enroll_{timestamp}.wav"
            
            # 转换为 int16 并保存
            # 只做必要的 clamp 到 [-1, 1]，不做归一化，确保动态范围不被压缩
            audio_clamped = np.clip(self.enroll_audio_buffer, -1.0, 1.0)
            audio_int16 = (audio_clamped * 32767.0).astype(np.int16)
            with wave.open(str(enroll_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(STREAMING_TARGET_SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            logger.info(f"✅ 注册样本已保存：{enroll_path}")
            return str(enroll_path)
        except Exception as e:
            logger.error(f"❌ 保存注册样本失败：{e}", exc_info=True)
            return None
    
    def _verify_speaker_sync(self, audio_buffer: np.ndarray, buffer_type: str = "chunk") -> bool:
        """实验性：同步版本的声纹验证（用于chunk级别的实时验证）
        
        参数:
            audio_buffer: 要验证的音频缓冲区（numpy数组）
            buffer_type: 缓冲区类型（"chunk" 或 "accumulated"），用于日志标识
        
        注意：这是实验性功能，用于在process_chunk中实时验证声纹
        """
        if len(audio_buffer) == 0:
            logger.warning(f"🔬 [实验性SV验证] {buffer_type}缓冲区为空，跳过验证")
            return False
        
        if not self.enroll_audio_path or not os.path.exists(self.enroll_audio_path):
            logger.warning(f"🔬 [实验性SV验证] 注册样本不存在，跳过验证")
            return False
        
        try:
            # 1. 保存实验性验证缓冲区为临时文件
            from datetime import datetime
            from pathlib import Path
            import wave
            
            save_dir = Path("/workspace/voice-service/generated/sv_experimental")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            temp_audio_path = save_dir / f"experimental_sv_{buffer_type}_{timestamp}.wav"
            
            # 转换为 int16 并保存
            # 只做必要的 clamp 到 [-1, 1]，不做归一化，确保动态范围不被压缩
            audio_clamped = np.clip(audio_buffer, -1.0, 1.0)
            audio_int16 = (audio_clamped * 32767.0).astype(np.int16)
            with wave.open(str(temp_audio_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(STREAMING_TARGET_SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            # 2. 初始化SV pipeline（同步）
            sv_pipeline = self._init_sv_pipeline()
            
            # 3. 调用声纹验证
            sv_res = sv_pipeline([self.enroll_audio_path, str(temp_audio_path)])
            
            # 4. 解析验证结果
            verdict_text, score = self._parse_sv_result(sv_res)
            
            # 5. 判定是否通过
            is_verified = self._is_sv_verified(verdict_text, score)
            
            # 6. 输出详细的验证信息（用于测试chunk级别验证的可行性）
            buffer_duration = len(audio_buffer) / STREAMING_TARGET_SAMPLE_RATE
            logger.info(
                f"🔬 [实验性SV验证] {buffer_type}验证详情: "
                "结果=%s, verdict=%s, score=%.5f, 阈值=%.2f, "
                "音频长度=%.2fs, 注册样本=%s, 当前音频=%s",
                "通过" if is_verified else "失败",
                verdict_text if verdict_text else "N/A",
                score if score is not None else float('nan'),
                self.sv_threshold,
                buffer_duration,
                self.enroll_audio_path,
                str(temp_audio_path)
            )
            
            # 7. 清理临时文件（可选，保留用于调试）
            # os.remove(str(temp_audio_path))
            
            return is_verified
            
        except Exception as e:
            logger.error(f"❌ [实验性SV验证] 验证异常：{e}", exc_info=True)
            return False
    
    async def _verify_speaker(self, current_audio_path: str) -> bool:
        """声纹验证：比对注册样本和当前音频"""
        try:
            sv_pipeline = self._init_sv_pipeline()
            
            # 调用声纹验证
            sv_res = sv_pipeline([self.enroll_audio_path, current_audio_path])
            
            # 解析验证结果
            verdict_text, score = self._parse_sv_result(sv_res)
            
            # 判定是否通过
            is_verified = self._is_sv_verified(verdict_text, score)
            
            if is_verified:
                logger.info(f"✅ 声纹验证通过 (text={verdict_text}, score={score})")
            else:
                logger.warning(f"❌ 声纹验证失败 (text={verdict_text}, score={score})")
            
            return is_verified
            
        except Exception as e:
            logger.error(f"❌ 声纹验证异常：{e}", exc_info=True)
            # 验证异常时，可以选择继续 ASR 或跳过
            # 这里选择跳过（更安全）
            return False
    
    def _parse_sv_result(self, sv_res: Any) -> Tuple[Optional[str], Optional[float]]:
        """解析声纹验证结果"""
        verdict_text = None
        score = None
        
        if isinstance(sv_res, dict):
            verdict_text = sv_res.get('text')
            for k in ('score', 'similarity', 'sim'):
                if k in sv_res:
                    try:
                        score = float(sv_res[k])
                        break
                    except Exception:
                        pass
        elif isinstance(sv_res, (list, tuple)) and sv_res:
            first = sv_res[0]
            if isinstance(first, dict):
                verdict_text = first.get('text')
                for k in ('score', 'similarity', 'sim'):
                    if k in first:
                        try:
                            score = float(first[k])
                            break
                        except Exception:
                            pass
            elif isinstance(first, str):
                verdict_text = first
            elif isinstance(first, (int, float)):
                score = float(first)
        
        if isinstance(verdict_text, str):
            verdict_text = verdict_text.strip().lower()
        
        return verdict_text, score
    

    def _is_sv_verified(self, verdict_text, score):
        if score is None:
            return False

        # 强通过
        if score >= self.sv_threshold:
            return True

        # 强拒绝
        if score < self.sv_threshold:
            return False

        # 模糊区间，用 text 辅助
        if verdict_text == 'yes':
            return True
        if verdict_text == 'no':
            return False

        return False
