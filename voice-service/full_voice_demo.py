import os
import time
import tempfile
import threading
import queue
import collections
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from funasr import AutoModel
from modelscope.pipelines import pipeline
from myLLM import correct_text, chat_with_qwen
from typing import Optional, Callable, Dict, Any

# ---------------- 配置 ----------------
SAMPLE_RATE = 16000 # 采样率（Hz）
CHANNELS = 1 # 声道数：webrtcvad需要输入声道数
FRAME_MS = 30 # 每帧时长（毫秒）：webrtcvad 只接受 10、20 或 30
VAD_MODE = 3 # webrtcvad 模式（0-3）：数值越大越严格（更不容易把噪声判为语音）
MIN_START_VOICED_FRAMES = 3 # 连续有声音帧数（帧数 * FRAME_MS = 实际毫秒数）
SILENCE_TIMEOUT_S = 1.0 # 静音检测时长
MAX_SEGMENT_SECONDS = 60.0 # 最大录音时长
QUEUE_MAXSIZE = 8 # 队列最大长度，超出会丢弃新段
# 背景能量乘子：calibrate_background 测得的 rms * ENERGY_MULTIPLIER = 能量阈值
# 值越大对能量判定越苛刻（需要更高能量才认为是“有声”）
ENERGY_MULTIPLIER = 2.5
# webrtcvad 与 能量阈值 的组合方式：
# True -> 同时满足 webrtcvad 判定 AND 能量阈值（更严格）
# False -> webrtcvad 判定 OR 能量阈值（更宽松）
USE_AND_DECISION = True
# 背景校准时长（秒）：启动时用于采集环境噪声以计算 energy_threshold
CALIBRATE_SECONDS = 1.0
MIN_ENROLL_SECONDS = 2.0 # enroll（声纹注册）要求的最短时长
SV_THRESHOLD = 0.31 # 声纹判定阈值

ENROLL_FILENAME = os.path.join(tempfile.gettempdir(), "enroll_voice.wav")

FUNASR_KW = dict(
    model="paraformer-zh",
    model_revision="v2.0.4",
    vad_model="fsmn-vad",
    vad_model_revision="v2.0.4",
    punc_model="ct-punc-c",
    punc_model_revision="v2.0.4",
    spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
    spk_model_revision="v2.0.2",
    disable_update=True,
)

SV_MODEL_ID = 'iic/speech_campplus_sv_zh-cn_16k-common'
SV_MODEL_REV = 'v1.0.0'

GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# 语音唤醒模型初始化（KWS）
kws_model = AutoModel(
    model="iic/speech_charctc_kws_phone-xiaoyun",
    keywords="小云小云",
    output_dir="./outputs/debug",
    device='cpu',
    disable_update=True,
)

def voice_wake(input_wave):
    wake_res = kws_model.generate(input=input_wave, cache={},)
    print(wake_res)
    if wake_res and isinstance(wake_res, (list, tuple)) and len(wake_res) > 0:
        first = wake_res[0]
        wake_txt = first.get('text', None) if isinstance(first, dict) else None
        if wake_txt and wake_txt != 'rejected':
            color_print(f"[KWS] 唤醒识别通过：{wake_txt}", GREEN)
            return True
        else:
            color_print(f"[KWS] 唤醒失败：{wake_txt}", RED)
            return False
    else:
        color_print(f"[KWS_ERROR] 唤醒结果解析错误：{wake_res}", RED)
        return False

def color_print(text: str, color: Optional[str] = None):
    """带颜色打印，若终端不支持颜色则回退普通打印。"""
    if color:
        try:
            print(color + text + RESET)
        except Exception:
            print(text)
    else:
        print(text)

# ---------------- 运行时队列与信号 ----------------
recognize_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=QUEUE_MAXSIZE)

# ---------------- VADRecorder：按帧判定并切段（带 pre-roll） ----------------
class VADRecorder:
    """
    读取回调传入的 bytes，按 frame_ms 切帧，
    每帧通过 webrtcvad + 能量阈值 判定是否“有声”，
    通过状态机（waiting/recording）组装语音段并把 numpy int16 放入 recognize_queue。
    新增 pre-roll（默认 300 ms）：在 waiting 状态下保持最近若干帧，一旦进入 recording
    会把 pre-roll 中的帧一并作为段的开头，避免丢失说话起始几帧。
    如果传入了 processing_event，当该事件被置位时，不会从 waiting 进入 recording（阻止新的段开始）。
    """

    def __init__(self,
                 sr: int = SAMPLE_RATE,
                 channels: int = CHANNELS,
                 frame_ms: int = FRAME_MS,
                 vad_mode: int = VAD_MODE,
                 min_voiced_frames: int = MIN_START_VOICED_FRAMES,
                 silence_timeout_s: float = SILENCE_TIMEOUT_S,
                 max_segment_s: float = MAX_SEGMENT_SECONDS,
                 energy_multiplier: float = ENERGY_MULTIPLIER,
                 use_and: bool = USE_AND_DECISION,
                 processing_event: Optional[threading.Event] = None,
                 pre_roll_ms: int = 300):   # 新增：预留前置时长（毫秒）
        assert frame_ms in (10, 20, 30)
        self.sr = sr
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_samples = int(sr * frame_ms / 1000.0)
        self.frame_bytes = self.frame_samples * 2 * channels  # int16 每样本2字节
        self.vad = webrtcvad.Vad(vad_mode)
        self.min_voiced_frames = min_voiced_frames
        self.silence_timeout_s = silence_timeout_s
        self.max_segment_s = max_segment_s
        self.energy_multiplier = energy_multiplier
        self.use_and = use_and

        self.energy_threshold: Optional[float] = None

        # 外部用于阻止在回调处理中启动新段（由 main 创建并传入）
        self.processing_event = processing_event

        # pre-roll buffer，用 deque 保存最近若干帧（frame 为 bytes）
        self.pre_roll_ms = pre_roll_ms
        self._prebuffer_max_frames = max(1, int(self.pre_roll_ms / self.frame_ms))
        self._prebuffer = collections.deque(maxlen=self._prebuffer_max_frames)

        # 状态
        self._buf = bytearray()
        self._segment_frames: list[bytes] = []
        self._state = "waiting"
        self._consec_voiced = 0
        self._consec_silence_ms = 0
        self._segment_start_ts: Optional[float] = None
        self._lock = threading.Lock()

    def calibrate_background(self, seconds: float = CALIBRATE_SECONDS) -> float:
        """录制短时背景音并计算能量阈值（rms * multiplier）。"""
        print(f"[CALIBRATE] 采集 {seconds:.2f}s 背景噪声，请保持安静...")
        try:
            rec = sd.rec(int(seconds * self.sr), samplerate=self.sr, channels=self.channels, dtype='int16')
            sd.wait()
            arr = rec.flatten().astype(np.int16)
        except Exception as e:
            print("[CALIBRATE] 采样失败：", e)
            arr = None

        if arr is None or arr.size == 0:
            self.energy_threshold = 500.0 / 32768.0
            print("[CALIBRATE] 使用回退能量阈值 =", self.energy_threshold)
            return self.energy_threshold

        f = arr.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(f * f)))
        self.energy_threshold = rms * self.energy_multiplier
        print(f"[CALIBRATE] 背景 rms={rms:.6f}, energy_threshold={self.energy_threshold:.6f}")
        return self.energy_threshold

    def _frame_rms(self, frame_bytes: bytes) -> float:
        """计算单帧 RMS（归一化到 [-1,1]）。"""
        if not frame_bytes:
            return 0.0
        arr = np.frombuffer(frame_bytes, dtype=np.int16)
        if arr.size == 0:
            return 0.0
        f = arr.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(f * f)))

    def _is_voiced(self, frame_bytes: bytes) -> bool:
        """结合 webrtcvad 与能量阈值返回布尔判定。"""
        try:
            vad_decision = self.vad.is_speech(frame_bytes, self.sr)
        except Exception:
            vad_decision = False
        rms = self._frame_rms(frame_bytes)
        if self.energy_threshold is None:
            return vad_decision
        return (vad_decision and (rms >= self.energy_threshold)) if self.use_and else (vad_decision or (rms >= self.energy_threshold))

    def _process_frame(self, frame_bytes: bytes):
        # 先把帧放到 prebuffer（保证进入 recording 时能取回之前若干帧）
        try:
            self._prebuffer.append(frame_bytes)
        except Exception:
            pass

        # 如果外部 callback 正在运行（processing_event 已置位），则在 waiting 状态下不触发新的 recording
        if self._state == "waiting" and self.processing_event is not None and self.processing_event.is_set():
            # 将该帧视为静音，重置连续有声计数
            self._consec_voiced = 0
            return

        is_speech = self._is_voiced(frame_bytes)
        if self._state == "waiting":
            if is_speech:
                self._consec_voiced += 1
            else:
                self._consec_voiced = 0
            if self._consec_voiced >= self.min_voiced_frames:
                # 进入 recording，开始新段
                self._state = "recording"
                # 把 prebuffer 的内容（包括当前帧）作为段的开头
                self._segment_frames = list(self._prebuffer)
                self._segment_start_ts = time.time()
                self._consec_silence_ms = 0
                print("[VAD] 语音开始（包含 pre-roll）")
                # 清空 prebuffer（避免重复）
                self._prebuffer.clear()
        else:  # recording
            self._segment_frames.append(frame_bytes)
            if is_speech:
                self._consec_silence_ms = 0
            else:
                self._consec_silence_ms += self.frame_ms
            elapsed = time.time() - (self._segment_start_ts or time.time())
            if self._consec_silence_ms >= int(self.silence_timeout_s * 1000):
                print(f"[VAD] 语音结束（静音 {self._consec_silence_ms} ms）")
                self._finalize()
            elif elapsed >= self.max_segment_s:
                print(f"[VAD] 语音结束（超时 {elapsed:.1f}s）")
                self._finalize()

    def _finalize(self):
        """把已收集的帧拼接成 numpy int16，放入队列。"""
        if not self._segment_frames:
            self._reset()
            return
        seg_bytes = b"".join(self._segment_frames)
        arr = np.frombuffer(seg_bytes, dtype=np.int16)
        try:
            recognize_queue.put_nowait(arr)
            print(f"[QUEUE] 已入队段，长度 {len(arr)/self.sr:.3f}s")
        except queue.Full:
            print("[WARN] recognize_queue 已满，丢弃该段")
        self._reset()

    def _reset(self):
        self._state = "waiting"
        self._consec_voiced = 0
        self._consec_silence_ms = 0
        self._segment_frames = []
        self._segment_start_ts = None
        # 清空 prebuffer，保证下一次从干净状态开始
        try:
            self._prebuffer.clear()
        except Exception:
            pass

    def audio_callback(self, indata, frames, time_info, status):
        """sounddevice 回调，快速把 bytes 累积并按帧处理（注意回调应尽量快）。"""
        if status:
            print("[InputStream status]:", status)
        try:
            chunk = indata.tobytes()
        except Exception:
            chunk = bytes(indata)
        with self._lock:
            self._buf.extend(chunk)
            while len(self._buf) >= self.frame_bytes:
                frame = bytes(self._buf[:self.frame_bytes])
                del self._buf[:self.frame_bytes]
                try:
                    self._process_frame(frame)
                except Exception as e:
                    print("[ERROR] process_frame:", e)
                    self._reset()

# ---------------- 识别工作线程：声纹比对 + ASR ----------------
def recognition_worker(asr_model: Any,
                       sv_pipeline: Any,
                       enroll_holder: Dict[str, Any],
                       stop_evt: threading.Event,
                       chat_callback: Optional[Callable[[str], Any]],
                       processing_event: threading.Event,
                       use_speaker_verification: bool = True):
    """
    消费 recognize_queue：
      - 如果 use_speaker_verification True（默认），行为跟原先相同：首段 >= MIN_ENROLL_SECONDS 作为 enroll，后续段做比对通过才 ASR。
      - 如果 use_speaker_verification False，跳过 enroll 与声纹比对，对每个段直接做 ASR，并在 iat_result 非空时调用回调。
    """
    while not stop_evt.is_set():
        try:
            arr = recognize_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        tmp_path = None
        try:
            # 若多通道，需要 reshape 成 (n_frames, channels)
            if CHANNELS > 1 and arr.ndim == 1:
                arr = arr.reshape((-1, CHANNELS))

            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_path = tmpf.name
            tmpf.close()

            sf.write(tmp_path, arr, SAMPLE_RATE, subtype='PCM_16')
            seg_sec = len(arr) / SAMPLE_RATE
            print(f"[WORKER] 临时 wav 已保存: {tmp_path} ({seg_sec:.3f}s)")


            # 如果不使用声纹验证，直接做 ASR（跳过 enroll 与 SV）
            if not use_speaker_verification:
                # 语音唤醒
                normal_wake=voice_wake(tmp_path)
                # 如果没有唤醒/唤醒词,不进行语音识别
                if not normal_wake:
                    continue
                
                def do_asr_direct_and_maybe_callback():
                    t0 = time.time()
                    try:
                        res = asr_model.generate(input=tmp_path, batch_size_s=300, hotword=None, sentence_timestamp=True, is_final=True)
                    except Exception as e:
                        res = f"[ERROR] asr.generate 失败: {e}"
                    dt = time.time() - t0
                    print(f"[ASR] 完成，耗时 {dt:.2f}s，结果：\n{res}")

                    # 解析 iat_result
                    iat_result = ""
                    try:
                        if isinstance(res, (list, tuple)) and res:
                            iat_result = res[0].get('text', '') if isinstance(res[0], dict) else str(res[0])
                        else:
                            iat_result = str(res)
                    except Exception:
                        iat_result = str(res)

                    # 若非空则调用回调（期间置位 processing_event）
                    if iat_result and chat_callback is not None:
                        print(iat_result)
                        print(f"\033[95m>>>>>用户: {iat_result}  \033[0m\n")
                        print(f"\033[95m>>>>>智能体:\033[0m\n")
                        if processing_event is not None:
                            processing_event.set()
                        try:
                            try:
                                chat_res = chat_callback(iat_result)
                            except TypeError:
                                chat_res = chat_callback()
                        finally:
                            if processing_event is not None:
                                processing_event.clear()
                    else:
                        if iat_result:
                            print(iat_result)

                do_asr_direct_and_maybe_callback()
                continue  # 处理完当前段，继续下一个

            # use_speaker_verification == True 的逻辑：
            # enroll 流程
            if enroll_holder.get('path') is None:
                if seg_sec < MIN_ENROLL_SECONDS:
                    color_print(f"[ENROLL] 录音太短 ({seg_sec:.2f}s)，请至少说 {MIN_ENROLL_SECONDS:.1f}s 完成注册。", RED)
                    continue
                try:
                    sf.write(ENROLL_FILENAME, arr, SAMPLE_RATE, subtype='PCM_16')
                    # 语音唤醒
                    verification_wake=voice_wake(ENROLL_FILENAME)
                    if verification_wake:
                        enroll_holder['path'] = ENROLL_FILENAME
                        color_print(f"[ENROLL] enroll 已保存到 {ENROLL_FILENAME}（{seg_sec:.2f}s），后续段将与此比对。", GREEN)
                    
                except Exception as e:
                    print("[ENROLL] 保存失败：", e)
                finally:
                    continue  # enroll 段不做 ASR

            # 已 enroll：做声纹比对
            try:
                sv_res = sv_pipeline([enroll_holder['path'], tmp_path])
            except Exception as e:
                print("[SV] pipeline 错误：", e)
                sv_res = None

            # 解析 sv_res：优先取 'text'，其次尝试提取 numeric score
            verdict_text = None
            score = None
            if isinstance(sv_res, dict):
                verdict_text = sv_res.get('text')
                for k in ('score', 'similarity', 'sim'):
                    if k in sv_res:
                        try:
                            score = float(sv_res[k]); break
                        except Exception:
                            pass
            elif isinstance(sv_res, (list, tuple)) and sv_res:
                first = sv_res[0]
                if isinstance(first, dict):
                    verdict_text = first.get('text')
                    for k in ('score', 'similarity', 'sim'):
                        if k in first:
                            try:
                                score = float(first[k]); break
                            except Exception:
                                pass
                elif isinstance(first, str):
                    verdict_text = first
                elif isinstance(first, (int, float)):
                    score = float(first)

            if isinstance(verdict_text, str):
                verdict_text = verdict_text.strip().lower()

            # 判定与后续处理
            def do_asr_and_maybe_call_callback():
                """执行 ASR；若 iat_result 非空则调用回调。调用回调前置位 processing_event，返回后清除。"""
                t0 = time.time()
                try:
                    res = asr_model.generate(input=tmp_path, batch_size_s=300, hotword=None, sentence_timestamp=True, is_final=True)
                except Exception as e:
                    res = f"[ERROR] asr.generate 失败: {e}"
                dt = time.time() - t0
                print(f"[ASR] 完成，耗时 {dt:.2f}s，结果：\n{res}")

                # 尝试解析文本结果
                iat_result = ""
                try:
                    if isinstance(res, (list, tuple)) and res:
                        iat_result = res[0].get('text', '') if isinstance(res[0], dict) else str(res[0])
                    else:
                        iat_result = str(res)
                except Exception:
                    iat_result = str(res)

                # 若 iat_result 非空则调用回调
                if iat_result and chat_callback is not None:
                    print(iat_result)
                    print(f"\033[95m>>>>>用户: {iat_result}  \033[0m\n")
                    print(f"\033[95m>>>>>智能体:\033[0m\n")
                    # 在调用回调前置位 processing_event，阻止 VAD 开始新段
                    if processing_event is not None:
                        processing_event.set()
                    try:
                        try:
                            if iat_result=="退出。":
                                print("结束语音识别！！！！")
                                stop_speech_recognition()
                            else:
                                chat_res = chat_callback(iat_result)
                        except TypeError:
                            # 如果回调签名无参数，尝试无参调用
                            chat_res = chat_callback()
                    finally:
                        if processing_event is not None:
                            processing_event.clear()
                else:
                    # 不调用回调，直接输出或忽略
                    if iat_result:
                        print(iat_result)

            if verdict_text == 'yes':
                color_print(f"[SV] 验证通过 (text='yes', score={score})，开始 ASR", GREEN)
                do_asr_and_maybe_call_callback()

            elif verdict_text == 'no':
                color_print(f"[SV] 验证失败 (text='no', score={score})，跳过 ASR", RED)
            else:
                # 回退到 numeric score 判定
                if score is not None:
                    if score >= SV_THRESHOLD:
                        color_print(f"[SV] numeric score {score:.4f} >= {SV_THRESHOLD}，视为通过，开始 ASR", GREEN)
                        do_asr_and_maybe_call_callback()
                    else:
                        color_print(f"[SV] numeric score {score:.4f} < {SV_THRESHOLD}，视为失败，跳过 ASR", RED)
                else:
                    print("[SV] 无法解析 SV 结果（既无 text 也无数值），跳过 ASR。SV 返回：", sv_res)

        except Exception as e:
            print("[WORKER] 未知错误：", e)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            try:
                recognize_queue.task_done()
            except Exception:
                pass


# ---------------- main：加载模型、启动录音与工作线程 ----------------
def load_recognition_model():
    print("[MAIN] 加载模型（若未缓存则可能下载）...")
    asr_model = AutoModel(**FUNASR_KW)
    sv_pipeline = pipeline(task='speaker-verification', model=SV_MODEL_ID, model_revision=SV_MODEL_REV)
    print("[MAIN] 模型加载完成。")
    return asr_model, sv_pipeline

# 初始化模型
asr_model, sv_pipeline = load_recognition_model()

# ===== 全局保存运行状态（便于 stop 调用） =====
running_state = {
    "local_stop": None,
    "processing_event": None,
    "worker": None,
    "stream": None,
}

def start_speech_recognition(model=asr_model, pipeline=sv_pipeline,
                             chat_callback: Optional[Callable[[str], Any]] = None,
                             use_speaker_verification: bool = True):
    global running_state

    local_stop = threading.Event()
    processing_event = threading.Event()
    enroll_holder: Dict[str, Any] = {'path': None}

    # 删除旧的 enroll
    if os.path.exists(ENROLL_FILENAME):
        try:
            os.remove(ENROLL_FILENAME)
            print(f"[MAIN] 删除旧 enroll 文件 {ENROLL_FILENAME}")
        except Exception:
            pass

    # VAD 校准
    recorder = VADRecorder(processing_event=processing_event)
    try:
        recorder.calibrate_background(seconds=CALIBRATE_SECONDS)
    except Exception as e:
        print("[MAIN] 校准失败：", e)

    # 启动 worker 线程
    worker = threading.Thread(
        target=recognition_worker,
        args=(model, pipeline, enroll_holder, local_stop, chat_callback,
              processing_event, use_speaker_verification),
        daemon=True,
    )
    worker.start()

    blocksize = recorder.frame_samples
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16',
            blocksize=blocksize, callback=recorder.audio_callback
        )
        stream.start()

        # 保存状态，方便 stop 使用
        running_state.update({
            "local_stop": local_stop,
            "processing_event": processing_event,
            "worker": worker,
            "stream": stream,
        })

        if use_speaker_verification:
            print(f"[MAIN] 正在监听... 首个 >= {MIN_ENROLL_SECONDS:.1f}s 的语音段将作为 enroll。调用 stop_speech_recognition() 可关闭。")

        while not local_stop.is_set():
            time.sleep(0.2)

    except Exception as e:
        print("[MAIN] InputStream 错误：", e)
        stop_speech_recognition()

def stop_speech_recognition():
    """安全关闭语音识别与声纹线程（防止 worker 自己 join 自己）。"""
    global running_state

    local_stop = running_state.get("local_stop")
    processing_event = running_state.get("processing_event")
    worker = running_state.get("worker")
    stream = running_state.get("stream")

    # 1) 通知 worker 退出（非阻塞）
    if local_stop:
        local_stop.set()

    # 2) 先尝试关闭/停止 InputStream（若存在）
    if stream:
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            print("[MAIN] 关闭 InputStream 出错：", e)

    # 3) 等待正在进行的回调（如果有）
    if processing_event and processing_event.is_set():
        print("[MAIN] 等待回调处理完成...")
        processing_event.wait(timeout=10.0)

    # 4) 如果 worker 存在且仍然是另一个线程，则 join 等待其结束；否则跳过 join
    if worker:
        current = threading.current_thread()
        if worker is current:
            # 如果我们正好在 worker 内部调用 stop（比如 callback 内），不能 join 自己
            print("[MAIN] stop 在 worker 线程内被调用，跳过 join（worker 会自行退出）。")
        else:
            print("[MAIN] 等待 worker 线程结束...")
            worker.join(timeout=3.0)

    # 5) 清理全局状态
    running_state.update({"local_stop": None, "processing_event": None,
                          "worker": None, "stream": None})
    print("[MAIN] 已安全关闭语音识别。")

if __name__ == "__main__":
    # 将 chat_with_qwen 作为回调传入，默认开启声纹比对
    # start_speech_recognition(chat_callback=chat_with_qwen, use_speaker_verification=True)

    # 跳过声纹比对，直接识别并调用回调，运行：
    start_speech_recognition(chat_callback=chat_with_qwen, use_speaker_verification=True)
