import logging
import re
import time
import uuid
import json
import numpy as np

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.models.voice import (
    ASRRequest, 
    ASRResponse,
    WebSocketAudioMessage,
    WebSocketResultMessage,
    WebSocketErrorMessage,
    WebSocketWelcomeMessage
)
from app.services.voice_service import VoiceService
from app.services.voice_interface import (
    base64_to_audio_np,
    get_streaming_models,
    STREAMING_VAD_ENERGY_THRESHOLD,
    STREAMING_VAD_MAX_THRESHOLD,
    STREAMING_VAD_USE_AND_LOGIC,
    STREAMING_SILENCE_THRESHOLD
)
from app.services.hg_deepseek import process_speech_result, correct_text_only, load_hotwords_list
from app.services.full_hotwords import SYMS
from app.config import config

logger = logging.getLogger(__name__)
router = APIRouter()

# 延迟初始化 VoiceService，避免启动时模型加载失败导致服务无法启动
_voice_service = None

def get_voice_service() -> VoiceService:
    """获取 VoiceService 实例（延迟初始化）"""
    global _voice_service
    if _voice_service is None:
        try:
            _voice_service = VoiceService()
            logger.info("VoiceService 初始化成功")
        except Exception as e:
            logger.error("VoiceService 初始化失败: %s", e, exc_info=True)
            # 即使初始化失败，也创建一个实例，避免后续调用失败
            _voice_service = VoiceService.__new__(VoiceService)
    return _voice_service


@router.post("/asr", response_model=ASRResponse)
async def speech_recognition(request: ASRRequest):
    """语音识别接口"""
    asr_start = time.perf_counter()
    logger.info("ASR请求开始: use_wake=%s, audio_data_length=%d", request.use_wake, len(request.audio_data))

    try:
        # 验证音频数据
        validation_start = time.perf_counter()
        if not request.audio_data or len(request.audio_data) < 100:
            logger.warning("音频数据过短或为空: %d 字节", len(request.audio_data))
            raise HTTPException(status_code=400, detail="音频数据无效或过短")
        validation_time = (time.perf_counter() - validation_start) * 1000
        logger.info("耗时统计 - 数据验证: %.2f ms", validation_time)

        # 获取 VoiceService 实例（延迟初始化）
        voice_service = get_voice_service()
        result = await voice_service.recognize_speech(request)

        logger.info("ASR识别结果: success=%s, text_length=%d", result.success, len(result.text))

        # Convert Pydantic model to dictionary
        serialize_start = time.perf_counter()
        result_dict = result.model_dump()
        serialize_time = (time.perf_counter() - serialize_start) * 1000
        logger.info("耗时统计 - 序列化响应: %.2f ms", serialize_time)

        total_asr_time = (time.perf_counter() - asr_start) * 1000
        logger.info("耗时统计 - ASR接口总耗时: %.2f ms", total_asr_time)

        return JSONResponse(content=result_dict, headers={"Content-Type": "application/json; charset=utf-8"})
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        total_asr_time = (time.perf_counter() - asr_start) * 1000
        logger.error("ASR处理异常: %s (耗时: %.2f ms)", e, total_asr_time, exc_info=True)
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}") from e


@router.websocket("/asr/ws")
async def speech_recognition_websocket(websocket: WebSocket):
    """
    WebSocket 流式语音识别接口
    
    协议说明:
    1. 客户端连接到 /api/v1/voice/asr/ws
    2. 服务端发送欢迎消息
    3. 客户端每200ms发送一段音频片段（base64编码的WAV）
    4. 服务端进行VAD检测和流式ASR识别
    5. 服务端返回中间结果（流式识别中）或最终结果（静默1秒后）
    6. 可以持续发送多个音频片段进行识别
    
    消息格式:
    
    欢迎消息 (Server -> Client):
    {
        "type": "welcome",
        "message": "Connected to voice recognition service",
        "timestamp": 1234567890.123
    }
    
    音频消息 (Client -> Server):
    {
        "wav_base64": "base64编码的WAV音频片段（200ms）"
    }
    或（兼容旧格式）:
    {
        "type": "audio",
        "audio_data": "base64编码的音频数据"
    }
    
    中间结果 (Server -> Client):
    {
        "type": "processing",
        "status": "processing",
        "intermediate_text": "累积的识别文本"
    }
    
    处理中状态 (Server -> Client):
    {
        "type": "processing",
        "status": "finalizing",
        "message": "正在处理音频..."
    }
    
    最终结果 (Server -> Client):
    {
        "type": "result",
        "status": "completed",
        "text": "最终识别文本（带标点）",
        "success": true
    }
    
    错误消息 (Server -> Client):
    {
        "type": "error",
        "message": "错误描述",
        "code": "ERROR_CODE"
    }
    """
    client_id = None
    session = None
    try:
        logger.info("收到 WebSocket 连接请求，准备接受连接...")
        # 接受 WebSocket 连接
        await websocket.accept()
        client_id = str(uuid.uuid4())
        logger.info("WebSocket客户端连接成功: %s", client_id)
        
        # 获取 VoiceService 实例（延迟初始化）
        voice_service = get_voice_service()
        
        # 创建流式处理会话
        try:
            session = voice_service.create_streaming_session()
            logger.info("流式处理会话创建成功: %s", client_id)
        except Exception as e:
            logger.error("创建流式处理会话失败: %s", e, exc_info=True)
            await websocket.send_json({
                "type": "error",
                "message": f"创建会话失败: {str(e)}",
                "code": "SESSION_CREATE_ERROR"
            })
            await websocket.close(code=1011, reason="Failed to create session")
            return
        
        # 发送欢迎消息（包含当前唤醒模式状态）
        welcome_msg = WebSocketWelcomeMessage(
            message="Connected to voice recognition service",
            timestamp=time.time()
        )
        welcome_data = welcome_msg.model_dump()
        welcome_data["use_wake"] = session.use_wake  # 告知前端当前唤醒模式状态
        welcome_data["mode"] = session.mode  # 告知前端当前模式
        await websocket.send_json(welcome_data)
        logger.info("已发送欢迎消息给客户端: %s (use_wake=%s, mode=%s)", 
                   client_id, session.use_wake, session.mode)
        
        # 消息处理循环
        while True:
            try:
                # 接收客户端消息
                data_str = await websocket.receive_text()
                if not data_str:
                    await websocket.send_json({
                        "type": "error",
                        "message": "空消息，忽略",
                        "code": "EMPTY_MESSAGE"
                    })
                    continue
                
                # 解析JSON
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "消息格式错误，需为 JSON 字符串",
                        "code": "INVALID_JSON"
                    })
                    continue
                
                # 记录收到的消息类型（用于调试）
                message_type = data.get("type", "unknown")
                logger.info("📨 [消息接收] 收到消息类型: %s (client_id=%s, 当前模式=%s)", 
                           message_type, client_id, session.mode if session else "N/A")
                
                # # 处理 interrupt 消息（前端发送的打断信号，用于打断当前说话过程）
                # if data.get("type") == "interrupt":
                #     logger.info("收到 interrupt 信号（打断当前说话过程），重置会话: %s", client_id)
                #     session.reset()  # reset() 会根据 use_wake 决定重置后的模式
                #     continue
                
                # 处理 end_conversation 消息（前端发送的对话结束信号，表示用户离开，需要完全重置）
                if data.get("type") == "end_conversation":
                    logger.info("收到 end_conversation 信号（对话结束，用户离开），完全重置会话状态: %s", client_id)
                    session.reset()  # 完全重置所有状态（包括KWS和SV注册状态）
                    # 发送确认消息给前端
                    await websocket.send_json({
                        "type": "status",
                        "status": "conversation_ended",
                        "message": "会话已结束，状态已重置"
                    })
                    logger.info("已发送对话结束确认消息: %s", client_id)
                    continue
                
                # 处理 cancel_enrollment 消息（前端发送的信号，表示用户提前取消声纹录制/对话，需强制回到等待唤醒）
                if data.get("type") == "cancel_enrollment":
                    logger.info("🛑 [取消声纹录制] ========== 收到 cancel_enrollment 信号 ==========")
                    logger.info("🛑 [取消声纹录制] 用户提前取消声纹录制 (client_id=%s, 当前模式=%s)", 
                               client_id, session.mode if session else "N/A")
                    
                    if session:
                        old_mode = session.mode
                        # 强制回到需要 KWS 唤醒的模式
                        session.set_use_wake(True)
                        session.reset()  # reset 会清除 KWS / SV / ASR 全部状态，并根据 use_wake 设为 WAITING_FOR_WAKEUP
                        session.mode = "WAITING_FOR_WAKEUP"  # 再次显式设置，确保状态一致
                        
                        logger.info("🔄 [取消声纹录制] ✅ 已重置所有状态，回退到等待唤醒模式: %s -> WAITING_FOR_WAKEUP", old_mode)
                        
                        # 发送确认消息给前端
                        await websocket.send_json({
                            "type": "status",
                            "status": "enrollment_cancelled",
                            "message": "声纹录制已取消，已回退到等待唤醒状态"
                        })
                        logger.info("🔄 [取消声纹录制] ✅ 已发送声纹录制取消确认消息给前端: %s", client_id)
                        logger.info("🛑 [取消声纹录制] ========== cancel_enrollment 处理完成 ==========")
                    else:
                        logger.warning("⚠️ [取消声纹录制] 收到 cancel_enrollment 信号但 session 不存在 (client_id=%s)", client_id)
                    continue
                
                # 处理 start_asr 消息（前端发送的信号，表示声纹录制完成/弹窗已关闭，可以开始ASR识别）
                if data.get("type") == "start_asr":
                    logger.info("收到 start_asr 信号（前端确认，开始ASR识别）: %s", client_id)
                    # 在 WAITING_FOR_ENROLLMENT 或 WAITING_FOR_ENROLLMENT_CONFIRM 模式下都可以处理此信号
                    if session.mode == "WAITING_FOR_ENROLLMENT" or session.mode == "WAITING_FOR_ENROLLMENT_CONFIRM":
                        old_mode = session.mode
                        session.mode = "ASR_ACTIVE"
                        logger.info("🔄 [模式切换] 会话模式: %s -> ASR_ACTIVE (前端确认，开始ASR识别)", old_mode)
                        
                        # 清空ASR音频缓冲区和相关状态（确保声纹录制期间的音频不参与ASR识别）
                        session.audio_buffer = np.array([], dtype=np.float32)
                        session.vad_cache = {}  # 清空VAD cache
                        session.asr_cache = {}  # 清空ASR cache
                        session.accumulated_intermediate_text = ""
                        # ⚠️ 关键：在状态切换时重置计时参考点，避免跨状态污染
                        current_monotonic = time.monotonic()
                        session.silence_timer = 0.0
                        session.last_voice_time = current_monotonic  # 重置计时参考点
                        session.tail_protection_start_time = None
                        session.is_completed = False
                        session.pre_speech_buffer = np.array([], dtype=np.float32)  # 清空前向保护缓冲区
                        session.has_detected_speech = False  # 重置语音检测标记
                        # 注意：enrollment相关状态在自动切换时已清理，这里不需要再次清理
                        
                        logger.info("🔄 [ASR准备] 已清空所有ASR相关状态，准备接收新的ASR音频: %s", client_id)
                        
                        # 发送确认消息给前端
                        await websocket.send_json({
                            "type": "status",
                            "status": "asr_started",
                            "message": "已切换到ASR识别模式"
                        })
                        logger.info("已发送ASR启动确认消息: %s", client_id)
                    else:
                        logger.warning("⚠️ 收到 start_asr 信号但当前模式不是 WAITING_FOR_ENROLLMENT 或 WAITING_FOR_ENROLLMENT_CONFIRM (当前模式: %s): %s", 
                                      session.mode, client_id)
                    continue
                
                # 处理 use_wake 参数（前端可以动态控制是否启用唤醒）
                use_wake_param = data.get("use_wake")
                if use_wake_param is not None:
                    # 前端明确指定了 use_wake 参数
                    use_wake = bool(use_wake_param)
                    if session.use_wake != use_wake:
                        logger.info("收到 use_wake 参数变更: %s -> %s (client_id=%s)", 
                                   session.use_wake, use_wake, client_id)
                        session.set_use_wake(use_wake)
                
                # 处理 use_sv 参数（前端可以动态控制是否启用声纹验证）
                use_sv_param = data.get("use_sv")
                if use_sv_param is not None:
                    use_sv = bool(use_sv_param)
                    if session.use_speaker_verification != use_sv:
                        logger.info("收到 use_sv 参数变更: %s -> %s (client_id=%s)", 
                                   session.use_speaker_verification, use_sv, client_id)
                        session.use_speaker_verification = use_sv
                        # 如果禁用声纹验证，清空注册状态
                        if not use_sv:
                            old_enrolled = session.is_enrolled
                            old_enroll_path = session.enroll_audio_path
                            old_buffer_len = len(session.enroll_audio_buffer)
                            session.is_enrolled = False
                            session.enroll_audio_path = None
                            session.enroll_audio_buffer = np.array([], dtype=np.float32)
                            logger.info("🔄 [SV清除] 已禁用声纹验证，清空注册状态: is_enrolled=%s->False, enroll_audio_path=%s, buffer=%d样本 (%.2fs)", 
                                       old_enrolled, old_enroll_path, old_buffer_len, 
                                       old_buffer_len / 16000 if old_buffer_len > 0 else 0.0)
                
                # 处理 use_llm 参数（前端可以动态控制是否启用LLM后处理）
                # 默认值：读取配置，如果配置未禁用则默认启用
                voice_config = getattr(config, "voice_service", None)
                default_use_llm = not (voice_config and voice_config.voice_disable_llm) if voice_config else True
                use_llm_param = data.get("use_llm")
                use_llm = bool(use_llm_param) if use_llm_param is not None else default_use_llm
                # 如果配置全局禁用，则覆盖
                if voice_config and voice_config.voice_disable_llm:
                    use_llm = True
                
                # 提取音频数据（兼容两种格式）
                wav_base64 = data.get("wav_base64") or data.get("audio_data")
                if not wav_base64 or not isinstance(wav_base64, str):
                    await websocket.send_json({
                        "type": "error",
                        "message": "缺少有效字段 'wav_base64' 或 'audio_data'",
                        "code": "MISSING_AUDIO_DATA"
                    })
                    continue
                
                # 如果上一轮已完成，重置状态准备下一轮（清除所有cache）
                if session.is_completed:
                    session.reset()  # reset() 会根据 use_wake 决定重置后的模式
                    logger.info("会话已重置，准备下一轮识别: %s (use_wake=%s, mode=%s)", 
                               client_id, session.use_wake, session.mode)
                
                # base64解码为numpy数组
                try:
                    audio_np, sr = base64_to_audio_np(wav_base64)
                    logger.debug("音频解码成功: shape=%s, sr=%sHz, 时长=%.1fms", 
                               audio_np.shape, sr, len(audio_np)/sr*1000)
                except Exception as e:
                    logger.error("音频解码失败: %s", e, exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"音频解码失败: {str(e)}",
                        "code": "AUDIO_DECODE_ERROR"
                    })
                    continue
                
                # ========== KWS 唤醒模式：只运行 KWS 检测，不运行 ASR ==========
                if session.mode == "WAITING_FOR_WAKEUP":
                    # 只运行 KWS 检测（不运行ASR）
                    # 第一句话的所有chunk都只做KWS检测，不做ASR识别
                    logger.info("🔍 [WAITING_FOR_WAKEUP] 只运行KWS检测，不运行ASR: %s (音频长度: %.2fms)", 
                               client_id, len(audio_np) / 16000 * 1000)
                    is_wakeup = await session.process_wakeup_chunk(audio_np)
                    
                    if is_wakeup:
                        # 唤醒成功，设置激活标记并切换到 WAITING_FOR_ENROLLMENT 模式
                        # 注意：KWS 音频已在 _perform_kws_detection() 中保存并清空，并转移到enroll_audio_buffer
                        old_activated = session.is_activated
                        old_mode = session.mode
                        session.is_activated = True
                        session.mode = "WAITING_FOR_ENROLLMENT"  # 切换到等待声纹录制状态
                        logger.info("🔄 [状态更新] KWS 激活状态: %s -> True", old_activated)
                        logger.info("🔄 [模式切换] 会话模式: %s -> WAITING_FOR_ENROLLMENT (KWS已激活，等待声纹录制完成)", old_mode)
                        
                        # 注意：ASR相关状态已在 _perform_kws_detection() 中清空
                        # 这里只需要确认状态已清空（避免重复清空）
                        if len(session.audio_buffer) > 0 or session.accumulated_intermediate_text:
                            logger.warning("⚠️ [KWS激活] ASR状态未完全清空，再次清空: audio_buffer=%d样本, accumulated_text='%s'", 
                                          len(session.audio_buffer), session.accumulated_intermediate_text)
                            session.audio_buffer = np.array([], dtype=np.float32)
                            session.accumulated_intermediate_text = ""
                            session.vad_cache = {}
                            session.asr_cache = {}
                            # ⚠️ 使用 monotonic 时间，避免系统时间调整影响
                            current_monotonic = time.monotonic()
                            session.silence_timer = 0.0
                            session.last_voice_time = current_monotonic
                            session.tail_protection_start_time = None
                            session.is_completed = False
                        else:
                            logger.debug("🔄 [KWS激活] ASR相关状态已清空（在_perform_kws_detection中完成）")
                        
                        logger.info("🎤 系统已唤醒，设置激活标记并切换到等待声纹录制状态: %s", client_id)
                        
                        # 发送唤醒成功消息给前端
                        await websocket.send_json({
                            "type": "wakeup",
                            "status": "activated",
                            "message": "系统已唤醒，等待声纹录制"
                        })
                        
                        # ✅ 关键修复：KWS激活后，跳过当前chunk的后续处理
                        # 避免当前chunk（可能还包含唤醒词）被ASR识别
                        # 从下一个chunk开始进入WAITING_FOR_ENROLLMENT模式
                        logger.info("🔄 [KWS激活] 跳过当前chunk的后续处理，从下一个chunk开始等待声纹录制: %s", client_id)
                        continue  # 跳过当前chunk，等待下一个chunk
                    else:
                        # KWS 检测失败，如果之前已激活，重置激活状态（防止状态污染）
                        if session.is_activated:
                            logger.warning("⚠️ KWS 检测失败但 is_activated=True（可能是之前会话的状态），重置为 False: %s", client_id)
                            session.is_activated = False
                            logger.info("🔄 [KWS清除] KWS 激活状态已清除: True -> False (检测失败)")
                        logger.debug("等待唤醒模式 - 未检测到唤醒词，继续等待: %s", client_id)
                    
                    # 注意：WAITING_FOR_WAKEUP 模式下不运行ASR，不累积audio_buffer
                    # 第一句话（包含唤醒词）不会被ASR识别，避免"小护"等词干扰业务
                
                # ========== 等待声纹录制模式：KWS已激活，等待前端完成声纹录制 ==========
                elif session.mode == "WAITING_FOR_ENROLLMENT":
                    # 在这个模式下：
                    # 1. 通过VAD检测到声音后才开始累积音频到enroll_audio_buffer（用于SV注册）
                    # 2. 不运行ASR，不累积audio_buffer
                    # 3. 结束条件（"与"逻辑）：同时满足
                    #    a. 从VAD第一次检测到声音开始，累积够5秒内容
                    #    b. 出现足够的静音检测（2秒）
                    # 4. 前端也可以手动发送"start_asr"信号切换
                    logger.debug("等待声纹录制模式 - VAD检测并累积音频到SV注册buffer: %s (音频长度: %.2fms)", 
                               client_id, len(audio_np) / 16000 * 1000)
                    
                    try:
                        # 1. VAD检测（用于判断是否有语音，用于控制何时开始累积和静音检测）
                        vad_model, _, _ = get_streaming_models()
                        current_time = time.time()
                        
                        audio_energy = np.mean(np.abs(audio_np))
                        audio_max = np.max(np.abs(audio_np))
                        
                        # 能量检测
                        if STREAMING_VAD_USE_AND_LOGIC:
                            is_speech_energy = audio_energy > STREAMING_VAD_ENERGY_THRESHOLD and audio_max > STREAMING_VAD_MAX_THRESHOLD
                        else:
                            is_speech_energy = audio_energy > STREAMING_VAD_ENERGY_THRESHOLD or audio_max > STREAMING_VAD_MAX_THRESHOLD
                        
                        # VAD模型检测
                        is_speech_vad = False
                        try:
                            chunk_duration_ms = len(audio_np) / 16000 * 1000
                            vad_res = vad_model.generate(
                                input=audio_np,
                                cache=session.vad_cache,
                                is_final=False,
                                chunk_size=int(chunk_duration_ms)
                            )
                            
                            if isinstance(vad_res, list) and len(vad_res) > 0:
                                vad_item = vad_res[0]
                                if isinstance(vad_item, dict):
                                    value = vad_item.get("value", [])
                                    if isinstance(value, list):
                                        is_speech_vad = len(value) > 0
                                    elif isinstance(value, str):
                                        is_speech_vad = value.lower() == "speech"
                        except Exception as vad_error:
                            logger.warning("WAITING_FOR_ENROLLMENT VAD检测异常（使用能量检测）: %s", vad_error)
                        
                        # 综合判断
                        is_speech = is_speech_energy or is_speech_vad
                        
                        # 2. 检测到语音时，标记已开始累积，记录第一次语音时间
                        if is_speech:
                            if not session.enroll_has_detected_speech:
                                # 第一次检测到语音，开始累积
                                session.enroll_has_detected_speech = True
                                session.enroll_first_speech_time = current_time
                                logger.info("🎤 [Enrollment] 第一次检测到语音，开始累积声纹注册音频")
                            
                            # 更新最后语音时间（用于静音检测）
                            if not hasattr(session, 'enroll_last_voice_time'):
                                session.enroll_last_voice_time = current_time
                            session.enroll_last_voice_time = current_time
                            session.enroll_silence_timer = 0.0
                        else:
                            # 检测到静音：更新静音计时器
                            if hasattr(session, 'enroll_last_voice_time') and session.enroll_last_voice_time:
                                session.enroll_silence_timer = current_time - session.enroll_last_voice_time
                            else:
                                # 如果还没有记录过语音时间，初始化
                                if not hasattr(session, 'enroll_last_voice_time'):
                                    session.enroll_last_voice_time = current_time
                                session.enroll_silence_timer = 0.0
                        
                        # 3. 只有检测到语音后才累积音频到enroll_audio_buffer（类似ASR逻辑）
                        if session.enroll_has_detected_speech:
                            old_buffer_len = len(session.enroll_audio_buffer)
                            session.enroll_audio_buffer = np.concatenate([session.enroll_audio_buffer, audio_np])
                            enroll_duration = len(session.enroll_audio_buffer) / 16000
                            
                            # 计算从第一次检测到语音开始的累积时长
                            enroll_duration_from_first_speech = 0.0
                            if session.enroll_first_speech_time:
                                enroll_duration_from_first_speech = current_time - session.enroll_first_speech_time
                            
                            logger.debug("等待声纹录制模式 - 累积SV注册音频: %s, 总长度=%.2fs, 从首次语音开始=%.2fs, 静音时长=%.2fs, 累积了%d样本 (%.2fs)", 
                                       "有语音" if is_speech else "静音",
                                       enroll_duration,
                                       enroll_duration_from_first_speech,
                                       session.enroll_silence_timer if hasattr(session, 'enroll_silence_timer') else 0.0,
                                       len(audio_np), len(audio_np) / 16000)
                            
                            # 4. 检查是否应该自动切换到ASR状态（"与"逻辑：同时满足两个条件）
                            should_auto_switch = False
                            auto_switch_reason = ""
                            
                            # 条件1：从VAD第一次检测到声音开始，累积够5秒内容
                            condition1_met = enroll_duration_from_first_speech >= session.min_enroll_seconds
                            
                            # 条件2：出现足够的静音检测（2秒）
                            condition2_met = (hasattr(session, 'enroll_silence_timer') and 
                                            session.enroll_silence_timer >= 2.0)
                            
                            # "与"逻辑：两个条件都满足
                            if condition1_met and condition2_met:
                                should_auto_switch = True
                                auto_switch_reason = f"满足结束条件（从首次语音开始={enroll_duration_from_first_speech:.2f}s≥5s 且 静音={session.enroll_silence_timer:.2f}s≥2s）"
                            elif condition1_met:
                                silence_time = session.enroll_silence_timer if hasattr(session, 'enroll_silence_timer') else 0.0
                                logger.debug("等待声纹录制模式 - 条件1满足（时长≥5s），但条件2未满足（静音=%.2fs<2s），继续等待", 
                                           silence_time)
                            elif condition2_met:
                                logger.debug("等待声纹录制模式 - 条件2满足（静音≥2s），但条件1未满足（从首次语音开始=%.2fs<5s），继续等待", 
                                           enroll_duration_from_first_speech)
                        else:
                            # 还没有检测到语音，不累积，只记录日志
                            logger.debug("等待声纹录制模式 - 尚未检测到语音，不累积音频（等待VAD检测到声音）")
                            should_auto_switch = False
                        
                        # 5. 如果满足自动切换条件，保存注册样本并切换到过渡状态
                        if should_auto_switch and not session.is_enrolled:
                            # 保存所有累积的音频（不截取，超过5秒更好）
                            enroll_path = session._save_enroll_sample()
                            if enroll_path:
                                session.enroll_audio_path = enroll_path
                                old_enrolled = session.is_enrolled
                                session.is_enrolled = True
                                saved_buffer_len = len(session.enroll_audio_buffer)
                                saved_duration = saved_buffer_len / 16000
                                logger.info(f"✅ 声纹注册完成：{enroll_path} ({saved_duration:.2f}s, {saved_buffer_len}样本)")
                                logger.info(f"🔄 [自动切换] {auto_switch_reason}，自动切换到ASR状态")
                                
                                # 清空注册缓冲区
                                session.enroll_audio_buffer = np.array([], dtype=np.float32)
                                logger.info("🔄 [SV清除] 声纹注册缓冲区已清空: %d 样本 (%.2fs), 注册状态: %s -> True", 
                                           saved_buffer_len, saved_duration, old_enrolled)
                                
                                # 切换到过渡状态，等待前端确认（不立即切换到ASR）
                                old_mode = session.mode
                                session.mode = "WAITING_FOR_ENROLLMENT_CONFIRM"
                                logger.info("🔄 [模式切换] 会话模式: %s -> WAITING_FOR_ENROLLMENT_CONFIRM (等待前端确认: %s)", old_mode, auto_switch_reason)
                                
                                # 清理enrollment相关的计时器和状态（但保留注册状态）
                                if hasattr(session, 'enroll_last_voice_time'):
                                    delattr(session, 'enroll_last_voice_time')
                                if hasattr(session, 'enroll_silence_timer'):
                                    delattr(session, 'enroll_silence_timer')
                                session.enroll_has_detected_speech = False  # 重置enrollment语音检测标记
                                session.enroll_first_speech_time = None  # 重置enrollment首次语音时间
                                
                                # 发送声纹录制完成信号给前端（类似wakeup信号），让前端关闭弹窗
                                await websocket.send_json({
                                    "type": "enrollment_completed",
                                    "status": "completed",
                                    "message": f"Enrollment is completed, please close the window."
                                })
                                logger.info("已发送声纹录制完成信号给前端，等待前端确认: %s", client_id)
                    
                    except Exception as e:
                        logger.error("等待声纹录制模式处理异常: %s", e, exc_info=True)
                    continue
                
                # ========== 等待前端确认模式：声纹录制已完成，等待前端关闭弹窗并发送确认信号 ==========
                elif session.mode == "WAITING_FOR_ENROLLMENT_CONFIRM":
                    # 在这个模式下：
                    # 1. 不处理音频，不累积任何buffer
                    # 2. 只等待前端发送 start_asr 信号
                    # 3. 收到信号后切换到 ASR_ACTIVE 模式
                    logger.debug("等待前端确认模式 - 声纹录制已完成，等待前端关闭弹窗并发送确认信号: %s", client_id)
                    # 不处理音频，直接跳过，等待前端信号
                    continue
                
                # ========== ASR 处理模式：正常的语音识别流程 ==========
                elif session.mode == "ASR_ACTIVE":
                    # 处理音频片段（KWS已激活，只运行ASR识别）
                    logger.debug("🔍 [ASR_ACTIVE] 只运行ASR识别，不运行KWS: %s (音频长度: %.2fms)", 
                               client_id, len(audio_np) / 16000 * 1000)
                    try:
                        result = session.process_chunk(audio_np)
                        
                        # 注释：不再发送中间结果给前端（测试用，不稳定）
                        # 发送中间结果（如果检测到语音且有文本）
                        # if result["is_speech"] and result["intermediate_text"]:
                        #     await websocket.send_json({
                        #         "type": "processing",
                        #         "status": "processing",
                        #         "intermediate_text": result["intermediate_text"]
                        #     })
                        
                        # 检查是否应该触发最终识别
                        if result["should_finalize"]:
                            logger.info("静默达到阈值，开始最终识别: %s", client_id)
                            # 发送处理中状态，让前端显示"正在处理音频"特效
                            await websocket.send_json({
                                "type": "processing",
                                "status": "finalizing",
                                "message": "正在处理音频..."
                            })
                            final_text = await session.finalize()  # finalize() 现在是 async
                            
                            if final_text == "__SV_VERIFICATION_FAILED__":
                                # 声纹验证失败
                                await websocket.send_json({
                                    "type": "result",
                                    "status": "completed",
                                    "text": "",
                                    "success": False,
                                    "message": "抱歉，请再说一遍！"
                                })
                                logger.info("已发送声纹验证失败消息 (client_id=%s)", client_id)
                            elif final_text == "__SV_NOT_ACTIVATED__":
                                # 未激活状态下不允许声纹验证
                                await websocket.send_json({
                                    "type": "result",
                                    "status": "completed",
                                    "text": "",
                                    "success": False,
                                    "message": "非认证注册声音，拒绝访问。"
                                })
                                logger.info("已发送未激活状态消息 (client_id=%s)", client_id)
                            elif final_text == "__ASR_RESULT_EMPTY__":
                                # ASR识别结果为空
                                await websocket.send_json({
                                    "type": "result",
                                    "status": "completed",
                                    "text": "",
                                    "success": False,
                                    "message": "抱歉，请再说一遍！"
                                })
                                logger.info("已发送识别结果为空消息 (client_id=%s)", client_id)
                            elif final_text:
                                # 正常识别结果 - 添加硬性修正逻辑
                                corrected_text = final_text
                                
                                # 硬性修正规则：
                                # 1. "五"及其同音字/相似字只有在单独出现时才改成"无"（完全匹配，去除空格和标点后）
                                #    例如："五"、"五。"、"五，"、"乌"、"吴"、"屋"、"舞"、"5"、"午"、"吾"、"芜" -> "无"
                                #    但"无其他"、"无其他伴随"、"无既往史"等不需要改
                                # 先去除标点符号和空白字符，再检查是否等于这些字
                                text_without_punct = re.sub(r'[，。！？、；：""''（）【】《》〈〉「」『』〔〕〖〗…—～·\s]', '', corrected_text.strip())
                                # 需要修正为"无"的字列表：五、乌、吴、屋、舞、5（数字）、午、吾、芜
                                should_correct_to_wu = text_without_punct in ["五", "乌", "吴", "屋", "舞", "5", "午", "吾", "芜"]
                                if should_correct_to_wu:
                                    corrected_text = "无"
                                    logger.info("🔧 [硬性修正] '%s' -> '无' (单独出现，去除标点后匹配，原文本: '%s', client_id=%s)", 
                                              text_without_punct, final_text, client_id)
                                
                                # 2. "前妻"无论出现在哪里，都必须改成"前期"（全局替换）
                                #    例如："前妻"、"前妻的"、"有前妻"等都要替换
                                if "前妻" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("前妻", "前期")
                                    logger.info("🔧 [硬性修正] '前妻' -> '前期' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 3. "黑边"和"黑变"无论出现在哪里，都必须改成"黑便"（全局替换）
                                #    例如："黑边"、"黑变"、"有黑边"、"黑变便"等都要替换
                                if "黑边" in corrected_text or "黑变" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("黑边", "黑便")
                                    corrected_text = corrected_text.replace("黑变", "黑便")
                                    logger.info("🔧 [硬性修正] '黑边'/'黑变' -> '黑便' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 4. "腾"、"藤"、"滕"、"誊"无论出现在哪里，都必须改成"疼"（全局替换）
                                #    例如："腾"、"藤"、"滕"、"誊"、"肚子腾"、"腿藤"等都要替换
                                if "腾" in corrected_text or "藤" in corrected_text or "滕" in corrected_text or "誊" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("腾", "疼")
                                    corrected_text = corrected_text.replace("藤", "疼")
                                    corrected_text = corrected_text.replace("滕", "疼")
                                    corrected_text = corrected_text.replace("誊", "疼")
                                    logger.info("🔧 [硬性修正] '腾'/'藤'/'滕'/'誊' -> '疼' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 5. "壳"无论出现在哪里，都必须改成"咳"（全局替换）
                                #    例如："壳黄色粘痰" -> "咳黄色粘痰"
                                if "壳" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("壳", "咳")
                                    logger.info("🔧 [硬性修正] '壳' -> '咳' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 6. "气势"无论出现在哪里，都必须改成"前期"（全局替换）
                                #    例如："气势" -> "前期"
                                if "气势" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("气势", "前期")
                                    logger.info("🔧 [硬性修正] '气势' -> '前期' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 7. "串"和"川"无论出现在哪里，都必须改成"喘"（全局替换）
                                #    例如："串"、"川"、"气喘串"、"串气"等都要替换
                                if "串" in corrected_text or "川" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("串", "喘")
                                    corrected_text = corrected_text.replace("川", "喘")
                                    logger.info("🔧 [硬性修正] '串'/'川' -> '喘' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 8. "涨"和"账"无论出现在哪里，都必须改成"胀"（全局替换）
                                #    例如："涨"、"账"、"肚子涨"、"账气"等都要替换
                                if "涨" in corrected_text or "账" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("涨", "胀")
                                    corrected_text = corrected_text.replace("账", "胀")
                                    logger.info("🔧 [硬性修正] '涨'/'账' -> '胀' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 9. "脱腾"、"拖腾"、"拖疼"、"脱疼"无论出现在哪里，都必须改成"头疼"（全局替换）
                                #    例如："脱腾"、"拖腾"、"拖疼"、"脱疼"、"我脱腾"、"拖疼得很"等都要替换
                                if "脱腾" in corrected_text or "拖腾" in corrected_text or "拖疼" in corrected_text or "脱疼" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("脱腾", "头疼")
                                    corrected_text = corrected_text.replace("拖腾", "头疼")
                                    corrected_text = corrected_text.replace("拖疼", "头疼")
                                    corrected_text = corrected_text.replace("脱疼", "头疼")
                                    logger.info("🔧 [硬性修正] '脱腾'/'拖腾'/'拖疼'/'脱疼' -> '头疼' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 10. "游离"和"游历"无论出现在哪里，都必须改成"油腻"（全局替换）
                                #    例如："游离"、"游历"、"食物游离"、"游历的食物"等都要替换
                                if "游离" in corrected_text or "游历" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("游离", "油腻")
                                    corrected_text = corrected_text.replace("游历", "油腻")
                                    logger.info("🔧 [硬性修正] '游离'/'游历' -> '油腻' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 11. "颜面不通"无论出现在哪里，都必须改成"颜面部痛"（全局替换）
                                #    例如："颜面不通"、"我颜面不通"等都要替换
                                if "颜面不通" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("颜面不通", "颜面部痛")
                                    logger.info("🔧 [硬性修正] '颜面不通' -> '颜面部痛' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 12. "即性"无论出现在哪里，都必须改成"急性"（全局替换）
                                #    例如："即性"、"即性疾病"等都要替换
                                if "即性" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("即性", "急性")
                                    logger.info("🔧 [硬性修正] '即性' -> '急性' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 13. "犯罪症状"无论出现在哪里，都必须改成"伴随症状"（全局替换）
                                #    例如："犯罪症状"、"有犯罪症状"等都要替换
                                if "犯罪症状" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("犯罪症状", "伴随症状")
                                    logger.info("🔧 [硬性修正] '犯罪症状' -> '伴随症状' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 14. "树叶"、"书页"、"术业"、"树业"无论出现在哪里，都必须改成"输液"（全局替换）
                                #    例如："树叶"、"书页"、"术业"、"树业"、"正在树叶"等都要替换
                                if "树叶" in corrected_text or "书页" in corrected_text or "术业" in corrected_text or "树业" in corrected_text:
                                    old_text = corrected_text
                                    corrected_text = corrected_text.replace("树叶", "输液")
                                    corrected_text = corrected_text.replace("书页", "输液")
                                    corrected_text = corrected_text.replace("术业", "输液")
                                    corrected_text = corrected_text.replace("树业", "输液")
                                    logger.info("🔧 [硬性修正] '树叶'/'书页'/'术业'/'树业' -> '输液' (全局替换): '%s' -> '%s' (client_id=%s)", 
                                              old_text, corrected_text, client_id)
                                
                                # 15. 全局去掉所有拟声词/语气词（保留标点符号）
                                #    例如："啊，我头疼。" -> "，我头疼。"
                                #    例如："我呃呃不知道呜呜呜呜怎么说" -> "我不知道怎么说"
                                #    例如："这个症状啊，其他的" -> "这个症状，其他的"
                                #    例如："嗯哎呦妈呀。" -> "。"（全部是拟声词，保留标点）
                                
                                # 定义拟声词的正则模式（匹配连续的拟声词，包括单个和重复）
                                # 注意：使用字符集合匹配，会匹配所有连续的拟声词字符
                                interjection_pattern = r'[嗯哈哼噗砰呀嗷啊哦额呃诶唉哎呦妈]+'
                                
                                # 全局替换所有拟声词为空字符串（保留标点符号）
                                original_text = corrected_text
                                corrected_text = re.sub(interjection_pattern, '', corrected_text)
                                
                                if corrected_text != original_text:
                                    logger.info("🔧 [硬性修正] 去掉所有拟声词: '%s' -> '%s' (client_id=%s)", 
                                              original_text, corrected_text, client_id)
                                
                                # 16. LLM大模型纠错（如果启用）
                                if use_llm and corrected_text:
                                    try:
                                        llm_start = time.perf_counter()
                                        # 记录修改前的文本
                                        text_before_llm = corrected_text
                                        logger.info("🔍 [LLM大模型纠错] 开始处理 (client_id=%s)", client_id)
                                        logger.info("📥 [LLM大模型纠错] 修改前文本: '%s' (client_id=%s)", text_before_llm, client_id)
                                        
                                        # 加载热词列表
                                        hotwords = load_hotwords_list()
                                        logger.debug("📋 [LLM大模型纠错] 已加载 %d 个热词 (client_id=%s)", len(hotwords), client_id)
                                        
                                        # 调用LLM大模型纠错（仅修正文本，不进行匹配）
                                        llm_corrected_text = correct_text_only(
                                            latest_context=None,
                                            latest_options=hotwords,
                                            text=corrected_text,
                                            DEBUG=False
                                        )
                                        llm_time = (time.perf_counter() - llm_start) * 1000
                                        
                                        # 记录修改后的文本
                                        logger.info("📤 [LLM大模型纠错] 修改后文本: '%s' (耗时: %.2f ms, client_id=%s)", 
                                                  llm_corrected_text, llm_time, client_id)
                                        
                                        if llm_corrected_text and llm_corrected_text != text_before_llm:
                                            logger.info("✅ [LLM大模型纠错] 文本已修改: '%s' -> '%s' (耗时: %.2f ms, client_id=%s)", 
                                                      text_before_llm, llm_corrected_text, llm_time, client_id)
                                            corrected_text = llm_corrected_text
                                        else:
                                            logger.info("➡️  [LLM大模型纠错] 文本未修改，保持原样: '%s' (耗时: %.2f ms, client_id=%s)", 
                                                      text_before_llm, llm_time, client_id)
                                    except Exception as e:
                                        logger.warning("❌ [LLM大模型纠错] 异常，使用硬性修正结果: %s (原文本: '%s', client_id=%s)", 
                                                     e, corrected_text, client_id, exc_info=True)
                                
                                await websocket.send_json({
                                    "type": "result",
                                    "status": "completed",
                                    "text": corrected_text,
                                    "success": True
                                })
                                if corrected_text != final_text:
                                    logger.info("已发送最终识别结果（已修正）: '%s' -> '%s' (client_id=%s)", 
                                              final_text, corrected_text, client_id)
                                else:
                                    logger.info("已发送最终识别结果: '%s' (client_id=%s)", corrected_text, client_id)
                            else:
                                # 兜底：不应该到达这里，但以防万一
                                await websocket.send_json({
                                    "type": "result",
                                    "status": "completed",
                                    "text": "",
                                    "success": False,
                                    "message": "抱歉，请再说一遍！"
                                })
                                logger.warning("⚠️ final_text为空，发送默认空结果消息 (client_id=%s)", client_id)
                            
                            # 一句话识别完成，重置 ASR 状态，但保持在 ASR_ACTIVE 模式
                            # 继续监听下一句话，直到前端通过 use_wake 参数或 interrupt 消息取消激活
                            session.reset_asr_state()  # 只重置 ASR 状态，不改变模式
                            logger.info("识别完成，已重置 ASR 状态，继续监听下一句话（模式: %s）: %s", session.mode, client_id)
                    
                    except Exception as e:
                        logger.error("处理音频片段异常: %s", e, exc_info=True)
                        await websocket.send_json({
                            "type": "error",
                            "message": f"处理音频片段时发生错误: {str(e)}",
                            "code": "PROCESSING_ERROR"
                        })
                        continue
                
                # 未知模式（理论上不应该发生）
                else:
                    logger.warning("未知的会话模式: %s，重置为等待唤醒模式: %s", session.mode, client_id)
                    session.mode = "WAITING_FOR_WAKEUP"
                    session.reset()
                    continue
                
            except WebSocketDisconnect:
                logger.info("WebSocket客户端断开连接: %s", client_id)
                break
            except Exception as e:
                logger.error("处理WebSocket消息异常: client_id=%s, error=%s", client_id, e, exc_info=True)
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"处理请求时发生错误: {str(e)}",
                        "code": "PROCESSING_ERROR"
                    })
                except Exception:
                    # 如果无法发送错误消息，可能连接已断开
                    logger.warning("无法发送错误消息，可能连接已断开: %s", client_id)
                    break
                
    except WebSocketDisconnect:
        logger.info("WebSocket连接断开: %s", client_id)
    except Exception as e:
        logger.error("WebSocket连接异常: client_id=%s, error=%s", client_id, e, exc_info=True)
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass
    finally:
        # 清理会话状态（reset() 会根据 use_wake 决定模式）
        if session:
            session.reset()
            logger.info("会话状态已清理: %s (use_wake=%s, mode=%s)", 
                       client_id, session.use_wake, session.mode)
