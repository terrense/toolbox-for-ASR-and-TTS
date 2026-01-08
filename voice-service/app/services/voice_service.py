"""
voice_service.

This module provides the VoiceService class for handling voice interactions,
including Automatic Speech Recognition (ASR) functionality.
"""

import base64
import logging
import os
import tempfile
import time
from datetime import datetime
import os

from app.models.voice import ASRRequest, ASRResponse
from app.config import config
from shared.core.paths import GENERATED_DIR
# Import the voice interface functions
from .voice_interface import asr_wake, StreamingASRSession, init_streaming_models


logger = logging.getLogger(__name__)


class VoiceService:
    """语音服务 - 处理语音识别"""

    def __init__(self):
        logger.info("VoiceService initialized")
        # 注意：模型初始化已在服务启动时完成（main.py的lifespan事件）
        # 这里只需要确保模型已初始化（如果启动时失败，这里会重试）
        try:
            from app.services.voice_interface import get_streaming_models
            # 尝试获取模型，如果未初始化会自动初始化
            get_streaming_models()
            logger.debug("VoiceService: 流式处理模型已就绪")
        except Exception as e:
            logger.warning("VoiceService: 流式处理模型初始化失败（将在首次使用时重试）: %s", e)
    
    def create_streaming_session(self) -> StreamingASRSession:
        """
        创建流式处理会话（用于WebSocket流式识别）
        
        Returns:
            StreamingASRSession: 流式ASR会话实例
        """
        return StreamingASRSession()

    async def recognize_speech(self, request: ASRRequest) -> ASRResponse:
        """
        处理语音识别请求
        """
        service_start = time.perf_counter()
        logger.info("VoiceService.recognize_speech开始: use_wake=%s, use_llm=%s", 
                   request.use_wake, request.use_llm)

        try:
            # 优先复用已保存样本
            samples_dir = GENERATED_DIR / "asr_samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            sample_id_to_return = None

            voice_config = getattr(config, "voice_service", None)
            always_save = voice_config.voice_always_save_sample if voice_config else False

            if request.sample_id:
                sample_start = time.perf_counter()
                candidate = samples_dir / request.sample_id
                if not candidate.exists() and not str(candidate).lower().endswith('.wav'):
                    candidate = candidate.with_suffix('.wav')
                if candidate.exists():
                    temp_audio_path = str(candidate)
                    logger.info("复用样本文件: %s", temp_audio_path)
                    sample_time = (time.perf_counter() - sample_start) * 1000
                    logger.info("耗时统计 - 查找样本文件: %.2f ms", sample_time)
                else:
                    raise FileNotFoundError(f"找不到样本文件: {candidate}")
            else:
                # Decode base64 audio data to temporary file
                decode_start = time.perf_counter()
                logger.info("解码音频数据: %d 字符", len(request.audio_data))
                audio_bytes = base64.b64decode(request.audio_data)
                decode_time = (time.perf_counter() - decode_start) * 1000
                logger.info("解码后音频大小: %d 字节", len(audio_bytes))
                logger.info("耗时统计 - Base64解码: %.2f ms", decode_time)

                # Create temporary audio file
                file_create_start = time.perf_counter()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_audio.write(audio_bytes)
                    temp_audio_path = temp_audio.name
                file_create_time = (time.perf_counter() - file_create_start) * 1000
                logger.info("创建临时音频文件: %s", temp_audio_path)
                logger.info("耗时统计 - 创建临时文件: %.2f ms", file_create_time)

            try:
                # Use the voice interface for recognition
                # 处理 None 值：如果为 None，使用默认值 True（保持向后兼容）
                effective_use_wake = request.use_wake if request.use_wake is not None else True
                effective_use_llm = request.use_llm if request.use_llm is not None else True
                
                # 读取配置：全局LLM开关（如果配置禁用，则覆盖）
                config_read_start = time.perf_counter()
                voice_config = getattr(config, "voice_service", None)
                if voice_config and voice_config.voice_disable_llm:
                    effective_use_llm = False  # 配置全局禁用时覆盖请求参数
                config_read_time = (time.perf_counter() - config_read_start) * 1000
                logger.info("调用asr_wake: audio_file=%s, use_wake=%s, use_llm=%s", 
                           temp_audio_path, effective_use_wake, effective_use_llm)
                logger.info("耗时统计 - 读取配置: %.2f ms", config_read_time)

                asr_wake_start = time.perf_counter()
                result_text = await asr_wake(
                    audio_file=temp_audio_path,
                    use_wake=effective_use_wake,
                    use_LLM=effective_use_llm
                )
                asr_wake_time = (time.perf_counter() - asr_wake_start) * 1000
                logger.info("ASR识别完成: 结果长度=%d", len(result_text))
                logger.info("耗时统计 - asr_wake总耗时: %.2f ms", asr_wake_time)

                # 保存样本（保存上传的原始wav，便于复用对比）
                if not request.sample_id and (request.save_sample or always_save):
                    save_start = time.perf_counter()
                    from hashlib import sha1
                    digest = sha1(temp_audio_path.encode('utf-8')).hexdigest()[:8]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"sample_{timestamp}_{digest}.wav"
                    dst_path = samples_dir / filename
                    try:
                        import shutil
                        shutil.copyfile(temp_audio_path, dst_path)
                        sample_id_to_return = filename
                        logger.info("保存样本: %s", dst_path)
                        save_time = (time.perf_counter() - save_start) * 1000
                        logger.info("耗时统计 - 保存样本: %.2f ms", save_time)
                    except Exception as e:
                        logger.warning("保存样本失败: %s", e)

                # 清理仅在非复用样本情况下创建的临时文件
                if not request.sample_id:
                    cleanup_start = time.perf_counter()
                    try:
                        os.unlink(temp_audio_path)
                        logger.info("临时文件已清理")
                        cleanup_time = (time.perf_counter() - cleanup_start) * 1000
                        logger.info("耗时统计 - 清理临时文件: %.2f ms", cleanup_time)
                    except Exception:
                        pass

                total_service_time = (time.perf_counter() - service_start) * 1000
                logger.info("耗时统计 - VoiceService.recognize_speech总耗时: %.2f ms", total_service_time)

                return ASRResponse(
                    text=result_text,
                    success=True,
                    message="语音识别成功",
                    sample_id=sample_id_to_return
                )

            except Exception as e:
                logger.error("ASR处理异常: %s", e, exc_info=True)
                # Clean up temporary file in case of error
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    logger.info("异常时临时文件已清理")
                raise e

        except Exception as e:
            logger.error("VoiceService.recognize_speech异常: %s", e, exc_info=True)
            return ASRResponse(
                text="",
                success=False,
                message=f"语音识别失败: {str(e)}")

    def _save_audio_to_generated_dir(self, audio_bytes: bytes, diagnosis_session_id: str):
        """
        保存音频文件到GENERATED_DIR目录，文件操作失败不影响语音识别

        Args:
            audio_bytes: 音频字节数据
            diagnosis_session_id: 诊断会话ID
        """
        try:
            # 验证会话ID格式
            import uuid
            uuid.UUID(diagnosis_session_id)

            # 创建保存目录 GENERATED_DIR/{diagnosis_session_id}/
            session_dir = GENERATED_DIR / diagnosis_session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            # 生成格式化的时间字符串作为文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"audio_{timestamp}.wav"
            file_path = session_dir / filename

            # 写入音频数据到文件
            with open(file_path, 'wb') as f:
                f.write(audio_bytes)

            logger.info("✅ 成功保存音频文件: %s", file_path)

        except ValueError:
            logger.warning("⚠️ 无效的诊断会话ID格式: %s", diagnosis_session_id)
        except Exception as e:
            logger.warning("⚠️ 保存音频文件失败: %s", str(e))
            # 文件操作失败不影响语音识别，只记录日志
