from typing import Optional

from pydantic import BaseModel


class ASRRequest(BaseModel):
    """语音识别请求"""
    audio_data: str  # base64 encoded audio data
    use_wake: Optional[bool] = None  # 是否使用唤醒词检测，None时使用默认值True
    use_llm: Optional[bool] = None  # 是否使用LLM后处理，None时使用默认值True
    save_sample: bool = False
    sample_id: Optional[str] = None
    diagnosis_session_id: str  # 诊断会话ID，用于保存音频文件


class ASRResponse(BaseModel):
    """语音识别响应"""
    text: str
    success: bool
    message: Optional[str] = None
    sample_id: Optional[str] = None


# WebSocket 消息模型
class WebSocketAudioMessage(BaseModel):
    """WebSocket 音频消息"""
    type: str = "audio"  # 消息类型
    audio_data: str  # base64 encoded audio data
    use_wake: Optional[bool] = None
    use_llm: Optional[bool] = None
    use_sv: Optional[bool] = True  # 是否启用声纹验证，默认True
    save_sample: bool = False
    sample_id: Optional[str] = None
    diagnosis_session_id: str


class WebSocketResultMessage(BaseModel):
    """WebSocket 识别结果消息"""
    type: str = "result"  # 消息类型
    text: str
    success: bool
    message: Optional[str] = None
    sample_id: Optional[str] = None


class WebSocketErrorMessage(BaseModel):
    """WebSocket 错误消息"""
    type: str = "error"  # 消息类型
    message: str
    code: Optional[str] = None


class WebSocketWelcomeMessage(BaseModel):
    """WebSocket 欢迎消息"""
    type: str = "welcome"  # 消息类型
    message: str
    timestamp: float

