"""
TTS Service 数据模型
"""
from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """TTS 请求模型"""
    text: str = Field(..., description="要转换的文本内容", min_length=1)
    voice: str = Field(default="zhitian_emo", description="语音类型（可选）")


class CancelRequest(BaseModel):
    """取消任务请求模型"""
    job_id: str = Field(..., description="任务 ID")


class TTSResponse(BaseModel):
    """TTS 响应模型"""
    status: str = Field(..., description="任务状态")
    job_id: str = Field(..., description="任务 ID")
    message: str = Field(default="", description="消息")


class TTSResultResponse(BaseModel):
    """TTS 结果响应模型"""
    status: str = Field(..., description="任务状态")
    job_id: str = Field(..., description="任务 ID")
    audio_base64: str = Field(default="", description="Base64 编码的音频数据")
    text: str = Field(default="", description="原始文本")
    audio_size: int = Field(default=0, description="音频大小（字节）")
    message: str = Field(default="", description="消息")
    error: str = Field(default="", description="错误信息")

