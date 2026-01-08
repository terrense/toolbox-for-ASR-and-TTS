"""
配置管理模块
从环境变量中加载配置
"""

from typing import List, Union, Optional, Any, Dict

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """服务器配置"""
    host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    port: int = Field(default=8001, alias="SERVER_PORT")  # Different port from diagnostic service


class SSLConfig(BaseSettings):
    """SSL/TLS配置"""
    cert_path: str = Field(default="certs/cert.pem", alias="SSL_CERT_PATH")
    key_path: str = Field(default="certs/key.pem", alias="SSL_KEY_PATH")


class SecurityConfig(BaseSettings):
    """安全配置"""
    # 使用 str 类型避免 pydantic-settings 的 JSON 解析问题
    # pydantic-settings 对 List[str] 会先尝试 JSON 解析，导致逗号分隔字符串解析失败
    allowed_hosts_str: Optional[str] = Field(
        default=None,
        alias="SECURITY_ALLOWED_HOSTS"
    )
    cors_origins_str: Optional[str] = Field(
        default=None,
        alias="SECURITY_CORS_ORIGINS"
    )
    
    @field_validator('allowed_hosts_str', 'cors_origins_str', mode='before')
    @classmethod
    def parse_list_from_string(cls, v: Union[str, List[str], None]) -> Optional[str]:
        """
        将环境变量值转换为字符串（如果已经是字符串则保持）
        如果传入的是列表（从 JSON 解析来的），转换为逗号分隔字符串
        """
        if v is None:
            return None
        if isinstance(v, list):
            # 如果已经是列表（从 JSON 解析来的），转换为逗号分隔字符串
            return ','.join(str(item) for item in v)
        # 如果已经是字符串，直接返回
        return str(v) if v else None
    
    @property
    def allowed_hosts(self) -> List[str]:
        """将字符串转换为列表"""
        if self.allowed_hosts_str and self.allowed_hosts_str.strip():
            hosts = [item.strip() for item in self.allowed_hosts_str.split(',') if item.strip()]
            # 如果包含 "*"，则允许所有主机（用于开发环境）
            if "*" in hosts:
                return ["*"]
            return hosts
        # 使用默认值（包含端口号变体以支持 WebSocket）
        return ["localhost", "127.0.0.1", "*.local", "*"]
    
    @property
    def cors_origins(self) -> List[str]:
        """将字符串转换为列表"""
        if self.cors_origins_str and self.cors_origins_str.strip():
            return [item.strip() for item in self.cors_origins_str.split(',') if item.strip()]
        # 使用默认值
        return ["http://localhost:3000", "https://localhost:3000", "http://127.0.0.1:3000"]


class VoiceServiceConfig(BaseSettings):
    """语音服务功能开关配置"""
    # FunASR相关
    funasr_disable_lm: bool = Field(default=False, alias="FUNASR_DISABLE_LM")
    """是否禁用FunASR的LM模块（True=仅AM，False=AM+LM）"""
    
    # LLM后处理
    voice_disable_llm: bool = Field(default=False, alias="VOICE_DISABLE_LLM")
    """是否全局禁用LLM后处理（True=禁用，False=允许）"""
    
    # 样本保存
    voice_always_save_sample: bool = Field(default=False, alias="VOICE_ALWAYS_SAVE_SAMPLE")
    """是否自动保存所有ASR样本（True=总是保存，False=按请求参数）"""
    
    # KWS唤醒
    voice_require_wake: bool = Field(default=False, alias="VOICE_REQUIRE_WAKE")
    """是否全局强制要求KWS唤醒（True=强制唤醒，False=按请求参数）"""
    
    @field_validator('funasr_disable_lm', 'voice_disable_llm', 'voice_always_save_sample', 'voice_require_wake', mode='before')
    @classmethod
    def parse_bool_from_string(cls, v: Union[str, bool, int]) -> bool:
        """将字符串/数字转换为布尔值"""
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        if isinstance(v, str):
            return v.lower() in ('1', 'true', 'yes', 'on')
        return False
#这样组合确保了在创建VoiceServiceConfig实例时，相关字段的值会自动经过parse_bool_from_string方法的验证和转换


class AppConfig(BaseSettings):
    """应用配置"""
    name: str = Field(default="HGDoctor Voice Service", alias="APP_NAME")
    version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="APP_ENVIRONMENT")

    voice_server: ServerConfig = Field(default_factory=ServerConfig)
    ssl: SSLConfig = Field(default_factory=SSLConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    voice_service: VoiceServiceConfig = Field(default_factory=VoiceServiceConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例 - 直接从环境变量加载
config = AppConfig()
