import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.api import voice
from app.config import config
from app.services.voice_interface import init_streaming_models, init_speaker_diarization_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI生命周期管理：在服务启动时初始化模型，在服务关闭时清理资源
    """
    # 启动时：读取并记录FunASR LM配置状态
    try:
        from app.config import config
        voice_config = getattr(config, "voice_service", None)
        if voice_config:
            funasr_disable_lm = voice_config.funasr_disable_lm
            lm_status = "已禁用" if funasr_disable_lm else "已启用"
            logger.info("🔧 [FunASR LM配置] funasr_disable_lm=%s (%s) - 注意：此配置仅用于记录，实际LM控制需在FunASR服务端配置", 
                       funasr_disable_lm, lm_status)
        else:
            logger.warning("⚠️ [FunASR LM配置] 无法读取voice_service配置")
    except Exception as e:
        logger.warning("⚠️ [FunASR LM配置] 读取配置异常: %s", e)
    
    # 启动时：初始化流式处理模型
    logger.info("🚀 服务启动中，开始初始化流式处理模型...")
    try:
        # 在后台线程中初始化模型（避免阻塞服务启动）
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # 在线程池中执行模型初始化（同步函数）
            # 使用run_in_executor避免阻塞事件循环
            future = loop.run_in_executor(executor, init_streaming_models)
            await future
        logger.info("✅ 流式处理模型初始化完成")
        
        # 初始化说话人分离模型
        logger.info("🚀 开始初始化说话人分离模型...")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(executor, init_speaker_diarization_model)
                await future
        except Exception as e:
            logger.error("❌ 说话人分离模型初始化失败: %s", e, exc_info=True)
            logger.warning("⚠️ 服务将继续启动，但首次使用时可能需要等待模型加载")
        
        logger.info("✅ 所有模型初始化完成，服务已就绪")
    except Exception as e:
        logger.error("❌ 流式处理模型初始化失败: %s", e, exc_info=True)
        logger.warning("⚠️ 服务将继续启动，但首次WebSocket连接时可能需要等待模型加载")
    
    yield  # 服务运行中
    
    # 关闭时：清理资源（如果需要）
    logger.info("🛑 服务关闭中，清理资源...")


app = FastAPI(
    title=config.name, 
    version=config.version,
    lifespan=lifespan
)

# 安全中间件配置
# 注意：TrustedHostMiddleware 可能会阻止 WebSocket 连接，所以暂时禁用
# 如果需要启用，应该确保 allowed_hosts 包含 "*" 或所有可能的 Host 头
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=config.security.allowed_hosts
# )

# 2. Gzip压缩中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 3. CORS配置（参考 test_voice.py，使用更宽松的配置以支持 WebSocket）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in config.security.cors_origins else config.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法，包括 WebSocket 升级请求需要的 OPTIONS
    allow_headers=["*"],
)

# API路由
app.include_router(voice.router, prefix="/api/v1/voice", tags=["voice"])

# 错误处理中间件（跳过 WebSocket 升级请求）


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    # 跳过 WebSocket 升级请求（避免干扰 WebSocket 握手）
    is_websocket_upgrade = (
        request.headers.get("upgrade", "").lower() == "websocket" or
        "upgrade" in request.headers.get("connection", "").lower()
    )
    
    if is_websocket_upgrade:
        # WebSocket 升级请求直接通过，不进行错误处理
        return await call_next(request)
    
    try:
        response = await call_next(request)

        # 记录错误响应
        if response.status_code >= 400:
            logger.error("错误响应: %s - %s %s", response.status_code, request.method, request.url)

        return response
    except Exception as e:
        logger.error("未处理的异常: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误", "error": str(e)}
        )


# 请求日志中间件（简化版本，避免阻塞）
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # 跳过 WebSocket 升级请求的详细日志（避免干扰 WebSocket 连接）
    is_websocket_upgrade = (
        request.headers.get("upgrade", "").lower() == "websocket" or
        "upgrade" in request.headers.get("connection", "").lower()
    )
    
    # WebSocket 升级请求直接通过，不进行任何处理
    if is_websocket_upgrade:
        return await call_next(request)
    
    # 简化日志，避免阻塞
    logger.info("请求: %s %s", request.method, request.url)
    
    # 对于 GET 请求（如 /health），不读取请求体，直接处理
    if request.method == "GET":
        response = await call_next(request)
        logger.info("响应: %s %s", response.status_code, request.url)
        return response
    
    # 对于 POST 请求，简化处理，不读取请求体（避免阻塞）
    response = await call_next(request)
    logger.info("响应: %s %s", response.status_code, request.url)
    return response


# 安全响应头中间件（跳过 WebSocket 升级请求）
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    # 跳过 WebSocket 升级请求（避免干扰 WebSocket 握手）
    is_websocket_upgrade = (
        request.headers.get("upgrade", "").lower() == "websocket" or
        "upgrade" in request.headers.get("connection", "").lower()
    )
    
    response = await call_next(request)
    
    # WebSocket 升级请求不添加安全响应头
    if is_websocket_upgrade:
        return response

    # 添加安全响应头
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    return response


@app.get("/")
async def root():
    return {"message": "HGDoctor Voice Service is running", "service": "voice"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "voice"}
