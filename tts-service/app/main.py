"""
TTS Service 主应用入口
"""
import os
import sys
import logging
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# 确保当前目录在 Python 路径中（解决 IDE 和运行时路径问题）
# 获取当前文件的目录（app/）
_current_dir = Path(__file__).parent
# 获取服务根目录（services/tts-service/）
_service_root = _current_dir.parent
# 将服务根目录添加到 Python 路径
if str(_service_root) not in sys.path:
    sys.path.insert(0, str(_service_root))

# 现在可以安全地导入 app 模块
from app.api import tts  # type: ignore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True  # 强制重新配置，避免被其他模块覆盖
)
# 确保日志立即输出（不缓冲）
logging.getLogger().handlers[0].setLevel(logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理（替代已弃用的 on_event）
    - startup: 应用启动时执行
    - shutdown: 应用关闭时执行
    """
    # Startup: 应用启动时执行
    logger.info("TTS Service 启动中...")
    
    # 后台异步预加载模型（方案2：后台预热）
    # 不阻塞 FastAPI 启动，但模型在后台开始加载
    from app.api.tts import get_tts_manager  # type: ignore
    import asyncio
    
    tts_manager = get_tts_manager()
    
    # 在后台线程中异步加载模型（不阻塞启动）
    def preload_model():
        """后台预加载模型的同步函数"""
        try:
            logger.info("[TTS] 开始后台预加载模型...")
            tts_manager._ensure_pipeline(wait_if_loading=False)
            logger.info("[TTS] 模型后台预加载完成")
        except Exception as e:
            logger.error(f"[TTS] 模型后台预加载失败: {e}", exc_info=True)
    
    # 使用线程池执行器在后台加载（不阻塞）
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, preload_model)
    
    logger.info("TTS Service 启动完成（模型正在后台加载，首次请求时可能已就绪）")
    
    yield  # 应用运行期间
    
    # Shutdown: 应用关闭时执行
    logger.info("TTS Service 正在关闭...")


app = FastAPI(
    title="TTS Service",
    version="1.0.0",
    description="Text-to-Speech Service with async job management",
    lifespan=lifespan  # 使用新的 lifespan 参数
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    logger.error(f"[请求验证错误] {request.method} {request.url}")
    logger.error(f"错误详情: {exc.errors()}")
    logger.error(f"请求体: {await request.body()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "请求参数验证失败",
            "detail": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """处理所有未捕获的异常"""
    logger.error(f"[未捕获异常] {request.method} {request.url}")
    logger.error(f"异常类型: {type(exc).__name__}")
    logger.error(f"异常信息: {str(exc)}")
    logger.error(f"异常堆栈:\n{traceback.format_exc()}")
    
    # 尝试读取请求体（用于调试）
    try:
        body = await request.body()
        if body:
            logger.error(f"请求体: {body[:500]}")  # 只记录前500字符
    except Exception as e:
        logger.error(f"无法读取请求体: {e}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "服务器内部错误",
            "error": str(exc) if os.getenv("APP_ENVIRONMENT") == "development" else "Internal server error"
        }
    )

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有请求"""
    logger.info(f"[请求] {request.method} {request.url.path}")
    logger.info(f"[请求查询参数] {dict(request.query_params)}")
    
    # 对于 POST 请求，尝试记录 Content-Type
    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        logger.info(f"[请求 Content-Type] {content_type}")
    
    try:
        response = await call_next(request)
        logger.info(f"[响应] {request.method} {request.url.path} - 状态码: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"[请求异常] {request.method} {request.url.path} - {type(e).__name__}: {e}")
        raise

# 注册路由
# API路由
app.include_router(tts.router, prefix="/api/v1/tts", tags=["tts"])

@app.get("/health")
async def health():
    """健康检查"""
    from app.api.tts import get_tts_manager  # type: ignore
    
    # 获取全局 TTS 管理器实例（用于检查模型是否加载）
    tts_manager = get_tts_manager()
    
    return {
        "status": "healthy",
        "model_loaded": tts_manager._tts_pipeline is not None,
        "active_jobs": len([
            j for j in tts_manager.jobs.values()
            if j["status"] in ["pending", "processing"]
        ])
    }



