"""
TTS Service API 路由
"""
import logging
import time
from fastapi import APIRouter, HTTPException
from app.models.tts import TTSRequest, CancelRequest, TTSResponse, TTSResultResponse  # type: ignore
from app.services.tts_service import TTSManager  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tts"])  # prefix 在 main.py 中统一设置

_tts_manager_instance = None

def get_tts_manager() -> TTSManager:
    """获取 TTS 管理器实例（单例）"""
    global _tts_manager_instance
    if _tts_manager_instance is None:
        _tts_manager_instance = TTSManager()
    return _tts_manager_instance

# 导出供其他模块使用
tts_manager = get_tts_manager()


@router.post("/start", response_model=TTSResponse)
async def start_tts(request: TTSRequest):
    """启动 TTS 任务"""
    api_start_time = time.time()  # API 接收请求时间
    
    try:
        logger.info(f"[TTS API] 接收到 TTS 请求 - text: '{request.text[:100]}{'...' if len(request.text) > 100 else ''}' (长度: {len(request.text)} 字符), voice: '{request.voice}'")
        
        validation_start = time.time()
        if not request.text or not request.text.strip():
            logger.warning(f"[TTS API] 请求验证失败: text 为空")
            raise HTTPException(status_code=400, detail="text 不能为空")
        validation_elapsed = (time.time() - validation_start) * 1000
        
        task_start_time = time.time()
        job_id = await tts_manager.start_task(request.text, request.voice)
        task_creation_elapsed = (time.time() - task_start_time) * 1000
        
        api_total_elapsed = (time.time() - api_start_time) * 1000
        
        logger.info(
            f"[TTS API] TTS 任务已启动 - job_id: {job_id}, "
            f"API总耗时: {api_total_elapsed:.2f}ms "
            f"(验证: {validation_elapsed:.2f}ms, 任务创建: {task_creation_elapsed:.2f}ms)"
        )
        
        return TTSResponse(
            status="started",
            job_id=job_id,
            message="TTS 任务已启动"
        )
    except HTTPException:
        # 重新抛出 HTTPException，让 FastAPI 正常处理
        raise
    except Exception as e:
        # 捕获所有其他异常并记录
        logger.error(f"[TTS API] 启动 TTS 任务时发生异常: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"启动 TTS 任务失败: {str(e)}"
        )


@router.post("/cancel", response_model=TTSResponse)
async def cancel_tts(request: CancelRequest):
    """取消 TTS 任务"""
    result = await tts_manager.cancel_task(request.job_id)
    
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"任务 {request.job_id} 不存在")
    
    return TTSResponse(
        status=result["status"],
        job_id=request.job_id,
        message=result["message"]
    )


@router.get("/result/{job_id}", response_model=TTSResultResponse)
async def get_tts_result(job_id: str):
    """获取 TTS 结果（轮询）"""
    result = await tts_manager.get_result(job_id)
    
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    
    # 如果任务完成，记录总耗时（从任务中获取，不返回给客户端）
    if result["status"] == "completed":
        # 从任务管理器获取耗时信息（仅用于日志）
        job_info = tts_manager.jobs.get(job_id, {})
        elapsed_time = job_info.get("_elapsed_time_ms", 0)
        detailed_timing = job_info.get("_detailed_timing", {})
        audio_size = result.get("audio_size", 0)
        
        if elapsed_time > 0:
            if detailed_timing:
                logger.info(
                    f"[TTS API] 返回 TTS 结果 - job_id: {job_id}, "
                    f"总耗时: {elapsed_time:.2f}ms, "
                    f"详细: 线程等待={detailed_timing.get('thread_wait_ms', 0):.2f}ms, "
                    f"Pipeline检查={detailed_timing.get('pipeline_check_ms', 0):.2f}ms, "
                    f"TTS生成={detailed_timing.get('tts_generation_ms', 0):.2f}ms, "
                    f"Base64编码={detailed_timing.get('base64_encode_ms', 0):.2f}ms, "
                    f"音频大小: {audio_size} 字节"
                )
            else:
                logger.info(f"[TTS API] 返回 TTS 结果 - job_id: {job_id}, 总耗时: {elapsed_time:.2f}ms, 音频大小: {audio_size} 字节")
        else:
            logger.info(f"[TTS API] 返回 TTS 结果 - job_id: {job_id}, 音频大小: {audio_size} 字节")
    
    # 注意：error 状态应该返回 JSON，而不是抛出异常，这样客户端可以正常处理
    # 返回 TTSResultResponse，其中 status="error"，error 字段包含错误信息
    return TTSResultResponse(**result)


@router.delete("/jobs/{job_id}")
async def cleanup_job(job_id: str):
    """清理已完成的任务（可选，用于内存管理）"""
    result = await tts_manager.cleanup_job(job_id)
    
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    
    if result["status"] == "cannot_cleanup":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return {"status": "deleted", "job_id": job_id}

