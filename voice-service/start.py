#!/usr/bin/env python3
"""
Unified launcher for FunASR (WebSocket) + Voice Service (FastAPI).

Placed alongside start.py to simplify running inside/outside Docker.
"""

import os
import sys
import time
import signal
import socket
import subprocess


# --- Ensure imports work both locally and in container ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, "..", ".."))

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Prefer in-tree shared for container (Dockerfile copies to voice-service/shared),
# fallback to repo-level shared for local runs
SHARED_IN_APP = os.path.join(APP_DIR, "shared")
SHARED_IN_REPO = os.path.join(REPO_ROOT, "shared")
if os.path.isdir(SHARED_IN_APP) and SHARED_IN_APP not in sys.path:
    sys.path.insert(0, SHARED_IN_APP)
elif os.path.isdir(SHARED_IN_REPO) and SHARED_IN_REPO not in sys.path:
    sys.path.insert(0, SHARED_IN_REPO)


# Reuse existing service launcher utilities and config
from app.config import config  # type: ignore
from shared.service_launcher import (  # type: ignore
    setup_service_logging,
    setup_argument_parser,
    handle_tls_validation,
    setup_https_config,
    print_startup_info,
    print_service_info,
    build_uvicorn_config,
)
import uvicorn  # type: ignore


logger = setup_service_logging(
    service_name="voice_service",
    log_file="voice/app.log",
    error_file="voice/error.log",
)


# -------------------- FunASR helpers --------------------
def _cleanup_old_funasr_processes() -> None:
    try:
        subprocess.run(["pkill", "-f", "run_server.sh"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "funasr-wss-server"], stderr=subprocess.DEVNULL)
        time.sleep(2)
    except Exception:
        pass


def _start_funasr_websocket() -> subprocess.Popen:
    """Start FunASR WebSocket server in background using run_server.sh."""
    logger.info("[FunASR] 清理旧进程…")
    _cleanup_old_funasr_processes()

    funasr_runtime_dir = \
        "/workspace/FunASR/runtime" if os.path.isdir("/workspace/FunASR/runtime") \
        else os.path.join(REPO_ROOT, "services", "funasr-service", "runtime")

    # 从配置读取FunASR LM开关
    from app.config import config
    voice_config = getattr(config, "voice_service", None)
    disable_lm = voice_config.funasr_disable_lm if voice_config else False
    lm_status = "已禁用" if disable_lm else "已启用"
    logger.info("🔧 [FunASR LM配置] funasr_disable_lm=%s (%s) - 将%s LM模块", 
               disable_lm, lm_status, "禁用" if disable_lm else "启用")
    
    cmd = [
        "nohup", "bash", "run_server.sh",
        "--download-model-dir", "/workspace/models",
        "--vad-dir", "damo/speech_fsmn_vad_zh-cn-16k-common-onnx",
        "--model-dir", "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
        "--punc-dir", "damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx",
        "--itn-dir", "thuduj12/fst_itn_zh",
        "--hotword", "/workspace/models/hotwords.txt",
        "--certfile", "0",
    ]
    
    # ⚠️ 根据配置决定是否添加LM参数
    if not disable_lm:
        # 如果不禁用LM，添加--lm-dir参数
        cmd.extend(["--lm-dir", "damo/speech_ngram_lm_zh-cn-ai-wesp-fst"])
        logger.info("✅ [FunASR启动] 已添加LM参数: --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst")
    else:
        # 如果禁用LM，不添加--lm-dir参数
        logger.info("✅ [FunASR启动] 已禁用LM模块，不添加--lm-dir参数")

    log_path = "/workspace/funasr.log" if os.path.isdir("/workspace") else os.path.join(APP_DIR, "funasr.log")
    log_file = open(log_path, "w")

    logger.info("[FunASR] 启动WebSocket服务…")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=funasr_runtime_dir,
        start_new_session=True,
    )
    logger.info("[FunASR] 已启动，PID=%s，日志：%s", proc.pid, log_path)
    return proc


def _wait_for_funasr_ready(timeout_s: int = 60, host: str = "127.0.0.1", port: int = 10095) -> bool:
    logger.info("[FunASR] 等待服务就绪…")
    waited = 0
    interval = 3
    while waited < timeout_s:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                logger.info("[FunASR] ✅ 已就绪")
                return True
        except Exception:
            pass
        time.sleep(interval)
        waited += interval
        logger.info("[FunASR] 等待中… (%s/%s)s", waited, timeout_s)
    try:
        if os.path.exists(log_path := ("/workspace/funasr.log" if os.path.isdir("/workspace") else os.path.join(APP_DIR, "funasr.log"))):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                tail = f.readlines()[-50:]
                for line in tail:
                    sys.stderr.write(line)
    except Exception:
        pass
    logger.error("[FunASR] ❌ 启动超时")
    return False


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(2)
        except Exception:
            pass
        try:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# -------------------- Unified main --------------------
def main() -> None:
    app_config = config

    # 兼容性获取 host/port
    default_host = getattr(getattr(app_config, "voice_server", None), "host", "0.0.0.0")
    default_port = getattr(getattr(app_config, "voice_server", None), "port", 8001)

    parser = setup_argument_parser(
        service_name="语音服务(合并启动)",
        default_port=default_port,
        default_host=default_host,
        default_env=app_config.environment,
    )
    args = parser.parse_args()

    # TLS validation only
    if args.validate_tls:
        handle_tls_validation(args, logger, app_config.ssl.cert_path, app_config.ssl.key_path)
        return

    # Print startup info
    project_root_for_log = REPO_ROOT if os.path.isdir(REPO_ROOT) else APP_DIR
    print_startup_info(logger, "语音服务(合并)", project_root_for_log, APP_DIR, args)

    # Start FunASR first and wait for readiness
    funasr_proc = _start_funasr_websocket()
    ready = _wait_for_funasr_ready(timeout_s=60, host="127.0.0.1", port=10095)
    if not ready:
        _terminate_process_tree(funasr_proc)
        sys.exit(1)

    # Prepare HTTPS
    ssl_keyfile, ssl_certfile, ssl_context = setup_https_config(
        args, logger, app_config.ssl.cert_path, app_config.ssl.key_path
    )

    # Print service info
    print_service_info(logger, args)

    # Build uvicorn config and run (blocking)
    uvicorn_config = build_uvicorn_config(args, ssl_certfile, ssl_keyfile, ssl_context)

    # Handle termination to clean child process
    def _handle_term(sig, frame):
        try:
            logger.info("收到退出信号，正在关闭服务…")
        except Exception:
            pass
        _terminate_process_tree(funasr_proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_term)
    signal.signal(signal.SIGTERM, _handle_term)

    try:
        uvicorn.run(**uvicorn_config)
    finally:
        _terminate_process_tree(funasr_proc)


if __name__ == "__main__":
    main()


