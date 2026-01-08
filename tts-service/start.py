#!/usr/bin/env python3
"""
TTS Service 启动脚本
设置正确的 Python 路径并启动服务
"""
import os
import sys
from pathlib import Path

# 获取脚本所在目录（services/tts-service/）
SCRIPT_DIR = Path(__file__).parent.absolute()

# 将服务根目录添加到 Python 路径
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 设置工作目录
os.chdir(SCRIPT_DIR)

# 现在可以导入并运行应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7001, reload=True)

