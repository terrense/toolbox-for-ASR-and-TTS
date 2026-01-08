# 语音服务 Dockerfile - 基于FunASR官方镜像
FROM registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.4.7

# ✅ Debian 源切到清华（兼容 bookworm 的 debian.sources）
RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
    elif [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i -E 's|https?://deb\.debian\.org|https://mirrors.tuna.tsinghua.edu.cn|g; s|https?://security\.debian\.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        git \
        openssl \
    && rm -rf /var/lib/apt/lists/*

# ✅ pip/Poetry 走清华
RUN pip config set global.index-url http://172.24.25.31:8080/simple/ \
 && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple poetry \
 && poetry config virtualenvs.create false

# 使用pip直接安装依赖（避开Poetry版本问题）
# 安装项目基础依赖 + voice-service依赖 + funasr
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    fastapi \
    pydantic \
    pydantic-settings \
    python-dotenv \
    uvicorn \
    requests \
    websocket-client \
    'openai>=0.27.0' \
    httpx \
    funasr \
    websockets \
 && pip cache purge

## 模型与热词文件从 voice-service 目录复制，统一到一个服务目录下
COPY services/voice-service/app/services/models/ /workspace/models/

# 复制热词文件
COPY services/voice-service/app/services/hotwords.txt /workspace/models/

# 语音服务代码 - 放到 /workspace/voice-service 以保持官方结构
WORKDIR /workspace/voice-service

COPY services/voice-service/ ./ 
COPY shared/ ./shared/

ENV PYTHONPATH=/workspace/voice-service:/workspace/voice-service/shared
ENV LOG_TO_CONSOLE_ONLY=true
ENV FUNASR_HOST=localhost
ENV FUNASR_PORT=10095

# 端口：7701对外（voice-service API）
EXPOSE 7701

# 启动脚本：FunASR后台运行 + voice-service前台运行
#CMD ["bash", "-c", "python /workspace/start_funasr.py & cd /workspace/voice-service && python start.py"]
CMD ["python", "start.py"]