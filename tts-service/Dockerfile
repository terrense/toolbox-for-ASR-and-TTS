# CUDA 12.9 + cuDNN + Ubuntu 22.04（GPU 支持）
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELSCOPE_CACHE=/root/.cache/modelscope

WORKDIR /app

# ===== 1.1 Ubuntu 永久换清华源 =====
RUN sed -i 's@http://archive.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com/ubuntu@https://mirrors.tuna.tsinghua.edu.cn/ubuntu@g' /etc/apt/sources.list

# 1) 系统依赖（包含 sox/ffmpeg/libsndfile 等音频链路必备）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev \
    git curl ca-certificates \
    build-essential cmake pkg-config \
    libsndfile1 \
    sox libsox-fmt-all \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# 2) PyTorch：固定 cu129（确保支持 sm_120 / RTX 5070）
RUN python3 -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

# ===== pip 永久换清华源 =====
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.timeout 120

# 3) 服务端依赖
RUN python3 -m pip install \
    fastapi==0.116.1 \
    "uvicorn[standard]==0.35.0" \
    httpx==0.28.1 \
    simplejson==3.20.2 \
    pydantic==2.11.10

# 4) ModelScope + KAN-TTS 运行时依赖
RUN python3 -m pip install \
    modelscope==1.33.0 \
    numpy==1.26.4 \
    scipy==1.10.1 \
    addict==2.4.0 \
    sortedcontainers==2.4.0 \
    tqdm==4.67.1 \
    PyYAML==6.0.3 \
    requests==2.32.5 \
    tensorboardX==2.6.4 \
    pytorch-wavelets==1.3.0 \
    soundfile==0.13.1 \
    sox==1.5.0 \
    datasets==3.6.0

# 5) 安装 kantts / ttsfrd：强制从 ModelScope wheel 仓库拿
RUN python3 -m pip install \
    -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html \
    kantts==1.0.1 ttsfrd==0.2.1

# 6) 拷贝服务代码（新的目录结构）
COPY app/ /app/app/

# 7) 拷贝本地模型（如果存在）
# 注意：模型应该放在 services/tts-service/app/services/models/damo/speech_sambert-hifigan_tts_zh-cn_16k/
# 如果模型目录存在，则拷贝到容器中；否则容器启动时会从 ModelScope 下载
COPY app/services/models/ /app/app/services/models/

# 8) 端口
EXPOSE 7001

# 9) 启动（使用新的入口点）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7001"]

