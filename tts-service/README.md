# TTS Service

文本转语音（Text-to-Speech）服务，基于 ModelScope 的 SambertHifigan 模型。

## 目录结构

```
services/tts-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── api/
│   │   ├── __init__.py
│   │   └── tts.py           # TTS API 路由
│   ├── models/
│   │   ├── __init__.py
│   │   └── tts.py           # 数据模型（Pydantic）
│   └── services/
│       ├── __init__.py
│       ├── tts_service.py   # TTS 核心业务逻辑（TTSManager 类）
│       └── models/
│           └── damo/        # 本地模型目录（可选）
│               └── speech_sambert-hifigan_tts_zh-cn_16k/
├── Dockerfile               # Docker 镜像构建文件
├── requirements.txt         # Python 依赖
├── README.md               # 本文档
└── API_DOCUMENTATION.md    # API 详细文档
```

## 功能特性

- ✅ 异步任务模式：支持长时间运行的 TTS 任务
- ✅ 任务管理：启动、取消、查询、清理
- ✅ 本地模型支持：优先使用本地模型，避免每次下载
- ✅ 健康检查：监控服务状态和活跃任务数
- ✅ RESTful API：标准的 HTTP 接口

## 快速开始

### 1. 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 运行服务
uvicorn app.main:app --host 0.0.0.0 --port 7001
```

### 2. Docker 构建

```bash
# 构建镜像
docker build -f Dockerfile -t tts-service:latest .

# 运行容器
docker run -d \
  --name tts-service \
  -p 7001:7001 \
  tts-service:latest
```

### 3. 使用本地模型

**步骤 1：下载模型**

模型默认会从 ModelScope 自动下载到 `~/.cache/modelscope/`。如果你想将模型打包到 Docker 镜像中：

```bash
# 1. 首次运行服务，让模型自动下载
# 2. 找到模型目录（通常在 ~/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k/）
# 3. 复制到 tts-service 目录
cp -r ~/.cache/modelscope/hub/damo/speech_sambert-hifigan_tts_zh-cn_16k \
  services/tts-service/app/services/models/damo/
```

**步骤 2：更新 Dockerfile**

Dockerfile 已经配置为自动拷贝 `app/services/models/` 目录。如果模型存在，会优先使用本地模型。

## API 使用

详细 API 文档请参考 [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)

### 基本流程

1. **启动任务**：`POST /tts/start`
   ```json
   {
     "text": "你好，这是一个测试",
     "voice": "zhitian_emo"
   }
   ```

2. **轮询结果**：`GET /tts/result/{job_id}`
   - 持续轮询直到 `status` 为 `completed`

3. **解码音频**：将返回的 `audio_base64` 解码为 WAV 文件

### Python 示例

```python
import httpx
import base64

async def tts_example():
    async with httpx.AsyncClient() as client:
        # 启动任务
        resp = await client.post(
            "http://localhost:7001/tts/start",
            json={"text": "你好，这是一个测试"}
        )
        job_id = resp.json()["job_id"]
        
        # 轮询结果
        while True:
            resp = await client.get(f"http://localhost:7001/tts/result/{job_id}")
            data = resp.json()
            
            if data["status"] == "completed":
                # 解码音频
                audio_bytes = base64.b64decode(data["audio_base64"])
                with open("output.wav", "wb") as f:
                    f.write(audio_bytes)
                break
            elif data["status"] in ["cancelled", "error"]:
                print(f"任务失败: {data}")
                break
            
            await asyncio.sleep(0.5)
```

## 模型配置

### 本地模型路径

服务会按以下顺序查找模型：

1. **本地路径**：`app/services/models/damo/speech_sambert-hifigan_tts_zh-cn_16k/`
2. **ModelScope**：`damo/speech_sambert-hifigan_tts_zh-cn_16k`（自动下载）

### 模型目录结构

如果使用本地模型，目录结构应该是：

```
app/services/models/damo/speech_sambert-hifigan_tts_zh-cn_16k/
├── configuration.json
├── model.pb
├── am/
│   └── ...
└── voc/
    └── ...
```

## 环境变量

- `MODELSCOPE_CACHE`: ModelScope 模型缓存目录（默认：`/root/.cache/modelscope`）

## 健康检查

```bash
curl http://localhost:7001/health
```

响应：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_jobs": 2
}
```

## 注意事项

1. **模型大小**：TTS 模型较大（约 500MB+），首次下载可能需要较长时间
2. **内存占用**：模型加载后常驻内存，建议至少 2GB 可用内存
3. **并发限制**：当前配置最多 2 个并发任务（可在 `TTSManager` 中调整 `max_workers`）
4. **任务清理**：建议定期清理已完成的任务，避免内存泄漏

## 故障排查

### 模型加载失败

- 检查模型目录是否存在
- 查看日志确认是否从 ModelScope 下载
- 确认网络连接正常

### 任务一直处于 processing 状态

- 检查日志中的错误信息
- 确认模型已正确加载
- 检查系统资源（CPU/内存）

## 开发指南

### 添加新的语音类型

修改 `app/services/tts_service.py` 中的 `voice` 参数支持。

### 调整并发数

修改 `TTSManager.__init__()` 中的 `max_workers` 参数。

## License

[根据项目整体 License]

