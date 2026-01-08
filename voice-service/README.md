# Terrence Voice Service

语音交互服务，提供自动语音识别（ASR）功能。

## 📁 项目结构

```
voice-service/
├── app/                          # 应用主目录
│   ├── main.py                   # FastAPI应用入口，中间件配置
│   ├── config.py                 # 配置管理（环境变量、YAML）
│   ├── api/                      # API路由层
│   │   └── voice.py             # ASR接口定义
│   ├── models/                   # 数据模型
│   │   └── voice.py             # ASR请求响应模型
│   └── services/                 # 业务逻辑层
│       ├── voice_service.py      # 语音服务主类（ASR调度）
│       ├── voice_interface.py    # ASR接口实现（FunASR WebSocket）
│       ├── LLM_functions.py        # LLM后处理（语音识别结果修正）
│       ├── full_hotwords.py      # 热词处理工具
│       ├── hotwords.txt          # 热词配置文件
│       └── models/               # 模型文件目录
│           └── damo/            # DAMO模型（FunASR、KWS等）
│
├── start_test.py                 # Docker容器启动脚本（FunASR + Voice Service）
├── start.py                      # 开发环境启动脚本
├── test_http.py                  # HTTP接口测试脚本（开发用）
├── docker-compose.yml            # Docker Compose配置文件
├── Dockerfile                    # Docker镜像构建文件
├── .dockerignore                 # Docker构建忽略文件
└── README.md                     # 本文件
```

## 🔄 核心流程

### ASR（语音识别）流程

```
客户端请求
  ↓
app/api/voice.py (speech_recognition)
  ↓
app/services/voice_service.py (recognize_speech)
  ↓
解码Base64音频 → 创建临时文件
  ↓
app/services/voice_interface.py (asr_wake)
  ├─→ KWS唤醒检测（可选，speech_charctc_kws_phone-xiaoyun）
  └─→ recognize_voice_websocket
      ├─→ ensure_wav_mono_16k (ffmpeg转换)
      └─→ FunASR WebSocket (ws://localhost:10095)
          └─→ 加载热词 (hotwords.txt)
  ↓
LLM后处理（可选，hg_deepseek.py）
  ↓
返回识别结果
```

## 🚀 快速开始

### 开发环境启动

```bash
# 方式1: 使用start.py（仅启动voice-service，需单独启动FunASR）
cd services/voice-service
python start.py

# 方式2: 使用start_test.py（自动启动FunASR + voice-service）
python start_test.py
```

### Docker Compose部署

```bash
# 使用docker-compose启动
cd services/voice-service
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 配置说明

voice-service **不再使用外部配置文件**，所有配置通过环境变量提供。配置在 `docker-compose.yml` 中设置。

#### 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `SERVER_HOST` | 服务监听地址 | `0.0.0.0` |
| `SERVER_PORT` | 服务端口 | `8001` |
| `APP_ENVIRONMENT` | 运行环境 | `development` |
| `FUNASR_DISABLE_LM` | 禁用FunASR语言模型 | `false` |
| `VOICE_DISABLE_LLM` | 禁用LLM后处理 | `false` |
| `VOICE_REQUIRE_WAKE` | 强制要求KWS唤醒 | `false` |
| `VOICE_ALWAYS_SAVE_SAMPLE` | 自动保存所有ASR样本 | `false` |
| `AI_MODEL_API_KEY` | AI模型API密钥（用于LLM后处理） | - |
| `AI_MODEL_BASE_URL` | AI模型服务地址 | `http://172.24.27.11:5105/v1` |
| `AI_MODEL_MODEL_NAME` | AI模型名称 | `Qwen3-32B` |
| `SECURITY_ALLOWED_HOSTS` | 允许的主机列表（逗号分隔） | `localhost,127.0.0.1,*.local` |
| `SECURITY_CORS_ORIGINS` | CORS允许的源（逗号分隔） | - |

**修改配置**: 编辑 `docker-compose.yml` 中的 `environment` 部分，然后重启服务：
```bash
docker-compose down
docker-compose up -d
```

## 📡 API 接口

### 1. ASR 语音识别

**POST** `/api/v1/voice/asr`

请求体:
```json
{
  "audio_data": "base64编码的音频数据",
  "use_wake": true,
  "use_llm": true,
  "save_sample": false,
  "sample_id": null,
  "diagnosis_session_id": "会话ID"
}
```

响应:
```json
{
  "text": "识别结果文本",
  "success": true,
  "message": "语音识别成功",
  "sample_id": null
}
```

### 2. WebSocket 语音识别

**WebSocket** `/api/v1/voice/asr/ws`

连接后服务端会发送欢迎消息，然后客户端可以持续发送音频数据进行识别。

**连接示例:**
```javascript
const ws = new WebSocket('ws://localhost:8001/api/v1/voice/asr/ws');

// 接收消息
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'welcome') {
    console.log('连接成功:', msg.message);
  } else if (msg.type === 'result') {
    console.log('识别结果:', msg.text);
  } else if (msg.type === 'error') {
    console.error('错误:', msg.message);
  }
};

// 发送音频数据
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'audio',
    audio_data: 'base64编码的音频数据',
    use_wake: true,
    use_llm: true,
    save_sample: false,
    diagnosis_session_id: '会话ID'
  }));
};
```

**消息格式:**

欢迎消息 (Server -> Client):
```json
{
  "type": "welcome",
  "message": "Connected to voice recognition service",
  "timestamp": 1234567890.123
}
```

音频消息 (Client -> Server):
```json
{
  "type": "audio",
  "audio_data": "base64编码的音频数据",
  "use_wake": true,
  "use_llm": true,
  "save_sample": false,
  "sample_id": null,
  "diagnosis_session_id": "会话ID"
}
```

识别结果 (Server -> Client):
```json
{
  "type": "result",
  "text": "识别结果文本",
  "success": true,
  "message": "语音识别成功",
  "sample_id": null
}
```

错误消息 (Server -> Client):
```json
{
  "type": "error",
  "message": "错误描述",
  "code": "ERROR_CODE"
}
```

### 3. 健康检查

**GET** `/health`

响应:
```json
{
  "status": "healthy",
  "service": "voice"
}
```

## 🔧 依赖关系

### 核心依赖
- **FastAPI**: Web框架
- **FunASR**: 语音识别（通过WebSocket调用本地FunASR服务）
- **FFmpeg**: 音频格式转换
- **websockets**: WebSocket客户端
- **funasr**: FunASR Python SDK（KWS唤醒）

### 可选依赖
- **hg_deepseek**: LLM后处理（语音识别结果修正）

### 模型依赖
- **FunASR模型**: `app/services/models/damo/speech_paraformer-*`（ASR主模型）
- **KWS模型**: `app/services/models/damo/speech_charctc_kws_phone-xiaoyun`（唤醒词检测）

## 📝 文件说明

### 启动脚本

- **`start_test.py`**: Docker容器启动脚本，自动启动FunASR WebSocket服务（端口10095）和Voice Service（端口8001）
- **`start.py`**: 开发环境启动脚本，仅启动Voice Service（需手动启动FunASR）

### 核心服务

- **`app/services/voice_service.py`**: 语音服务主类，处理ASR请求
- **`app/services/voice_interface.py`**: ASR接口实现，包含：
  - `asr_wake()`: ASR主入口（KWS唤醒 + FunASR识别 + LLM修正）
  - `recognize_voice_websocket()`: FunASR WebSocket客户端
  - `kws_wakeup()`: KWS唤醒词检测
  - `ensure_wav_mono_16k()`: 音频格式转换

### 配置和模型

- **`app/config.py`**: 配置管理，从环境变量加载配置（不再使用外部配置文件）
- **`app/services/hotwords.txt`**: 热词配置文件（格式：`热词 权重`）
- **`app/services/models/damo/`**: DAMO模型目录（FunASR、KWS等）

## 🐛 调试

### 查看日志

服务启动后会输出详细日志，包含：
- 请求处理耗时统计（毫秒级）
- FunASR连接状态
- 音频转换过程
- 识别结果

### 测试接口

```bash
# 使用内置测试脚本
python test_http.py

# 或手动测试
curl -X POST http://localhost:8001/api/v1/voice/asr \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "base64音频数据", "diagnosis_session_id": "test"}'
```

## 🔍 性能监控

所有关键步骤都包含耗时统计（毫秒），日志格式：
```
耗时统计 - [步骤名称]: X.XX ms
```

例如：
- `耗时统计 - Base64解码: 2.34 ms`
- `耗时统计 - ffmpeg转换: 45.67 ms`
- `耗时统计 - asr_wake总耗时: 123.45 ms`

## ⚠️ 注意事项

1. **FunASR服务**: Voice Service依赖FunASR WebSocket服务（默认 `ws://localhost:10095`），确保FunASR已启动
2. **模型路径**: 确保模型文件在 `app/services/models/damo/` 目录下
3. **热词配置**: 修改 `app/services/hotwords.txt` 后需重启服务生效

## 📚 相关文档

- FunASR官方文档: https://github.com/alibaba-damo-academy/FunASR
- FastAPI文档: https://fastapi.tiangolo.com/
