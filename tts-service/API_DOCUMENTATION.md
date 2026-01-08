# TTS Service API 文档

## 接口概览

TTS Service 提供异步任务模式的 HTTP API，支持文本转语音（TTS）的生成、取消和结果查询。

## 接口详情

### 1. POST `/tts/start` - 启动 TTS 任务

**请求体（JSON）**：
```json
{
  "text": "要转换的文本内容",
  "voice": "zhitian_emo"  // 可选，默认 "zhitian_emo"
}
```

**响应（JSON）**：
```json
{
  "status": "started",
  "job_id": "uuid-string",
  "message": "TTS 任务已启动"
}
```

**说明**：
- `text` 不能为空
- 任务在后台异步执行
- 返回 `job_id` 用于后续查询和取消

---

### 2. POST `/tts/cancel` - 取消 TTS 任务

**请求体（JSON）**：
```json
{
  "job_id": "uuid-string"
}
```

**响应（JSON）**：
```json
{
  "status": "cancelled",
  "job_id": "uuid-string",
  "message": "任务已取消"
}
```

**状态说明**：
- `already_completed`: 任务已完成，无法取消
- `already_cancelled`: 任务已被取消
- `cancelled`: 成功取消

---

### 3. GET `/tts/result/{job_id}` - 获取 TTS 结果（轮询）

**路径参数**：
- `job_id`: 任务 ID

**响应（JSON）**：

**处理中**：
```json
{
  "status": "processing",
  "job_id": "uuid-string",
  "message": "任务处理中，请稍后重试"
}
```

**已完成**：
```json
{
  "status": "completed",
  "job_id": "uuid-string",
  "audio_base64": "base64编码的WAV音频数据",
  "text": "原始文本",
  "audio_size": 123456
}
```

**已取消**：
```json
{
  "status": "cancelled",
  "job_id": "uuid-string",
  "message": "任务已取消"
}
```

**错误**：
```json
{
  "status": "error",
  "job_id": "uuid-string",
  "error": "错误信息"
}
```

**说明**：
- 客户端需要轮询此接口直到 `status` 为 `completed`、`cancelled` 或 `error`
- `audio_base64` 是 base64 编码的 WAV 格式音频数据
- 可以直接解码后播放或保存

---

### 4. GET `/health` - 健康检查

**响应（JSON）**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_jobs": 2
}
```

---

### 5. DELETE `/tts/jobs/{job_id}` - 清理已完成的任务

**路径参数**：
- `job_id`: 任务 ID

**响应（JSON）**：
```json
{
  "status": "deleted",
  "job_id": "uuid-string"
}
```

**说明**：
- 只能清理已完成/已取消/失败的任务
- 用于内存管理，释放任务占用的资源

---

## 数据格式说明

### 输入数据
- **文本**：UTF-8 编码的中文文本字符串
- **语音类型**：可选，默认为 `zhitian_emo`

### 输出数据
- **音频格式**：WAV（16kHz，单声道）
- **编码方式**：Base64 字符串
- **解码示例**（Python）：
  ```python
  import base64
  audio_bytes = base64.b64decode(audio_base64)
  with open("output.wav", "wb") as f:
      f.write(audio_bytes)
  ```

---

## 使用流程示例

### Python 客户端示例

```python
import httpx
import base64
import time

async def tts_example():
    async with httpx.AsyncClient() as client:
        # 1. 启动任务
        response = await client.post(
            "http://localhost:7001/tts/start",
            json={"text": "你好，这是一个测试"}
        )
        job_id = response.json()["job_id"]
        print(f"任务已启动: {job_id}")
        
        # 2. 轮询结果
        while True:
            response = await client.get(
                f"http://localhost:7001/tts/result/{job_id}"
            )
            data = response.json()
            
            if data["status"] == "completed":
                # 3. 解码音频
                audio_bytes = base64.b64decode(data["audio_base64"])
                with open("output.wav", "wb") as f:
                    f.write(audio_bytes)
                print("音频已保存")
                break
            elif data["status"] in ["cancelled", "error"]:
                print(f"任务失败: {data}")
                break
            
            time.sleep(0.5)  # 等待 0.5 秒后重试
```

### 取消任务示例

```python
async def cancel_tts_example(job_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:7001/tts/cancel",
            json={"job_id": job_id}
        )
        print(response.json())
```

