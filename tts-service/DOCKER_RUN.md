# TTS Service Docker 运行指南

## 方式 1：使用 docker-compose（推荐）

### 启动服务

```bash
# 在项目根目录执行
docker-compose up -d tts-service
```

### 查看日志

```bash
docker-compose logs -f tts-service
```

### 停止服务

```bash
docker-compose stop tts-service
```

### 重启服务

```bash
docker-compose restart tts-service
```

---

## 方式 2：单独使用 Docker 命令

### 1. 构建镜像

```bash
# 在项目根目录执行
cd services/tts-service
docker build -f Dockerfile -t hgdoctor-tts-service:latest .
```

或者从项目根目录：

```bash
docker build -f services/tts-service/Dockerfile -t hgdoctor-tts-service:latest services/tts-service/
```

### 2. 运行容器

```bash
docker run -d \
  --name tts-service \
  -p 19001:7001 \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  --restart unless-stopped \
  hgdoctor-tts-service:latest
```

**参数说明**：
- `-d`: 后台运行
- `--name tts-service`: 容器名称
- `-p 19001:7001`: 端口映射（宿主机 19001 -> 容器 7001）
- `-v ~/.cache/modelscope:/root/.cache/modelscope`: 挂载模型缓存（避免重复下载）
- `--restart unless-stopped`: 自动重启

### 3. 如果需要 GPU 支持

```bash
docker run -d \
  --name tts-service \
  -p 19001:7001 \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  --gpus all \
  --restart unless-stopped \
  hgdoctor-tts-service:latest
```

### 4. 查看日志

```bash
docker logs -f tts-service
```

### 5. 停止容器

```bash
docker stop tts-service
```

### 6. 删除容器

```bash
docker rm tts-service
```

---

## 验证服务

### 健康检查

```bash
curl http://localhost:19001/health
```

应该返回：
```json
{
  "status": "healthy",
  "model_loaded": false,  // 首次使用时才会加载
  "active_jobs": 0
}
```

### 测试 TTS

```bash
# 启动任务
curl -X POST http://localhost:19001/tts/start \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，这是一个测试"}'

# 返回 job_id，然后轮询结果
curl http://localhost:19001/tts/result/{job_id}
```

---

## 常见问题

### 1. 端口被占用

如果 19001 端口被占用，可以修改端口映射：

```bash
docker run -d \
  --name tts-service \
  -p 19002:7001 \  # 改为其他端口
  ...
```

然后修改 `test_voice.py` 中的 `TTS_SERVICE_URL` 为 `http://127.0.0.1:19002`

### 2. 模型下载慢

首次运行会从 ModelScope 下载模型（约 500MB+），可能需要较长时间。模型会缓存在 `~/.cache/modelscope`，下次启动会更快。

### 3. GPU 不可用

如果不需要 GPU，可以移除 `--gpus all` 参数。TTS 模型也可以在 CPU 上运行，只是速度较慢。

### 4. 容器启动失败

查看日志：
```bash
docker logs tts-service
```

常见原因：
- 端口被占用
- 内存不足（模型需要至少 2GB 可用内存）
- 依赖安装失败

---

## 与 test_voice.py 集成

确保 `test_voice.py` 中的配置：

```python
TTS_SERVICE_URL = "http://127.0.0.1:19001"  # 与 Docker 端口映射一致
```

然后 `test_voice.py` 就可以正常调用 TTS 服务了。

