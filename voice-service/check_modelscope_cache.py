#!/usr/bin/env python3
"""
临时脚本：检查 ModelScope 缓存路径
用于确认容器内实际使用的缓存目录
"""
import os
from pathlib import Path

# 方法1：检查环境变量
modelscope_cache_env = os.getenv("MODELSCOPE_CACHE")
print(f"环境变量 MODELSCOPE_CACHE: {modelscope_cache_env}")

# 方法2：检查默认路径（Linux/Ubuntu）
default_cache_paths = [
    "/root/.cache/modelscope",
    os.path.expanduser("~/.cache/modelscope"),
    "/home/.cache/modelscope",
]

print("\n检查默认缓存路径：")
for path in default_cache_paths:
    path_obj = Path(path)
    exists = path_obj.exists()
    is_dir = path_obj.is_dir() if exists else False
    print(f"  {path}: {'存在' if exists else '不存在'} {'(目录)' if is_dir else ''}")
    if exists:
        # 列出目录内容
        try:
            items = list(path_obj.iterdir())
            print(f"    包含 {len(items)} 个项目")
            for item in items[:5]:  # 只显示前5个
                print(f"      - {item.name}")
            if len(items) > 5:
                print(f"      ... 还有 {len(items) - 5} 个项目")
        except Exception as e:
            print(f"    无法读取目录: {e}")

# 方法3：尝试导入 modelscope 并查看其配置
print("\n尝试从 modelscope 库获取缓存路径：")
try:
    import modelscope
    # 查看 modelscope 的缓存配置
    cache_dir = getattr(modelscope, "cache_dir", None)
    if cache_dir:
        print(f"  modelscope.cache_dir: {cache_dir}")
    else:
        print("  modelscope 没有 cache_dir 属性")
    
    # 尝试查看 modelscope 的配置文件
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        print("  modelscope.hub.snapshot_download 可用")
    except Exception as e:
        print(f"  无法导入 snapshot_download: {e}")
        
except ImportError:
    print("  modelscope 未安装或无法导入")

# 方法4：检查 FunASR 可能使用的路径
print("\n检查 FunASR 相关路径：")
funasr_paths = [
    "/workspace/.cache/modelscope",
    "/workspace/modelscope_cache",
    "/root/.cache/funasr",
]

for path in funasr_paths:
    path_obj = Path(path)
    exists = path_obj.exists()
    print(f"  {path}: {'存在' if exists else '不存在'}")

print("\n建议：")
print("1. 如果看到某个路径存在且有内容，那就是实际使用的缓存路径")
print("2. 如果都不存在，说明模型还没有下载，需要先运行一次服务")
print("3. 可以通过环境变量 MODELSCOPE_CACHE 指定自定义路径")

