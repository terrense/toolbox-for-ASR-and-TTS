# import httpx
# import base64

# async def tts_example():
#     async with httpx.AsyncClient() as client:
#         # 启动任务
#         resp = await client.post(
#             "http://localhost:7001/tts/start",
#             json={"text": "你好，这是一个测试"}
#         )
#         job_id = resp.json()["job_id"]
        
#         # 轮询结果
#         while True:
#             resp = await client.get(f"http://localhost:7001/tts/result/{job_id}")
#             data = resp.json()
            
#             if data["status"] == "completed":
#                 # 解码音频
#                 audio_bytes = base64.b64decode(data["audio_base64"])
#                 with open("output.wav", "wb") as f:
#                     f.write(audio_bytes)
#                 break
#             elif data["status"] in ["cancelled", "error"]:
#                 print(f"任务失败: {data}")
#                 break
            
#             await asyncio.sleep(0.5)


import time
import torch
from modelscope.pipelines import pipeline
from modelscope.tasks import Tasks

# -------------------------- 关键配置（按你的实际情况修改）--------------------------
# 你的本地模型绝对路径（复制你提供的路径，Windows下用原始字符串r""避免转义）
LOCAL_MODEL_PATH = r"E:\hgdoctor-dev-whu\hgdoctor-dev-whu\services\tts-service\app\services\models\damo\speech_sambert-hifigan_tts_zh-cn_16k"
# 测试文本（和日志中一致，含换行符，模拟真实场景）
TEST_TEXT = """参考诊断：
1. 前庭运动障碍
2. 良性阵发性位置性眩晕
3. 中毒性眩晕
4. 颅内感染
5. ..."""
# 测试次数（取平均值，避免偶然误差）
TEST_TIMES = 5
# ----------------------------------------------------------------------------------

def test_tts_speed():
    # 1. 打印环境信息（确认GPU和路径）
    print("="*50)
    print(f"本地模型路径: {LOCAL_MODEL_PATH}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f}MB")
    else:
        print("警告：未检测到GPU，将使用CPU（速度极慢！）")
    print("="*50)

    # 2. 初始化TTS Pipeline（和你业务代码的配置一致，启用FP16优化）
    print("初始化本地TTS模型...")
    tts = pipeline(
        task=Tasks.text_to_speech,
        model=LOCAL_MODEL_PATH,  # 直接使用本地模型路径
        device="cuda:0" if torch.cuda.is_available() else "cpu",  # 强制GPU
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # GPU启用FP16
    )
    print("模型初始化完成！")
    print("="*50)

    # 3. 预热推理（首次调用会有GPU内核初始化开销，不计入统计）
    print("预热推理中...（不计入耗时统计）")
    tts("测试预热，消除初始化开销")
    print("预热完成！开始正式测试...")
    print("="*50)

    # 4. 正式测试（多次运行取平均值）
    total_time = 0.0
    for i in range(TEST_TIMES):
        start_time = time.time()
        # 调用模型（添加优化参数，和业务代码优化保持一致）
        result = tts(
            TEST_TEXT,
            forward_params={
                "beam_size": 1,  # 减少解码计算量
                "sampling_rate": 16000  # 匹配模型16k采样率，避免插值耗时
            }
        )
        end_time = time.time()
        cost_time = (end_time - start_time) * 1000  # 转毫秒
        total_time += cost_time
        print(f"第{i+1}次测试：耗时 {cost_time:.2f}ms")

    # 5. 输出统计结果
    avg_time = total_time / TEST_TIMES
    print("="*50)
    print(f"测试完成！共测试{TEST_TIMES}次")
    print(f"平均推理耗时：{avg_time:.2f}ms")
    print(f"文本长度：{len(TEST_TEXT)}字符")
    print("="*50)

    # 验证音频是否正常生成（可选）
    audio_data = result["output_wav"]
    print(f"生成音频长度：{len(audio_data)}字节（正常应为几十KB）")
    # 可选：保存音频到本地验证音质
    with open("test_tts_output.wav", "wb") as f:
        f.write(audio_data)
    print("音频已保存为 test_tts_output.wav（可验证音质是否正常）")

if __name__ == "__main__":
    test_tts_speed()