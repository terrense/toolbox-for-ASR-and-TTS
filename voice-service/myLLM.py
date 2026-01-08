import os
from openai import OpenAI
import time
import json

# 非流式输出
def correct_text(input_text):
    """
    使用阿里云百炼平台的Qwen模型纠正文本中的错别字和逻辑问题
    
    参数:
    input_text (str): 需要纠正的文本
    
    返回:
    str: 纠正后的文本
    """
    client = OpenAI(
        api_key="sk-56690a31e6cf4ff3a466b7d2dccda6bc",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个智能对话助手，根据我的提问进行对话"},
                {"role": "user", "content": input_text},
            ],
        )
        end_time = time.time()
        total_time = end_time - start_time
        print(f"API调用耗时: {total_time:.2f}秒")
        
        # 提取纠正后的文本
        corrected_text = completion.choices[0].message.content
        return corrected_text
        
    except Exception as e:
        print(f"API调用出错: {e}")
        # 出错时返回原文本
        return input_text

# 使用示例
# if __name__ == "__main__":
#     input_text = "我头通好几田了，就像有真在扎，太杨穴这边蹦着腾，记意里也变茶了，睡不捉觉，盗汉，白天也没精神，老是米米胡胡想睡觉，眼精干瑟发养"
#     result = correct_text(input_text)
#     print("Robot:", result)

#---流式-----

# 初始化客户端
client = OpenAI(
    api_key="sk-56690a31e6cf4ff3a466b7d2dccda6bc",  #
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def chat_with_qwen(user_input: str, model: str = "qwen-plus", stream: bool = True) -> str:
    """
    使用 Qwen 模型对话
    :param user_input: str, 用户输入
    :param model: str, 模型名称
    :param stream: bool, 是否流式输出
    :return: str, 模型的完整回答
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. 用简洁自然的方式回答用户问题。"},
        {"role": "user", "content": user_input}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        stream_options={"include_usage": True} if stream else None
    )

    full_text = ""

    if stream:
        for chunk in completion:
            # 有些 chunk 没有 choices，要先判断
            if not hasattr(chunk, "choices") or not chunk.choices:
                # 打印调试信息，方便确认
                # print("跳过非内容 chunk:", chunk.model_dump_json())
                continue

            delta = chunk.choices[0].delta.content or ""
            if delta:
                #print(delta, end="", flush=True)  # 实时打印
                print(f"\033[95m{delta}\033[0m", end="", flush=True)
                full_text += delta
        print()  # 换行
    else:
        full_text = completion.choices[0].message.content
        print(full_text)


    return full_text


# 使用示例
if __name__ == "__main__":
    result = chat_with_qwen("你是谁？")
    #print("\n最终返回结果：", result)





