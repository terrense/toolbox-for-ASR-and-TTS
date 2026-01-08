from openai import OpenAI
from typing import List, Dict, Any, Callable, Optional
import time
import re
import json
import logging
import os
import traceback

logger = logging.getLogger(__name__)
# deepseek_with_context_no_heuristic.py

# 尝试导入 httpx（用于强制 HTTP/1.1）
try:
    import httpx
except Exception:
    httpx = None

# 从环境变量读取配置（不再依赖配置文件）
ai_model_config = {
    'api_key': os.getenv('AI_MODEL_API_KEY', 'GHVHTT9meytovgGA3eAotzAfmeSE_5CLA1NVJ0cOWPVEGWc8sw'),
    'base_url': os.getenv('AI_MODEL_BASE_URL', 'http://172.24.27.11:5105/v1'),
    'model_name': os.getenv('AI_MODEL_MODEL_NAME', 'Qwen3-32B')
}

logger.info("配置加载完成:")
logger.info("  API Key: %s...", ai_model_config.get('api_key', '未设置')[:20])
logger.info("  Base URL: %s", ai_model_config.get('base_url', '未设置'))
logger.info("  Model Name: %s", ai_model_config.get('model_name', '未设置'))
print("="*50)


#====== 配置区（API_KEY） ======
client = OpenAI(
    base_url='http://172.24.27.11:4457/v1',
    api_key='eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyZWIwYjliZGZkNGI0NjJhOTczM2UzODAyNDM1ZWZlYyIsIm1ldGhvZHMiOiJQT1NUIiwiaWF0IjoiMTc1NDUzMTgyMiIsImV4cCI6IjE3NTQ1MzU0MjIifQ.JiNIZPgvfAFQIl0shSghYJQe1Sf_xzANTcVlFCK97EWNNp8wpXx9MSlchVAJXSyER-3_Z_0nAgN5dNGrTO8zyWUkUwJZ4qdrXLbGcANCuyOaK2UHfaRJFVhKwWYU32B1sj16dWvTzd6OQ-xUxFSH2RH4kDAyy1sYYUsgByFZELVwgZNmL6MqbRgYbtFmR8CnLQ6hutkTLfn9tlIxpahW1JJCWBXUphoECB4RfmwcmAh0Khv5F030TFcRc-UsLt7qLA-v2-34ITXZwrZkLBFAtf75_00Q9TNEyX1YHnvylKtzxVL8uwV4gITLB7zsqhf5QMGK1s1BaUNspQUh-owCVA',  # 密钥
)
MODEL = 'ds-v3-085'

# -------qwen-------
# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
#     api_key='sk-56690a31e6cf4ff3a466b7d2dccda6bc',
# )
# #MODEL = 'qwen-plus'
# MODEL = 'qwen2.5-32b-instruct'

# client = OpenAI(
#     api_key='eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyZWIwYjliZGZkNGI0NjJhOTczM2UzODAyNDM1ZWZlYyIsIm1ldGhvZHMiOiJQT1NUIiwiaWF0IjoiMTc1NDAxNzc3NCIsImV4cCI6IjE3NTQwMjEzNzQifQ.zlPOn8hmjzyzW9uh2e7Z0Uw1P4nHugz7JeEQeY0yiQuxARPoD5uhQi41NFYIZQZZJ0oErWWw0hZ1iDAWhnW1ICxXAaxqdCa0t130TYnPHNP6tdrqoMXCfShjd7JOKMBPb7wqFO4MgddGtLyixW2aPgD32FSBsTEAKQYIJaMOxbgwexsQzotbwe54-w4BfGKHn9WrQSDAVqzI-T1zVpyaRU6e9gaQjpv8mYKQ51hwhkl_xJCP6qSfHwJOTgpH4kVvefckRl56OgzFsRBHIRJuUXV3QpkDEcocBAxNzANMwuhBx4KCR1NDSZJyLg1r8ija5ejR3TaHrBtknJiX220TFg',  # 密钥
#     base_url='http://172.24.27.11:5104/v1/chat/completions',
# )
# MODEL = 'Qwen2.5-VL-72B-Instruct-hg'

# 使用配置文件中的值，如果没有则使用默认值
client = OpenAI(
    api_key=ai_model_config.get('api_key', 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyZWIwYjliZGZkNGI0NjJhOTczM2UzODAyNDM1ZWZlYyIsIm1ldGhvZHMiOiJQT1NUIiwiaWF0IjoiMTc1Nzk4ODA3MyIsImV4cCI6IjE3NTc5OTE2NzMifQ.mQ-g1g5ICkZ_8_kFEx-AABgPvYGrtGOsZCcz0ad6pSicmw6H6SMLFl6iq3-Q9WERN9GK70jIzssQdOlsuw7IFz4pd52hwL5n73Ha8ujAQndLMvlCaqt7gaZGI7E9NqqwY6yba3O3IBMiG7rqxp50kACBw0U4XmZZ7LV0RA--3jRCUF5pv5-ksBAVMb6d-yL5HEDp9FHgggfV_1EGC3hxjw-E4lUyvCuzUjMWNj9NNqa9Sy7pJSPNfeLCgg1QOFTKyJlUxb_Snu0GZ3NAtYBi8Woka-18DXswhZf7FNyPxz5Nlq0yQXveqfxyujaFAMRDUYolGn7bZnyQbQbf3QdGVw'),
    base_url=ai_model_config.get('base_url', 'http://172.24.27.11:5105/v1'),
)
MODEL = ai_model_config.get('model_name', 'Qwen3-32B')

DEBUG = False

# =====================================================

# ====== prompt 构建 ======



PROMPT_HEADER = (
    "请以 n o t h i n k 模式工作：不要输出推理过程、解释、额外文字；只输出最终 JSON。\n"
    "你是医院内的就医预问诊与院内流程/导航问询助手（hospital pre-triage & in-hospital navigation）。\n\n"

    "唯一目标：对输入的中文 ASR 文本做“最小必要纠错”，修正明显错误，使其在医院问询场景下更自然、清晰、可理解。\n"
    "重点任务：处理同音/近音误识别（homophones），并优先使用我提供的【热词列表】来纠正。\n\n"

    "硬规则（必须遵守）：\n"
    "1) 最小编辑优先（minimal edit）：只改明显错误片段，尽量保留原句结构与信息；不要随意改写整句。\n"
    "2) 语义类型守恒（do not change symptom category）：\n"
    "   - 不要为了命中热词而改变症状/事件类型。\n"
    "   - 尤其是“出血相关”词：必须根据上下文判断是咯血/呕血/黑便/血便等，不能随意替换。\n"
    "3) 热词优先（highest priority）：若某处疑似同音误识别，且热词列表中存在读音相近且语义更合理的候选，优先替换为该热词。\n"
    "4) 热词权重规则：热词列表的每一行可能形如 “词语 权重”。\n"
    "   - 权重为正：候选冲突时优先选择权重更高者；\n"
    "   - 权重为负：该词为禁止词，禁止输出到 corrected；\n"
    "   - corrected 中不得包含权重数字。\n"
    "5) 若热词中没有合理候选，才做常规中文纠错；仍需遵守最小编辑与语义类型守恒。\n"
    "6) 语义守恒（硬规则）：\n"
    " - 部位守恒：若原句或邻近句包含“胸/前胸/左前胸/胸闷/胸痛/咳嗽/呼吸困难”等线索，则纠错后应优先保持为胸部/呼吸系统相关表达；\n"
    " 除非原句明确提到“腹/肚子/胃/拉肚子/排便/恶心呕吐”等线索，否则禁止把疼痛改成“腹痛”。\n"
    " - 出血类型守恒：若出血相关片段与“咳嗽/咳痰/胸部不适”相邻，优先纠正为“咯血/咳血”；\n"
    " - 解剖部位守恒：除非用户原句中明确出现某个身体部位（如脚/腿/手/背/腰），否则禁止在纠错后新增该部位词；疼痛描述优先用“疼痛性质词”（绞痛/刺痛/闷痛/压榨痛）而不是新增部位词。\n"
    " 若与“呕吐/胃/排便/黑便/血便”相邻，才可纠正为“呕血/黑便/血便”。\n"
    "- 若无法确定，应宁可保留原片段（或做更保守的纠错），不要擅自替换为另一类症状。\n"
    "7) 疼痛短语类型判定（用于消歧，必须执行）：\n"
    "   - 当出现“X痛/…痛”的片段时，先判断 X 属于哪一类：\n"
    "     A. 身体部位类（body-part）→ 例如“脚痛/腹痛/胸痛/耳痛/关节痛”等；\n"
    "     B. 疼痛性质类（pain-quality）→ 例如“绞痛/刺痛/闷痛/胀痛/压榨样疼痛”等。\n"
    "   - 判定依据只能使用“局部结构 + 原句信息”，不得凭空猜测：\n"
    "     • 若原句已经出现明确部位词（如“胸/腹/脚/腿/耳/腰/背/关节”等）并在同一句或相邻短语中指向同一处疼痛，则优先输出部位类；\n"
    "     • 若原句同时出现“针刺样/闷胀/压榨样/烧灼样”等性质描述，且该“X痛”用于描述感觉性质（如“那种___的感觉”），优先输出性质类（如绞痛/刺痛/闷痛/胀痛）。\n"
    "   - 对同音歧义必须做保守选择：\n"
    "     • 当 “脚痛/绞痛” 这类同音候选同时合理时，优先选择与原句整体一致且“新增信息更少”的那个；我的脚痛非常脚痛 -> 我的脚痛非常绞痛，而不是疼痛，因为拼音要尽量相似！\n"
    "     • 若仍无法可靠判断，改为更中性的表达：“疼痛/痛感/不适”，不要强行选一个具体词。\n"
    "8) 去除明显异常标点：如句首孤立标点、重复逗号等；但不要过度文学化润色。\n"
    "9) 加一个强制约束：拼音xiaohu的单词比如小户，小虎，小胡必须都改成小护，并且小护小护最后显示一遍即可。\n\n"

    "输出必须严格为 JSON（只输出 JSON，不要代码块）：\n"
    "{\n"
    "  \"corrected\": \"...\",\n"
    "  \"changed\": true_or_false\n"
    "}\n"
)


def build_prompt(context: List[Dict[str, str]], options: List[str], original: str) -> str:
    ctx_lines = []
    for m in context:
        role = m.get("role", "")
        content = m.get("content", "")
        ctx_lines.append(f"{role}: {content}")
    ctx_text = "\n".join(ctx_lines)
    options_text = "\n".join(f"- {opt}" for opt in options)

    prompt = (
        PROMPT_HEADER
        + "上下文（近对话历史）：\n"
        + ctx_text
        + "\n\n当前可选项：\n"
        + options_text
        + "\n\n请修正的原句：\n"
        + f"'{original}'\n"
    )
    return prompt


# ====== 复写的辅助函数（独立部署，不依赖 shared 模块） ======

def _get_voice_model_config() -> dict:
    """
    获取模型配置（voice-service 独立版本）
    优先级：环境变量 > 默认值
    """
    return {
        "base_url": os.getenv("AI_MODEL_BASE_URL", "http://172.24.27.11:5105/v1"),
        "api_key": os.getenv("AI_MODEL_API_KEY", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyZWIwYjliZGZkNGI0NjJhOTczM2UzODAyNDM1ZWZlYyIsIm1ldGhvZHMiOiJQT1NUIiwiaWF0IjoiMTc1Nzk4ODA3MyIsImV4cCI6IjE3NTc5OTE2NzMifQ.mQ-g1g5ICkZ_8_kFEx-AABgPvYGrtGOsZCcz0ad6pSicmw6H6SMLFl6iq3-Q9WERN9GK70jIzssQdOlsuw7IFz4pd52hwL5n73Ha8ujAQndLMvlCaqt7gaZGI7E9NqqwY6yba3O3IBMiG7rqxp50kACBw0U4XmZZ7LV0RA--3jRCUF5pv5-ksBAVMb6d-yL5HEDp9FHgggfV_1EGC3hxjw-E4lUyvCuzUjMWNj9NNqa9Sy7pJSPNfeLCgg1QOFTKyJlUxb_Snu0GZ3NAtYBi8Woka-18DXswhZf7FNyPxz5Nlq0yQXveqfxyujaFAMRDUYolGn7bZnyQbQbf3QdGVw"),
        "model_name": os.getenv("AI_MODEL_MODEL_NAME", "Qwen3-32B"),
    }


def _extract_json_from_text_voice(content: str) -> Optional[Any]:
    """Try multiple strategies to parse JSON from a model's text content."""
    if not content:
        return None
    # 1) Direct parse
    try:
        return json.loads(content)
    except Exception:
        pass
    # 2) ```json ... ``` block
    m = re.search(r"```json\s*(.*?)\s*```", content, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3) Strip common fences
    stripped = re.sub(r"```json|```", "", content).strip()
    if stripped:
        try:
            return json.loads(stripped)
        except Exception:
            pass
    # 4) Fallback: first outermost JSON-looking braces
    m2 = re.search(r"\{[\s\S]*\}", content)
    if m2:
        try:
            return json.loads(m2.group(0))
        except Exception:
            pass
    return None


def _with_retries_voice(fn: Callable[[], Any], *, retries: int = 3, base_delay: float = 0.8, model_info: str = '') -> Any:
    """Run a callable with simple exponential backoff retries on transient failures."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return fn()
        except Exception as e:  # Capture and retry transient failures
            logger.error("%s 第%d/%d次尝试调用失败：%s", model_info, attempt, retries, e)
            last_exc = e
            if attempt == retries:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))  # Exponential backoff
    if last_exc:
        raise last_exc
    return None


def _build_openai_client_voice(api_key: str, base_url: str) -> OpenAI:
    """Create OpenAI client forcing HTTP/1.1 (disable HTTP/2) when possible."""
    # Also disable HTTP/2 via env as a fallback for libraries that honor it
    os.environ.setdefault("HTTPX_HTTP2", "0")
    if httpx is not None:
        # Force HTTP/1.1 and set sane defaults
        transport = httpx.HTTPTransport(http2=False, retries=0)
        http_client = httpx.Client(http2=False, transport=transport, timeout=60)
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    # Fallback: default client (may still work if server handles negotiation)
    return OpenAI(api_key=api_key, base_url=base_url)


def text_to_json_voice(prompt, api_key=None, base_url=None, model_name=None) -> Any:
    """
    将文本转换为JSON格式（voice-service 独立版本）
    增强：HTTP/1.1 + 重试 + 稳健JSON解析
    使用环境变量中的ai_model配置
    Args:
        prompt (str): 提示文本
        api_key (str, optional): API密钥
        base_url (str, optional): API基础URL
        model_name (str, optional): 模型名称
    Returns:
        any: 包含分析结果的JSON字典或列表
    """
    # 获取模型配置（优先使用传入值，其次使用环境变量，否则使用默认值）
    config = _get_voice_model_config()
    api_key = api_key or config["api_key"]
    base_url = base_url or config["base_url"]
    model_name = model_name or config["model_name"]

    # 增强提示词：添加JSON关键字和/nothink提示
    enhanced_prompt = f"{prompt}\n\n请直接输出JSON格式，不要包含任何思考过程或解释。/nothink"
    logger.info("调用模型: %s, BASE_URL: %s", model_name, base_url)
    try:
        client = _build_openai_client_voice(api_key, base_url)

        def _do_call():
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": enhanced_prompt}],
                stream=False,
                temperature=0,
                top_p=1,
                response_format={"type": "json_object"},
                max_tokens=840,
                extra_body={"enable_thinking": False},
                seed=42,
            )
            content = (completion.choices[0].message.content or "") 
            if not content.strip():
                raise Exception(f"AI模型返回空响应，模型: {model_name}")
            parsed = _extract_json_from_text_voice(content)
            if parsed is None:
                raise Exception(f"JSON解析失败，原始内容: '{content}'")
            if isinstance(parsed, (dict, list)):
                logger.info("模型成功输出json：%s", parsed)
                return parsed
            raise Exception(f"解析后的类型异常: {type(parsed)}")

        return _with_retries_voice(_do_call, retries=3, base_delay=0.8, model_info=f"模型: {model_name}, BASE_URL: {base_url}")

    except Exception:
        logger.error("text_to_json_voice三次尝试全部失败，模型: %s, BASE_URL: %s\n%s", model_name, base_url, traceback.format_exc())
        return None


# ====== 简化的同步调用（不使用流式） ======
def query_final(prompt: str, max_tokens: int = 150) -> str:
    """使用text_to_json_voice调用LLM API（voice-service 独立版本）"""
    try:
        # 使用text_to_json_voice获取JSON格式的响应
        result = text_to_json_voice(prompt)
        
        if result is None:
            if DEBUG:
                logger.info("text_to_json_voice返回None")
            return ""
        
        # 将JSON结果转换为字符串返回
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        else:
            return str(result)
            
    except Exception as e:
        if DEBUG:
            logger.info("调用模型发生异常： %s", e)
        return ""

# ====== 解析模型输出（简化版） ======


def _extract_via_regex(raw: str):
    """用简单正则尝试从非严格 JSON 的文本中抓取 corrected 和 matches。"""
    corrected = None
    matches = []
    # corrected
    m = re.search(r'"corrected"\s*:\s*"((?:\\.|[^"\\])*)"', raw, re.S)
    if m:
        corrected = m.group(1).encode('utf-8').decode('unicode_escape') if '\\' in m.group(1) else m.group(1)

    # matches array - 抓所有双引号内元素
    m2 = re.search(r'"matches"\s*:\s*\[\s*((?:.|\s)*?)\s*\]', raw, re.S)
    if m2:
        inner = m2.group(1)
        items = re.findall(r'"((?:\\.|[^"\\])*)"', inner)
        for it in items:
            val = it.encode('utf-8').decode('unicode_escape') if '\\' in it else it
            matches.append(val)
    return corrected, matches


def parse_model_output(raw: str, original: str = "") -> Dict[str, Any]:
    raw = (raw or "").strip()
    logger.info("🔍 开始解析LLM输出:")
    logger.info("原始输入: %s", repr(raw))
    logger.info("输入长度: %s", len(raw))
    
    # 由于使用了text_to_json，raw已经是JSON字符串，直接解析
    try:
        data = json.loads(raw)
        logger.info("✅ JSON解析成功: %s", data)
        if isinstance(data, dict):
            matches = data.get("matches", [])
            has_match = bool(matches)
            result = {
                "success": True,
                "corrected": data.get("corrected", original),
                "matches": matches,
                "has_match": has_match,
                "raw": raw,
                "matched_via": "model",
                "error": None,
            }
            logger.info("✅ 解析结果: %s", result)
            return result
    except Exception as e:
        logger.error("❌ JSON解析失败: %s", e)
        pass

    # 2) 简单正则抓取
    corr, matches = _extract_via_regex(raw)
    if corr is not None or matches:
        has_match = bool(matches)
        return {
            "success": True,
            "corrected": corr or original,
            "matches": matches or [],
            "has_match": has_match,
            "raw": raw,
            "matched_via": "regex",
            "error": "parsed_via_regex",
        }

    # 3) 全部失败
    return {
        "success": False,
        "corrected": original,
        "matches": [],
        "has_match": False,
        "raw": raw,
        "matched_via": "none",
        "error": "invalid json from model",
    }

# ====== 校验/归一化工具（保留用于验证模型返回的 matches） ======


def normalize_str_for_match(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = re.sub(r'\s+', '', s)
    return t.lower()

# ====== 对外主接口（已移除启发式降级，纯模型返回 + 客户端验证） ======


def process_with_context(context: List[Dict[str, str]],
                         options: List[str],
                         text: str) -> Dict[str, Any]:
    original = (text or "").strip()
    if not original:
        return {
            "success": False,
            "corrected": "",
            "matches": [],
            "has_match": False,
            "matched_via": "none",
            "raw": "",
            "error": "empty input",
        }

    prompt = build_prompt(context, options, original)
    start_time = time.time()
    raw = query_final(prompt, max_tokens=200)
    end_time = time.time()
    if DEBUG:
        logger.info("Model call time: %.3fs", end_time - start_time)
        logger.info("Raw model output preview: %s", raw[:400])
        print("="*50)
        logger.info("🔍 LLM原始输出:")
        logger.info("类型: %s", type(raw))
        logger.info("长度: %s", len(raw) if raw else 0)
        logger.info("内容: %s", repr(raw))
        print("="*50)

    parsed = parse_model_output(raw, original=original)

    # 校验并归一化模型给的 matches（若有）
    if parsed.get("success") and parsed.get("matches"):
        validated = []
        seen = set()
        def norm(x): return normalize_str_for_match(x)
        for m in parsed.get("matches", []):
            if not isinstance(m, str):
                continue
            if m in options:
                cand = m
            else:
                cand = next((o for o in options if norm(m) in norm(o) or norm(o) in norm(m)), None)
            if cand and cand not in seen:
                seen.add(cand)
                validated.append(cand)
        parsed["matches"] = validated
        parsed["has_match"] = bool(validated)
        if parsed["has_match"]:
            parsed["success"] = True
            parsed["matched_via"] = parsed.get("matched_via", "model")
            parsed["error"] = None
            return parsed
        else:
            # 如果模型给了 matches 但校验未通过，则不再尝试任何启发式，直接返回无匹配的结果
            parsed["matches"] = []
            parsed["has_match"] = False
            parsed["success"] = parsed.get("success", False)
            parsed["matched_via"] = parsed.get("matched_via", "model")
            parsed["error"] = "model_matches_not_validated"
            return parsed

    # 若模型未返回 matches 或解析失败，则直接返回 parsed（可能没有匹配）
    return {
        "success": parsed.get("success", False),
        "corrected": parsed.get("corrected", original),
        "matches": parsed.get("matches", []),
        "has_match": parsed.get("has_match", False),
        "matched_via": parsed.get("matched_via", "none"),
        "raw": parsed.get("raw", ""),
        "error": parsed.get("error", "no match found"),
    }


def load_hotwords_list() -> List[str]:
    """
    从 hotwords.txt 文件加载热词列表（仅返回热词，忽略权重）
    
    返回:
        List[str]: 热词列表
    """
    hotwords = []
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        hotwords_file = os.path.join(current_dir, "hotwords.txt")
        
        # 如果当前目录不存在，尝试使用相对路径
        if not os.path.exists(hotwords_file):
            hotwords_file = "app/services/hotwords.txt"
        
        with open(hotwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 检查是否有权重（包含空格且最后一部分是数字，可能带负号）
                if ' ' in line:
                    parts = line.rsplit(' ', 1)
                    word = parts[0].strip()
                    weight_str = parts[1].strip()
                    # 检查是否是数字（可能带负号）
                    try:
                        int(weight_str)  # 尝试转换为整数
                        # 如果是数字，则 word 是热词
                        if word:
                            hotwords.append(word)
                    except ValueError:
                        # 如果不是数字，整行作为热词
                        if line:
                            hotwords.append(line)
                else:
                    # 没有空格，整行作为热词
                    hotwords.append(line)
        
        logger.info("从 hotwords.txt 加载了 %d 个热词", len(hotwords))
    except Exception as e:
        logger.error("加载热词文件失败: %s", e)
        # 使用默认热词
        hotwords = ["小护", "胸闷", "胸痛", "发热", "呕吐"]
    
    return hotwords


def correct_text_only(latest_context=None, latest_options=None, text=None, DEBUG=False):
    """
    仅对文本进行修正，不进行匹配操作。
    修正包括：错别字、发音相似但不符合医疗场景的词（结合hotwords判断）

    参数:
        latest_context (list | None): 历史上下文，可以为空。
        latest_options (list | None): 热词列表，用于帮助修正发音相似的词。
        text (str | None): 原始语音识别结果，可以为空。
        DEBUG (bool): 是否输出调试信息。

    返回:
        str: 修正后的文本
    """
    # 如果 text 为空，直接返回空字符串
    if not text or str(text).strip() == "":
        return ""

    # 确保 context 和 options 不为空
    latest_context = latest_context or []
    latest_options = latest_options or []

    # 调用外部纠错函数
    post_text = process_with_context(latest_context, latest_options, text)
    correct_text = post_text.get("corrected", text)

    if DEBUG:
        logger.info("LLM处理前的文本: %s", text)
        logger.info("LLM修正后的文本: %s", correct_text)

    return correct_text


def correct_text_only(latest_context=None, latest_options=None, text=None, DEBUG=False):
    """
    仅对文本进行修正，不进行匹配操作。
    修正包括：错别字、发音相似但不符合医疗场景的词（结合hotwords判断）

    参数:
        latest_context (list | None): 历史上下文，可以为空。
        latest_options (list | None): 热词列表，用于帮助修正发音相似的词。
        text (str | None): 原始语音识别结果，可以为空。
        DEBUG (bool): 是否输出调试信息。

    返回:
        str: 修正后的文本
    """
    # 如果 text 为空，直接返回空字符串
    if not text or str(text).strip() == "":
        return ""

    # 确保 context 和 options 不为空
    latest_context = latest_context or []
    latest_options = latest_options or []

    # 调用外部纠错函数
    post_text = process_with_context(latest_context, latest_options, text)
    correct_text = post_text.get("corrected", text)

    if DEBUG:
        logger.info("LLM处理前的文本: %s", text)
        logger.info("LLM修正后的文本: %s", correct_text)

    return correct_text


def process_speech_result(latest_context=None, latest_options=None, text=None, useQwen=None, DEBUG=False):
    """
    处理语音识别结果，并根据上下文和可选项进行纠正与匹配。

    参数:
        latest_context (list | None): 历史上下文，可以为空。
        latest_options (list | None): 可选项，可以为空。
        text (str | None): 原始语音识别结果，可以为空。

    返回:
        tuple: (latest_context, latest_options, corrected_text)
    """

    # 如果 text 为空，直接返回空字符串
    if not text or str(text).strip() == "":
        return "", ""

    # 确保 context 和 options 不为空
    latest_context = latest_context or []
    latest_options = latest_options or []

    # 调用外部纠错函数
    post_text = process_with_context(latest_context, latest_options, text)
    correct_text = post_text.get("corrected", text)

    # 匹配意图
    if "matches" in post_text and post_text["matches"]:

        # -----------Qwen--------------
        if useQwen:
            try:
                raw_data = post_text.get("raw", "")
                if raw_data and raw_data.strip():
                    parsed_data = json.loads(raw_data)
                    match_string = '", "'.join(parsed_data.get("matches", []))
                else:
                    match_string = '", "'.join(post_text.get("matches", []))
                logger.info("\033[95m>>>>>>>>>>>>>>>>>>基于原始文本_意图匹配结果: %s  \033[0m\n", match_string)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("JSON解析失败，使用备用匹配: %s", e)
                match_string = '", "'.join(post_text.get("matches", []))
        # ------------deepseek------------------
        else:
            match_string = '", "'.join(post_text["matches"])
    else:
        match_string = ""

    if DEBUG:
        logger.info("\033[95m>>>>>>>>>>>>>>>>>>LLM处理前的语音识别结果: %s  \033[0m\n", text)
        logger.info("\033[95m>>>>>>>>>>>>>>>>>>LLM修正后的语音识别结果: %s  \033[0m\n", correct_text)
        if match_string:
            logger.info("\033[95m>>>>>>>>>>>>>>>>>>基于选项_意图匹配结果: %s  \033[0m\n", match_string)
        else:
            logger.error("\033[91m>>>>>>>>>>>>>>>>>>未匹配结果: 无匹配项 \033[0m\n")

    return match_string, correct_text






############################################################################################################
# ====== CLI 示例（本地测试） ======
if __name__ == "__main__":
    # 配置日志输出到控制台
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    # 获取配置信息（使用text_to_json_voice的配置逻辑）
    config = _get_voice_model_config()
    base_url = config["base_url"]
    model_name = config["model_name"]
    api_key = config["api_key"]
    
    print("="*80)
    print("🚀 开始测试语音识别修正功能")
    print("="*80)
    print(f"📋 配置信息:")
    print(f"   API地址: {base_url}")
    print(f"   模型名称: {model_name}")
    print(f"   API密钥: {api_key[:20] if api_key else '未设置'}...")
    print("="*80)

    # 从 hotwords.txt 文件加载热词列表
    hotwords = load_hotwords_list()
    
    # 测试案例列表 - 包含错别字和发音相似但不符合医疗场景的词
    # test_cases = [
    #     "我的眼镜肿胀睁不开眼，不舒服，已近有4天了",  # 眼镜→眼睛，已近→已经
    #     "手被别人菜了一觉，又，红肿、疼痛看什么科",  # 菜→踩
    #     "肚皮皮肤杨，怎么搞？",  # 杨→痒
    #     "皮肤氧得不行，老是想挠",  # 氧→痒
    #     "兄痛，像被石头压着一样",  # 兄→胸
    #     "胸焖气短，爬楼梯就喘",  # 焖→闷
    #     "胸动，左边疼得厉害",  # 动→痛
    #     "药疼，站久了就疼",  # 药→腰
    #     "小云，我皮养，怎么办？",  # 皮养→皮肤痒
    #     "胸门，感觉透不过气",  # 门→闷
    #     "我需要打树叶",  # 树叶→输液（发音相似但不符合医疗场景）
    #     "医生让我吃要",  # 要→药
    #     "我有点发绕",  # 绕→热
    #     "肚子疼，想吐",  # 正确，测试不修正
    #     "胸闷得慌，喘不上气",  # 正确，测试不修正
    #     "我腰子疼得厉害，已经一周了",  # 腰子→腰
    #     "皮肤国明，痒得受不了",  # 国明→发痒/痒
    #     "小云小云，我头晕眼花",  # 正确
    #     "胸痛得厉害，像针扎一样",  # 正确
    #     "腰酸背疼，坐久了就难受",
    #     "我最近祭天，嗯嗯啊就是 嗓子样地不行，很难受，颜面布筒，浑身不舒服"# 正确
    # ]
    
    test_cases=["，小户，小胡，我最近祭天，，确实有过剧烈的咳，有一点沸疼。见突下，还有左前胸痛吧，是那这脚痛的感觉，背部也脚痛，是突发的闷胀的那种感觉，也有点针刺痒。前期疼的厉害一点，现在好点了。还有滑血"]
    
    # 测试所有案例
    print(f"\n📝 热词数量: {len(hotwords)}")
    print(f"📝 测试案例数量: {len(test_cases)}")
    print("="*80)
    
    total_time = 0
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"🧪 测试案例 {i}/{len(test_cases)}")
        print(f"📥 原始文本: {test_text}")
        print("-" * 80)
        
        start_time = time.time()
        try:
            corrected_text = correct_text_only(
                latest_context=None, 
                latest_options=hotwords, 
                text=test_text, 
                DEBUG=False
            )
            end_time = time.time()
            
            test_time = end_time - start_time
            total_time += test_time
            
            # 判断是否有修正
            is_corrected = corrected_text != test_text
            
            print(f"✅ 修正后文本: {corrected_text}")
            print(f"🔄 是否修正: {'是' if is_corrected else '否'}")
            print(f"⏱️  耗时: {test_time:.3f}秒")
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        print("="*80)
    
    # 总结报告
    print("\n" + "="*80)
    print("📈 测试总结")
    print("="*80)
    print(f"总案例数: {len(test_cases)}")
    print(f"总耗时: {total_time:.3f}秒")
    print(f"平均耗时: {total_time / len(test_cases):.3f}秒/案例")
    print("="*80)
    