"""JPTranslator 的单元测试。

设计
----
``translate.py`` 不依赖 ``maibot_sdk``，``llm_generate`` 是注入回调，
所以测试里直接构造一个简单的假函数即可，不引入 SDK 的 mock 框架。

加载方式：``importlib.util.spec_from_file_location`` 把 ``translate.py``
当作独立模块加载，避免污染工作区全局命名空间（也方便在临时目录里跑）。
"""

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Tuple

import asyncio
import importlib.util
import os
import sys

# 解析待测模块路径。``tests/`` 与 ``translate.py`` 同级，
# 用 ``__file__`` 锚定，避免被 pytest 的 rootdir 改变工作目录时跑偏。
PLUGIN_DIR = Path(__file__).resolve().parent.parent
TRANSLATE_PATH = PLUGIN_DIR / "translate.py"


def _load_translate_module() -> Any:
    """从绝对路径加载 ``translate.py``，返回模块对象。"""
    spec = importlib.util.spec_from_file_location(
        "sbv2_tts_translate_under_test", TRANSLATE_PATH,
    )
    assert spec is not None and spec.loader is not None, "加载 translate.py 失败"
    module = importlib.util.module_from_spec(spec)
    # 不注册到 sys.modules，避免污染全局命名空间。
    spec.loader.exec_module(module)
    return module


# 预加载，被多个测试用例复用。
translate_mod = _load_translate_module()
JPTranslator = translate_mod.JPTranslator


# ---------------------------------------------------------------------------
# 假 LLM：记录每次调用收到的 prompt/model，返回预设响应。
# ---------------------------------------------------------------------------


class FakeLLM:
    """记录调用历史的可编程假 LLM 回调。"""

    def __init__(self, response: Dict[str, Any]) -> None:
        self._response = response
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    async def __call__(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append((prompt, dict(kwargs)))
        return self._response


def _run(coro: Awaitable[Any]) -> Any:
    """同步跑 coroutine 的小工具。

    不用 ``asyncio.get_event_loop().run_until_complete``——Python 3.14
    在主线程里 ``get_event_loop`` 已经不再自动创建 loop。
    直接 ``asyncio.run`` 最稳。
    """
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


def test_translate_success_returns_translated_text() -> None:
    """假 LLM 返回 success=True 且 response 非空时，应原样返回译文。"""
    fake = FakeLLM({"success": True, "response": "こんにちは"})
    ok, payload = _run(JPTranslator().translate("你好", "[t]", fake))

    assert ok is True
    assert payload == "こんにちは"
    # 调用历史必须有一次。
    assert len(fake.calls) == 1


def test_translate_success_strips_whitespace() -> None:
    """译文两端空白应被剔除，避免影响下游分句。"""
    fake = FakeLLM({"success": True, "response": "  おはよう  \n"})
    ok, payload = _run(JPTranslator().translate("早上好", "[t]", fake))

    assert ok is True
    assert payload == "おはよう"


def test_translate_success_uses_content_fallback_field() -> None:
    """当 LLM 用 ``content`` 字段（部分任务模型）时也应能拿到译文。"""
    fake = FakeLLM({"success": True, "content": "ありがとう"})
    ok, payload = _run(JPTranslator().translate("谢谢", "[t]", fake))

    assert ok is True
    assert payload == "ありがとう"


# ---------------------------------------------------------------------------
# Prompt 内容校验：保证传给 LLM 的指令确实包含日文关键字与原文
# ---------------------------------------------------------------------------


def test_translate_prompt_contains_japanese_keyword_and_source() -> None:
    """Prompt 必须含"日文"关键字与原文，否则下游模型可能误以为是润色任务。"""
    fake = FakeLLM({"success": True, "response": "さようなら"})
    _run(JPTranslator().translate("再见", "[t]", fake))

    assert len(fake.calls) == 1
    prompt = fake.calls[0][0]
    assert "日文" in prompt
    assert "再见" in prompt


def test_translate_prompt_includes_max_length_constraint() -> None:
    """Prompt 必须含长度约束，避免 LLM 输出超出 TTS 限制的长句。"""
    fake = FakeLLM({"success": True, "response": "テスト"})
    _run(JPTranslator(max_length=80).translate("测试", "[t]", fake))

    prompt = fake.calls[0][0]
    assert "80" in prompt
    assert "字符" in prompt


def test_translate_forwards_translate_model_to_llm() -> None:
    """translate_model 应透传到 llm_generate 的 kwargs。"""
    fake = FakeLLM({"success": True, "response": "テスト"})
    _run(JPTranslator().translate("测试", "[t]", fake, translate_model="custom-ja"))

    assert fake.calls[0][1].get("model") == "custom-ja"


def test_translate_defaults_to_replyer_task_when_model_empty() -> None:
    """未指定 translate_model 时，model 应回退 replyer。

    不能发空任务名：Host 对空任务名取 model_task_config 字母序首个任务
    （embedding），用聊天请求打 embedding 模型会得到 404 Not Found。
    """
    fake = FakeLLM({"success": True, "response": "テスト"})
    _run(JPTranslator().translate("测试", "[t]", fake))

    assert fake.calls[0][1].get("model") == "replyer"


def test_translate_forwards_model_override() -> None:
    """model_override（具体模型名）应原样透传到 llm_generate 的 kwargs。"""
    fake = FakeLLM({"success": True, "response": "テスト"})
    _run(
        JPTranslator().translate(
            "测试", "[t]", fake,
            translate_model="replyer",
            model_override="gemini-3.7-flash-low",
        )
    )

    kwargs = fake.calls[0][1]
    assert kwargs.get("model") == "replyer"
    assert kwargs.get("model_override") == "gemini-3.7-flash-low"


# ---------------------------------------------------------------------------
# 失败路径
# ---------------------------------------------------------------------------


def test_translate_returns_false_on_success_false() -> None:
    """LLM 显式失败时，返回 (False, error_detail) 且不含原文中文。"""
    fake = FakeLLM({"success": False, "error": "模拟失败"})
    ok, payload = _run(JPTranslator().translate("你好世界", "[t]", fake))

    assert ok is False
    assert payload == "模拟失败"
    # 关键断言：失败时不能把中文原文"透传"给上层当译文。
    assert "你好世界" not in payload


def test_translate_returns_false_on_timeout() -> None:
    """LLM 超时异常应被捕获并返回 (False, ...)。"""

    async def timeout_llm(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        raise asyncio.TimeoutError("LLM timeout")

    ok, payload = _run(JPTranslator().translate("你好", "[t]", timeout_llm))

    assert ok is False
    assert payload  # 应包含错误描述


def test_translate_returns_false_on_unexpected_exception() -> None:
    """未预期异常（如 KeyError）也应被捕获，不让调用栈冒到 plugin 主流程。"""

    async def boom_llm(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        raise KeyError("字段缺失")

    ok, payload = _run(JPTranslator().translate("你好", "[t]", boom_llm))

    assert ok is False
    assert "字段缺失" in payload


def test_translate_returns_false_on_empty_response_with_success_true() -> None:
    """success=True 但 response 为空：不能当作成功返回。"""
    fake = FakeLLM({"success": True, "response": ""})
    ok, payload = _run(JPTranslator().translate("你好", "[t]", fake))

    assert ok is False
    assert payload  # 应有错误描述
    assert "你好" not in payload  # 绝不能回退到原文


def test_translate_returns_false_on_non_dict_response() -> None:
    """极端情况：LLM 返回非 dict（如 None 或字符串），不能崩。"""
    async def weird_llm(prompt: str, **kwargs: Any) -> Any:
        return None

    ok, payload = _run(JPTranslator().translate("你好", "[t]", weird_llm))

    assert ok is False


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------


def test_translate_empty_text_short_circuits() -> None:
    """空文本不应调用 LLM，直接返回 (False, "")。"""
    fake = FakeLLM({"success": True, "response": "何か"})
    ok, payload = _run(JPTranslator().translate("", "[t]", fake))

    assert ok is False
    assert payload == ""
    assert len(fake.calls) == 0  # 不应触发 LLM


if __name__ == "__main__":
    # 允许 ``python tests/test_translate.py`` 直接跑（pytest 不在 CI 里也能兜底）。
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))