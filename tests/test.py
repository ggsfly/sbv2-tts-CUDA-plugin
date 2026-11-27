"""Style-Bert-VITS2 (CUDA) 日文语音合成插件 · 合并单元测试。

单文件覆盖：utils 层、translate 层、backend 层（参数拼装 / 422 解析 / profile 解析 /
合成与投递拆分）、命令路线回归、ReplyExtension 路线、配置模型与重建、manifest 不变量。
插件目录名含连字符无法直接 import，故用 importlib 构造伪包加载。

运行：
    uv run pytest plugins/ggsfly_sbv2-tts-CUDA-plugin/tests/test.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

from maibot_sdk.config import generate_plugin_config_schema, rebuild_plugin_config_data
from pydantic import ValidationError

import aiohttp
import pytest

# ============================================================
# 路径常量与伪包加载
# ============================================================

PLUGIN_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PACKAGE_NAME: str = "sbv2_cuda_tts_plugin"

# 构造伪包本体（插件根目录无 __init__.py，按 namespace 包处理）
_pkg = types.ModuleType(PACKAGE_NAME)
_pkg.__path__ = [PLUGIN_DIR]  # type: ignore[attr-defined]
sys.modules[PACKAGE_NAME] = _pkg

# 子包占位（保留 __path__ 以解析相对导入）
for _sub in ("utils", "backends"):
    _subpkg = types.ModuleType(f"{PACKAGE_NAME}.{_sub}")
    _subpkg.__path__ = [os.path.join(PLUGIN_DIR, _sub)]  # type: ignore[attr-defined]
    sys.modules[f"{PACKAGE_NAME}.{_sub}"] = _subpkg


def _load_module(alias: str, file_path: str) -> Any:
    """按伪包别名加载普通模块并注册到 sys.modules。"""

    full = f"{PACKAGE_NAME}.{alias}"
    spec = importlib.util.spec_from_file_location(full, file_path)
    assert spec is not None and spec.loader is not None, f"无法构造 spec: {file_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


# 按依赖顺序加载（被依赖者先加载）
config_keys_mod = _load_module("config_keys", os.path.join(PLUGIN_DIR, "config_keys.py"))
_load_module("utils.text", os.path.join(PLUGIN_DIR, "utils", "text.py"))
_load_module("utils.file", os.path.join(PLUGIN_DIR, "utils", "file.py"))
_load_module("utils.session", os.path.join(PLUGIN_DIR, "utils", "session.py"))
base_mod = _load_module("backends.base", os.path.join(PLUGIN_DIR, "backends", "base.py"))
voice_mod = _load_module("backends.voice", os.path.join(PLUGIN_DIR, "backends", "voice.py"))
_load_module("backends", os.path.join(PLUGIN_DIR, "backends", "__init__.py"))
translate_mod = _load_module("translate", os.path.join(PLUGIN_DIR, "translate.py"))
# plugin.py 导入会触发 maibot_sdk 装饰器与后端注册；其相对导入依赖已加载的伪包。
plugin_mod = _load_module("plugin", os.path.join(PLUGIN_DIR, "plugin.py"))

# 暴露被测类型
TTSTextUtils = sys.modules[f"{PACKAGE_NAME}.utils.text"].TTSTextUtils
TTSFileManager = sys.modules[f"{PACKAGE_NAME}.utils.file"].TTSFileManager
TTSSessionManager = sys.modules[f"{PACKAGE_NAME}.utils.session"].TTSSessionManager
VoiceBackend = voice_mod.VoiceBackend
VoiceProfile = voice_mod.VoiceProfile
TTSResult = base_mod.TTSResult
JPTranslator = translate_mod.JPTranslator
IndexTranslateTranslator = translate_mod.IndexTranslateTranslator
ConfigKeys = config_keys_mod.ConfigKeys
ComponentsConfig = plugin_mod.ComponentsConfig
GeneralConfig = plugin_mod.GeneralConfig
PluginSectionConfig = plugin_mod.PluginSectionConfig
VoiceConfig = plugin_mod.VoiceConfig
TranslateConfig = plugin_mod.TranslateConfig
SBV2TTSPluginConfig = plugin_mod.SBV2TTSPluginConfig
SBV2TTSPlugin = plugin_mod.SBV2TTSPlugin

PLUGIN_ID = "ggsfly.sbv2-tts-cuda-plugin"
EXTENSION_NAME = f"{PLUGIN_ID}.sbv2_ja_voice"
COMMAND_NAME = f"{PLUGIN_ID}.sbv2_tts_command"


def _run(coro):
    """同步跑 coroutine。"""
    return asyncio.run(coro)


# ============================================================
# 假宿主设施：HTTP / 发送 / LLM / 组件管理
# ============================================================

WAV_BYTES: bytes = b"RIFF" + b"\x00" * 1024


class FakeHTTPResponse:
    """最小 aiohttp.ClientResponse 假体：仅提供 status / read() / text()。"""

    def __init__(
        self,
        status: int = 200,
        body: bytes = WAV_BYTES,
        text_body: str = "",
    ) -> None:
        self.status = status
        self._body = body
        self._text_body = text_body

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._text_body

    def release(self) -> None:
        return None


class _FakePostContext:
    """与 ``TTSSessionManager.post`` 同构的异步上下文。"""

    def __init__(self, manager: "FakeSessionManager") -> None:
        self._manager = manager

    async def __aenter__(self) -> FakeHTTPResponse:
        if self._manager.raise_in_enter is not None:
            raise self._manager.raise_in_enter
        return self._manager.response

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False


class FakeSessionManager:
    """记录 ``/voice`` 请求，并按预设返回响应或抛异常。"""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.response = FakeHTTPResponse()
        self.raise_in_enter: Optional[BaseException] = None

    def post(
        self,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
        params: Optional[Dict[str, Any]] = None,
        backend_name: str = "default",
        timeout: Optional[int] = None,
    ) -> _FakePostContext:
        del json, headers, data
        self.calls.append(
            {
                "url": url,
                "params": dict(params or {}),
                "backend_name": backend_name,
                "timeout": timeout,
            }
        )
        return _FakePostContext(self)


class RecordingSend:
    """记录 ``ctx.send.text`` / ``ctx.send.custom`` 的调用与返回值。"""

    def __init__(self) -> None:
        self.text_calls: List[Dict[str, Any]] = []
        self.custom_calls: List[Tuple[str, Any, str]] = []
        self.text_result: Any = True
        self.text_exception: Optional[BaseException] = None
        self.custom_result: Any = True
        self.custom_exception: Optional[BaseException] = None

    async def text(self, text: str, stream_id: str, **kwargs: Any) -> Any:
        self.text_calls.append(
            {"text": text, "stream_id": stream_id, "kwargs": dict(kwargs)}
        )
        if self.text_exception is not None:
            raise self.text_exception
        return self.text_result

    async def custom(self, message_type: str, content: Any, stream_id: str, **kwargs: Any) -> Any:
        del kwargs
        self.custom_calls.append((message_type, content, stream_id))
        if self.custom_exception is not None:
            raise self.custom_exception
        return self.custom_result


class RecordingLLM:
    """记录 ``ctx.llm.generate`` / ``ctx.llm.get_available_models``。"""

    def __init__(self) -> None:
        self.generate_calls: List[Tuple[str, Dict[str, Any]]] = []
        self.generate_result: Dict[str, Any] = {
            "success": True,
            "response": "こんにちは",
        }
        self.generate_exception: Optional[BaseException] = None
        self.available_models: List[str] = ["replyer", "utils"]
        self.available_models_calls: int = 0

    async def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.generate_calls.append((prompt, dict(kwargs)))
        if self.generate_exception is not None:
            raise self.generate_exception
        return self.generate_result

    async def get_available_models(self) -> List[str]:
        self.available_models_calls += 1
        return list(self.available_models)


class ComponentRecorder:
    """记录组件 enable / disable 调用。"""

    def __init__(self, result: Any = None) -> None:
        self.calls: List[Tuple[str, str, str]] = []
        self._result = {"success": True} if result is None else result

    async def enable_component(self, name: str, component_type: str) -> Any:
        self.calls.append(("enable", name, component_type))
        return self._result

    async def disable_component(self, name: str, component_type: str) -> Any:
        self.calls.append(("disable", name, component_type))
        return self._result


def _make_plugin(
    *,
    config_data: Optional[Dict[str, Any]] = None,
    send: Optional[RecordingSend] = None,
    llm: Optional[RecordingLLM] = None,
    component: Optional[ComponentRecorder] = None,
) -> Tuple[Any, RecordingSend, RecordingLLM]:
    """构造注入了假宿主的插件实例。"""

    plugin = SBV2TTSPlugin()
    plugin.set_plugin_config(
        config_data if config_data is not None else SBV2TTSPlugin.build_default_config()
    )
    # 任务名缓存是类属性，逐实例复位避免测试互染
    plugin._available_task_names = None
    send = send if send is not None else RecordingSend()
    llm = llm if llm is not None else RecordingLLM()
    plugin._set_context(
        types.SimpleNamespace(
            plugin_id=PLUGIN_ID,
            logger=logging.getLogger("test.sbv2_plugin"),
            send=send,
            llm=llm,
            component=component if component is not None else ComponentRecorder(),
        )
    )
    return plugin, send, llm


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> FakeSessionManager:
    """把 TTSSessionManager 单例替换为假实现。"""

    fake = FakeSessionManager()
    monkeypatch.setattr(TTSSessionManager, "_instance", fake)
    return fake


def _voice_backend(http: FakeSessionManager) -> Tuple[Any, RecordingSend]:
    """构造直接注入假发送回调的 Voice 后端。"""

    del http
    table = {
        ConfigKeys.VOICE_API_URL: "http://127.0.0.1:5000/voice",
        ConfigKeys.VOICE_DEFAULT_VOICE: "Ling v2",
        ConfigKeys.VOICE_LENGTH: 1.0,
        ConfigKeys.VOICE_VOICES: ["Ling v2", "Fusetsu_v1.5"],
        ConfigKeys.GENERAL_TIMEOUT: 60,
    }
    recorder = RecordingSend()

    async def _send(message_type: str, content: Any, **kwargs: Any) -> Any:
        del kwargs
        return await recorder.custom(message_type, content, "stream-1")

    backend = VoiceBackend(lambda key, default=None: table.get(key, default), log_prefix="[test]")
    backend.set_send_custom(_send)
    return backend, recorder


# ============================================================
# utils 层
# ============================================================


class TestTextUtils:
    def test_clean_text_strips_whitespace(self):
        assert TTSTextUtils.clean_text("  你好  ") == "你好"
        assert TTSTextUtils.clean_text("hello world") == "hello world"

    def test_clean_text_empty(self):
        assert TTSTextUtils.clean_text("") == ""
        assert TTSTextUtils.clean_text(None) == ""  # type: ignore[arg-type]

    def test_detect_language(self):
        assert TTSTextUtils.detect_language("こんにちは") == "ja"
        assert TTSTextUtils.detect_language("你好世界") == "zh"
        assert TTSTextUtils.detect_language("hello world") == "en"
        assert TTSTextUtils.detect_language("") == "zh"


class TestFileManager:
    def test_audio_to_base64(self):
        assert TTSFileManager.audio_to_base64(b"abc") == "YWJj"

    def test_audio_to_base64_exception_safe(self):
        assert TTSFileManager.audio_to_base64("not bytes") == ""  # type: ignore[arg-type]

    def test_validate_audio_data_empty(self):
        is_valid, msg = TTSFileManager.validate_audio_data(b"")
        assert is_valid is False
        assert msg

    def test_validate_audio_data_none(self):
        is_valid, msg = TTSFileManager.validate_audio_data(None)
        assert is_valid is False
        assert msg

    def test_validate_audio_data_valid(self):
        is_valid, msg = TTSFileManager.validate_audio_data(b"x" * 1024)
        assert is_valid is True
        assert msg == ""


# ============================================================
# translate 层
# ============================================================


class FakeLLM:
    """记录调用历史的可编程假 LLM 回调。"""

    def __init__(self, response: Dict[str, Any]) -> None:
        self._response = response
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    async def __call__(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append((prompt, dict(kwargs)))
        return self._response


class TestTranslate:
    def test_success_returns_translated_text(self):
        fake = FakeLLM({"success": True, "response": "こんにちは"})
        ok, payload = _run(JPTranslator().translate("你好", "[t]", fake))
        assert ok is True
        assert payload == "こんにちは"
        assert len(fake.calls) == 1

    def test_success_strips_whitespace(self):
        fake = FakeLLM({"success": True, "response": "  おはよう  \n"})
        ok, payload = _run(JPTranslator().translate("早上好", "[t]", fake))
        assert ok is True
        assert payload == "おはよう"

    def test_success_uses_content_field(self):
        fake = FakeLLM({"success": True, "content": "ありがとう"})
        ok, payload = _run(JPTranslator().translate("谢谢", "[t]", fake))
        assert ok is True
        assert payload == "ありがとう"

    def test_prompt_contains_japanese_keyword_and_source(self):
        fake = FakeLLM({"success": True, "response": "さようなら"})
        _run(JPTranslator().translate("再见", "[t]", fake))
        prompt = fake.calls[0][0]
        assert "日文" in prompt
        assert "再见" in prompt

    def test_prompt_includes_max_length_constraint(self):
        fake = FakeLLM({"success": True, "response": "テスト"})
        _run(JPTranslator(max_length=80).translate("测试", "[t]", fake))
        prompt = fake.calls[0][0]
        assert "80" in prompt
        assert "字符" in prompt

    def test_forwards_task_name_to_llm(self):
        """task_name 应透传到 llm_generate 的 kwargs。"""
        fake = FakeLLM({"success": True, "response": "テスト"})
        _run(JPTranslator().translate("测试", "[t]", fake, task_name="utils"))
        assert fake.calls[0][1].get("task_name") == "utils"

    def test_forwards_model_name_to_llm(self):
        """model_name 应透传到 llm_generate 的 kwargs。"""
        fake = FakeLLM({"success": True, "response": "テスト"})
        _run(JPTranslator().translate("测试", "[t]", fake, model_name="gemini-flash"))
        assert fake.calls[0][1].get("model_name") == "gemini-flash"

    def test_defaults_to_replyer_task_when_empty(self):
        """未指定 task_name 时，默认 replyer。"""
        fake = FakeLLM({"success": True, "response": "テスト"})
        _run(JPTranslator().translate("测试", "[t]", fake))
        assert fake.calls[0][1].get("task_name") == "replyer"

    def test_empty_model_name_not_sent(self):
        """model_name 为空时不应出现在 kwargs 中。"""
        fake = FakeLLM({"success": True, "response": "テスト"})
        _run(JPTranslator().translate("测试", "[t]", fake))
        assert "model_name" not in fake.calls[0][1]

    def test_returns_false_on_success_false(self):
        fake = FakeLLM({"success": False, "error": "模拟失败"})
        ok, payload = _run(JPTranslator().translate("你好世界", "[t]", fake))
        assert ok is False
        assert payload == "模拟失败"
        assert "你好世界" not in payload

    def test_returns_false_on_timeout(self):
        async def timeout_llm(prompt: str, **kwargs: Any) -> Dict[str, Any]:
            raise asyncio.TimeoutError("LLM timeout")

        ok, payload = _run(JPTranslator().translate("你好", "[t]", timeout_llm))
        assert ok is False
        assert payload

    def test_returns_false_on_exception(self):
        async def boom_llm(prompt: str, **kwargs: Any) -> Dict[str, Any]:
            raise KeyError("字段缺失")

        ok, payload = _run(JPTranslator().translate("你好", "[t]", boom_llm))
        assert ok is False
        assert "字段缺失" in payload

    def test_returns_false_on_empty_response_with_success_true(self):
        fake = FakeLLM({"success": True, "response": ""})
        ok, payload = _run(JPTranslator().translate("你好", "[t]", fake))
        assert ok is False
        assert payload
        assert "你好" not in payload

    def test_empty_text_short_circuits(self):
        fake = FakeLLM({"success": True, "response": "何か"})
        ok, payload = _run(JPTranslator().translate("", "[t]", fake))
        assert ok is False
        assert payload == ""
        assert len(fake.calls) == 0


class FakeResponse:
    """最小 aiohttp.ClientResponse 假体：仅提供 status / read()。"""

    def __init__(self, status: int = 200, body: bytes = b"") -> None:
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        return self._body


class FakePostFactory:
    """与 TTSSessionManager.post 同构的可编程假 post 上下文工厂。

    记录每次调用的 url / json / timeout；raise_in_enter 模拟连接层异常。
    """

    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.calls: List[Dict[str, Any]] = []
        self.raise_in_enter: Optional[Exception] = None

    def __call__(self, url: str, *, json: Dict[str, Any], timeout: int) -> "FakePostFactory":
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self

    async def __aenter__(self) -> FakeResponse:
        if self.raise_in_enter is not None:
            raise self.raise_in_enter
        return self._response

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False


def _openai_body(content: str) -> bytes:
    """构造 OpenAI chat/completions 成功响应字节。"""

    return json.dumps(
        {"choices": [{"message": {"content": content}}]},
        ensure_ascii=False,
    ).encode("utf-8")


# translate.py 的剥离正则匹配 `&lt;think&gt;...&lt;/think&gt;`；这里用拼接写出标签，
# 避免源码里出现会被误处理的尖括号标签字面量。
_THINK_OPEN: str = "<" + "think" + ">"
_THINK_CLOSE: str = "</" + "think" + ">"


def _make_translator(post_factory: FakePostFactory, **kwargs: Any) -> IndexTranslateTranslator:
    """构造注入假 post 工厂的翻译器，默认参数对齐生产装配。"""

    defaults: Dict[str, Any] = {
        "base_url": "http://127.0.0.1:8020/v1",
        "model": "Index-Translate-2B",
        "max_length": 100,
        "timeout": 30,
        "max_tokens": 512,
    }
    defaults.update(kwargs)
    return IndexTranslateTranslator(post_factory=post_factory, **defaults)


class TestIndexTranslateTranslator:
    def test_success_returns_translated_text(self):
        fake = FakePostFactory(FakeResponse(200, _openai_body("こんにちは")))
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is True
        assert payload == "こんにちは"
        assert len(fake.calls) == 1

    def test_request_payload_matches_official_recommendation(self):
        """请求体必须对齐官方要点：temperature=0、关思考、非流式、模型名。"""
        fake = FakePostFactory(FakeResponse(200, _openai_body("テスト")))
        _run(_make_translator(fake).translate("测试", "[t]"))
        sent = fake.calls[0]["json"]
        assert sent["model"] == "Index-Translate-2B"
        assert sent["temperature"] == 0
        assert sent["stream"] is False
        assert sent["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent["messages"][0]["role"] == "user"
        assert "测试" in sent["messages"][0]["content"]

    def test_url_and_per_call_timeout_forwarded(self):
        """URL 拼 chat/completions；translate.timeout 按次透传不吃默认 60s。"""
        fake = FakePostFactory(FakeResponse(200, _openai_body("テスト")))
        _run(_make_translator(fake, timeout=17).translate("测试", "[t]"))
        assert fake.calls[0]["url"] == "http://127.0.0.1:8020/v1/chat/completions"
        assert fake.calls[0]["timeout"] == 17

    def test_strips_think_blocks(self):
        fake = FakePostFactory(
            FakeResponse(200, _openai_body(f"{_THINK_OPEN}推理过程{_THINK_CLOSE}こんにちは"))
        )
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is True
        assert payload == "こんにちは"
        assert "推理" not in payload

    def test_plain_prompt_when_no_constraints(self):
        """无术语表/风格时用无约束直译模板，不含约束段。"""
        fake = FakePostFactory(FakeResponse(200, _openai_body("さようなら")))
        _run(_make_translator(fake, max_length=0).translate("再见", "[t]"))
        prompt = fake.calls[0]["json"]["messages"][0]["content"]
        assert "请将以下文本翻译为日语" in prompt
        assert "【约束要求】" not in prompt
        assert "再见" in prompt

    def test_constrained_prompt_contains_glossary_style_length(self):
        """instTrans 格式：术语表与长度进【硬性要求】，风格进【注意】。"""
        fake = FakePostFactory(FakeResponse(200, _openai_body("クリーパー")))
        translator = _make_translator(
            fake,
            glossary=("苦力怕→クリーパー", "僵尸→ゾンビ"),
            style="保持口语化、自然的直播说话风格",
            max_length=80,
        )
        _run(translator.translate("我玩苦力怕", "[t]"))
        prompt = fake.calls[0]["json"]["messages"][0]["content"]
        assert "【约束要求】" in prompt
        assert "苦力怕→クリーパー、僵尸→ゾンビ" in prompt
        assert "【硬性要求】" in prompt
        assert "80" in prompt
        assert "【注意】保持口语化、自然的直播说话风格" in prompt
        assert "【源文】" in prompt

    def test_non_200_returns_failure_with_status(self):
        fake = FakePostFactory(FakeResponse(504, b"gateway timeout"))
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is False
        assert "504" in payload
        assert "你好" not in payload

    def test_malformed_json_returns_failure(self):
        fake = FakePostFactory(FakeResponse(200, b"not json"))
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is False
        assert "JSON" in payload

    def test_missing_choices_returns_failure(self):
        fake = FakePostFactory(FakeResponse(200, json.dumps({"error": "bad"}).encode()))
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is False
        assert payload
        assert "你好" not in payload

    def test_empty_content_returns_failure(self):
        fake = FakePostFactory(FakeResponse(200, _openai_body("")))
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is False
        assert payload
        assert "你好" not in payload

    def test_think_only_content_returns_failure(self):
        """enable_thinking 失效导致整段是思维链时：失败暴露，不回原文。"""
        fake = FakePostFactory(
            FakeResponse(200, _openai_body(f"{_THINK_OPEN}只有思维链{_THINK_CLOSE}"))
        )
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is False
        assert "你好" not in payload

    def test_timeout_returns_failure(self):
        fake = FakePostFactory(FakeResponse(200, _openai_body("こんにちは")))
        fake.raise_in_enter = asyncio.TimeoutError("request timeout")
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is False
        assert payload

    def test_connection_error_returns_failure(self):
        fake = FakePostFactory(FakeResponse(200, _openai_body("こんにちは")))
        fake.raise_in_enter = ConnectionRefusedError("connection refused")
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is False
        assert payload

    def test_generic_exception_returns_failure(self):
        fake = FakePostFactory(FakeResponse(200, _openai_body("こんにちは")))
        fake.raise_in_enter = KeyError("boom")
        ok, payload = _run(_make_translator(fake).translate("你好", "[t]"))
        assert ok is False
        assert "boom" in payload

    def test_empty_text_short_circuits(self):
        fake = FakePostFactory(FakeResponse(200, _openai_body("何か")))
        ok, payload = _run(_make_translator(fake).translate("", "[t]"))
        assert ok is False
        assert payload == ""
        assert len(fake.calls) == 0

    def test_missing_post_factory_exposes_error(self):
        translator = IndexTranslateTranslator(
            base_url="http://127.0.0.1:8020/v1",
            model="Index-Translate-2B",
        )
        ok, payload = _run(translator.translate("你好", "[t]"))
        assert ok is False
        assert "post_factory" in payload

    def test_base_url_trailing_slash_normalized(self):
        fake = FakePostFactory(FakeResponse(200, _openai_body("テスト")))
        translator = _make_translator(fake, base_url="http://127.0.0.1:8020/v1/")
        _run(translator.translate("测试", "[t]"))
        assert fake.calls[0]["url"] == "http://127.0.0.1:8020/v1/chat/completions"


class TestTranslateConfigDefaults:
    def test_engine_default_host_llm(self):
        assert TranslateConfig().engine == "host_llm"

    def test_engine_schema_renders_select_choices(self):
        """engine 必须是 Literal 双值：WebUI 依据 choices 渲染下拉而非文本输入。"""
        schema = generate_plugin_config_schema(SBV2TTSPluginConfig)
        engine_field = schema["sections"]["translate"]["fields"]["engine"]
        assert engine_field["choices"] == ["host_llm", "index_translate"]

    def test_engine_rejects_invalid_value(self):
        """非法 engine 值在加载校验时直接报错（不静默回落）。"""
        with pytest.raises(ValidationError):
            TranslateConfig(engine="openai")

    def test_translate_to_japanese_default_on(self):
        """v1.2.1 自 [general] 迁入，默认值保持 true。"""
        assert TranslateConfig().translate_to_japanese is True

    def test_translate_model_default_empty(self):
        """v1.2.1 自 [general] 迁入，默认值保持留空 = replyer 任务。"""
        assert TranslateConfig().translate_model == ""

    def test_base_url_default_uses_port_8020(self):
        """默认端口必须是 8020——8000 被 MaiBot 主消息服务器占用。"""
        assert TranslateConfig().base_url == "http://127.0.0.1:8020/v1"

    def test_model_default(self):
        assert TranslateConfig().model == "Index-Translate-2B"

    def test_glossary_default_empty(self):
        assert TranslateConfig().glossary == []


# ============================================================
# backend 层：profile / 422 解析 / 合成与投递拆分
# ============================================================


class TestVoiceProfile:
    def test_defaults_style_neutral(self):
        p = VoiceProfile(name="x", model="m", speaker="s")
        assert p.style == "Neutral"

    def test_custom_style(self):
        p = VoiceProfile(name="x", model="m", speaker="s", style="Happy")
        assert p.style == "Happy"


class TestExtract422Message:
    def test_extracts_msg_from_detail(self):
        text = json.dumps({"detail": [{"type": "invalid_params", "msg": "speaker_name=XXX not found", "loc": ["query", "speaker_name"]}]})
        assert VoiceBackend._extract_422_message(text) == "speaker_name=XXX not found"

    def test_extracts_type_when_no_msg(self):
        text = json.dumps({"detail": [{"type": "invalid_params"}]})
        assert VoiceBackend._extract_422_message(text) == "invalid_params"

    def test_fallback_on_invalid_json(self):
        assert VoiceBackend._extract_422_message("not json") == "not json"

    def test_fallback_on_empty(self):
        assert VoiceBackend._extract_422_message("") == "未知参数错误"


class TestBackendProfileResolution:
    def test_default_profile_resolves(self, http: FakeSessionManager):
        backend, _ = _voice_backend(http)
        profile = backend._get_default_profile()
        assert profile.name == "Ling v2"
        assert profile.model == "Ling-v2"

    def test_empty_text_returns_failure(self, http: FakeSessionManager):
        backend, _ = _voice_backend(http)
        result = _run(backend.execute(""))
        assert result.success is False
        assert "为空" in result.message


class TestBackendSynthesize:
    """合成与投递拆分：synthesize 零投递，execute 恰好投递一次。"""

    def test_synthesize_returns_wav_without_delivery(self, http: FakeSessionManager):
        backend, send = _voice_backend(http)
        profile = backend._get_default_profile()

        wav, error = _run(backend.synthesize("こんにちは", profile))

        assert wav == WAV_BYTES
        assert error == ""
        assert send.custom_calls == []
        assert len(http.calls) == 1
        params = http.calls[0]["params"]
        assert params["text"] == "こんにちは"
        assert params["language"] == "JP"
        assert params["model_name"] == "Ling-v2"

    def test_synthesize_rejects_empty_text(self, http: FakeSessionManager):
        backend, send = _voice_backend(http)
        wav, error = _run(backend.synthesize("  ", backend._get_default_profile()))

        assert wav is None
        assert error == "待合成的文本为空"
        assert send.custom_calls == []
        assert http.calls == []

    def test_execute_delivers_once(self, http: FakeSessionManager):
        backend, send = _voice_backend(http)

        result = _run(backend.execute("こんにちは"))

        assert result.success is True
        assert len(send.custom_calls) == 1
        message_type, content, _ = send.custom_calls[0]
        assert message_type == "voice"
        assert content == TTSFileManager.audio_to_base64(WAV_BYTES)

    def test_execute_propagates_delivery_failure(self, http: FakeSessionManager):
        backend, send = _voice_backend(http)
        send.custom_result = False

        result = _run(backend.execute("こんにちは"))

        assert result.success is False
        assert result.message == "语音投递失败: 宿主发送层返回未成功"
        assert len(send.custom_calls) == 1

    def test_execute_maps_delivery_timeout(self, http: FakeSessionManager):
        backend, send = _voice_backend(http)
        send.custom_exception = asyncio.TimeoutError("send timeout")

        result = _run(backend.execute("こんにちは"))

        assert result.success is False
        assert result.message == "Style-Bert-VITS2 调用超时"
        assert len(send.custom_calls) == 1

    def test_execute_maps_delivery_network_error(self, http: FakeSessionManager):
        backend, send = _voice_backend(http)
        send.custom_exception = aiohttp.ClientConnectionError("boom")

        result = _run(backend.execute("こんにちは"))

        assert result.success is False
        assert result.message == "网络错误: ClientConnectionError"
        assert len(send.custom_calls) == 1

    def test_rejects_invalid_wav_magic(self, http: FakeSessionManager):
        http.response = FakeHTTPResponse(200, body=b"XXXX" + b"\x00" * 200)
        backend, send = _voice_backend(http)

        wav, error = _run(backend.synthesize("こんにちは", backend._get_default_profile()))

        assert wav is None
        assert error == "返回数据不是有效的 WAV（缺少 RIFF 头）"
        assert send.custom_calls == []

    def test_rejects_too_small_body(self, http: FakeSessionManager):
        http.response = FakeHTTPResponse(200, body=b"RIFF")
        backend, _ = _voice_backend(http)

        wav, error = _run(backend.synthesize("こんにちは", backend._get_default_profile()))

        assert wav is None
        assert error == "音频数据无效: 音频数据过小(4字节 < 100字节)"

    def test_422_detail_exposed(self, http: FakeSessionManager):
        body_text = json.dumps({"detail": [{"msg": "text too long"}]})
        http.response = FakeHTTPResponse(422, text_body=body_text)
        backend, send = _voice_backend(http)

        wav, error = _run(backend.synthesize("こんにちは", backend._get_default_profile()))

        assert wav is None
        assert error == "参数错误: text too long"
        assert send.custom_calls == []

    def test_non_2xx_status_exposed(self, http: FakeSessionManager):
        http.response = FakeHTTPResponse(500, text_body="internal error")
        backend, _ = _voice_backend(http)

        wav, error = _run(backend.synthesize("こんにちは", backend._get_default_profile()))

        assert wav is None
        assert error == "Style-Bert-VITS2 API调用失败: HTTP 500"

    def test_request_timeout_exposed(self, http: FakeSessionManager):
        http.raise_in_enter = asyncio.TimeoutError("request timeout")
        backend, _ = _voice_backend(http)

        wav, error = _run(backend.synthesize("こんにちは", backend._get_default_profile()))

        assert wav is None
        assert error == "Style-Bert-VITS2 调用超时"

    def test_request_network_error_exposed(self, http: FakeSessionManager):
        http.raise_in_enter = aiohttp.ClientConnectionError("boom")
        backend, _ = _voice_backend(http)

        wav, error = _run(backend.synthesize("こんにちは", backend._get_default_profile()))

        assert wav is None
        assert error == "网络错误: ClientConnectionError"


class TestSendAudioResultPropagation:
    """语音投递结果必须如实传播（不得吞成「成功发送」）。"""

    def _backend(self, send_ok: bool, http: FakeSessionManager) -> Any:
        backend, _ = _voice_backend(http)

        async def _send_custom(message_type: str, content: Any, **kwargs: Any) -> bool:
            del message_type, content, kwargs
            return send_ok

        backend.set_send_custom(_send_custom)
        return backend

    def test_success_returns_success_result(self, http: FakeSessionManager):
        result = _run(
            self._backend(True, http).send_audio(
                b"RIFF" + b"x" * 1024, voice_info="Ling v2"
            )
        )
        assert result.success is True

    def test_failure_is_not_swallowed(self, http: FakeSessionManager):
        result = _run(
            self._backend(False, http).send_audio(
                b"RIFF" + b"x" * 1024, voice_info="Ling v2"
            )
        )
        assert result.success is False
        assert "投递失败" in result.message


# ============================================================
# 命令路线：回归与失败文案精确断言
# ============================================================


def _command_groups(text: str = "你好", voice: str = "") -> Dict[str, Any]:
    return {"text": text, "voice": voice}


class TestCommandRoute:
    def test_success_delivers_once_and_translates_once(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(),
            )
        )

        assert result == (True, "成功发送voice语音 (Ling v2)", 1)
        assert len(send.custom_calls) == 1
        assert send.text_calls == []
        assert len(llm.generate_calls) == 1
        assert http.calls[0]["params"]["text"] == "こんにちは"
        assert http.calls[0]["params"]["language"] == "JP"

    def test_delivery_refused_returns_precise_failure(self, http: FakeSessionManager):
        plugin, send, _ = _make_plugin()
        send.custom_result = False

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(),
            )
        )

        assert result == (False, "语音投递失败: 宿主发送层返回未成功", 1)
        assert len(send.custom_calls) == 1
        assert [call["text"] for call in send.text_calls] == [
            "语音合成失败: 语音投递失败: 宿主发送层返回未成功"
        ]

    def test_delivery_timeout_returns_precise_failure(self, http: FakeSessionManager):
        plugin, send, _ = _make_plugin()
        send.custom_exception = asyncio.TimeoutError("send timeout")

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(),
            )
        )

        assert result == (False, "Style-Bert-VITS2 调用超时", 1)
        assert len(send.custom_calls) == 1
        assert [call["text"] for call in send.text_calls] == [
            "语音合成失败: Style-Bert-VITS2 调用超时"
        ]

    def test_delivery_network_error_returns_precise_failure(self, http: FakeSessionManager):
        plugin, send, _ = _make_plugin()
        send.custom_exception = aiohttp.ClientConnectionError("boom")

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(),
            )
        )

        assert result == (False, "网络错误: ClientConnectionError", 1)
        assert len(send.custom_calls) == 1
        assert [call["text"] for call in send.text_calls] == [
            "语音合成失败: 网络错误: ClientConnectionError"
        ]

    def test_http_422_returns_precise_failure(self, http: FakeSessionManager):
        http.response = FakeHTTPResponse(
            422, text_body=json.dumps({"detail": [{"msg": "text too long"}]})
        )
        plugin, send, _ = _make_plugin()

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(),
            )
        )

        assert result == (False, "参数错误: text too long", 1)
        assert send.custom_calls == []
        assert [call["text"] for call in send.text_calls] == [
            "语音合成失败: 参数错误: text too long"
        ]

    def test_request_timeout_returns_precise_failure(self, http: FakeSessionManager):
        http.raise_in_enter = asyncio.TimeoutError("request timeout")
        plugin, send, _ = _make_plugin()

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(),
            )
        )

        assert result == (False, "Style-Bert-VITS2 调用超时", 1)
        assert send.custom_calls == []
        assert [call["text"] for call in send.text_calls] == [
            "语音合成失败: Style-Bert-VITS2 调用超时"
        ]

    def test_request_network_error_returns_precise_failure(self, http: FakeSessionManager):
        http.raise_in_enter = aiohttp.ClientConnectionError("boom")
        plugin, send, _ = _make_plugin()

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(),
            )
        )

        assert result == (False, "网络错误: ClientConnectionError", 1)
        assert send.custom_calls == []
        assert [call["text"] for call in send.text_calls] == [
            "语音合成失败: 网络错误: ClientConnectionError"
        ]

    def test_translate_disabled_sends_original_text_to_voice(self, http: FakeSessionManager):
        """translate_to_japanese=false：命令把清理后的原文交给 language=JP 的后端。"""
        config_data = SBV2TTSPlugin.build_default_config()
        config_data["translate"]["translate_to_japanese"] = False
        plugin, send, llm = _make_plugin(config_data=config_data)

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(text="  你好世界  "),
            )
        )

        assert result[0] is True
        assert llm.generate_calls == []
        assert http.calls[0]["params"]["text"] == "你好世界"
        assert http.calls[0]["params"]["language"] == "JP"
        assert len(send.custom_calls) == 1

    def test_error_notification_respects_send_error_messages(self, http: FakeSessionManager):
        config_data = SBV2TTSPlugin.build_default_config()
        config_data["general"]["send_error_messages"] = False
        plugin, send, _ = _make_plugin(config_data=config_data)
        send.custom_result = False

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(),
            )
        )

        assert result == (False, "语音投递失败: 宿主发送层返回未成功", 1)
        assert send.text_calls == []

    def test_unknown_voice_reports_available_voices(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(voice="Nope"),
            )
        )

        expected = "未知音色: Nope，可选: Ling v2、Fusetsu_v1.5"
        assert result == (False, expected, 1)
        assert llm.generate_calls == []
        assert http.calls == []
        assert [call["text"] for call in send.text_calls] == [
            f"语音合成参数错误: {expected}"
        ]

    def test_explicit_voice_is_forwarded(self, http: FakeSessionManager):
        plugin, _, _ = _make_plugin()

        _run(
            plugin.handle_tts_command(
                stream_id="stream-1",
                matched_groups=_command_groups(voice="Fusetsu_v1.5"),
            )
        )

        assert http.calls[0]["params"]["model_name"] == "Fusetsu-v1.5"
        assert http.calls[0]["params"]["speaker_name"] == "Fusetsu_v1.5"

    def test_help_does_not_touch_network_or_llm(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()

        result = _run(
            plugin.handle_tts_command(
                stream_id="stream-1", matched_groups=_command_groups(text="help"),
            )
        )

        assert result == (True, "显示帮助信息", 1)
        assert "帮助" in send.text_calls[0]["text"]
        assert "echo_original_text" not in send.text_calls[0]["text"]
        assert llm.generate_calls == []
        assert http.calls == []

    def test_missing_stream_id_is_reported(self, http: FakeSessionManager):
        plugin, send, _ = _make_plugin()

        result = _run(
            plugin.handle_tts_command(stream_id="", matched_groups=_command_groups())
        )

        assert result == (False, "缺少 stream_id", 1)
        assert send.text_calls == []
        assert http.calls == []

    def test_command_pattern_covers_both_aliases(self):
        metadata = _component_metadata("COMMAND")
        pattern = re.compile(metadata["command_pattern"])

        for text in ("/sbv2 你好世界", "/voice 你好世界 -v Fusetsu_v1.5"):
            match = pattern.match(text)
            assert match is not None, text
            assert match.group("text")

    def test_command_description_shows_trigger_not_component_name(self):
        """面板只显示组件名与 description，description 必须给出真实触发方式。"""
        metadata = _component_metadata("COMMAND")
        description = metadata["description"]

        assert "/sbv2" in description
        assert "/voice" in description
        # 组件名 sbv2_tts_command 不是可输入的命令，description 里要说明这一点
        assert "内部组件名" in description


def _component_metadata(component_type: str) -> Dict[str, Any]:
    """取指定类型组件的元数据。"""

    plugin = SBV2TTSPlugin()
    for component in plugin.get_components():
        if component["type"] == component_type:
            return dict(component["metadata"])
    raise AssertionError(f"未注册 {component_type} 组件")


# ============================================================
# ReplyExtension 路线
# ============================================================

CHAT_CONTEXT: Dict[str, Any] = {
    "session_id": "sess-1",
    "stream_id": "stream-1",
    "platform": "qq",
    "is_group_chat": True,
    "group_id": "group-1",
    "user_id": "user-1",
    "is_focus": False,
}


def _text_message(text: str = "你好") -> Dict[str, Any]:
    return {"segments": [{"type": "text", "data": text}], "quote_previous": True}


def _call_extension(
    plugin: Any,
    *,
    phase: str = "before_send",
    text: str = "你好",
    messages: Optional[List[Dict[str, Any]]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    chat: Optional[Dict[str, Any]] = None,
    reply_id: str = "reply-1",
) -> Dict[str, Any]:
    return _run(
        plugin.ja_voice(
            phase=phase,
            text=text,
            messages=messages if messages is not None else [_text_message()],
            parameters=parameters if parameters is not None else {},
            chat=dict(chat) if chat is not None else dict(CHAT_CONTEXT),
            reply_id=reply_id,
        )
    )


class TestReplyExtensionSelection:
    """选不选扩展：未选中时零翻译/零合成/零通知；选中后各恰好一次。"""

    def test_no_effects_before_selection(self, http: FakeSessionManager):
        """构造插件与 prepare 阶段都不产生副作用（宿主仅在选中时进 before_send）。"""
        plugin, send, llm = _make_plugin()

        assert plugin.get_components()
        assert _call_extension(plugin, phase="prepare", text="", messages=[]) == {}
        assert llm.generate_calls == []
        assert http.calls == []
        assert send.text_calls == []
        assert send.custom_calls == []

    def test_selected_empty_options_runs_pipeline_once(self, http: FakeSessionManager):
        """选中且省略参数：默认音色、翻译一次、合成一次、无通知。"""
        plugin, send, llm = _make_plugin()

        result = _call_extension(plugin, parameters={})

        assert len(result["messages"]) == 2
        assert len(llm.generate_calls) == 1
        assert len(http.calls) == 1
        assert http.calls[0]["params"]["model_name"] == "Ling-v2"
        assert send.text_calls == []

    def test_disabled_extension_stays_registered_but_disabled(self):
        """extension_enabled=false：扩展仍声明，只是被组件开关禁用。"""
        config_data = SBV2TTSPlugin.build_default_config()
        config_data["components"]["extension_enabled"] = False
        recorder = ComponentRecorder()
        plugin, _, _ = _make_plugin(config_data=config_data, component=recorder)

        declared = {
            component["name"]
            for component in plugin.get_components()
            if component["type"] == "REPLY_EXTENSION"
        }
        assert declared == {"sbv2_ja_voice"}
        _run(plugin._sync_component_switches())
        assert ("disable", EXTENSION_NAME, "REPLY_EXTENSION") in recorder.calls


class TestReplyExtensionRegistration:
    def test_only_command_and_reply_extension_registered(self):
        plugin = SBV2TTSPlugin()
        types_registered = {component["type"] for component in plugin.get_components()}

        assert "REPLY_EXTENSION" in types_registered
        assert "COMMAND" in types_registered
        assert "TOOL" not in types_registered

    def test_extension_metadata(self):
        plugin = SBV2TTSPlugin()
        extension = next(
            component
            for component in plugin.get_components()
            if component["type"] == "REPLY_EXTENSION"
        )

        assert extension["name"] == "sbv2_ja_voice"
        metadata = extension["metadata"]
        assert metadata["timeout_ms"] == 120000
        assert metadata["generation_prompt"]
        assert "仅当" in metadata["generation_prompt"]
        assert "attach_voice" in metadata["description"]
        schema = metadata["parameters_schema"]
        assert schema["type"] == "object"
        assert schema["properties"]["voice"]["type"] == "string"
        assert "required" not in schema

    def test_load_has_no_extension_side_effects(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()

        assert send.text_calls == []
        assert send.custom_calls == []
        assert llm.generate_calls == []
        assert http.calls == []


class TestReplyExtensionSuccess:
    def test_prepare_returns_empty_without_side_effects(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()

        result = _call_extension(plugin, phase="prepare", text="", messages=[])

        assert result == {}
        assert send.text_calls == []
        assert send.custom_calls == []
        assert llm.generate_calls == []
        assert http.calls == []

    def test_other_phases_return_empty(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()

        assert _call_extension(plugin, phase="after_send") == {}
        assert send.text_calls == []
        assert llm.generate_calls == []
        assert http.calls == []

    def test_before_send_appends_single_voice_message(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()
        original = _text_message("你好呀")

        result = _call_extension(plugin, messages=[original])

        assert len(result["messages"]) == 2
        assert result["messages"][0] is original
        assert original == _text_message("你好呀")
        voice_message = result["messages"][1]
        assert voice_message["quote_previous"] is False
        assert len(voice_message["segments"]) == 1
        segment = voice_message["segments"][0]
        assert segment["type"] == "voice"
        assert segment["data"] == ""
        assert segment["binary_data_base64"] == TTSFileManager.audio_to_base64(WAV_BYTES)

        # 扩展不自己投递，也不发失败提示
        assert send.custom_calls == []
        assert send.text_calls == []
        assert len(llm.generate_calls) == 1
        assert http.calls[0]["params"]["text"] == "こんにちは"
        assert http.calls[0]["params"]["model_name"] == "Ling-v2"

    def test_explicit_voice_is_used(self, http: FakeSessionManager):
        plugin, _, _ = _make_plugin()

        result = _call_extension(plugin, parameters={"voice": "Fusetsu_v1.5"})

        assert len(result["messages"]) == 2
        assert http.calls[0]["params"]["model_name"] == "Fusetsu-v1.5"
        assert http.calls[0]["params"]["speaker_name"] == "Fusetsu_v1.5"

    def test_translate_off_still_translates_for_extension(self, http: FakeSessionManager):
        """translate_to_japanese 只约束命令路线；扩展始终走翻译引擎。"""
        config_data = SBV2TTSPlugin.build_default_config()
        config_data["translate"]["translate_to_japanese"] = False
        plugin, send, llm = _make_plugin(config_data=config_data)

        result = _call_extension(plugin)

        assert len(result["messages"]) == 2
        assert len(llm.generate_calls) == 1
        assert len(http.calls) == 1
        assert http.calls[0]["params"]["text"] == "こんにちは"
        assert send.custom_calls == []


class TestReplyExtensionFailure:
    def _assert_single_notice(self, send: RecordingSend, expected: str) -> None:
        assert len(send.text_calls) == 1
        call = send.text_calls[0]
        assert call["text"] == expected
        assert call["stream_id"] == "stream-1"
        assert call["kwargs"] == {
            "sync_to_maisaka_history": False,
            "storage_message": False,
        }

    def test_translate_failure(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()
        llm.generate_result = {"success": False, "error": "额度耗尽"}

        result = _call_extension(plugin)

        assert result == {}
        self._assert_single_notice(send, "本次未附加日语语音：日文翻译失败。")
        assert http.calls == []
        assert send.custom_calls == []

    def test_422_failure(self, http: FakeSessionManager):
        http.response = FakeHTTPResponse(
            422, text_body=json.dumps({"detail": [{"msg": "text too long"}]})
        )
        plugin, send, llm = _make_plugin()

        result = _call_extension(plugin)

        assert result == {}
        self._assert_single_notice(
            send, "本次未附加日语语音：合成失败（参数错误: text too long）。"
        )
        assert len(llm.generate_calls) == 1
        assert len(http.calls) == 1
        assert send.custom_calls == []

    def test_unknown_voice_skips_translate_and_synthesize(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()

        result = _call_extension(plugin, parameters={"voice": "Nope"})

        assert result == {}
        self._assert_single_notice(send, "本次未附加日语语音：未知音色「Nope」。")
        assert llm.generate_calls == []
        assert http.calls == []

    def test_empty_base64_skips_voice_segment(
        self,
        http: FakeSessionManager,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            TTSFileManager,
            "audio_to_base64",
            classmethod(lambda cls, data: ""),
        )
        plugin, send, _ = _make_plugin()

        result = _call_extension(plugin)

        assert result == {}
        self._assert_single_notice(
            send, "本次未附加日语语音：合成失败（音频数据转 base64 失败）。"
        )
        assert len(http.calls) == 1

    def test_notice_failure_does_not_break_handler(self, http: FakeSessionManager):
        plugin, send, _ = _make_plugin()
        send.text_result = False

        result = _call_extension(plugin, parameters={"voice": "Nope"})

        assert result == {}
        assert len(send.text_calls) == 1

    def test_notice_exception_does_not_break_handler(self, http: FakeSessionManager):
        plugin, send, _ = _make_plugin()
        send.text_exception = RuntimeError("send layer down")

        result = _call_extension(plugin, parameters={"voice": "Nope"})

        assert result == {}
        assert len(send.text_calls) == 1

    def test_notice_ignores_send_error_messages_switch(self, http: FakeSessionManager):
        config_data = SBV2TTSPlugin.build_default_config()
        config_data["general"]["send_error_messages"] = False
        plugin, send, _ = _make_plugin(config_data=config_data)

        result = _call_extension(plugin, parameters={"voice": "Nope"})

        assert result == {}
        self._assert_single_notice(send, "本次未附加日语语音：未知音色「Nope」。")

    def test_empty_text_notifies_once(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()

        result = _call_extension(plugin, text="   ")

        assert result == {}
        self._assert_single_notice(
            send, "本次未附加日语语音：正文为空。"
        )
        assert llm.generate_calls == []
        assert http.calls == []

    def test_missing_stream_id_only_logs(self, http: FakeSessionManager):
        plugin, send, _ = _make_plugin()
        chat = dict(CHAT_CONTEXT)
        chat["stream_id"] = ""

        result = _call_extension(plugin, parameters={"voice": "Nope"}, chat=chat)

        assert result == {}
        assert send.text_calls == []

    def test_cli_platform_skips_translate_and_synthesize(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()
        chat = dict(CHAT_CONTEXT)
        chat["platform"] = "maisaka_cli"

        result = _call_extension(plugin, chat=chat)

        assert result == {}
        assert llm.generate_calls == []
        assert http.calls == []
        assert send.text_calls == []
        assert send.custom_calls == []

    def test_cancelled_error_propagates(self, http: FakeSessionManager):
        plugin, send, llm = _make_plugin()
        llm.generate_exception = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            _call_extension(plugin)

        assert send.text_calls == []
        assert http.calls == []


# ============================================================
# manifest 不变量
# ============================================================


class TestManifest:
    @pytest.fixture
    def manifest(self) -> dict:
        path = os.path.join(PLUGIN_DIR, "_manifest.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_id_is_cuda_plugin(self, manifest: dict):
        """id 必须与留档插件区分，避免 loader 重复拉黑。"""
        assert manifest["id"] == "ggsfly.sbv2-tts-cuda-plugin"

    def test_version_is_cuda_plugin_release(self, manifest: dict):
        assert manifest["version"] == "1.3.0"

    def test_capabilities_include_required(self, manifest: dict):
        caps = set(manifest.get("capabilities", []))
        required = {
            "send.custom",
            "send.text",
            "llm.generate",
            "llm.get_available_models",
            # 宿主按声明逐项授权组件开关，enable/disable 是两个 capability，缺一即被拒。
            "component.enable",
            "component.disable",
        }
        assert required.issubset(caps), f"缺少能力: {required - caps}"

    def test_dependencies_include_aiohttp(self, manifest: dict):
        deps = [d.get("name") for d in manifest.get("dependencies", [])]
        assert "aiohttp" in deps

    def test_version_ranges_are_pinned_to_reply_extension_family(self, manifest: dict):
        """ReplyExtension 与 plugin_options 需要 Host 1.4.8 / SDK 2.23.0 起。"""
        assert manifest["host_application"]["min_version"] == "1.4.8"
        assert manifest["host_application"]["max_version"] == "1.4.99"
        assert manifest["sdk"]["min_version"] == "2.23.0"
        assert manifest["sdk"]["max_version"] == "2.23.99"

    def test_display_icon_is_lucide(self, manifest: dict):
        icon = manifest["display"]["icon"]
        assert icon["type"] == "lucide"
        assert icon["value"]

    def test_changelog_points_to_existing_file(self, manifest: dict):
        changelog = manifest["changelog"]
        assert changelog == "CHANGELOG.md"
        assert os.path.isfile(os.path.join(PLUGIN_DIR, changelog))


# ============================================================
# 配置模型、重建与组件开关
# ============================================================


class TestConfigDefaults:
    def test_split_sentences_field_removed(self):
        assert "split_sentences" not in GeneralConfig.model_fields

    def test_split_delay_field_removed(self):
        assert "split_delay" not in GeneralConfig.model_fields

    def test_echo_original_text_removed(self):
        assert "echo_original_text" not in GeneralConfig.model_fields

    def test_tool_enabled_removed(self):
        assert "tool_enabled" not in ComponentsConfig.model_fields

    def test_extension_enabled_default_on(self):
        assert ComponentsConfig().extension_enabled is True

    def test_deprecated_config_keys_removed(self):
        assert not hasattr(ConfigKeys, "GENERAL_ECHO_ORIGINAL_TEXT")
        assert not hasattr(ConfigKeys, "COMPONENTS_TOOL_ENABLED")

    def test_max_text_length_still_present(self):
        assert "max_text_length" in GeneralConfig.model_fields

    def test_config_version_matches_schema(self):
        assert PluginSectionConfig().config_version == "1.3.0"

    def test_translate_fields_moved_out_of_general(self):
        """v1.2.1：翻译开关与 host_llm 模型选择归拢到 [translate] 节。"""
        assert "translate_to_japanese" not in GeneralConfig.model_fields
        assert "translate_model" not in GeneralConfig.model_fields

    def test_voice_language_field_removed(self):
        """v1.2.1：language 配置移除——当前模型仅支持日文，多语种无意义。"""
        assert "language" not in VoiceConfig.model_fields

    def test_section_order_follows_pipeline(self):
        """v1.2.1：节顺序对齐数据流 通用→翻译→合成，组件开关殿后。"""
        expected = ["plugin", "general", "translate", "voice", "components"]
        assert list(plugin_mod.SBV2TTSPluginConfig.model_fields) == expected


class TestConfigRebuild:
    """宿主按新骨架重建旧配置：补齐新键、丢弃废弃键、保留仍存在的旧值。"""

    OLD_CONFIG: Dict[str, Any] = {
        "plugin": {"enabled": True, "config_version": "1.2.1"},
        "general": {
            "timeout": 45,
            "max_text_length": 80,
            "strip_voice_placeholder": False,
            "send_error_messages": False,
            "echo_original_text": True,
        },
        "translate": {
            "translate_to_japanese": False,
            "engine": "index_translate",
            "translate_model": "utils",
            "base_url": "http://127.0.0.1:8123/v1",
            "model": "Index-Translate-2B",
            "style": "口语化",
            "glossary": ["苦力怕→クリーパー"],
            "timeout": 11,
            "max_tokens": 256,
        },
        "voice": {
            "api_url": "http://127.0.0.1:5001/voice",
            "default_voice": "Fusetsu_v1.5",
            "length": 1.2,
            "voices": ["Ling v2", "Fusetsu_v1.5"],
        },
        "components": {"tool_enabled": False, "command_enabled": False},
    }

    def _rebuilt(self) -> Dict[str, Any]:
        return rebuild_plugin_config_data(
            SBV2TTSPlugin.build_default_config(), self.OLD_CONFIG
        )

    def test_new_key_is_filled(self):
        assert self._rebuilt()["components"]["extension_enabled"] is True

    def test_deprecated_keys_dropped(self):
        rebuilt = self._rebuilt()
        assert "echo_original_text" not in rebuilt["general"]
        assert "tool_enabled" not in rebuilt["components"]

    def test_old_tool_switch_is_not_migrated(self):
        """旧 tool_enabled=false 不迁移，扩展按默认 true 补齐。"""
        assert self._rebuilt()["components"]["extension_enabled"] is True
        assert self._rebuilt()["components"]["command_enabled"] is False

    def test_translate_section_values_preserved(self):
        rebuilt = self._rebuilt()
        assert rebuilt["translate"] == self.OLD_CONFIG["translate"]

    def test_rebuilt_config_validates(self):
        config = SBV2TTSPluginConfig.model_validate(self._rebuilt())
        assert config.components.extension_enabled is True
        assert config.plugin.config_version == "1.3.0"


class TestComponentSwitches:
    def test_uses_fully_qualified_names_with_reply_extension_type(self):
        recorder = ComponentRecorder()
        plugin, _, _ = _make_plugin(component=recorder)

        _run(plugin._sync_component_switches())

        assert recorder.calls == [
            ("enable", EXTENSION_NAME, "REPLY_EXTENSION"),
            ("enable", COMMAND_NAME, "COMMAND"),
        ]

    def test_disabled_extension_is_turned_off(self):
        config_data = SBV2TTSPlugin.build_default_config()
        config_data["components"]["extension_enabled"] = False
        recorder = ComponentRecorder()
        plugin, _, _ = _make_plugin(config_data=config_data, component=recorder)

        _run(plugin._sync_component_switches())

        assert ("disable", EXTENSION_NAME, "REPLY_EXTENSION") in recorder.calls
        assert ("enable", COMMAND_NAME, "COMMAND") in recorder.calls

    def test_disabled_command_is_turned_off(self):
        config_data = SBV2TTSPlugin.build_default_config()
        config_data["components"]["command_enabled"] = False
        recorder = ComponentRecorder()
        plugin, _, _ = _make_plugin(config_data=config_data, component=recorder)

        _run(plugin._sync_component_switches())

        assert ("disable", COMMAND_NAME, "COMMAND") in recorder.calls
        assert ("enable", EXTENSION_NAME, "REPLY_EXTENSION") in recorder.calls


# ============================================================
# LLM 任务名缓存：拉取失败 / 空任务表都不得写缓存
# ============================================================


class TestTranslateLlmTaskCache:
    def test_failed_lookup_is_not_cached(self):
        config_data = SBV2TTSPlugin.build_default_config()
        config_data["translate"]["translate_model"] = "utils"
        plugin, _, llm = _make_plugin(config_data=config_data)

        async def _boom() -> List[str]:
            llm.available_models_calls += 1
            raise RuntimeError("boom")

        llm.get_available_models = _boom  # type: ignore[method-assign]

        assert _run(plugin._resolve_translate_llm()) == ("replyer", "utils")
        assert _run(plugin._resolve_translate_llm()) == ("replyer", "utils")
        assert llm.available_models_calls == 2, "拉取失败不应写缓存，下一次调用必须重试"
        assert plugin._available_task_names is None

    def test_empty_task_list_is_not_cached_then_recovers(self):
        config_data = SBV2TTSPlugin.build_default_config()
        config_data["translate"]["translate_model"] = "utils"
        plugin, _, llm = _make_plugin(config_data=config_data)
        llm.available_models = []

        assert _run(plugin._resolve_translate_llm()) == ("replyer", "utils")
        assert plugin._available_task_names is None
        assert llm.available_models_calls == 1

        llm.available_models = ["utils"]
        assert _run(plugin._resolve_translate_llm()) == ("utils", "")
        assert llm.available_models_calls == 2

        llm.available_models = ["planner"]
        assert _run(plugin._resolve_translate_llm()) == ("utils", "")
        assert llm.available_models_calls == 2, "命中缓存后不应再次拉取任务表"


# ============================================================
# 源码不变量：plugin.py 不得 import src.*
# ============================================================


class TestNoSrcImport:
    def test_plugin_py_no_src_import(self):
        """新版 SDK 禁止插件直接 import src.* 模块。"""
        path = os.path.join(PLUGIN_DIR, "plugin.py")
        with open(path, encoding="utf-8") as f:
            source = f.read()
        assert "from src" not in source, "plugin.py 不得 import src.*（SDK 硬约束）"
        assert "import src" not in source, "plugin.py 不得 import src.*（SDK 硬约束）"


# ============================================================
# Hook：宿主把 modified_kwargs 当整体替换，必须原样带回全部入参
# ============================================================


class TestStripVoicePlaceholderHook:
    """`maisaka.reply.before_post_process` 的回归测试。

    宿主 `hook_dispatcher` 用 `dispatch_result.kwargs = dict(modified_kwargs)`
    整体替换，而后处理消费方要求 `skip_post_process` / `enable_splitter` /
    `enable_chinese_typo` 必须是布尔值。钩子若只回传 `response`，剥离会因
    `TypeError` 被宿主吞掉、退回默认策略，功能等于失效。
    """

    HOOK_KWARGS: Dict[str, Any] = {
        "session_id": "sess-1",
        "reply_message_id": "msg-1",
        "reply_tool_args": {"foo": "bar"},
        "skip_post_process": False,
        "enable_splitter": True,
        "enable_chinese_typo": True,
    }

    def _plugin(self) -> Any:
        plugin, _, _ = _make_plugin()
        return plugin

    def test_strip_keeps_all_host_kwargs(self):
        result = _run(
            self._plugin().strip_voice_placeholder(
                response="[语音消息]你好呀", **self.HOOK_KWARGS
            )
        )

        assert result["action"] == "continue"
        modified = result["modified_kwargs"]
        assert modified["response"] == "你好呀"
        assert set(modified) == {"response", *self.HOOK_KWARGS}
        assert modified["skip_post_process"] is False
        assert modified["enable_splitter"] is True
        assert modified["enable_chinese_typo"] is True
        assert modified["reply_tool_args"] == {"foo": "bar"}

    def test_strip_to_empty_keeps_all_host_kwargs(self):
        result = _run(
            self._plugin().strip_voice_placeholder(
                response="[语音消息]", **self.HOOK_KWARGS
            )
        )

        assert result["modified_kwargs"]["response"] == ""
        assert set(result["modified_kwargs"]) == {"response", *self.HOOK_KWARGS}

    def test_no_placeholder_returns_plain_continue(self):
        result = _run(
            self._plugin().strip_voice_placeholder(response="你好呀", **self.HOOK_KWARGS)
        )

        assert result == {"action": "continue"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
