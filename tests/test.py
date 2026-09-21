"""Style-Bert-VITS2 (CUDA) 日文语音合成插件 · 合并单元测试。

单文件覆盖：utils 层、translate 层、backend 层（参数拼装/422 解析/profile 解析）、
manifest 不变量。插件目录名含连字符无法直接 import，故用 importlib 构造伪包加载。

运行：
    uv run pytest plugins/ggsfly_sbv2-tts-CUDA-plugin/tests/test.py -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import asyncio
from typing import Any, Dict, List, Tuple

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
_load_module("backends.base", os.path.join(PLUGIN_DIR, "backends", "base.py"))
voice_mod = _load_module("backends.voice", os.path.join(PLUGIN_DIR, "backends", "voice.py"))
_load_module("backends", os.path.join(PLUGIN_DIR, "backends", "__init__.py"))
translate_mod = _load_module("translate", os.path.join(PLUGIN_DIR, "translate.py"))
# plugin.py 导入会触发 maibot_sdk 装饰器与后端注册；其相对导入依赖已加载的伪包。
plugin_mod = _load_module("plugin", os.path.join(PLUGIN_DIR, "plugin.py"))

# 暴露被测类型
TTSTextUtils = sys.modules[f"{PACKAGE_NAME}.utils.text"].TTSTextUtils
TTSFileManager = sys.modules[f"{PACKAGE_NAME}.utils.file"].TTSFileManager
VoiceBackend = voice_mod.VoiceBackend
VoiceProfile = voice_mod.VoiceProfile
JPTranslator = translate_mod.JPTranslator
ConfigKeys = config_keys_mod.ConfigKeys
GeneralConfig = plugin_mod.GeneralConfig
PluginSectionConfig = plugin_mod.PluginSectionConfig
SBV2TTSPlugin = plugin_mod.SBV2TTSPlugin


def _run(coro):
    """同步跑 coroutine。"""
    return asyncio.run(coro)


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


# ============================================================
# backend 层
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
    def _fake_config_getter(self, voice_name: str = "Ling v2"):
        voices = ["Ling v2", "Fusetsu_v1.5"]
        table = {
            ConfigKeys.VOICE_API_URL: "http://127.0.0.1:5000/voice",
            ConfigKeys.VOICE_DEFAULT_VOICE: voice_name,
            ConfigKeys.VOICE_LANGUAGE: "JP",
            ConfigKeys.VOICE_LENGTH: 1.0,
            ConfigKeys.VOICE_VOICES: voices,
            ConfigKeys.GENERAL_TIMEOUT: 60,
        }
        return lambda key, default=None: table.get(key, default)

    def test_default_profile_resolves(self):
        backend = VoiceBackend(self._fake_config_getter(), log_prefix="[test]")
        profile = backend._get_default_profile()
        assert profile.name == "Ling v2"
        assert profile.model == "Ling-v2"

    def test_empty_text_returns_failure(self):
        backend = VoiceBackend(self._fake_config_getter(), log_prefix="[test]")
        result = _run(backend.execute(""))
        assert result.success is False
        assert "为空" in result.message


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

    def test_version_initial(self, manifest: dict):
        assert manifest["version"] == "1.1.1"

    def test_capabilities_include_required(self, manifest: dict):
        caps = set(manifest.get("capabilities", []))
        required = {"send.custom", "send.text", "llm.generate", "llm.get_available_models", "component.disable"}
        assert required.issubset(caps), f"缺少能力: {required - caps}"

    def test_dependencies_include_aiohttp(self, manifest: dict):
        deps = [d.get("name") for d in manifest.get("dependencies", [])]
        assert "aiohttp" in deps

    def test_sdk_range(self, manifest: dict):
        assert manifest["sdk"]["min_version"] == "2.8.1"


# ============================================================
# 配置模型（v1.1.1：分段字段已移除；回显开关保留且默认关闭）
# ============================================================


class TestConfigDefaults:
    def test_split_sentences_field_removed(self):
        assert "split_sentences" not in GeneralConfig.model_fields

    def test_split_delay_field_removed(self):
        assert "split_delay" not in GeneralConfig.model_fields

    def test_echo_original_text_default_off(self):
        assert GeneralConfig().echo_original_text is False

    def test_max_text_length_still_present(self):
        assert "max_text_length" in GeneralConfig.model_fields

    def test_config_version_bumped(self):
        assert PluginSectionConfig().config_version == "1.1.1"


# ============================================================
# 语音前回显文本构造
# ============================================================


class TestBuildEchoText:
    def test_returns_stripped_text(self):
        assert SBV2TTSPlugin._build_echo_text("  你好世界  ") == "你好世界"

    def test_multiline_preserved(self):
        assert SBV2TTSPlugin._build_echo_text("第一行\n第二行") == "第一行\n第二行"

    def test_empty_returns_empty(self):
        assert SBV2TTSPlugin._build_echo_text("") == ""


# ============================================================
# 回显作用域不变量：仅 @Tool、整段一条、不回 planner、受多重门控
# ============================================================


class TestEchoScope:
    @pytest.fixture
    def source(self) -> str:
        path = os.path.join(PLUGIN_DIR, "plugin.py")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_run_pipeline_has_default_off_echo_param(self, source: str):
        assert "echo_enabled: bool = False" in source

    def test_only_tool_enables_echo(self, source: str):
        # echo_enabled=True 只应出现在 @Tool 调用点（恰好 1 次）
        assert source.count("echo_enabled=True") == 1

    def test_command_does_not_enable_echo(self, source: str):
        # Command 调用点用 raw_text=user_text，且其参数块不含 echo_enabled
        idx = source.index("raw_text=user_text")
        call_block = source[idx: idx + 200]
        assert "echo_enabled" not in call_block

    def test_echo_guarded_by_translate_and_sync_false(self, source: str):
        norm = " ".join(source.split())
        assert (
            "echo_enabled and self.config.general.echo_original_text "
            "and self.config.general.translate_to_japanese"
        ) in norm
        assert "sync_to_maisaka_history=False" in norm


# ============================================================
# 分段子系统已移除（源码不变量）
# ============================================================


class TestSegmentationRemoved:
    @pytest.fixture
    def plugin_source(self) -> str:
        path = os.path.join(PLUGIN_DIR, "plugin.py")
        with open(path, encoding="utf-8") as f:
            return f.read()

    @pytest.fixture
    def translate_source(self) -> str:
        path = os.path.join(PLUGIN_DIR, "translate.py")
        with open(path, encoding="utf-8") as f:
            return f.read()

    @pytest.fixture
    def text_source(self) -> str:
        path = os.path.join(PLUGIN_DIR, "utils", "text.py")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_plugin_no_split_symbols(self, plugin_source: str):
        for token in (
            "split_sentences",
            "_SPLIT_MARKER",
            "|||SPLIT|||",
            "clamp_sentences",
            "_send_in_segments",
            "split_delay",
        ):
            assert token not in plugin_source, f"plugin.py 仍残留 {token}"

    def test_translate_no_split_marker(self, translate_source: str):
        assert "|||SPLIT|||" not in translate_source

    def test_text_utils_split_clamp_removed(self, text_source: str):
        assert "def split_sentences" not in text_source
        assert "def clamp_sentences" not in text_source


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
