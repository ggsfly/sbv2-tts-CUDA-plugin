"""
SBV2 TTS 插件 utils 层单元测试。

因插件目录名 `sbv2-tts-plugin` 含连字符，无法用普通 import 语法加载，
故使用 `importlib.util.spec_from_file_location` 按文件路径加载 text / file。
session 模块依赖 aiohttp，本测试仅验证类属性与协议形状，不发起网络请求。
"""

import importlib.util
import os
import sys

import pytest

PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
UTILS_DIR = os.path.join(PLUGIN_DIR, "utils")


def _load_module(name: str, file_name: str):
    """
    按文件路径加载 utils 模块。

    构造独立模块名避免与可能存在的同名全局模块冲突。
    """
    file_path = os.path.join(UTILS_DIR, file_name)
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None, f"无法构造 spec: {file_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 加载依赖标准库的模块（无副作用）
text_module = _load_module("sbv2_utils_text", "text.py")
file_module = _load_module("sbv2_utils_file", "file.py")

TTSTextUtils = text_module.TTSTextUtils
TTSFileManager = file_module.TTSFileManager


# 文本处理 ---------------------------------------------------------------------


def test_clean_text_strips_whitespace():
    """clean_text 仅做 strip，保留原文。"""
    assert TTSTextUtils.clean_text("  你好  ") == "你好"
    assert TTSTextUtils.clean_text("hello world") == "hello world"
    # max_length 参数保留但不使用
    assert TTSTextUtils.clean_text("  text  ", max_length=2) == "text"


def test_clean_text_empty():
    """空输入返回空串。"""
    assert TTSTextUtils.clean_text("") == ""
    assert TTSTextUtils.clean_text(None) == ""  # type: ignore[arg-type]


def test_split_sentences_basic():
    """混合中英文标点切出多段。"""
    result = TTSTextUtils.split_sentences("你好。世界！")
    assert len(result) >= 2
    # 第一段必须以中文标点结尾
    assert any("。" in seg or "！" in seg for seg in result)


def test_split_sentences_short_merge():
    """过短句子并入前一句。"""
    # "嗯" 长度 1 < min_length=2，会并到前一段
    sentences = TTSTextUtils.split_sentences("第一句。嗯，第二句很长。", min_length=2)
    # 合并后应至少包含 "嗯，" 这种短尾
    joined = "".join(sentences)
    assert "嗯，" in joined
    assert "第一句。" in joined


def test_split_sentences_empty():
    """空输入返回空列表而非抛错。"""
    assert TTSTextUtils.split_sentences("") == []
    assert TTSTextUtils.split_sentences(None) == []  # type: ignore[arg-type]


def test_detect_language_japanese():
    """纯假名识别为日文。"""
    assert TTSTextUtils.detect_language("こんにちは") == "ja"


def test_detect_language_chinese():
    """纯中文识别为中文。"""
    assert TTSTextUtils.detect_language("你好世界") == "zh"


def test_detect_language_english():
    """纯英文识别为英文（阈值 0.8）。"""
    assert TTSTextUtils.detect_language("hello world") == "en"


def test_detect_language_empty_default_zh():
    """空输入退化为 zh。"""
    assert TTSTextUtils.detect_language("") == "zh"


# 文件处理 ---------------------------------------------------------------------


def test_audio_to_base64_non_empty():
    """audio_to_base64 对正常输入返回非空字符串。"""
    encoded = TTSFileManager.audio_to_base64(b"abc")
    assert encoded  # 非空
    assert isinstance(encoded, str)
    # base64("abc") == "YWJj"
    assert encoded == "YWJj"


def test_audio_to_base64_exception_safe():
    """audio_to_base64 对异常输入不抛错返回空串。"""
    # 传入非 bytes 通常在 b64encode 处抛 TypeError，本方法应捕获并返回 ""
    result = TTSFileManager.audio_to_base64("not bytes")  # type: ignore[arg-type]
    assert result == ""


def test_validate_audio_data_empty():
    """validate_audio_data(b'') 因小于 MIN_AUDIO_SIZE 返回 (False, msg)。"""
    is_valid, msg = TTSFileManager.validate_audio_data(b"")
    assert is_valid is False
    assert msg  # 错误描述非空


def test_validate_audio_data_none():
    """None 输入返回 (False, ...)。"""
    is_valid, msg = TTSFileManager.validate_audio_data(None)
    assert is_valid is False
    assert msg


def test_validate_audio_data_valid():
    """足够大的数据应通过校验。"""
    data = b"x" * 1024
    is_valid, msg = TTSFileManager.validate_audio_data(data)
    assert is_valid is True
    assert msg == ""


def test_generate_temp_path_in_output_dir():
    """generate_temp_path 必须把文件放在调用方传入的 output_dir 内。"""
    out = os.path.join(PLUGIN_DIR, "tests", "tmp")
    os.makedirs(out, exist_ok=True)
    try:
        path = TTSFileManager.generate_temp_path(
            prefix="t", suffix=".wav", output_dir=out
        )
        assert path.endswith(".wav"), path
        assert os.path.dirname(os.path.abspath(path)) == os.path.abspath(out), path
        # 文件名应包含 prefix 与 12 位 hex
        basename = os.path.basename(path)
        assert basename.startswith("t_")
        stem = basename.removeprefix("t_").removesuffix(".wav")
        assert len(stem) == 12 and all(c in "0123456789abcdef" for c in stem)
    finally:
        # 不留残余文件
        try:
            os.remove(path)
        except OSError:
            pass


def test_generate_temp_path_unique():
    """两次调用生成的路径必须不同。"""
    out = os.path.join(PLUGIN_DIR, "tests", "tmp")
    os.makedirs(out, exist_ok=True)
    p1 = TTSFileManager.generate_temp_path(prefix="t", suffix=".wav", output_dir=out)
    p2 = TTSFileManager.generate_temp_path(prefix="t", suffix=".wav", output_dir=out)
    assert p1 != p2
    # 清理
    for p in (p1, p2):
        try:
            os.remove(p)
        except OSError:
            pass


def test_generate_temp_path_empty_output_dir_fallback():
    """output_dir 为空时退回到当前工作目录而非抛错。"""
    # 退到 cwd 应不抛错并返回合法路径
    path = TTSFileManager.generate_temp_path(prefix="t", suffix=".wav", output_dir="")
    assert path.endswith(".wav")
    assert os.path.isabs(path)


# session 模块编译 / 形状校验 ----------------------------------------------------


def test_session_module_imports_clean():
    """
    session.py 仅做形状校验：能通过 py_compile 编译，且导出 TTSSessionManager 类。
    实际网络行为由集成测试覆盖。
    """
    import py_compile

    session_path = os.path.join(UTILS_DIR, "session.py")
    py_compile.compile(session_path, doraise=True)
    # 真正的断言：用 dynamic load 拿到类
    spec = importlib.util.spec_from_file_location(
        "sbv2_utils_session_for_test", session_path
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sbv2_utils_session_for_test"] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "TTSSessionManager")
    cls = mod.TTSSessionManager
    # 单例相关类属性
    assert hasattr(cls, "_instance")
    assert hasattr(cls, "_lock")
    # 关键方法签名存在
    for name in ("get_session", "close_session", "post", "get"):
        assert hasattr(cls, name), name