"""
SBV2 后端集成测试（端到端）。

对**正在运行的本地 SBV2 推理服务**（默认监听 ``127.0.0.1:3000``）发起
真实的 ``POST /synthesize`` HTTP 调用，验证 :class:`Sbv2Backend` 的
成功 / 失败 / 异常三类路径。

加载策略
--------
插件目录名 ``sbv2-tts-plugin`` 含连字符，普通 ``import`` 无法直接加载，
而 :mod:`backends.sbv2` 使用相对导入（``from ..config_keys`` 等）。
故先用 :mod:`importlib` 构造一个名为 ``sbv2_tts_plugin`` 的伪包，
把 ``config_keys``、``utils``、``backends`` 子包逐个注册到 ``sys.modules``，
让相对导入可以解析。

前置条件
--------
SBV2 推理服务必须手动启动并监听 ``127.0.0.1:3000``，且至少已加载以下
说话人模型：``Ling v2``、``Fusetsu_v1.5``。

注意
----
测试**不会**停止 / 重启 SBV2 服务——服务由用户手动管理。
"""

from __future__ import annotations

from typing import Any, List, Tuple

import asyncio
import importlib.util
import logging
import os
import sys

import pytest

# ============================================================
# 路径常量
# ============================================================

PLUGIN_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PACKAGE_NAME: str = "sbv2_tts_plugin"  # 伪包名（绕开连字符目录名）
TMP_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "tmp"))

# 测试目标服务地址（SBV2 推理服务）
SBV2_BASE_URL: str = "http://127.0.0.1:3000"
SBV2_API_URL: str = f"{SBV2_BASE_URL}/synthesize"

# 已加载的说话人（SBV2 服务中已部署）
KNOWN_VOICE: str = "Ling v2"
ALT_VOICE: str = "Fusetsu_v1.5"
UNKNOWN_VOICE: str = "nonexistent_voice_for_test"

# ============================================================
# 通过 importlib 构造可解析相对导入的伪包
# ============================================================


def _load_as_subpackage(alias: str, init_file: str) -> Any:
    """
    以 *alias* 作为模块名加载子包 / 子模块，并注册到 ``sys.modules``。

    关键：包级模块必须保留 ``__path__``，否则其内部的相对导入无法解析。
    """
    spec = importlib.util.spec_from_file_location(
        alias,
        init_file,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None, f"无法构造 spec: {init_file}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _load_module(alias: str, file_path: str) -> Any:
    """以 *alias* 作为模块名加载普通（无 ``__init__.``）模块。"""
    spec = importlib.util.spec_from_file_location(alias, file_path)
    assert spec is not None and spec.loader is not None, f"无法构造 spec: {file_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# 1) 伪包本体（``sbv2_tts_plugin``）
# 注意：插件根目录没有 ``__init__.py``，因此这里只构造一个 namespace 包
# 风格的模块占位，不调用 ``exec_module``。
import types

_pkg_module = types.ModuleType(PACKAGE_NAME)
_pkg_module.__path__ = [PLUGIN_DIR]  # type: ignore[attr-defined]
sys.modules[PACKAGE_NAME] = _pkg_module

# 2) 各子包 / 模块
config_keys_module = _load_module(
    f"{PACKAGE_NAME}.config_keys",
    os.path.join(PLUGIN_DIR, "config_keys.py"),
)
utils_pkg = _load_as_subpackage(
    f"{PACKAGE_NAME}.utils",
    os.path.join(PLUGIN_DIR, "utils", "__init__.py"),
)
utils_pkg.__path__ = [os.path.join(PLUGIN_DIR, "utils")]

utils_text_module = _load_module(
    f"{PACKAGE_NAME}.utils.text",
    os.path.join(PLUGIN_DIR, "utils", "text.py"),
)
utils_file_module = _load_module(
    f"{PACKAGE_NAME}.utils.file",
    os.path.join(PLUGIN_DIR, "utils", "file.py"),
)
utils_session_module = _load_module(
    f"{PACKAGE_NAME}.utils.session",
    os.path.join(PLUGIN_DIR, "utils", "session.py"),
)

backends_pkg = _load_as_subpackage(
    f"{PACKAGE_NAME}.backends",
    os.path.join(PLUGIN_DIR, "backends", "__init__.py"),
)
backends_pkg.__path__ = [os.path.join(PLUGIN_DIR, "backends")]

backends_base_module = _load_module(
    f"{PACKAGE_NAME}.backends.base",
    os.path.join(PLUGIN_DIR, "backends", "base.py"),
)
backends_sbv2_module = _load_module(
    f"{PACKAGE_NAME}.backends.sbv2",
    os.path.join(PLUGIN_DIR, "backends", "sbv2.py"),
)

# 暴露被测类型
Sbv2Backend = backends_sbv2_module.Sbv2Backend
TTSResult = backends_base_module.TTSResult
ConfigKeys = config_keys_module.ConfigKeys
TTSSessionManager = utils_session_module.TTSSessionManager

# ============================================================
# 测试夹具：构造可独立工作的 Sbv2Backend 实例
# ============================================================


def _run_in_fresh_loop(coro_factory):
    """
    在全新的 event loop 中执行异步协程，并在退出前关闭并清理
    :class:`TTSSessionManager` 单例持有的 aiohttp session。

    :class:`TTSSessionManager` 是模块级单例，绑定到首次 ``asyncio.run``
    的 loop。后续 ``asyncio.run`` 创建新 loop 后，旧的 ``ClientSession``
    与之失联，会触发 ``Event loop is closed``。因此每次调用都重置
    单例，让其在当前 loop 内重建。
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        # 重置单例，使其在本 loop 内首次访问时重建
        TTSSessionManager._instance = None  # type: ignore[attr-defined]
        return loop.run_until_complete(coro_factory())
    finally:
        # 取消 loop 内尚未完成的延迟清理 task（send_audio 创建的 180s 延迟任务），
        # 避免 pytest 出现 "Task was destroyed but it is pending!" 噪音。
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        except Exception:
            pass
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()
        asyncio.set_event_loop(None)
        TTSSessionManager._instance = None  # type: ignore[attr-defined]


def _fake_config_getter(key: str, default: Any = None) -> Any:
    """针对本测试场景定制的配置 getter。"""
    # ``general.audio_output_dir`` 在 :class:`backends.base` 中以扁平字符串
    # 形式直接访问，未在 :class:`ConfigKeys` 中暴露常量；此处使用字面量。
    table: dict[str, Any] = {
        ConfigKeys.SBV2_API_URL: SBV2_API_URL,
        ConfigKeys.SBV2_DEFAULT_IDENT: KNOWN_VOICE,
        ConfigKeys.GENERAL_TIMEOUT: 60,
        ConfigKeys.GENERAL_USE_BASE64_AUDIO: False,
        "general.audio_output_dir": TMP_DIR,
    }
    return table.get(key, default)


def _build_backend(collected: List[Tuple[str, Any]]) -> Sbv2Backend:
    """构造后端实例并注入 ``send_custom`` 收集闭包。"""

    async def fake_send_custom(
        message_type: str, content: Any, **kwargs: Any
    ) -> bool:
        collected.append((message_type, content))
        return True

    backend = Sbv2Backend(_fake_config_getter, log_prefix="[sbv2-test]")
    backend.set_send_custom(fake_send_custom)
    return backend


@pytest.fixture
def backend() -> Sbv2Backend:
    """提供已注入 ``send_custom`` 的后端实例；``collected`` 通过闭包共享。"""
    collected: List[Tuple[str, Any]] = []
    backend = _build_backend(collected)
    # 把 collected 绑到 backend 上便于测试断言
    backend._collected_for_test = collected  # type: ignore[attr-defined]
    return backend


# ============================================================
# 守护：测试开始前确认 SBV2 服务可达
# ============================================================


def _ensure_service_reachable() -> None:
    """确保 SBV2 服务可达；不可达则跳过整个文件。"""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{SBV2_BASE_URL}/", timeout=5) as resp:
            if resp.status != 200:
                pytest.skip(f"SBV2 服务异常: HTTP {resp.status}")
    except Exception as exc:
        pytest.skip(f"SBV2 服务不可达 ({SBV2_BASE_URL}): {exc}")


# 模块级执行一次预检（pytest 会以 ``session`` 形式跑）
_ensure_service_reachable()


# ============================================================
# 测试用例
# ============================================================


def test_happy_path_ling_v2(backend: Sbv2Backend) -> None:
    """成功路径：使用 ``Ling v2`` 合成日文，断言 file 模式 + RIFF 头。"""
    text: str = "こんにちは、マスター"

    result: TTSResult = _run_in_fresh_loop(
        lambda: backend.execute(text, voice=KNOWN_VOICE)
    )

    collected: List[Tuple[str, Any]] = backend._collected_for_test  # type: ignore[attr-defined]
    assert result.success is True, (
        f"合成应成功，实际 message={result.message!r}, "
        f"audio_path={result.audio_path!r}"
    )
    assert result.backend_name == "sbv2"
    assert collected, "send_custom 未被调用"
    last_type, last_content = collected[-1]
    # 走文件路径模式（use_base64_audio=False）
    assert last_type == "voiceurl", f"期望 voiceurl 通道，实际 {last_type!r}"
    assert isinstance(last_content, str) and os.path.isfile(last_content), (
        f"语音文件路径无效: {last_content!r}"
    )
    # 文件应以 RIFF 开头（WAV 文件魔数）
    with open(last_content, "rb") as fp:
        head: bytes = fp.read(4)
    assert head == b"RIFF", f"WAV 头部应为 b'RIFF'，实际 {head!r}"


def test_happy_path_fusetsu(backend: Sbv2Backend) -> None:
    """成功路径：使用 ``Fusetsu_v1.5`` 音色。"""
    result: TTSResult = _run_in_fresh_loop(
        lambda: backend.execute("おはようございます", voice=ALT_VOICE)
    )
    assert result.success is True, (
        f"Fusetsu_v1.5 合成应成功，实际 message={result.message!r}"
    )
    assert result.backend_name == "sbv2"


def test_unknown_voice_returns_failure(backend: Sbv2Backend) -> None:
    """
    失败路径：传入不存在的音色，断言返回失败结果。

    注意：SBV2 推理服务对未知说话人返回 ``HTTP 500``（``model not found``），
    后端会落入 ``其他非 2xx`` 分支，返回 ``"SBV2 API调用失败: 500"``。
    若服务将来改成返回 ``422``，则会进入 ``SBV2 参数错误`` 分支。
    两种消息均表明“参数/音色错误”，因此断言使用 ``or`` 兼容两种契约。
    """
    result: TTSResult = _run_in_fresh_loop(
        lambda: backend.execute("テスト", voice=UNKNOWN_VOICE)
    )
    assert result.success is False
    assert (
        "参数错误" in result.message
        or "API调用失败" in result.message
    ), f"失败消息不符合预期: {result.message!r}"


def test_empty_text_returns_failure(backend: Sbv2Backend) -> None:
    """失败路径：空文本，应在请求前直接返回失败，不打到服务。"""
    collected_before: List[Tuple[str, Any]] = list(
        backend._collected_for_test  # type: ignore[attr-defined]
    )
    result: TTSResult = _run_in_fresh_loop(lambda: backend.execute(""))
    assert result.success is False
    assert "为空" in result.message, f"失败消息不符合预期: {result.message!r}"
    # 空文本校验在请求之前，send_custom 不应被调用
    assert backend._collected_for_test == collected_before  # type: ignore[attr-defined]


# ============================================================
# 收尾：清理 tmp 目录中产生的 wav 文件
# ============================================================


def _cleanup_tmp_wav() -> None:
    """测试结束后清理本模块产生的临时 wav。"""
    if not os.path.isdir(TMP_DIR):
        return
    for name in os.listdir(TMP_DIR):
        if name.lower().endswith(".wav"):
            try:
                os.remove(os.path.join(TMP_DIR, name))
            except OSError:
                pass


@pytest.fixture(scope="module", autouse=True)
def _cleanup_after_module() -> None:
    """模块级收尾：在所有用例跑完后清理 ``tests/tmp/*.wav``。"""
    yield
    _cleanup_tmp_wav()


# ============================================================
# 抑制 HTTP 噪音日志（aiohttp / asyncio 的 INFO）
# ============================================================

logging.getLogger("aiohttp.access").setLevel(logging.WARNING)