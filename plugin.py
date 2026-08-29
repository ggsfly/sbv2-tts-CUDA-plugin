"""SBV2 日文语音合成插件。

本插件调用本地 SBV2 推理服务合成日文语音。SBV2 是日文推理模型，
因此插件先调用 LLM 把用户输入翻译为自然日文，再送入本地推理服务合成。

触发面：
- ``@Tool``：由 LLM 自主决定调用，对应 ``sbv2_tts_tool`` 组件；
- ``@Command``：由用户通过 ``/sbv2`` / ``/voice`` 命令触发，对应
  ``sbv2_tts_command`` 组件。

管线（Tool 与 Command 共用）：
1. 文本清理（去除首尾空白）
2. 长度校验，超出 ``general.max_text_length`` 时降级 / 发错误提示
3. 智能分割：``|||SPLIT|||`` 标记优先切分 > ``TTSTextUtils.split_sentences`` > 单段
4. 若 ``general.translate_to_japanese`` 开启，调用 :class:`JPTranslator` 把每段
   中文译为日文（翻译失败时 **绝不** 用中文喂 SBV2）
5. 对每段译文再走 ``TTSTextUtils.split_sentences`` 自动切分（避免单段过长）
6. 调用 :class:`Sbv2Backend` 逐段合成，并通过 ``ctx.send.custom`` 把
   ``voiceurl`` 投递到聊天流
"""

import sys

sys.dont_write_bytecode = True

from typing import Any, Callable, Dict, List, Tuple

import asyncio
import logging

from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ToolParameterInfo, ToolParamType

import aiohttp

from .backends import Sbv2Backend, TTSBackendRegistry
from .backends.base import TTSResult
from .config_keys import ConfigKeys
from .translate import JPTranslator
from .utils.text import TTSTextUtils
from .utils.session import TTSSessionManager

logger = logging.getLogger("plugin.sbv2_tts")

# 智能分割标记，与 xuqian13_tts-voice-plugin 保持一致
_SPLIT_MARKER = "|||SPLIT|||"

# 后端注册表注册名（与 ``Sbv2Backend.backend_name`` 保持一致）
_BACKEND_NAME = "sbv2"

# 连接性探测目标（仅探测根路径；``/synthesize`` 才是合成入口）
_PROBE_URL = "http://127.0.0.1:3000/"
_PROBE_TIMEOUT_SECONDS = 3


# ─── 配置模型 ────────────────────────────────────────────────────────────


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=False, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本")


class GeneralConfig(PluginConfigBase):
    """通用配置。"""

    __ui_label__ = "通用"
    __ui_icon__ = "settings"
    __ui_order__ = 1

    timeout: int = Field(default=60, description="请求超时（秒）")
    max_text_length: int = Field(default=200, description="单次合成的最大文本长度")
    use_base64_audio: bool = Field(default=False, description="是否以 base64 形式回传音频")
    split_sentences: bool = Field(default=True, description="是否按句子拆分合成")
    split_delay: float = Field(default=0.3, description="分句之间的发送间隔（秒）")
    send_error_messages: bool = Field(default=True, description="是否向聊天流回显错误提示")
    translate_to_japanese: bool = Field(
        default=True,
        description="是否先把中文翻译为日文再合成（SBV2 为日文推理模型）",
    )
    translate_model: str = Field(
        default="",
        description="翻译用 LLM 模型名，留空用 Host 默认",
    )


class ComponentsConfig(PluginConfigBase):
    """组件开关。"""

    __ui_label__ = "组件"
    __ui_icon__ = "blocks"
    __ui_order__ = 2

    tool_enabled: bool = Field(default=True, description="是否启用 Tool 组件")
    command_enabled: bool = Field(default=True, description="是否启用 Command 组件")


class Sbv2Config(PluginConfigBase):
    """SBV2 推理服务配置。"""

    __ui_label__ = "SBV2"
    __ui_icon__ = "server"
    __ui_order__ = 3

    api_url: str = Field(default="http://127.0.0.1:3000/synthesize", description="SBV2 推理服务地址")
    default_ident: str = Field(default="Ling v2", description="默认说话人")
    idents: List[str] = Field(
        default_factory=lambda: ["Ling v2", "Fusetsu_v1.5"],
        description="可选说话人列表",
    )


class SBV2TTSPluginConfig(PluginConfigBase):
    """SBV2 日文语音合成插件总配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    sbv2: Sbv2Config = Field(default_factory=Sbv2Config)


# ─── 插件主体 ────────────────────────────────────────────────────────────


class Sbv2TTSPlugin(MaiBotPlugin):
    """SBV2 日文语音合成插件（MaiBot SDK 2.x 版）。

    由 Tool / Command 共同驱动 SBV2 推理服务的 ``/synthesize`` 接口，
    并通过 :meth:`_send_in_segments` 把多段语音依次投递到当前聊天流。
    """

    config_model = SBV2TTSPluginConfig

    # ─── 生命周期 ────────────────────────────────────────────────────────

    async def on_load(self) -> None:
        """插件加载钩子。

        1. 用 aiohttp 短超时探测 SBV2 服务存活（任意状态码均视为存活，
           仅连接被拒 / 超时 / DNS 失败视为未连通）。
        2. 按 ``[components]`` 配置禁用 Tool / Command 组件。
        3. 不阻断插件加载：未连通只发 warning。
        """

        await self._probe_sbv2_service()

        if not self.config.components.tool_enabled:
            await self.ctx.component.disable_component(
                name="sbv2_tts_tool",
                component_type="TOOL",
            )
            self.ctx.logger.info("SBV2 Tool 组件已按配置禁用")

        if not self.config.components.command_enabled:
            await self.ctx.component.disable_component(
                name="sbv2_tts_command",
                component_type="COMMAND",
            )
            self.ctx.logger.info("SBV2 Command 组件已按配置禁用")

        self.ctx.logger.info(
            "SBV2 日文 TTS 插件已加载，已注册后端=%s",
            TTSBackendRegistry.list_backends(),
        )

    async def on_unload(self) -> None:
        """插件卸载钩子：关闭 aiohttp 单例 session。"""

        session_manager = await TTSSessionManager.get_instance()
        await session_manager.close_session()
        self.ctx.logger.info("SBV2 日文 TTS 插件已卸载")

    async def on_config_update(
        self,
        scope: str,
        config_data: Dict[str, Any],
        version: str,
    ) -> None:
        """配置热更新回调。

        Args:
            scope: 配置变更范围。
            config_data: 最新配置数据。
            version: 配置版本号。
        """

        del scope
        del config_data
        self.ctx.logger.info("SBV2 插件配置已更新: version=%s", version)

    # ─── 内部：连接性探测 ────────────────────────────────────────────────

    async def _probe_sbv2_service(self) -> None:
        """短超时探测 SBV2 服务存活。

        任意 HTTP 状态码（404 / 405 / 422 等）都视为"服务监听中"，仅
        连接被拒 / 超时 / DNS 失败视为未连通。未连通仅 warning，不抛错。
        """

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_PROBE_TIMEOUT_SECONDS)
            ) as session:
                try:
                    async with session.get(_PROBE_URL) as resp:
                        # 任何状态码都说明 TCP 握手成功 + 服务在监听
                        del resp
                except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
                    self.ctx.logger.warning(
                        "SBV2 服务未连通，路径=%s，原因=%s",
                        _PROBE_URL, exc,
                    )
        except (aiohttp.ClientError, OSError) as exc:
            # ClientSession 构造阶段就失败（极少见），同样只 warning
            self.ctx.logger.warning(
                "SBV2 服务未连通，路径=%s，原因=%s",
                _PROBE_URL, exc,
            )

    # ─── 内部：配置桥接 ──────────────────────────────────────────────────

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """根据扁平 key（``section.field``）从 Pydantic 配置取值。

        backends 模块基于 ``config_getter(key, default)`` 读取配置，
        这里把 Pydantic 嵌套配置展开成扁平访问。

        特殊处理：``general.audio_output_dir``（``ConfigKeys._GENERAL_AUDIO_OUTPUT_DIR``）
        不在 Pydantic 模型里，直接返回 :class:`PluginPaths.runtime_dir` 的字符串。
        这样 backends 的 ``send_audio`` 能把临时音频落到运行时目录下。

        Args:
            key: 形如 ``"sbv2.api_url"`` 的扁平配置键。
            default: 当配置不存在时返回的默认值。

        Returns:
            Any: 对应配置项的当前值；找不到则返回 ``default``。
        """

        if "." not in key:
            return default

        # 音频输出目录桥接到运行时目录，不走 Pydantic
        if key == "general.audio_output_dir":
            return str(self.ctx.paths.runtime_dir)

        section_name, _, field_name = key.partition(".")
        section = getattr(self.config, section_name, None)
        if section is None:
            return default
        if not hasattr(section, field_name):
            return default
        return getattr(section, field_name)

    # ─── 内部：发送闭包 ──────────────────────────────────────────────────

    def _make_send_custom(self, stream_id: str) -> Callable[..., Any]:
        """构造发送自定义消息的闭包。

        backends 通过 ``set_send_custom`` 注入该闭包，
        把 ``voice`` / ``voiceurl`` 投到当前聊天流。

        Args:
            stream_id: 当前聊天流 ID，由 Tool / Command handler 注入。

        Returns:
            Callable[..., Any]: 与 :class:`TTSBackendBase._send_custom` 兼容的闭包。
        """

        async def _send_custom(
            message_type: str,
            content: Any,
            **kwargs: Any,
        ) -> bool:
            """向当前会话发送自定义类型消息。"""

            del kwargs
            result = await self.ctx.send.custom(
                message_type, content, stream_id,
            )
            return bool(result)

        return _send_custom

    # ─── 内部：后端构造 ──────────────────────────────────────────────────

    def _create_backend(
        self,
        stream_id: str,
        log_prefix: str,
    ) -> Any:
        """创建 SBV2 后端实例并注入 ``send_custom`` 闭包。

        Args:
            stream_id: 当前聊天流 ID。
            log_prefix: 日志前缀。

        Returns:
            :class:`Sbv2Backend` 实例，或未注册时返回 ``None``。
        """

        backend = TTSBackendRegistry.create(
            _BACKEND_NAME, self._get_config_value, log_prefix,
        )
        if backend is None:
            return None
        if hasattr(backend, "set_send_custom"):
            backend.set_send_custom(self._make_send_custom(stream_id))
        return backend

    async def _execute_backend(
        self,
        text: str,
        stream_id: str,
        log_prefix: str,
        voice: str = "",
    ) -> TTSResult:
        """调用 SBV2 后端合成一段文本。

        Args:
            text: 待合成日文文本。
            stream_id: 当前聊天流 ID。
            log_prefix: 日志前缀。
            voice: 音色 ident；空串则用 ``sbv2.default_ident``。

        Returns:
            :class:`TTSResult`。后端未注册时返回失败结果。
        """

        backend = self._create_backend(stream_id, log_prefix)
        if backend is None:
            return TTSResult(
                success=False,
                message=f"未注册的 TTS 后端: {_BACKEND_NAME}",
                backend_name=_BACKEND_NAME,
            )
        return await backend.execute(text, voice=voice)

    # ─── 内部：翻译 ──────────────────────────────────────────────────────

    async def _translate_text(
        self,
        text: str,
        log_prefix: str,
    ) -> Tuple[bool, str]:
        """根据配置决定是否翻译为日文。

        Args:
            text: 待翻译中文文本（已清理非空）。
            log_prefix: 日志前缀。

        Returns:
            ``(success, payload)``：
            - 不需要翻译时：``(True, 原文本)``
            - 翻译成功：``(True, 译文)``
            - 翻译失败：``(False, error_detail)``——绝不返回原文
        """

        if not self.config.general.translate_to_japanese:
            return True, text

        translator = JPTranslator(
            max_length=self.config.general.max_text_length,
        )
        return await translator.translate(
            text,
            log_prefix,
            self.ctx.llm.generate,
            translate_model=self.config.general.translate_model,
        )

    # ─── 内部：分段发送 ──────────────────────────────────────────────────

    async def _send_in_segments(
        self,
        sentences: List[str],
        stream_id: str,
        log_prefix: str,
        voice: str,
        split_delay: float,
    ) -> TTSResult:
        """逐段合成并投递。

        单句：直接调一次后端，失败时按 ``send_error_messages`` 配置发错误提示。
        多句：任一段失败→记录、停止后续段、汇总 ``success_count`` / ``total``。

        Args:
            sentences: 待合成段落列表。
            stream_id: 当前聊天流 ID。
            log_prefix: 日志前缀。
            voice: 音色 ident。
            split_delay: 段落间隔秒数。

        Returns:
            :class:`TTSResult`。
        """

        send_errors: bool = self.config.general.send_error_messages

        # 单段：直接调一次后端
        if len(sentences) <= 1:
            single_text = sentences[0] if sentences else ""
            if not single_text:
                return TTSResult(
                    success=False,
                    message="无可合成的文本",
                    backend_name=_BACKEND_NAME,
                )
            result = await self._execute_backend(
                single_text, stream_id, log_prefix, voice=voice,
            )
            if not result.success and send_errors:
                await self.ctx.send.text(
                    f"语音合成失败: {result.message}", stream_id,
                )
            return result

        # 多段：逐段合成，任一段失败→记录并停止
        total = len(sentences)
        success_count = 0
        last_error: str = ""
        for index, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            result = await self._execute_backend(
                sentence, stream_id, log_prefix, voice=voice,
            )
            if result.success:
                success_count += 1
                self.ctx.logger.debug(
                    "%s 分段 %d/%d 合成成功", log_prefix, index + 1, total,
                )
            else:
                # 失败：记录 + 停止后续段
                logger.error(
                    "%s 分段 %d/%d 合成失败: %s",
                    log_prefix, index + 1, total, result.message,
                )
                last_error = result.message
                break
            if index < total - 1 and split_delay > 0:
                await asyncio.sleep(split_delay)

        # T8 验收依赖此日志字符串
        logger.info(
            "%s 成功发送 %d/%d 条语音",
            log_prefix, success_count, total,
        )

        if success_count == 0:
            if send_errors:
                await self.ctx.send.text(
                    f"语音合成失败: {last_error or '未知错误'}", stream_id,
                )
            return TTSResult(
                success=False,
                message="所有语音发送失败",
                backend_name=_BACKEND_NAME,
            )
        return TTSResult(
            success=True,
            message=f"成功发送 {success_count}/{total} 条语音",
            backend_name=_BACKEND_NAME,
        )

    # ─── 内部：分割 + 翻译 + 合成 共享管线 ───────────────────────────────

    async def _run_pipeline(
        self,
        raw_text: str,
        stream_id: str,
        log_prefix: str,
        voice: str,
    ) -> TTSResult:
        """翻译-合成管线的共享实现。

        步骤：
        1. 文本清理 + 长度校验
        2. ``|||SPLIT|||`` 优先切分，否则按 ``split_sentences`` 自动切分
        3. 对每段中文翻译为日文（若开启）；失败的段会被丢弃
        4. 对译文再按标点切分（避免单段过长）
        5. 逐段合成投递

        Args:
            raw_text: 原始输入文本。
            stream_id: 当前聊天流 ID。
            log_prefix: 日志前缀。
            voice: 音色 ident。

        Returns:
            :class:`TTSResult`。
        """

        send_errors: bool = self.config.general.send_error_messages
        max_length: int = self.config.general.max_text_length

        # 1. 文本清理
        clean_text = TTSTextUtils.clean_text(raw_text)
        if not clean_text:
            if send_errors:
                await self.ctx.send.text(
                    "请输入要转换为语音的文本内容", stream_id,
                )
            return TTSResult(
                success=False,
                message="文本为空",
                backend_name=_BACKEND_NAME,
            )

        # 2. 长度校验：超长直接发错误提示，不静默截断
        if len(clean_text) > max_length:
            if send_errors:
                await self.ctx.send.text(
                    f"文本过长（{len(clean_text)}字符 > {max_length}），已取消合成",
                    stream_id,
                )
            return TTSResult(
                success=False,
                message="文本过长",
                backend_name=_BACKEND_NAME,
            )

        # 3. 智能分割：|||SPLIT||| 标记优先切分
        if _SPLIT_MARKER in clean_text:
            segments = [
                s.strip()
                for s in clean_text.split(_SPLIT_MARKER)
                if s.strip()
            ]
        elif self.config.general.split_sentences:
            segments = TTSTextUtils.split_sentences(clean_text)
        else:
            segments = [clean_text]

        if not segments:
            segments = [clean_text]

        # 4. 单段 vs 多段翻译策略
        # 单段时翻译一次；多段时逐段翻译（失败段丢弃，整体仍可继续）
        translated_segments: List[str] = []
        first_failure_message: str = ""
        if len(segments) == 1:
            ok, payload = await self._translate_text(segments[0], log_prefix)
            if not ok:
                if send_errors:
                    await self.ctx.send.text(
                        f"语音合成失败: 日文翻译失败: {payload}",
                        stream_id,
                    )
                return TTSResult(
                    success=False,
                    message=f"日文翻译失败: {payload}",
                    backend_name=_BACKEND_NAME,
                )
            translated_segments = [payload]
        else:
            for seg in segments:
                ok, payload = await self._translate_text(seg, log_prefix)
                if ok:
                    translated_segments.append(payload)
                else:
                    if not first_failure_message:
                        first_failure_message = payload
                    # 翻译失败的段丢弃，不送入 SBV2
            if not translated_segments:
                if send_errors:
                    await self.ctx.send.text(
                        f"语音合成失败: 日文翻译失败: {first_failure_message or '未知错误'}",
                        stream_id,
                    )
                return TTSResult(
                    success=False,
                    message=f"日文翻译失败: {first_failure_message or '未知错误'}",
                    backend_name=_BACKEND_NAME,
                )

        # 5. 对每段译文再切分（避免单段过长），并过滤空段
        final_sentences: List[str] = []
        for translated in translated_segments:
            if self.config.general.split_sentences:
                sub = TTSTextUtils.split_sentences(translated)
            else:
                sub = [translated]
            for s in sub:
                s = s.strip()
                if s:
                    final_sentences.append(s)

        if not final_sentences:
            return TTSResult(
                success=False,
                message="无可合成的文本",
                backend_name=_BACKEND_NAME,
            )

        # 6. 逐段合成投递
        return await self._send_in_segments(
            sentences=final_sentences,
            stream_id=stream_id,
            log_prefix=log_prefix,
            voice=voice,
            split_delay=self.config.general.split_delay,
        )

    # ─── Tool: 由 LLM 自主触发 ───────────────────────────────────────────

    @Tool(
        "sbv2_tts_tool",
        description=(
            "用日文语音回复（SBV2 日文 TTS）：先调用 LLM 把中文译为日文，"
            "再调用本地 SBV2 服务合成日文语音发送。"
        ),
        parameters=[
            ToolParameterInfo(
                name="text",
                param_type=ToolParamType.STRING,
                description="要转为日文语音的中文文本（必填）",
                required=True,
            ),
            ToolParameterInfo(
                name="ident",
                param_type=ToolParamType.STRING,
                description="SBV2 模型标识，'Ling v2' 或 'Fusetsu_v1.5'，默认 'Ling v2'",
                required=False,
                default="Ling v2",
            ),
        ],
    )
    async def handle_tts_tool(
        self,
        text: str = "",
        ident: str = "",
        stream_id: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """处理 LLM 自主调用的 TTS Tool。

        ``stream_id`` 由 Host 注入；不暴露在 ``parameters`` 中。

        Args:
            text: 要转为日文语音的中文文本。
            ident: SBV2 模型标识；空串则用默认音色。
            stream_id: 当前聊天流 ID（Host 注入）。
            **kwargs: 兼容其它 Host 注入字段。

        Returns:
            Dict[str, Any]: ``{"success": bool, "message": str}``。
        """

        del kwargs
        log_prefix = f"[sbv2_tts_plugin][tool][{stream_id}]"

        if not stream_id:
            return {"success": False, "message": "缺少 stream_id，无法发送"}

        # 解析音色（ident 必须在 sbv2.idents 列表里，否则回退到默认）
        voice: str = self._resolve_voice(ident)

        try:
            result = await self._run_pipeline(
                raw_text=text,
                stream_id=stream_id,
                log_prefix=log_prefix,
                voice=voice,
            )
            return {"success": result.success, "message": result.message}
        except asyncio.TimeoutError:
            timeout_sec = self.config.general.timeout
            self.ctx.logger.error(
                "%s TTS Tool 超时（%ss）", log_prefix, timeout_sec,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成超时（{timeout_sec}s）", stream_id,
                )
            return {"success": False, "message": "timeout"}
        except (aiohttp.ClientError, ConnectionError) as exc:
            self.ctx.logger.error(
                "%s TTS Tool 网络错误: %s", log_prefix, exc,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成网络错误: {type(exc).__name__}", stream_id,
                )
            return {"success": False, "message": str(exc)}
        except (KeyError, AttributeError, ValueError) as exc:
            self.ctx.logger.error(
                "%s TTS Tool 参数错误: %s", log_prefix, exc, exc_info=True,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成参数错误: {exc}", stream_id,
                )
            return {"success": False, "message": str(exc)}

    # ─── Command: 由用户手动触发 ─────────────────────────────────────────

    @Command(
        "sbv2_tts_command",
        description="将文本转换为日文语音（SBV2）",
        pattern=r"^/(?:sbv2|voice)\s+(?P<text>.+?)(?:\s+-v\s+(?P<ident>\S+))?$",
    )
    async def handle_tts_command(
        self,
        stream_id: str = "",
        matched_groups: Any = None,
        **kwargs: Any,
    ) -> Tuple[bool, str, bool]:
        """处理 ``/sbv2`` / ``/voice`` 命令。

        命中 ``text.lower() == "help"`` 时发帮助文本并返回成功；
        空文本时发错误提示。返回三元组 ``(success, message, intercept)``，
        ``intercept=True`` 表示拦截该消息不再向下传递。

        Args:
            stream_id: 当前聊天流 ID（Host 注入）。
            matched_groups: 正则命名捕获组，由 Host 注入。
            **kwargs: 兼容其它 Host 注入字段。

        Returns:
            Tuple[bool, str, bool]: ``(success, message, intercept)``。
        """

        del kwargs
        log_prefix = f"[sbv2_tts_plugin][cmd][{stream_id}]"
        groups: Dict[str, Any] = (
            matched_groups if isinstance(matched_groups, dict) else {}
        )
        user_text = (groups.get("text") or "").strip()
        user_ident = (groups.get("ident") or "").strip()

        try:
            if not stream_id:
                return False, "缺少 stream_id", True

            if user_text.lower() == "help":
                await self._send_help(stream_id)
                return True, "显示帮助信息", True

            if not user_text:
                if self.config.general.send_error_messages:
                    await self.ctx.send.text(
                        "请输入要转换为语音的文本内容", stream_id,
                    )
                return False, "缺少文本内容", True

            voice: str = self._resolve_voice(user_ident)

            result = await self._run_pipeline(
                raw_text=user_text,
                stream_id=stream_id,
                log_prefix=log_prefix,
                voice=voice,
            )

            return result.success, result.message, True
        except asyncio.TimeoutError:
            timeout_sec = self.config.general.timeout
            self.ctx.logger.error(
                "%s TTS 命令超时（%ss）", log_prefix, timeout_sec,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成超时（{timeout_sec}s）", stream_id,
                )
            return False, "timeout", True
        except (aiohttp.ClientError, ConnectionError) as exc:
            self.ctx.logger.error(
                "%s TTS 命令网络错误: %s", log_prefix, exc,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成网络错误: {type(exc).__name__}", stream_id,
                )
            return False, str(exc), True
        except (KeyError, AttributeError, ValueError) as exc:
            self.ctx.logger.error(
                "%s TTS 命令参数错误: %s", log_prefix, exc, exc_info=True,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成参数错误: {exc}", stream_id,
                )
            return False, str(exc), True

    # ─── 内部：音色解析 / 帮助 ───────────────────────────────────────────

    def _resolve_voice(self, ident: str) -> str:
        """解析音色 ident。

        命中 ``sbv2.idents`` 列表则原样返回；否则回退到 ``sbv2.default_ident``。

        Args:
            ident: 用户传入的音色标识。

        Returns:
            str: 最终使用的音色标识。
        """

        default_ident: str = self.config.sbv2.default_ident
        idents: List[str] = list(self.config.sbv2.idents or [])
        if ident and idents and ident in idents:
            return ident
        return default_ident

    async def _send_help(self, stream_id: str) -> None:
        """发送 ``/sbv2 help`` 帮助文本。"""

        default_ident: str = self.config.sbv2.default_ident
        help_text = (
            "【SBV2 日文语音合成插件帮助】\n\n"
            "📝 基本语法：\n"
            "/sbv2 <文本> [-v <音色>]\n"
            "/voice <文本> [-v <音色>]   # /sbv2 的别名\n\n"
            "🎵 可选音色（取自 [sbv2].idents）：\n"
            "  - Ling v2（默认）\n"
            "  - Fusetsu_v1.5\n"
            f"当前配置默认音色：{default_ident}\n\n"
            "🌐 翻译机制：\n"
            "插件默认先把中文翻译为日文，再送入本地 SBV2 推理合成。"
            "若 translate_to_japanese = false，则跳过翻译。\n\n"
            "✂️ 智能分割：\n"
            "文本中包含 |||SPLIT||| 时按标记精确分段；否则按句末标点自动切分。\n\n"
            "📌 示例：\n"
            "/sbv2 你好世界\n"
            "/sbv2 今天的天气真不错 -v Fusetsu_v1.5\n"
            "/voice 今天天气不错|||SPLIT|||适合出去玩\n"
        )
        await self.ctx.send.text(help_text, stream_id)


# ─── 后端注册 ────────────────────────────────────────────────────────────

# 在模块导入阶段就把 SBV2 后端注册到注册表，保证 ``create_plugin()`` 时可用。
TTSBackendRegistry.register(_BACKEND_NAME, Sbv2Backend)


def create_plugin() -> Sbv2TTSPlugin:
    """创建 SBV2 日文语音合成插件实例（SDK 入口）。

    Returns:
        Sbv2TTSPlugin: 新的插件实例。
    """

    return Sbv2TTSPlugin()