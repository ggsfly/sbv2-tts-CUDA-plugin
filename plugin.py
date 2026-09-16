"""Style-Bert-VITS2 (CUDA) 日文语音合成插件。

本插件调用本地 Style-Bert-VITS2 (CUDA) 推理服务合成日文语音。Style-Bert-VITS2
是日文推理模型，因此插件先调用 LLM 把用户输入翻译为自然日文，再送入本地
推理服务合成。

触发面：
- ``@Tool``：由 LLM 自主决定调用，对应 ``sbv2_tts_tool`` 组件；
- ``@Command``：由用户通过 ``/sbv2`` / ``/voice`` 命令触发，对应
  ``sbv2_tts_command`` 组件。

管线（Tool 与 Command 共用）：
1. 文本清理（去除首尾空白）
2. 智能分割：``|||SPLIT|||`` 标记优先切分 > ``TTSTextUtils.split_sentences`` > 单段
3. 若 ``general.translate_to_japanese`` 开启，调用 :class:`JPTranslator` 把每段
   中文译为日文（翻译失败时 **绝不** 用中文喂推理服务）
4. 对每段译文再走 ``TTSTextUtils.split_sentences`` 自动切分，并用
   ``clamp_sentences`` 保证每段不超过服务端 ``limit``（默认 100）
5. 调用 :class:`VoiceBackend` 逐段合成，并通过 ``ctx.send.custom("voice", base64)``
   把语音投递到聊天流

LLM 调取规范
------------
遵循 MaiBot SDK 2.x 规范，不导入 ``src.*``，统一使用 ``ctx.llm.generate``：
- 留空 → ``task_name="replyer"``
- 命中可用任务名 → ``task_name=<值>``
- 否则视为模型名/标识符 → ``task_name="replyer", model_name=<值>`` 直传 Host
  解析；模型无效时 Host 报错，翻译失败如实暴露（不静默回退）。
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import asyncio
import logging
import re
import sys

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import HookMode, ToolParameterInfo, ToolParamType

from .backends import TTSBackendRegistry, VoiceBackend, VoiceProfile
from .backends.base import TTSResult
from .translate import JPTranslator
from .utils.session import TTSSessionManager
from .utils.text import TTSTextUtils

sys.dont_write_bytecode = True

logger = logging.getLogger("plugin.sbv2_tts")

# replyer 会把聊天历史里语音消息的占位渲染（[语音消息]）模仿进回复正文开头，
# 导致首条分段变成无意义的"[语音消息]"引用消息。这里用正则剥离该占位回声：
# - 响应开头的连续占位（可带冒号/空格）：[语音消息]你音量... → 你音量...
# - 独立成行的占位行（连同行尾换行一起删除，避免残留空行）
_VOICE_PLACEHOLDER_LEADING_PATTERN = re.compile(r"^(?:\s*\[语音消息\]\s*)+")
_VOICE_PLACEHOLDER_LINE_PATTERN = re.compile(r"^[ \t]*\[语音消息\][ \t]*(?:\n|$)", re.MULTILINE)

# 智能分割标记，与 xuqian13_tts-voice-plugin 保持一致
_SPLIT_MARKER = "|||SPLIT|||"

# 后端注册表注册名（与 ``VoiceBackend.backend_name`` 保持一致）
_BACKEND_NAME = "voice"

# 连接性探测目标：Style-Bert-VITS2 的 /models/info（GET，返回已加载模型信息）
_PROBE_URL = "http://127.0.0.1:5000/models/info"
_PROBE_TIMEOUT_SECONDS = 5

# 内置默认音色档案映射（对应 ModelScope 官方整合包中的预置模型）
DEFAULT_VOICE_PROFILES: Dict[str, VoiceProfile] = {
    "Ling v2": VoiceProfile(name="Ling v2", model="Ling-v2", speaker="Ling v2", style="Neutral"),
    "Fusetsu_v1.5": VoiceProfile(name="Fusetsu_v1.5", model="Fusetsu-v1.5", speaker="Fusetsu_v1.5", style="Neutral"),
}


# ─── 配置模型 ────────────────────────────────────────────────────────────


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本")


class GeneralConfig(PluginConfigBase):
    """通用配置。"""

    __ui_label__ = "通用"
    __ui_icon__ = "settings"
    __ui_order__ = 1

    timeout: int = Field(default=60, description="请求超时（秒）")
    max_text_length: int = Field(
        default=100,
        description=(
            "单段合成文本的最大字符数，对齐 Style-Bert-VITS2 服务端 limit（默认 100）。"
            "超长段落会被自动二次切分，避免触发服务端 422。"
        ),
    )
    strip_voice_placeholder: bool = Field(
        default=True,
        description=(
            "是否剥离 replyer 正文中的 [语音消息] 占位回声。"
            "本插件发送语音后，聊天历史会把语音渲染为 [语音消息]，replyer 偶尔会模仿该占位并写进回复正文，"
            "经智能分段后产生一条无意义的引用消息。开启后通过 maisaka.reply.before_post_process 钩子剥离。"
        ),
    )
    split_sentences: bool = Field(default=True, description="是否按句子拆分合成")
    split_delay: float = Field(default=0.3, description="分句之间的发送间隔（秒）")
    send_error_messages: bool = Field(default=True, description="是否向聊天流回显错误提示")
    translate_to_japanese: bool = Field(
        default=True,
        description="是否先把中文翻译为日文再合成（Style-Bert-VITS2 为日文推理模型）",
    )
    translate_model: str = Field(
        default="",
        description=(
            "翻译用 LLM，支持两种填法："
            "① 任务名（replyer / utils / planner / memory 等）；"
            "② 模型名 / 模型标识符（model_config.toml 中 [[models]] 的 name 或 model_identifier）。"
            "留空 = replyer 任务。命中任务名按任务调用；否则视为模型名直传 Host 解析，"
            "模型无效时翻译失败如实暴露。"
        ),
        json_schema_extra={
            "hint": (
                "可填任务名（replyer/utils/planner 等）或模型名/标识符（如 "
                "gemini-3.7-flash-low）。留空使用 replyer 任务。翻译是小任务，"
                "推荐 utils 或快速小模型。"
            ),
        },
    )


class ComponentsConfig(PluginConfigBase):
    """组件开关。"""

    __ui_label__ = "组件"
    __ui_icon__ = "blocks"
    __ui_order__ = 2

    tool_enabled: bool = Field(default=True, description="是否启用 Tool 组件")
    command_enabled: bool = Field(default=True, description="是否启用 Command 组件")


class VoiceConfig(PluginConfigBase):
    """Style-Bert-VITS2 推理服务配置。"""

    __ui_label__ = "音色"
    __ui_icon__ = "mic"
    __ui_order__ = 3

    api_url: str = Field(
        default="http://127.0.0.1:5000/voice",
        description="Style-Bert-VITS2 API 地址，需指向 /voice 端点",
    )
    default_voice: str = Field(default="Ling v2", description="默认音色名（取自 voices 列表）")
    language: str = Field(default="JP", description="文本语言：JP / EN / ZH")
    length: float = Field(default=1.0, description="语速，基准 1.0，越大越慢")
    voices: List[str] = Field(
        default_factory=lambda: ["Ling v2", "Fusetsu_v1.5"],
        description="可选音色列表；-v 参数与 default_voice 从该列表按名称匹配",
    )


class SBV2TTSPluginConfig(PluginConfigBase):
    """Style-Bert-VITS2 日文语音合成插件总配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)


# ─── 插件主体 ────────────────────────────────────────────────────────────


class SBV2TTSPlugin(MaiBotPlugin):
    """Style-Bert-VITS2 (CUDA) 日文语音合成插件（MaiBot SDK 2.x 版）。

    由 Tool / Command 共同驱动 Style-Bert-VITS2 服务的 ``/voice`` 接口，
    并通过 :meth:`_send_in_segments` 把多段语音依次投递到当前聊天流。
    """

    config_model = SBV2TTSPluginConfig

    # 可用 LLM 任务名缓存（进程内一次；None=未拉取）。由 _resolve_translate_llm 维护。
    _available_task_names: Optional[frozenset] = None

    # ─── 生命周期 ────────────────────────────────────────────────────────

    async def on_load(self) -> None:
        """插件加载钩子。

        1. 短超时探测 Style-Bert-VITS2 服务存活（GET /models/info）。
        2. 按 ``[components]`` 配置禁用 Tool / Command 组件。
        3. 不阻断插件加载：未连通只发 warning。
        """

        await self._probe_service()

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
            "Style-Bert-VITS2 日文 TTS 插件已加载，已注册后端=%s",
            TTSBackendRegistry.list_backends(),
        )

    async def on_unload(self) -> None:
        """插件卸载钩子：关闭 aiohttp 单例 session。"""

        session_manager = await TTSSessionManager.get_instance()
        await session_manager.close_session()
        self.ctx.logger.info("Style-Bert-VITS2 日文 TTS 插件已卸载")

    async def on_config_update(
        self,
        scope: str,
        config_data: Dict[str, Any],
        version: str,
    ) -> None:
        """配置热更新回调。"""

        del scope
        del config_data
        self.ctx.logger.info("SBV2 插件配置已更新: version=%s", version)

    # ─── Hook：剥离回复正文中的语音占位回声 ──────────────────────────────

    @HookHandler(
        "maisaka.reply.before_post_process",
        name="sbv2_strip_voice_placeholder",
        mode=HookMode.BLOCKING,
    )
    async def strip_voice_placeholder(
        self,
        response: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """剥离 replyer 正文中的 [语音消息] 占位回声。

        本插件发送语音后，聊天历史会把该语音渲染为占位文本，replyer 在
        同一轮生成文字回复时可能把占位模仿进正文开头，经智能分段后会
        产生一条只含"[语音消息]"的引用消息。此钩子在文本后处理前把它剥掉。

        Args:
            response: 即将执行后处理的回复正文。
            **kwargs: Host 注入的其余钩子参数（session_id 等，本处理器不使用）。

        Returns:
            Dict[str, Any]: 钩子返回值；正文有变化时通过 modified_kwargs 改写。
        """

        del kwargs
        if not self.config.general.strip_voice_placeholder:
            return {"action": "continue"}
        if not response or "[语音消息]" not in response:
            return {"action": "continue"}

        # 先剥离正文中间独立成行的占位，再剥离开头连续的占位前缀
        cleaned = _VOICE_PLACEHOLDER_LINE_PATTERN.sub("", response)
        cleaned = _VOICE_PLACEHOLDER_LEADING_PATTERN.sub("", cleaned).strip()

        if cleaned == response:
            return {"action": "continue"}
        if not cleaned:
            # 整条回复只有占位回声：如实暴露为空正文（语音已由工具发出），
            # 由 Host 走"生成可见回复失败"分支，不静默保留占位。
            self.ctx.logger.info("回复正文仅含 [语音消息] 占位回声，已剥离为空")
            return {"action": "continue", "modified_kwargs": {"response": ""}}

        self.ctx.logger.info("已剥离回复正文中的 [语音消息] 占位回声")
        return {"action": "continue", "modified_kwargs": {"response": cleaned}}

    # ─── 内部：连接性探测 ────────────────────────────────────────────────

    async def _probe_service(self) -> None:
        """短超时探测 Style-Bert-VITS2 服务存活。

        任意 HTTP 状态码都视为"服务监听中"，仅连接被拒 / 超时 / DNS 失败视为
        未连通。未连通仅 warning，不抛错。
        """

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_PROBE_TIMEOUT_SECONDS)
            ) as session:
                try:
                    async with session.get(_PROBE_URL) as resp:
                        del resp
                except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
                    self.ctx.logger.warning(
                        "Style-Bert-VITS2 服务未连通，路径=%s，原因=%s",
                        _PROBE_URL, exc,
                    )
        except (aiohttp.ClientError, OSError) as exc:
            self.ctx.logger.warning(
                "Style-Bert-VITS2 服务未连通，路径=%s，原因=%s",
                _PROBE_URL, exc,
            )

    # ─── 内部：配置桥接 ──────────────────────────────────────────────────

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """根据扁平 key（``section.field``）从 Pydantic 配置取值。

        backends 模块基于 ``config_getter(key, default)`` 读取配置，
        这里把 Pydantic 嵌套配置展开成扁平访问。

        Args:
            key: 形如 ``"voice.api_url"`` 的扁平配置键。
            default: 当配置不存在时返回的默认值。

        Returns:
            Any: 对应配置项的当前值；找不到则返回 ``default``。
        """

        if "." not in key:
            return default

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

        backends 通过 ``set_send_custom`` 注入该闭包，把 ``voice`` base64
        投到当前聊天流。

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
        """创建 Voice 后端实例并注入 ``send_custom`` 闭包。

        Args:
            stream_id: 当前聊天流 ID。
            log_prefix: 日志前缀。

        Returns:
            :class:`VoiceBackend` 实例，或未注册时返回 ``None``。
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
        voice: Optional[VoiceProfile] = None,
    ) -> TTSResult:
        """调用 Voice 后端合成一段文本。

        Args:
            text: 待合成日文文本。
            stream_id: 当前聊天流 ID。
            log_prefix: 日志前缀。
            voice: 音色档案；None 时后端用默认档案。

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

        task_name, model_name = await self._resolve_translate_llm()
        translator = JPTranslator(
            max_length=self.config.general.max_text_length,
        )
        return await translator.translate(
            text,
            log_prefix,
            self.ctx.llm.generate,
            task_name=task_name,
            model_name=model_name,
        )

    async def _resolve_translate_llm(self) -> Tuple[str, str]:
        """把 ``translate_model`` 配置解析为 ``(task_name, model_name)``。

        规则：
        - 留空 → ``("replyer", "")``，默认使用 replyer 任务。
        - 命中已注册任务名 → ``(值, "")``，按任务路由。
        - 否则视为模型名/标识符 → ``("replyer", 值)``，直传 Host 解析；
          模型无效时 Host 报错，翻译失败如实暴露，**不静默回退**。

        Returns:
            Tuple[str, str]: ``(task_name, model_name)``。
        """

        raw = (self.config.general.translate_model or "").strip()
        if not raw:
            return "replyer", ""

        if self._available_task_names is None:
            try:
                names = await self.ctx.llm.get_available_models()
                self._available_task_names = frozenset(
                    str(n).strip() for n in names if str(n).strip()
                )
                self.ctx.logger.debug(
                    "已缓存可用 LLM 任务名: %s", sorted(self._available_task_names),
                )
            except Exception as exc:
                # 拉取失败不写缓存（保持 None），下次调用重试；本次视为模型名直传。
                self.ctx.logger.warning(
                    "获取可用 LLM 任务名列表失败 (%s)，translate_model 将按模型名直传 Host",
                    exc,
                )
                self._available_task_names = frozenset()

        if raw in self._available_task_names:
            return raw, ""

        # 视为模型名/标识符，直传 Host 解析；无效即报错暴露
        self.ctx.logger.info(
            "translate_model `%s` 未命中任务名，将作为模型名直传 Host", raw,
        )
        return "replyer", raw

    # ─── 内部：分段发送 ──────────────────────────────────────────────────

    async def _send_in_segments(
        self,
        sentences: List[str],
        stream_id: str,
        log_prefix: str,
        voice: Optional[VoiceProfile],
        split_delay: float,
    ) -> TTSResult:
        """逐段合成并投递。

        单句：直接调一次后端，失败时按 ``send_error_messages`` 配置发错误提示。
        多句：任一段失败→记录、停止后续段、汇总 ``success_count`` / ``total``。

        Args:
            sentences: 待合成段落列表。
            stream_id: 当前聊天流 ID。
            log_prefix: 日志前缀。
            voice: 音色档案。
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
                logger.error(
                    "%s 分段 %d/%d 合成失败: %s",
                    log_prefix, index + 1, total, result.message,
                )
                last_error = result.message
                break
            if index < total - 1 and split_delay > 0:
                await asyncio.sleep(split_delay)

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
        voice: Optional[VoiceProfile],
    ) -> TTSResult:
        """翻译-合成管线的共享实现。

        步骤：
        1. 文本清理
        2. ``|||SPLIT|||`` 优先切分，否则按 ``split_sentences`` 自动切分
        3. 对每段中文翻译为日文（若开启）；失败的段会被丢弃
        4. 对译文再按标点切分 + ``clamp_sentences`` 保证每段不超过服务端 limit
        5. 逐段合成投递

        Args:
            raw_text: 原始输入文本。
            stream_id: 当前聊天流 ID。
            log_prefix: 日志前缀。
            voice: 音色档案。

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

        # 2. 智能分割：|||SPLIT||| 标记优先切分
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

        # 3. 单段 vs 多段翻译策略
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
                    # 翻译失败的段丢弃，不送入推理服务
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

        # 4. 对每段译文再切分（避免单段过长），并 clamp 到服务端 limit
        final_sentences: List[str] = []
        for translated in translated_segments:
            if self.config.general.split_sentences:
                sub = TTSTextUtils.split_sentences(translated)
            else:
                sub = [translated]
            # 兜底：保证每段不超过 max_length（服务端 limit），否则会 422
            sub = TTSTextUtils.clamp_sentences(sub, max_length)
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

        # 5. 逐段合成投递
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
            "用日文语音回复（Style-Bert-VITS2 日文 TTS）：先调用 LLM 把中文译为日文，"
            "再调用本地 Style-Bert-VITS2 服务合成日文语音发送。"
        ),
        parameters=[
            ToolParameterInfo(
                name="text",
                param_type=ToolParamType.STRING,
                description="要转为日文语音的中文文本（必填）",
                required=True,
            ),
            ToolParameterInfo(
                name="voice",
                param_type=ToolParamType.STRING,
                description="音色名，如 'Ling v2' 或 'Fusetsu_v1.5'，默认 'Ling v2'",
                required=False,
                default="Ling v2",
            ),
        ],
    )
    async def handle_tts_tool(
        self,
        text: str = "",
        voice: str = "",
        stream_id: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """处理 LLM 自主调用的 TTS Tool。

        Args:
            text: 要转为日文语音的中文文本。
            voice: 音色名；空串则用配置中的默认音色。
            stream_id: 当前聊天流 ID（Host 注入）。
            **kwargs: 兼容其它 Host 注入字段。

        Returns:
            Dict[str, Any]: ``{"success": bool, "message": str}``。
        """

        del kwargs
        log_prefix = f"[sbv2_tts_plugin][tool][{stream_id}]"

        if not stream_id:
            return {"success": False, "message": "缺少 stream_id，无法发送"}

        try:
            profile = self._resolve_voice(voice)
        except ValueError as exc:
            if self.config.general.send_error_messages:
                await self.ctx.send.text(str(exc), stream_id)
            return {"success": False, "message": str(exc)}

        try:
            result = await self._run_pipeline(
                raw_text=text,
                stream_id=stream_id,
                log_prefix=log_prefix,
                voice=profile,
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
        description="将文本转换为日文语音（Style-Bert-VITS2）",
        pattern=r"^/(?:sbv2|voice)\s+(?P<text>.+?)(?:\s+-v\s+(?P<voice>.+))?$",
    )
    async def handle_tts_command(
        self,
        stream_id: str = "",
        matched_groups: Any = None,
        **kwargs: Any,
    ) -> Tuple[bool, str, int]:
        """处理 ``/sbv2`` / ``/voice`` 命令。

        命中 ``text.lower() == "help"`` 时发帮助文本并返回成功；
        空文本时发错误提示。返回三元组 ``(success, message, intercept)``，
        ``intercept`` 为 1 表示拦截该消息不再向下传递。

        Args:
            stream_id: 当前聊天流 ID（Host 注入）。
            matched_groups: 正则命名捕获组，由 Host 注入。
            **kwargs: 兼容其它 Host 注入字段。

        Returns:
            Tuple[bool, str, int]: ``(success, message, intercept)``。
        """

        del kwargs
        log_prefix = f"[sbv2_tts_plugin][cmd][{stream_id}]"
        groups: Dict[str, Any] = (
            matched_groups if isinstance(matched_groups, dict) else {}
        )
        user_text = (groups.get("text") or "").strip()
        user_voice = (groups.get("voice") or "").strip()

        try:
            if not stream_id:
                return False, "缺少 stream_id", 1

            if user_text.lower() == "help":
                await self._send_help(stream_id)
                return True, "显示帮助信息", 1

            if not user_text:
                if self.config.general.send_error_messages:
                    await self.ctx.send.text(
                        "请输入要转换为语音的文本内容", stream_id,
                    )
                return False, "缺少文本内容", 1

            profile = self._resolve_voice(user_voice)

            result = await self._run_pipeline(
                raw_text=user_text,
                stream_id=stream_id,
                log_prefix=log_prefix,
                voice=profile,
            )

            return result.success, result.message, 1
        except asyncio.TimeoutError:
            timeout_sec = self.config.general.timeout
            self.ctx.logger.error(
                "%s TTS 命令超时（%ss）", log_prefix, timeout_sec,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成超时（{timeout_sec}s）", stream_id,
                )
            return False, "timeout", 1
        except (aiohttp.ClientError, ConnectionError) as exc:
            self.ctx.logger.error(
                "%s TTS 命令网络错误: %s", log_prefix, exc,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成网络错误: {type(exc).__name__}", stream_id,
                )
            return False, str(exc), 1
        except ValueError as exc:
            # _resolve_voice 抛出的未知音色错误
            self.ctx.logger.error(
                "%s TTS 命令参数错误: %s", log_prefix, exc,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成参数错误: {exc}", stream_id,
                )
            return False, str(exc), 1
        except (KeyError, AttributeError) as exc:
            self.ctx.logger.error(
                "%s TTS 命令参数错误: %s", log_prefix, exc, exc_info=True,
            )
            if self.config.general.send_error_messages:
                await self.ctx.send.text(
                    f"语音合成参数错误: {exc}", stream_id,
                )
            return False, str(exc), 1

    # ─── 内部：音色解析 / 帮助 ───────────────────────────────────────────

    def _resolve_voice(self, voice_name: str) -> VoiceProfile:
        """解析音色名到 :class:`VoiceProfile`。

        校验目标音色是否在 ``voice.voices`` 配置列表中，
        命中则优先使用内置的 ``DEFAULT_VOICE_PROFILES`` 映射，
        如未预置则按同名推导参数构造档案；未命中 **抛 ValueError** 如实暴露，
        并在消息中列出可选音色（符合 AGENTS"不无脑兜底"原则）。

        Args:
            voice_name: 用户/LLM 传入的音色名。

        Returns:
            :class:`VoiceProfile`。

        Raises:
            ValueError: 音色名未命中配置列表。
        """

        voices: List[str] = list(self.config.voice.voices or [])
        # 空名 → 默认音色
        target = (voice_name or "").strip() or self.config.voice.default_voice

        if target not in voices:
            available = "、".join(voices) or "（未配置任何音色）"
            raise ValueError(
                f"未知音色: {target or '<空>'}，可选: {available}"
            )

        if target in DEFAULT_VOICE_PROFILES:
            return DEFAULT_VOICE_PROFILES[target]

        return VoiceProfile(
            name=target,
            model=target,
            speaker=target,
            style="Neutral",
        )

    async def _send_help(self, stream_id: str) -> None:
        """发送 ``/sbv2 help`` 帮助文本。"""

        voices: List[str] = list(self.config.voice.voices or [])
        voice_lines = "\n".join(f"  - {name}" for name in voices) or "  （未配置任何音色）"

        help_text = (
            "【Style-Bert-VITS2 日文语音合成插件帮助】\n\n"
            "📝 基本语法：\n"
            "/sbv2 <文本> [-v <音色名>]\n"
            "/voice <文本> [-v <音色名>]   # /sbv2 的别名\n\n"
            "🎵 可选音色（取自 [voice].voices）：\n"
            f"{voice_lines}\n"
            f"当前默认音色：{self.config.voice.default_voice}\n\n"
            "🌐 翻译机制：\n"
            "插件默认先把中文翻译为日文，再送入本地 Style-Bert-VITS2 推理合成。"
            "若 translate_to_japanese = false，则跳过翻译。\n\n"
            "✂️ 智能分割：\n"
            "文本中包含 |||SPLIT||| 时按标记精确分段；否则按句末标点自动切分，"
            "并对超长段落按服务端 limit（默认 100 字符）二次切分。\n\n"
            "📌 示例：\n"
            "/sbv2 你好世界\n"
            "/sbv2 今天的天气真不错 -v Fusetsu_v1.5\n"
            "/voice 今天天气不错|||SPLIT|||适合出去玩\n"
        )
        await self.ctx.send.text(help_text, stream_id)


# ─── 后端注册 ────────────────────────────────────────────────────────────

# 在模块导入阶段就把 Voice 后端注册到注册表，保证 ``create_plugin()`` 时可用。
TTSBackendRegistry.register(_BACKEND_NAME, VoiceBackend)


def create_plugin() -> SBV2TTSPlugin:
    """创建 Style-Bert-VITS2 日文语音合成插件实例（SDK 入口）。

    Returns:
        SBV2TTSPlugin: 新的插件实例。
    """

    return SBV2TTSPlugin()
