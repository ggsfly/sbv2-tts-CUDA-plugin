"""Style-Bert-VITS2 (CUDA) 日文语音合成插件。

本插件调用本地 Style-Bert-VITS2 (CUDA) 推理服务合成日文语音。Style-Bert-VITS2
是日文推理模型，因此插件先把文本翻译为自然日文，再送入本地推理服务合成。

触发面：
- ``@ReplyExtension``：planner 在 ``reply`` 的 ``plugin_options`` 里选中
  ``sbv2_ja_voice`` 时触发，对应 ``sbv2_ja_voice`` 组件。扩展在 ``before_send``
  把本条回复正文译为日文、合成一条语音，**保留全部已有消息**并在末尾追加一条
  ``voice`` 段，由宿主统一发送（正文即字幕）；
- ``@Command``：由用户通过 ``/sbv2`` / ``/voice`` 命令触发，对应
  ``sbv2_tts_command`` 组件。

命令管线（一次性整段合成单条语音，插件自己投递）：
1. 文本清理（去除首尾空白）
2. 若 ``translate.translate_to_japanese`` 开启，按 ``[translate].engine`` 选择
   翻译引擎把整段中文译为日文（:class:`JPTranslator` 走宿主 LLM，或
   :class:`IndexTranslateTranslator` 走本地 Index-Translate OpenAI 兼容端点；
   翻译失败时 **绝不** 用中文喂推理服务，直接如实暴露错误）
3. 调用 :class:`VoiceBackend` 合成唯一一条语音，并通过
   ``ctx.send.custom("voice", base64)`` 投递到聊天流

自动回复管线（ReplyExtension，仅在 planner 选中本扩展时运行）：
1. 把 ``before_send`` 收到的回复正文交给中译日引擎（不看
   ``translate.translate_to_japanese``，日文是硬约束）
2. :meth:`VoiceBackend.synthesize` 只合成不投递，把 WAV 编码为 base64 后作为
   一条追加消息返回，由宿主在正文之后发送
3. 任一步失败只向当前聊天发一条提示并返回 ``{}``，宿主原样发送正文（无语音）

LLM 调取规范
------------
遵循 MaiBot SDK 2.x 规范，不导入 ``src.*``，统一使用 ``ctx.llm.generate``：
- 留空 → ``task_name="replyer"``
- 命中可用任务名 → ``task_name=<值>``
- 否则视为模型名 → ``task_name="replyer", model_name=<值>`` 直传 Host
  解析；模型无效时 Host 报错，翻译失败如实暴露（不静默回退）。
"""

from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast

import asyncio
import logging
import re
import sys

from maibot_sdk import (
    Command,
    Field,
    HookHandler,
    MaiBotPlugin,
    PluginConfigBase,
    ReplyExtension,
)
from maibot_sdk.types import HookMode, ToolParameterInfo, ToolParamType

import aiohttp

from .backends import TTSBackendRegistry, VoiceBackend, VoiceProfile
from .backends.base import TTSResult
from .translate import IndexTranslateTranslator, JPTranslator
from .utils.file import TTSFileManager
from .utils.session import TTSSessionManager
from .utils.text import TTSTextUtils

sys.dont_write_bytecode = True

logger = logging.getLogger("plugin.sbv2_tts")

# replyer 会把聊天历史里语音消息的占位渲染（[语音消息]）模仿进回复正文开头，
# 导致首条回复消息变成无意义的"[语音消息]"引用。这里用正则剥离该占位回声：
# - 响应开头的连续占位（可带冒号/空格）：[语音消息]你音量... → 你音量...
# - 独立成行的占位行（连同行尾换行一起删除，避免残留空行）
_VOICE_PLACEHOLDER_LEADING_PATTERN = re.compile(r"^(?:\s*\[语音消息\]\s*)+")
_VOICE_PLACEHOLDER_LINE_PATTERN = re.compile(r"^[ \t]*\[语音消息\][ \t]*(?:\n|$)", re.MULTILINE)

# 后端注册表注册名（与 ``VoiceBackend.backend_name`` 保持一致）
_BACKEND_NAME = "voice"

# Index-Translate 翻译引擎在 TTSSessionManager 中的连接池隔离名
_INDEX_TRANSLATE_SESSION_NAME = "index_translate"

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
    config_version: str = Field(default="1.3.0", description="配置版本")


class GeneralConfig(PluginConfigBase):
    """合成超时、译文长度与回复处理。"""

    __ui_label__ = "通用"
    __ui_icon__ = "settings"
    __ui_order__ = 1

    timeout: int = Field(
        default=60,
        description="Style-Bert-VITS2 推理服务（/voice）请求超时（秒）；翻译请求超时见 [translate].timeout",
        json_schema_extra={
            "label": "合成超时（秒）",
            "hint": "单次 /voice 请求的等待上限。本地 CUDA 合成长句会变慢，超时就调大。",
        },
    )
    max_text_length: int = Field(
        default=100,
        description=(
            "译文长度上限（字符数），注入翻译 prompt 约束 LLM 生成的日文长度，"
            "对齐 Style-Bert-VITS2 服务端 limit（默认 100）。超长不再自动截断，"
            "由后端如实返回 422 错误。"
        ),
        json_schema_extra={
            "label": "译文长度上限（字符）",
            "hint": (
                "只写进翻译提示词，让模型自己控制长度。不截断、不做本地预检，"
                "也不会改动 SVB2 服务端的 limit——那边通常会返回 422。"
                "建议填成与服务端 limit 相同的值。"
            ),
        },
    )
    strip_voice_placeholder: bool = Field(
        default=True,
        description=(
            "是否剥离 replyer 正文中的 [语音消息] 占位回声。"
            "本插件发送语音后，聊天历史会把语音渲染为 [语音消息]，replyer 偶尔会模仿该占位并写进回复正文，"
            "形成一条无意义的引用消息。开启后通过 maisaka.reply.before_post_process 钩子剥离。"
        ),
        json_schema_extra={
            "label": "剥离 [语音消息] 占位回声",
            "hint": "开着就行。发过语音后 replyer 可能把 [语音消息] 学进正文，这里把它剥掉。",
        },
    )
    send_error_messages: bool = Field(
        default=True,
        description="是否向聊天流回显错误提示",
        json_schema_extra={
            "label": "命令失败时回显错误",
            "hint": "只影响 /sbv2 命令。自动回复路线失败时固定发一条「本次未附加日语语音」提示，不受此开关控制。",
        },
    )


class ComponentsConfig(PluginConfigBase):
    """自动回复扩展与命令开关。"""

    __ui_label__ = "组件"
    __ui_icon__ = "blocks"
    __ui_order__ = 4

    extension_enabled: bool = Field(
        default=True,
        description=(
            "是否允许 planner 在 reply 的 plugin_options 里选择 sbv2_ja_voice 扩展。"
            "关闭后扩展仍注册，但被禁用，模型看不到该选项；/sbv2 命令不受影响。"
        ),
        json_schema_extra={
            "label": "自动回复语音扩展",
            "hint": (
                "自动语音总开关。开启后 planner 可自主选择本扩展，把回复正文读成日文语音；"
                "关闭后只是模型看不到该选项，/sbv2 命令照常可用。"
            ),
        },
    )
    command_enabled: bool = Field(
        default=True,
        description="是否启用 Command 组件",
        json_schema_extra={
            "label": "斜杠命令",
            "hint": "是否启用 /sbv2 与 /voice 命令。",
        },
    )


class VoiceConfig(PluginConfigBase):
    """推理服务地址与音色。"""

    __ui_label__ = "音色"
    __ui_icon__ = "mic"
    __ui_order__ = 3

    api_url: str = Field(
        default="http://127.0.0.1:5000/voice",
        description="Style-Bert-VITS2 API 地址，需指向 /voice 端点",
        json_schema_extra={
            "label": "推理服务地址",
            "hint": "必须指向 /voice 端点。服务未启动时插件加载只发 warning，不会阻断。",
        },
    )
    default_voice: str = Field(
        default="Ling v2",
        description="默认音色名（取自 voices 列表）",
        json_schema_extra={
            "label": "默认音色",
            "hint": "必须出现在下面的可选音色列表里，否则不传 -v 时会报未知音色。",
        },
    )
    length: float = Field(
        default=1.0,
        description="语速，基准 1.0，越大越慢",
        json_schema_extra={
            "label": "语速",
            "hint": "1.0 为基准，数值越大念得越慢。",
        },
    )
    voices: List[str] = Field(
        default_factory=lambda: ["Ling v2", "Fusetsu_v1.5"],
        description="可选音色列表；-v 参数与 default_voice 从该列表按名称匹配",
        json_schema_extra={
            "label": "可选音色列表",
            "hint": "/sbv2 的 -v 参数与自动回复的音色参数都只能从这里选，未命中会如实报错并列出可选值。",
        },
    )


class TranslateConfig(PluginConfigBase):
    """中→日翻译，仅作用于 /sbv2、/voice 命令；自动回复路线始终翻译，不受本节开关影响。"""

    __ui_label__ = "翻译"
    __ui_icon__ = "languages"
    __ui_order__ = 2

    translate_to_japanese: bool = Field(
        default=True,
        description=(
            "命令路线是否免翻译：关闭后 /sbv2 把清理后的原文直接交给 language=JP 的后端合成；"
            "自动回复的日语语音扩展始终走中译日引擎，不受本开关影响"
        ),
        json_schema_extra={
            "label": "命令路线翻译开关",
            "hint": (
                "关掉后 /sbv2 会把中文原样交给日语后端，通常念成乱码——除非你本来就要发日文。"
                "自动回复路线永远翻译，不读这个开关。"
            ),
            "group": "总开关与引擎",
        },
    )
    engine: Literal["host_llm", "index_translate"] = Field(
        default="host_llm",
        description=(
            "翻译引擎（下拉选择）：host_llm = 宿主 LLM（translate_model，消耗大模型 API）；"
            "index_translate = 本地 Index-Translate 服务（OpenAI 兼容端点，"
            "需自行部署 llama-server，见 README）。其他值在加载校验时直接报错。"
        ),
        json_schema_extra={
            "label": "翻译引擎",
            "hint": (
                "index_translate 零 API 成本、延迟低（本机 2B 模型约 0.3~0.6 秒/条）；"
                "host_llm 借用宿主大模型，不需要额外部署。"
                "选定后只有对应分组里的字段生效，另一组会被忽略。"
            ),
            "group": "总开关与引擎",
        },
    )
    translate_model: str = Field(
        default="",
        description=(
            "（仅 engine = host_llm 时生效）翻译用 LLM，支持两种填法："
            "① 任务名（replyer / utils / planner / memory 等）；"
            "② 模型名（model_config.toml 中 [[models]] 的 name，须与 name 完全一致；"
            "宿主按 name 精确匹配，不认 model_identifier）。"
            "留空 = replyer 任务。命中任务名按任务调用；否则视为模型名直传 Host 解析，"
            "模型无效时翻译失败如实暴露。"
        ),
        json_schema_extra={
            "label": "翻译模型",
            "hint": (
                "填任务名（replyer/utils/planner 等）或模型的 name（须与 model_config.toml 里 "
                "[[models]].name 完全一致）。留空 = replyer 任务。翻译是小任务，推荐 utils 或快速小模型。"
            ),
            "group": "宿主 LLM 引擎（host_llm）",
        },
    )
    base_url: str = Field(
        default="http://127.0.0.1:8020/v1",
        description="（仅 index_translate 引擎）OpenAI 兼容端点根地址，不带末尾斜杠",
        json_schema_extra={
            "label": "端点地址",
            "hint": "llama-server 的 OpenAI 兼容根地址，不带末尾斜杠。默认端口 8020，避开消息服务器的 8000。",
            "group": "本地 Index-Translate 引擎（index_translate）",
        },
    )
    model: str = Field(
        default="Index-Translate-2B",
        description="（仅 index_translate 引擎）模型名，与 llama-server --alias 一致",
        json_schema_extra={
            "label": "模型名",
            "hint": "必须与启动 llama-server 时的 --alias 完全一致。",
            "group": "本地 Index-Translate 引擎（index_translate）",
        },
    )
    timeout: int = Field(
        default=30,
        description="（仅 index_translate 引擎）单次翻译请求超时（秒）",
        json_schema_extra={
            "label": "翻译超时（秒）",
            "hint": "这里的超时只管翻译请求，和「通用」里的合成超时是两个独立设置。",
            "group": "本地 Index-Translate 引擎（index_translate）",
        },
    )
    max_tokens: int = Field(
        default=512,
        description="（仅 index_translate 引擎）生成 token 上限；官方短台词建议 512",
        json_schema_extra={
            "label": "生成 token 上限",
            "hint": "官方短台词建议 512。设太小会截断译文，表现为译文不完整。",
            "group": "本地 Index-Translate 引擎（index_translate）",
        },
    )
    style: str = Field(
        default="",
        description=(
            "（仅 index_translate 引擎）风格软约束，进 prompt【注意】段，"
            "如：保持口语化、自然的直播说话风格，保留语气词与情绪，使用女性口吻。"
            "留空 = 无约束直译"
        ),
        json_schema_extra={
            "label": "文风软约束",
            "hint": (
                "写进 prompt 的风格要求，如「保持口语化、自然的直播说话风格，保留语气词与情绪」。"
                "留空 = 无约束直译。"
            ),
            "group": "本地 Index-Translate 引擎（index_translate）",
            "x-widget": "textarea",
            "rows": 2,
        },
    )
    glossary: List[str] = Field(
        default_factory=list,
        description=(
            "（仅 index_translate 引擎）术语表硬约束，每条形如 '中文→日本語'，"
            "如 '苦力怕→クリーパー'；多条时全部进 prompt【硬性要求】"
        ),
        json_schema_extra={
            "label": "术语表硬约束",
            "hint": "每条形如「中文→日本語」，如「苦力怕→クリーパー」。人名、专有名词保真靠这个。",
            "group": "本地 Index-Translate 引擎（index_translate）",
        },
    )


class SBV2TTSPluginConfig(PluginConfigBase):
    """Style-Bert-VITS2 日文语音合成插件总配置。

    节顺序对齐数据流：插件开关 → 通用 → 翻译 → 语音合成 → 组件开关
    （同时决定 config.toml 生成顺序与 WebUI 显示顺序）。
    """

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)


# ─── 插件主体 ────────────────────────────────────────────────────────────


class SBV2TTSPlugin(MaiBotPlugin):
    """Style-Bert-VITS2 (CUDA) 日文语音合成插件（MaiBot SDK 2.x 版）。

    由 ReplyExtension / Command 驱动 Style-Bert-VITS2 服务的 ``/voice`` 接口：
    命令路线每条输入一次性整段合成、由插件投递唯一一条语音；自动回复路线把
    回复正文译为日文后追加一条语音，交由宿主随正文一起发送。
    """

    config_model = SBV2TTSPluginConfig

    @property
    def config(self) -> SBV2TTSPluginConfig:
        """返回强类型配置实例。

        基类 ``MaiBotPlugin.config`` 标注为 ``PluginConfigBase``，静态检查看不到
        ``general`` / ``translate`` / ``voice`` / ``components`` 各节，这里按实际
        注入的配置模型收窄返回类型。
        """

        return cast(SBV2TTSPluginConfig, super().config)

    # 可用 LLM 任务名缓存（进程内一次；None=未拉取）。由 _resolve_translate_llm 维护。
    _available_task_names: Optional[frozenset] = None

    # ─── 生命周期 ────────────────────────────────────────────────────────

    async def on_load(self) -> None:
        """插件加载钩子。

        1. 短超时探测 Style-Bert-VITS2 服务存活（GET /models/info）。
        2. 若翻译引擎为 index_translate，短超时探测其 OpenAI 兼容端点
           （GET {base_url}/models）。
        3. 按 ``[components]`` 配置同步 ReplyExtension / Command 组件开关。
        4. 不阻断插件加载：未连通只发 warning。
        """

        await self._probe_service()

        if self.config.translate.engine == "index_translate":
            await self._probe_translate_service()

        await self._sync_component_switches()

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
        """配置热更新回调：按新的 ``[components]`` 开关同步组件状态。"""

        del scope
        del config_data
        self.ctx.logger.info("SBV2 插件配置已更新: version=%s", version)
        await self._sync_component_switches()

    async def _sync_component_switches(self) -> None:
        """按 ``[components]`` 配置启用/禁用 ReplyExtension 与 Command 组件。

        两组件都用 ``plugin_id.component_name`` 完全限定名定位，避免与其他插件重名；
        宿主返回 ``{"success": bool, "error": str}``，失败时如实记录，不假装已生效。
        扩展开关关闭时仍然注册扩展，只是通过 ``component.disable`` 禁用——
        模型在 ``reply.plugin_options`` 里看不到它，等价于关闭自动语音。
        """

        switches: Tuple[Tuple[str, str, bool], ...] = (
            ("sbv2_ja_voice", "REPLY_EXTENSION", self.config.components.extension_enabled),
            ("sbv2_tts_command", "COMMAND", self.config.components.command_enabled),
        )

        for short_name, component_type, enabled in switches:
            full_name = f"{self.ctx.plugin_id}.{short_name}"
            toggle = (
                self.ctx.component.enable_component
                if enabled
                else self.ctx.component.disable_component
            )
            result = await toggle(full_name, component_type)
            if result is True or (isinstance(result, dict) and bool(result.get("success"))):
                self.ctx.logger.info(
                    "组件状态已同步: %s (%s) -> %s",
                    full_name, component_type, "启用" if enabled else "禁用",
                )
                continue
            detail = result.get("error") if isinstance(result, dict) else result
            self.ctx.logger.error(
                "组件状态同步失败: %s (%s) -> %s, 原因=%s",
                full_name, component_type, "启用" if enabled else "禁用", detail,
            )

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

        本插件投递语音后，聊天历史会把该语音渲染为占位文本，replyer 在
        同一轮生成文字回复时可能把占位模仿进正文开头，从而
        产生一条只含"[语音消息]"的引用消息。此钩子在文本后处理前把它剥掉。
        本钩子在回复扩展之前执行，不处理扩展在正文之后追加的那条语音。

        Args:
            response: 即将执行后处理的回复正文。
            **kwargs: Host 注入的其余钩子参数（``session_id`` / ``reply_message_id`` /
                ``reply_tool_args`` / ``skip_post_process`` / ``enable_splitter`` /
                ``enable_chinese_typo``）。宿主把 ``modified_kwargs`` 当作**完整替换**，
                而非增量合并，因此修改正文时必须原样带回这些键。

        Returns:
            Dict[str, Any]: 钩子返回值；正文有变化时通过 modified_kwargs 改写。
        """

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
            # 整条回复只有占位回声：如实暴露为空正文，
            # 由 Host 走"生成可见回复失败"分支，不静默保留占位。
            self.ctx.logger.info("回复正文仅含 [语音消息] 占位回声，已剥离为空")
            return {"action": "continue", "modified_kwargs": {**kwargs, "response": ""}}

        self.ctx.logger.info("已剥离回复正文中的 [语音消息] 占位回声")
        return {"action": "continue", "modified_kwargs": {**kwargs, "response": cleaned}}

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

    async def _probe_translate_service(self) -> None:
        """短超时探测 Index-Translate 翻译服务存活（仅 index_translate 引擎调用）。

        与 :meth:`_probe_service` 同语义：任意 HTTP 状态码都视为"服务监听中"，
        仅连接被拒 / 超时 / DNS 失败视为未连通。未连通仅 warning，不抛错。
        目标 URL 由 ``[translate].base_url`` 动态拼出，不复用硬编码常量。
        """

        probe_url = f"{self.config.translate.base_url.rstrip('/')}/models"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_PROBE_TIMEOUT_SECONDS)
            ) as session:
                try:
                    async with session.get(probe_url) as resp:
                        del resp
                except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
                    self.ctx.logger.warning(
                        "Index-Translate 翻译服务未连通，路径=%s，原因=%s"
                        "（index_translate 引擎的翻译请求将失败，"
                        "请检查 llama-server 是否已启动）",
                        probe_url, exc,
                    )
        except (aiohttp.ClientError, OSError) as exc:
            self.ctx.logger.warning(
                "Index-Translate 翻译服务未连通，路径=%s，原因=%s",
                probe_url, exc,
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
            stream_id: 当前聊天流 ID，由命令 handler 注入。

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

        命令路线用 :meth:`VoiceBackend.execute` 直接投递；自动回复路线只调
        :meth:`VoiceBackend.synthesize` 取音频，闭包不会被触发。

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
        """命令路线翻译入口：``translate_to_japanese`` 关闭时免翻译。

        Args:
            text: 待翻译中文文本（已清理非空）。
            log_prefix: 日志前缀。

        Returns:
            ``(success, payload)``：
            - 开关关闭时：``(True, 原文本)``
            - 翻译成功：``(True, 译文)``
            - 翻译失败：``(False, error_detail)``——绝不返回原文
        """

        if not self.config.translate.translate_to_japanese:
            return True, text

        return await self._translate_by_engine(text, log_prefix)

    async def _translate_by_engine(
        self,
        text: str,
        log_prefix: str,
    ) -> Tuple[bool, str]:
        """按 ``[translate].engine`` 路由翻译引擎，无视免翻译开关。

        自动回复的日语语音扩展走本入口：日文是硬约束，失败也不回退原文。

        Args:
            text: 待翻译中文文本（已清理非空）。
            log_prefix: 日志前缀。

        Returns:
            ``(success, payload)``，失败时绝不返回原文。
        """

        # engine 为 Literal 校验字段，非法值在加载时即报错，此处无需兜底
        if self.config.translate.engine == "index_translate":
            return await self._translate_via_index_translate(text, log_prefix)
        return await self._translate_via_host_llm(text, log_prefix)

    async def _translate_via_index_translate(
        self,
        text: str,
        log_prefix: str,
    ) -> Tuple[bool, str]:
        """走本地 Index-Translate OpenAI 兼容端点翻译。

        HTTP 走 TTSSessionManager（隔离名 ``index_translate``），
        ``translate.timeout`` 作为按次超时传入，不吃 session 固化的默认 60 秒。

        Args:
            text: 待翻译中文文本（已清理非空）。
            log_prefix: 日志前缀。

        Returns:
            ``(success, payload)``，失败时绝不返回原文。
        """

        session_manager = await TTSSessionManager.get_instance()
        translator = IndexTranslateTranslator(
            base_url=self.config.translate.base_url,
            model=self.config.translate.model,
            max_length=self.config.general.max_text_length,
            style=self.config.translate.style,
            glossary=tuple(self.config.translate.glossary),
            timeout=self.config.translate.timeout,
            max_tokens=self.config.translate.max_tokens,
            post_factory=partial(
                session_manager.post,
                backend_name=_INDEX_TRANSLATE_SESSION_NAME,
            ),
        )
        return await translator.translate(text, log_prefix)

    async def _translate_via_host_llm(
        self,
        text: str,
        log_prefix: str,
    ) -> Tuple[bool, str]:
        """走宿主 LLM（``ctx.llm.generate``）翻译，即 v1.1.1 的既有路径。

        Args:
            text: 待翻译中文文本（已清理非空）。
            log_prefix: 日志前缀。

        Returns:
            ``(success, payload)``，失败时绝不返回原文。
        """

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
        - 否则视为模型名 → ``("replyer", 值)``，直传 Host 解析；
          模型无效时 Host 报错，翻译失败如实暴露，**不静默回退**。
        - 任务表拉取失败或为空 → 本次按模型名直传，且不写缓存（下次重试）。

        Returns:
            Tuple[str, str]: ``(task_name, model_name)``。
        """

        raw = (self.config.translate.translate_model or "").strip()
        if not raw:
            return "replyer", ""

        if self._available_task_names is None:
            try:
                names = await self.ctx.llm.get_available_models()
            except Exception as exc:
                # 拉取失败不写缓存（保持 None），下次调用重试；本次视为模型名直传。
                self.ctx.logger.warning(
                    "获取可用 LLM 任务名列表失败 (%s)，translate_model 将按模型名直传 Host",
                    exc,
                )
                return "replyer", raw
            task_names = frozenset(
                str(n).strip() for n in names if str(n).strip()
            )
            # 空列表说明宿主没给出可用任务表（SDK 会把失败形状塌缩成 []），
            # 一并保持未缓存，下次重试，避免把空集合钉死成「无任务名」。
            if not task_names:
                self.ctx.logger.warning(
                    "未取得可用 LLM 任务列表，translate_model 本次按模型名直传 Host",
                )
                return "replyer", raw
            self._available_task_names = task_names
            self.ctx.logger.debug(
                "已缓存可用 LLM 任务名: %s", sorted(self._available_task_names),
            )

        if raw in self._available_task_names:
            return raw, ""

        # 视为模型名，直传 Host 解析；无效即报错暴露
        self.ctx.logger.info(
            "translate_model `%s` 未命中任务名，将作为模型名直传 Host", raw,
        )
        return "replyer", raw

    # ─── 内部：翻译 + 合成 共享管线（一次性整段合成单条语音）──────────────

    async def _run_pipeline(
        self,
        raw_text: str,
        stream_id: str,
        log_prefix: str,
        voice: Optional[VoiceProfile],
    ) -> TTSResult:
        """命令路线的翻译-合成管线（一次性整段合成单条语音）。

        步骤：
        1. 文本清理
        2. 整段翻译为日文（``translate_to_japanese`` 开启时）；失败即如实暴露并
           返回，不分段、不回退原文
        3. 合成并投递唯一一条语音

        Args:
            raw_text: 原始输入文本。
            stream_id: 当前聊天流 ID。
            log_prefix: 日志前缀。
            voice: 音色档案。

        Returns:
            :class:`TTSResult`。
        """

        send_errors: bool = self.config.general.send_error_messages

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

        # 2. 整段翻译（失败即暴露，绝不回退原文、不分段）
        ok, payload = await self._translate_text(clean_text, log_prefix)
        if not ok:
            if send_errors:
                await self.ctx.send.text(
                    f"语音合成失败: 日文翻译失败: {payload}", stream_id,
                )
            return TTSResult(
                success=False,
                message=f"日文翻译失败: {payload}",
                backend_name=_BACKEND_NAME,
            )

        # 3. 一次性合成并投递唯一一条语音
        result = await self._execute_backend(
            payload, stream_id, log_prefix, voice=voice,
        )
        if not result.success and send_errors:
            await self.ctx.send.text(
                f"语音合成失败: {result.message}", stream_id,
            )
        return result

    # ─── ReplyExtension: planner 选中本条要走日语语音时触发 ─────────────

    @ReplyExtension(
        "sbv2_ja_voice",
        description=(
            "仅本条需要日语语音时选择。选中后本条回复正文会被朗读为日语语音——"
            "需要更短的台词时请把 reply_style 设为「简短表达」。"
            "省略 voice 使用默认音色；不要同时选择 attach_voice。"
        ),
        parameters=[
            ToolParameterInfo(
                name="voice",
                param_type=ToolParamType.STRING,
                description="音色名，如 'Ling v2'；省略用默认音色",
                required=False,
            ),
        ],
        generation_prompt="仅当本轮需要日语音频时，请用自然、简洁的中文口语书写回复正文。",
        # 宿主组件 RPC 默认 60s，本扩展最慢合法路径是「翻译 30s + 合成 60s」，放宽覆盖两者。
        timeout_ms=120000,
    )
    async def ja_voice(
        self,
        *,
        phase: str,
        text: str,
        messages: List[Dict[str, Any]],
        parameters: Dict[str, Any],
        chat: Dict[str, Any],
        reply_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """把本条回复正文译为日文并追加一条语音消息。

        Args:
            phase: ``prepare`` / ``before_send``；只有 ``before_send`` 产出语音。
            text: 回复正文（已过 ``before_post_process``，即待朗读的中文原文）。
            messages: 宿主待发送的整组消息（每项含 segments / quote_previous）。
            parameters: 经 ``prepare`` 校验后的扩展参数（可含 voice）。
            chat: ``ToolAvailabilityContext`` 的字典形式（platform / stream_id 等）。
            reply_id: 本次 reply 的标识，用于日志。
            **kwargs: 宿主注入的其余调用字段（call_id / session_id 等）。

        Returns:
            Dict[str, Any]: 成功时 ``{"messages": [*原消息, 一条 voice 消息]}``；
            其余情况返回 ``{}``（不改动宿主待发送的消息组）。
        """

        del kwargs
        log_prefix = f"[sbv2_tts_plugin][extension][{reply_id}]"

        if phase != "before_send":
            return {}
        # CLI 只渲染文本、不播音频，不做无效合成。
        if chat.get("platform") == "maisaka_cli":
            return {}

        clean_text = TTSTextUtils.clean_text(text)
        if not clean_text:
            await self._notify_extension_failure(
                chat, log_prefix, "正文为空",
            )
            return {}

        voice_name = str(parameters.get("voice") or "")
        try:
            profile = self._resolve_voice(voice_name)
        except ValueError as exc:
            self.ctx.logger.error("%s 音色解析失败: %s", log_prefix, exc)
            attempted = voice_name or self.config.voice.default_voice
            await self._notify_extension_failure(
                chat, log_prefix, f"未知音色「{attempted}」",
            )
            return {}

        ok, payload = await self._translate_by_engine(clean_text, log_prefix)
        if not ok:
            self.ctx.logger.error(
                "%s 日文翻译失败: %s", log_prefix, payload,
            )
            await self._notify_extension_failure(
                chat, log_prefix, "日文翻译失败",
            )
            return {}

        backend = self._create_backend(str(chat.get("stream_id") or ""), log_prefix)
        if backend is None:
            await self._notify_extension_failure(
                chat, log_prefix, f"合成失败（未注册的 TTS 后端: {_BACKEND_NAME}）",
            )
            return {}

        wav, error_message = await backend.synthesize(payload, profile)
        if wav is None:
            self.ctx.logger.error(
                "%s 语音合成失败: %s", log_prefix, error_message,
            )
            await self._notify_extension_failure(
                chat, log_prefix, f"合成失败（{error_message}）",
            )
            return {}

        # base64 编码随 WAV 体积增长，放到线程里，避免占用事件循环。
        b64 = await asyncio.to_thread(TTSFileManager.audio_to_base64, wav)
        if not b64:
            await self._notify_extension_failure(
                chat, log_prefix, "合成失败（音频数据转 base64 失败）",
            )
            return {}

        self.ctx.logger.info(
            "%s 已追加日语语音（音色=%s，音频=%d字节）",
            log_prefix, profile.name, len(wav),
        )
        return {
            "messages": [
                *messages,
                {
                    "segments": [
                        {"type": "voice", "data": "", "binary_data_base64": b64},
                    ],
                    "quote_previous": False,
                },
            ],
        }

    async def _notify_extension_failure(
        self,
        chat: Dict[str, Any],
        log_prefix: str,
        reason: str,
    ) -> None:
        """扩展失败通知出口：向当前聊天发一条提示，不改动宿主待发送的消息组。

        不受 ``general.send_error_messages`` 影响（该字段只控制命令路线）。
        平台没给 ``stream_id`` 时只记日志；通知返回 False 或抛异常也只记日志。

        Args:
            chat: ``ToolAvailabilityContext`` 的字典形式。
            log_prefix: 日志前缀（含 ``reply_id``）。
            reason: 失败原因文案，拼进固定提示。
        """

        stream_id = str(chat.get("stream_id") or "")
        notify_text = f"本次未附加日语语音：{reason}。"
        if not stream_id:
            self.ctx.logger.warning(
                "%s 缺少 stream_id，跳过失败通知: %s", log_prefix, notify_text,
            )
            return
        try:
            sent = await self.ctx.send.text(
                notify_text,
                stream_id,
                sync_to_maisaka_history=False,
                storage_message=False,
            )
        except Exception as exc:
            # 错误通知先于正文发出；这里失败不能阻断宿主发送正文。
            self.ctx.logger.error(
                "%s 失败通知发送异常: %s", log_prefix, exc,
            )
            return
        if not sent:
            self.ctx.logger.error(
                "%s 失败通知发送未成功: %s", log_prefix, notify_text,
            )

    # ─── Command: 由用户手动触发 ─────────────────────────────────────────

    @Command(
        "sbv2_tts_command",
        description=(
            "触发指令：/sbv2 <文本> [-v 音色] 或 /voice <文本> [-v 音色]；"
            "/sbv2 help 查看用法。（sbv2_tts_command 是内部组件名，不是输入的命令）"
        ),
        pattern=r"^/(?:sbv2|voice)\s+(?P<text>.+?)(?:\s+-v\s+(?P<voice>.+))?$",
        # 与扩展同一慢路径预算：翻译上限 30s + 合成默认 60s，宿主默认 60s 不够。
        timeout_ms=120000,
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
            "命令默认先把中文翻译为日文，再送入本地 Style-Bert-VITS2 推理合成。"
            "若 translate_to_japanese = false，则命令跳过翻译（自动回复的日语语音仍会翻译）。\n\n"
            "✂️ 合成方式：\n"
            "每条输入一次性整段翻译并合成为单条语音（不再按标点/标记分句）。"
            "译文长度由 general.max_text_length 约束，若仍超长将由服务端返回 422 错误。\n\n"
            "🤖 自动回复：\n"
            "在 reply 的 plugin_options 里选中 sbv2_ja_voice 后，本条回复正文会被朗读为"
            "日语语音，正文照常发送。\n\n"
            "📌 示例：\n"
            "/sbv2 你好世界\n"
            "/sbv2 今天的天气真不错 -v Fusetsu_v1.5\n"
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
