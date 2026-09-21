"""配置键常量定义。

集中管理 Style-Bert-VITS2 日文语音合成插件的所有配置键，避免在业务模块中硬编码字符串。
键名采用扁平 ``section.field`` 形式，与 ``config.toml`` 中的层级结构一一对应。
"""

from typing import Final


class ConfigKeys:
    """配置键常量类。

    所有键均为字符串字面量，遵循 ``section.field`` 的扁平命名约定，
    便于在不同业务模块之间共享引用，同时与 ``config.toml`` 的段/字段保持一致。
    """

    # ========== Plugin 配置 ==========
    PLUGIN_ENABLED: Final[str] = "plugin.enabled"
    PLUGIN_CONFIG_VERSION: Final[str] = "plugin.config_version"

    # ========== General 通用配置 ==========
    GENERAL_TIMEOUT: Final[str] = "general.timeout"
    GENERAL_MAX_TEXT_LENGTH: Final[str] = "general.max_text_length"
    GENERAL_SEND_ERROR_MESSAGES: Final[str] = "general.send_error_messages"
    GENERAL_ECHO_ORIGINAL_TEXT: Final[str] = "general.echo_original_text"
    GENERAL_TRANSLATE_TO_JAPANESE: Final[str] = "general.translate_to_japanese"
    GENERAL_TRANSLATE_MODEL: Final[str] = "general.translate_model"

    # ========== Components 组件配置 ==========
    COMPONENTS_TOOL_ENABLED: Final[str] = "components.tool_enabled"
    COMPONENTS_COMMAND_ENABLED: Final[str] = "components.command_enabled"

    # ========== Voice 推理服务配置 ==========
    VOICE_API_URL: Final[str] = "voice.api_url"
    VOICE_DEFAULT_VOICE: Final[str] = "voice.default_voice"
    VOICE_LANGUAGE: Final[str] = "voice.language"
    VOICE_LENGTH: Final[str] = "voice.length"
    VOICE_VOICES: Final[str] = "voice.voices"
