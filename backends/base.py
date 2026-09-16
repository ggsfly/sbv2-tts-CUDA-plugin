"""
TTS 后端抽象基类与注册表。

设计要点：
- 精简移植自 ``xuqian13_tts-voice-plugin.backends.base``；
- 所有 TTS 后端必须继承 ``TTSBackendBase`` 并实现 ``execute``；
- ``send_audio`` 统一走 base64 + ``ctx.send.custom("voice")`` 通道，
  不再支持文件路径 / ``voiceurl`` 模式（MaiBot 发送层不识别该自定义类型）；
- 输出目录由调用方按需处理，本基类不落盘。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

import logging

from ..utils.file import TTSFileManager

logger = logging.getLogger("plugin.sbv2_tts.backend")


@dataclass
class TTSResult:
    """TTS 执行结果。

    Attributes:
        success: 是否成功合成并完成投递。
        message: 描述信息，供日志或向用户回复使用。
        backend_name: 实际生效的后端名称，便于多后端场景下溯源。
    """

    success: bool
    message: str
    backend_name: str = ""

    def __iter__(self):
        """支持 ``success, message = result`` 解包。"""
        return iter((self.success, self.message))


class TTSBackendBase(ABC):
    """
    TTS 后端抽象基类。

    所有 TTS 后端必须继承此类并实现 :meth:`execute`。
    """

    # 后端名称（子类必须覆盖）
    backend_name: str = "base"

    # 后端描述
    backend_description: str = "TTS 后端基类"

    # 默认音频格式
    default_audio_format: str = "wav"

    def __init__(
        self,
        config_getter: Callable[[str, Any], Any],
        log_prefix: str = "",
    ) -> None:
        """
        初始化后端。

        Args:
            config_getter: 配置获取函数，签名为 ``get_config(key, default)``。
            log_prefix: 日志前缀；为空时使用 ``[<backend_name>]``。
        """
        self.get_config = config_getter
        self.log_prefix = log_prefix or f"[{self.backend_name}]"
        self._send_custom: Optional[Callable] = None

    def set_send_custom(self, send_custom_func: Callable) -> None:
        """设置发送自定义消息的回调（由 plugin 层注入）。"""
        self._send_custom = send_custom_func

    async def send_audio(
        self,
        audio_data: bytes,
        voice_info: str = "",
    ) -> TTSResult:
        """
        把音频二进制编码为 base64 后通过 ``voice`` 通道发送。

        Args:
            audio_data: 音频二进制数据（WAV）。
            voice_info: 音色信息（用于日志与用户提示）。

        Returns:
            :class:`TTSResult`。
        """
        logger.debug(
            "%s 开始发送音频 (原始大小: %d字节)",
            self.log_prefix, len(audio_data),
        )

        base64_audio = TTSFileManager.audio_to_base64(audio_data)
        if not base64_audio:
            return TTSResult(
                False, "音频数据转base64失败", backend_name=self.backend_name
            )

        if self._send_custom is None:
            logger.warning("%s send_custom未设置，无法发送语音", self.log_prefix)
            return TTSResult(
                False, "send_custom回调未设置", backend_name=self.backend_name
            )

        await self._send_custom(message_type="voice", content=base64_audio)
        logger.info(
            "%s 语音已通过send_custom发送 (base64模式, 音频大小: %d字节%s)",
            self.log_prefix, len(audio_data),
            f", 音色: {voice_info}" if voice_info else "",
        )
        return TTSResult(
            success=True,
            message=f"成功发送{self.backend_name}语音{(' (' + voice_info + ')') if voice_info else ''}",
            backend_name=self.backend_name,
        )

    @abstractmethod
    async def execute(
        self,
        text: str,
        voice: Any = None,
        **kwargs: Any,
    ) -> TTSResult:
        """
        执行 TTS 转换。

        Args:
            text: 待转换的文本。
            voice: 音色信息（具体类型由子类约定）。
            **kwargs: 其他后端特定参数。

        Returns:
            :class:`TTSResult`。
        """
        raise NotImplementedError

    def validate_config(self) -> tuple:
        """验证后端配置是否完整。默认放行；具体后端按需覆盖。"""
        return True, ""

    def is_available(self) -> bool:
        """检查后端是否可用。"""
        is_valid, _ = self.validate_config()
        return is_valid


class TTSBackendRegistry:
    """
    TTS 后端注册表。

    策略模式 + 工厂模式：后端类通过 :meth:`register` 注册到 ``_backends``，
    由 :meth:`create` 按名称实例化。本插件当前只实现 Voice 单后端，注册表
    机制保留以便后续扩展更多本地推理后端。
    """

    _backends: Dict[str, Type[TTSBackendBase]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[TTSBackendBase]) -> None:
        """注册后端。"""
        cls._backends[name] = backend_class
        logger.debug("注册TTS后端: %s", name)

    @classmethod
    def unregister(cls, name: str) -> None:
        """注销后端。"""
        cls._backends.pop(name, None)

    @classmethod
    def get(cls, name: str) -> Optional[Type[TTSBackendBase]]:
        """获取后端类，未注册返回 None。"""
        return cls._backends.get(name)

    @classmethod
    def create(
        cls,
        name: str,
        config_getter: Callable[[str, Any], Any],
        log_prefix: str = "",
    ) -> Optional[TTSBackendBase]:
        """
        创建后端实例。

        Args:
            name: 已注册的后端名称。
            config_getter: 配置获取回调。
            log_prefix: 日志前缀。

        Returns:
            后端实例或 None。
        """
        backend_class = cls.get(name)
        if backend_class is None:
            return None
        return backend_class(config_getter, log_prefix)

    @classmethod
    def list_backends(cls) -> list[str]:
        """列出所有已注册的后端名称。"""
        return list(cls._backends.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查后端是否已注册。"""
        return name in cls._backends
