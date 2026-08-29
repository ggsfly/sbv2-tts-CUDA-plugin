"""
TTS 后端抽象基类与注册表。

设计要点：
- 精简移植自 ``xuqian13_tts-voice-plugin.backends.base``；
- 所有 TTS 后端必须继承 ``TTSBackendBase`` 并实现 ``execute``；
- ``send_audio`` 统一处理 base64 与文件路径两种投递模式；
- 输出目录 ``output_dir`` 由调用方显式传入，避免在跨平台 / Docker 环境下误判项目根目录。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type

import asyncio
import logging

from ..utils.file import TTSFileManager

logger = logging.getLogger("plugin.sbv2_tts.backend")

# 本插件内部使用的扁平配置键（与 xuqian13 插件保持一致命名，便于阅读）
_GENERAL_USE_BASE64_AUDIO: str = "general.use_base64_audio"
_GENERAL_AUDIO_OUTPUT_DIR: str = "general.audio_output_dir"


@dataclass
class TTSResult:
    """TTS 执行结果。

    Attributes:
        success: 是否成功合成并完成投递。
        message: 描述信息，供日志或向用户回复使用。
        audio_path: 音频文件路径（文件模式下填充，base64 模式下为 None）。
        backend_name: 实际生效的后端名称，便于多后端场景下溯源。
    """

    success: bool
    message: str
    audio_path: Optional[str] = None
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
        audio_format: str = "wav",
        prefix: str = "tts",
        voice_info: str = "",
    ) -> TTSResult:
        """
        统一的音频发送方法。

        根据 ``general.use_base64_audio`` 配置决定走 ``voice``(base64) 还是
        ``voiceurl``(文件路径) 通道；文件模式下会安排 180 秒延迟清理。

        Args:
            audio_data: 音频二进制数据。
            audio_format: 音频扩展名（不含点号），如 ``wav`` / ``mp3``。
            prefix: 文件名前缀，用于区分后端或场景。
            voice_info: 音色信息（用于日志与用户提示）。

        Returns:
            :class:`TTSResult`。
        """
        # 是否走 base64 模式
        use_base64 = bool(self.get_config(_GENERAL_USE_BASE64_AUDIO, False))
        logger.debug(
            "%s 开始发送音频 (原始大小: %d字节, 格式: %s)",
            self.log_prefix, len(audio_data), audio_format,
        )

        if use_base64:
            # base64 模式：直接把音频编码后通过 voice 通道发送
            base64_audio = TTSFileManager.audio_to_base64(audio_data)
            if not base64_audio:
                return TTSResult(
                    False, "音频数据转base64失败", backend_name=self.backend_name
                )

            logger.debug("%s base64编码完成，准备通过send_custom发送", self.log_prefix)
            if self._send_custom:
                await self._send_custom(message_type="voice", content=base64_audio)
                logger.info(
                    "%s 语音已通过send_custom发送 "
                    "(%s, 音频大小: %d字节)",
                    self.log_prefix,
                    'base64模式' if use_base64 else '文件路径模式',
                    len(audio_data),
                )
            else:
                logger.warning("%s send_custom未设置，无法发送语音", self.log_prefix)
                return TTSResult(
                    False, "send_custom回调未设置", backend_name=self.backend_name
                )

            return TTSResult(
                success=True,
                message=(
                    f"成功发送{self.backend_name}语音"
                    f"{(' ('+voice_info+')') if voice_info else ''}, base64模式"
                ),
                backend_name=self.backend_name,
            )

        # 文件路径模式：写入临时文件后通过 voiceurl 通道发送
        output_dir = self.get_config(_GENERAL_AUDIO_OUTPUT_DIR, "")
        audio_path = TTSFileManager.generate_temp_path(
            prefix=prefix,
            suffix=f".{audio_format}",
            output_dir=output_dir,
        )

        if not await TTSFileManager.write_audio_async(audio_path, audio_data):
            return TTSResult(
                False, "保存音频文件失败", backend_name=self.backend_name
            )

        logger.debug("%s 音频文件已保存, 路径: %s", self.log_prefix, audio_path)
        if self._send_custom:
            await self._send_custom(message_type="voiceurl", content=audio_path)
            logger.info(
                "%s 语音已通过send_custom发送 "
                "(%s, 音频大小: %d字节)",
                self.log_prefix,
                'base64模式' if use_base64 else '文件路径模式',
                len(audio_data),
            )
            # 延迟清理临时文件：给 adapter 上传到 QQ 留充足时间，避免在慢传场景下被提前删除
            asyncio.create_task(TTSFileManager.cleanup_file_async(audio_path, delay=180))
        else:
            logger.warning("%s send_custom未设置，无法发送语音", self.log_prefix)
            return TTSResult(
                False, "send_custom回调未设置", backend_name=self.backend_name
            )

        return TTSResult(
            success=True,
            message=(
                f"成功发送{self.backend_name}语音"
                f"{(' ('+voice_info+')') if voice_info else ''}"
            ),
            audio_path=audio_path,
            backend_name=self.backend_name,
        )

    @abstractmethod
    async def execute(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs: Any,
    ) -> TTSResult:
        """
        执行 TTS 转换。

        Args:
            text: 待转换的文本。
            voice: 音色/风格标识；子类自行决定如何解释与回退。
            **kwargs: 其他后端特定参数。

        Returns:
            :class:`TTSResult`。
        """
        raise NotImplementedError

    def validate_config(self) -> Tuple[bool, str]:
        """验证后端配置是否完整。默认放行；具体后端按需覆盖。"""
        return True, ""

    def get_default_voice(self) -> str:
        """获取默认音色标识。默认空串，子类按需覆盖。"""
        return ""

    def is_available(self) -> bool:
        """检查后端是否可用。"""
        is_valid, _ = self.validate_config()
        return is_valid


class TTSBackendRegistry:
    """
    TTS 后端注册表。

    策略模式 + 工厂模式：后端类通过 :meth:`register` 注册到 ``_backends``，
    由 :meth:`create` 按名称实例化。本插件当前只实现 SBV2 单后端，注册表
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