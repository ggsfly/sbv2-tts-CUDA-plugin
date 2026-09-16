"""音频二进制数据处理工具。

提供 TTS 场景下的二进制处理能力：
- validate_audio_data：粗粒度校验音频数据是否过小
- audio_to_base64：把音频二进制编码为 base64 字符串

设计要点：本模块只负责内存数据校验与编码，不落盘、不生成临时文件——
插件始终走 ``ctx.send.custom("voice", base64)`` 通道发送语音。
"""

from typing import Optional, Tuple

import base64
import logging

logger = logging.getLogger("plugin.sbv2_tts.file")

# 音频数据最小有效大小（字节），小于此值通常意味着响应体不完整或解码失败
MIN_AUDIO_SIZE = 100


class TTSFileManager:
    """TTS 音频数据处理工具。"""

    @classmethod
    def validate_audio_data(
        cls,
        data: bytes,
        min_size: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        粗粒度校验音频数据有效性。

        Args:
            data: 音频二进制内容。
            min_size: 自定义最小字节数；为 None 时使用 MIN_AUDIO_SIZE。

        Returns:
            (is_valid, error_message) 元组。校验通过时 error_message 为空串。
        """
        if data is None:
            return False, "音频数据为空"

        threshold = min_size if min_size is not None else MIN_AUDIO_SIZE
        if len(data) < threshold:
            return False, f"音频数据过小({len(data)}字节 < {threshold}字节)"

        return True, ""

    @classmethod
    def audio_to_base64(cls, data: bytes) -> str:
        """
        将音频二进制编码为 base64 字符串。

        异常输入返回空串而非抛错，便于上层在 IPC 通道中容错。

        Args:
            data: 音频二进制内容。

        Returns:
            base64 字符串；失败时返回空串。
        """
        try:
            return base64.b64encode(data).decode("utf-8")
        except Exception as exc:
            logger.error("音频数据转 base64 失败: %s", exc)
            return ""
