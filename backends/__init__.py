"""
TTS 后端子包。

集中导出 TTS 后端抽象基类、注册表以及当前插件已实现的 Style-Bert-VITS2 后端。
外部代码统一通过 `from ..backends import ...` 引用。
"""

from .base import TTSBackendBase, TTSBackendRegistry, TTSResult
from .voice import VoiceBackend, VoiceProfile

__all__ = [
    "TTSResult",
    "TTSBackendBase",
    "TTSBackendRegistry",
    "VoiceBackend",
    "VoiceProfile",
]
