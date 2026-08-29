"""
TTS 后端子包。

集中导出 TTS 后端抽象基类、注册表以及当前插件已实现的 SBV2 后端。
外部代码统一通过 `from ..backends import ...` 引用。
"""

import sys

sys.dont_write_bytecode = True

from .base import TTSBackendBase, TTSBackendRegistry, TTSResult
from .sbv2 import Sbv2Backend

__all__ = [
    "TTSResult",
    "TTSBackendBase",
    "TTSBackendRegistry",
    "Sbv2Backend",
]