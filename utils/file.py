"""
文件操作工具类

提供 TTS 场景下的文件 / 二进制数据处理能力：
- generate_temp_path：生成唯一临时音频路径（由调用方指定 output_dir）
- write_audio_async：在线程池中异步写入音频二进制
- cleanup_file_async：可延迟清理临时音频
- validate_audio_data：粗粒度校验音频数据是否过小
- audio_to_base64：把音频二进制编码为 base64 字符串

设计要点：本模块不计算项目根目录，output_dir 必须由调用方显式传入
（一般是 `str(self.ctx.paths.runtime_dir)`）。这样可以避免在跨平台、
Docker、IDE 等不同运行环境下误判根目录。
"""

from typing import Optional, Tuple

import asyncio
import base64
import logging
import os
import uuid

logger = logging.getLogger("plugin.sbv2_tts.file")

# 音频数据最小有效大小（字节），小于此值通常意味着响应体不完整或解码失败
MIN_AUDIO_SIZE = 100


class TTSFileManager:
    """TTS 文件管理器。"""

    @classmethod
    def ensure_dir(cls, dir_path: str) -> bool:
        """
        确保目录存在，不存在则递归创建。

        Args:
            dir_path: 目标目录绝对路径或相对于当前工作目录的路径。

        Returns:
            创建成功或目录已存在则返回 True；否则返回 False。
        """
        try:
            os.makedirs(dir_path, exist_ok=True)
            return True
        except OSError as exc:
            logger.error("创建目录失败: %s, 错误: %s", dir_path, exc)
            return False

    @classmethod
    def generate_temp_path(
        cls,
        prefix: str = "tts",
        suffix: str = ".wav",
        output_dir: str = "",
    ) -> str:
        """
        生成唯一的临时音频文件路径。

        Args:
            prefix: 文件名前缀，用于区分不同后端或场景。
            suffix: 文件扩展名（含点号），例如 ".wav" / ".mp3"。
            output_dir: 输出目录绝对路径。**由调用方负责传入**
                （典型用法：`str(self.ctx.paths.runtime_dir)`）。
                传空字符串时退回到当前工作目录，避免静默写到未知位置。

        Returns:
            临时文件的绝对路径。
        """
        # 解析输出目录：调用方未提供则退到当前工作目录
        resolved_dir = output_dir if output_dir else os.getcwd()

        # 目录不存在则尝试创建
        if not cls.ensure_dir(resolved_dir):
            logger.warning(
                "无法创建输出目录 %s，使用当前工作目录 %s", resolved_dir, os.getcwd()
            )
            resolved_dir = os.getcwd()

        # 用 uuid 前 12 位保证并发场景下文件名不冲突
        unique_id = uuid.uuid4().hex[:12]
        filename = f"{prefix}_{unique_id}{suffix}"
        return os.path.join(resolved_dir, filename)

    @classmethod
    async def write_audio_async(cls, path: str, data: bytes) -> bool:
        """
        异步写入音频二进制数据到文件。

        实际写入放在默认线程池执行，避免阻塞事件循环。

        Args:
            path: 目标文件绝对路径。
            data: 音频二进制内容。

        Returns:
            写入成功返回 True；任何 IOError / 未知异常返回 False 并写日志。
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, cls._write_file_sync, path, data)
            logger.debug("音频文件写入成功: %s (%d bytes)", path, len(data))
            return True
        except IOError as exc:
            logger.error("写入音频文件失败: %s, 错误: %s", path, exc)
            return False
        except Exception as exc:
            logger.error("写入音频文件时发生未知错误: %s, 错误: %s", path, exc)
            return False

    @staticmethod
    def _write_file_sync(path: str, data: bytes) -> None:
        """在线程池中执行的同步写入逻辑。"""
        with open(path, "wb") as fp:
            fp.write(data)

    @classmethod
    def cleanup_file(cls, path: str, silent: bool = True) -> bool:
        """
        同步删除单个文件。

        Args:
            path: 文件路径。
            silent: 为 True 时忽略删除失败（推荐用于延迟清理路径）。

        Returns:
            文件存在并删除成功返回 True；否则返回 False。
        """
        try:
            if path and os.path.exists(path):
                os.remove(path)
                logger.debug("临时文件已清理: %s", path)
                return True
            return False
        except OSError as exc:
            if not silent:
                logger.warning("清理临时文件失败: %s, 错误: %s", path, exc)
            return False

    @classmethod
    async def cleanup_file_async(cls, path: str, delay: float = 180) -> bool:
        """
        异步清理临时文件，可延迟执行。

        Args:
            path: 文件路径。
            delay: 延迟秒数；默认 180 秒，覆盖大多数慢传场景。

        Returns:
            是否清理成功。
        """
        if delay > 0:
            await asyncio.sleep(delay)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, cls.cleanup_file, path, True)

    @classmethod
    def validate_audio_data(
        cls, data: bytes, min_size: Optional[int] = None
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