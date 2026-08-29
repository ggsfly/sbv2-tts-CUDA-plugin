"""
SBV2 后端实现。

调用本地 SBV2 推理服务的 ``/synthesize`` 接口：

- 请求体（POST JSON）：``{"text": str, "ident": str}``（DeBERTa 等参数由服务端自动处理，客户端无需下发）
- 响应：200 -> WAV 音频二进制；422 -> 参数错误文本

参考实现来自 `xuqian13_tts-voice-plugin.backends.gpt_sovits`（本地 HTTP 后端模式）。
"""

from typing import Any, Optional

import asyncio
import logging

import aiohttp

from ..config_keys import ConfigKeys
from ..utils.file import TTSFileManager
from ..utils.session import TTSSessionManager
from .base import TTSBackendBase, TTSResult

logger = logging.getLogger("plugin.sbv2_tts.backend")


class Sbv2Backend(TTSBackendBase):
    """本地 SBV2 推理服务后端。"""

    backend_name: str = "sbv2"
    backend_description: str = "本地 SBV2 推理服务"
    default_audio_format: str = "wav"

    def get_default_voice(self) -> str:
        """获取默认音色标识（``sbv2.default_ident`` 配置，默认 ``Ling v2``）。"""
        return self.get_config(ConfigKeys.SBV2_DEFAULT_IDENT, "Ling v2")

    async def execute(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs: Any,
    ) -> TTSResult:
        """
        执行 SBV2 语音合成。

        Args:
            text: 待合成文本。
            voice: 音色 ident（覆盖 ``sbv2.default_ident``）。
            **kwargs: 预留扩展参数（当前未使用）。

        Returns:
            :class:`TTSResult`。
        """
        # 文本校验
        if not text or not text.strip():
            return TTSResult(
                False, "待合成的文本为空", backend_name=self.backend_name
            )

        # 配置读取
        api_url = self.get_config(
            ConfigKeys.SBV2_API_URL, "http://127.0.0.1:3000/synthesize"
        )
        default_ident = self.get_config(ConfigKeys.SBV2_DEFAULT_IDENT, "Ling v2")
        timeout = self.get_config(ConfigKeys.GENERAL_TIMEOUT, 60)

        # 确定实际使用的 ident
        ident = voice if voice else default_ident

        logger.info(
            "%s SBV2请求: text='%s...', ident=%s",
            self.log_prefix, text[:50], ident,
        )

        session_manager = await TTSSessionManager.get_instance()
        try:
            async with session_manager.post(
                api_url,
                json={"text": text, "ident": ident},
                backend_name="sbv2",
                timeout=timeout,
            ) as response:
                if response.status == 200:
                    audio_data = await response.read()

                    # 校验音频数据有效性
                    is_valid, error_msg = TTSFileManager.validate_audio_data(
                        audio_data
                    )
                    if not is_valid:
                        return TTSResult(
                            False,
                            f"SBV2{error_msg}",
                            backend_name=self.backend_name,
                        )

                    return await self.send_audio(
                        audio_data=audio_data,
                        audio_format="wav",
                        prefix="tts_sbv2",
                        voice_info=ident,
                    )

                if response.status == 422:
                    error_text = await response.text()
                    logger.error(
                        "%s SBV2 参数错误: %s", self.log_prefix, error_text,
                    )
                    return TTSResult(
                        False,
                        f"SBV2 参数错误: {error_text}",
                        backend_name=self.backend_name,
                    )

                # 其他非 2xx 状态码
                error_text = await response.text()
                logger.error(
                    "%s SBV2 API失败[%s]: %s",
                    self.log_prefix, response.status, error_text[:200],
                )
                return TTSResult(
                    False,
                    f"SBV2 API调用失败: {response.status}",
                    backend_name=self.backend_name,
                )

        except asyncio.TimeoutError:
            logger.error("%s SBV2 调用超时 (>%ss)", self.log_prefix, timeout)
            return TTSResult(
                False, "SBV2 调用超时", backend_name=self.backend_name
            )
        except aiohttp.ClientError as e:
            logger.error("%s SBV2 网络错误: %s: %s", self.log_prefix, type(e).__name__, e)
            return TTSResult(
                False,
                f"SBV2 网络错误: {type(e).__name__}",
                backend_name=self.backend_name,
            )