"""
Style-Bert-VITS2 (CUDA) 后端实现。

调用本地 Style-Bert-VITS2 FastAPI 服务的 ``/voice`` 接口：

- 请求：POST，**全部参数走 query string**（FastAPI ``Query``，即使 POST 也不读 body）
- 参数：``text``、``model_name``、``speaker_name``、``style``、``language``、``length`` 等
- 响应：200 -> WAV 音频二进制（RIFF 头）；422 -> JSON ``{"detail":[{"msg":...}]}``
"""

from typing import Any, Optional

import asyncio
import json
import logging

import aiohttp
from pydantic import BaseModel

from ..config_keys import ConfigKeys
from ..utils.file import TTSFileManager
from ..utils.session import TTSSessionManager
from .base import TTSBackendBase, TTSResult

logger = logging.getLogger("plugin.sbv2_tts.backend")


class VoiceProfile(BaseModel):
    """音色档案：把用户友好的音色名映射到 Style-Bert-VITS2 的三个参数。

    同时作为 Pydantic 配置模型（``[voice].voices`` 列表元素）与后端运行时
    的音色描述对象，避免在 plugin / config / backend 之间重复定义同类结构。

    Attributes:
        name: 用户可见的音色名（``-v`` 与 ``default_voice`` 取值）。
        model: ``model_assets/`` 下的目录名，对应 ``model_name`` 参数。
        speaker: ``spk2id`` 的键，对应 ``speaker_name`` 参数。
        style: ``style2id`` 的键，对应 ``style`` 参数。
    """

    name: str
    model: str
    speaker: str
    style: str = "Neutral"


class VoiceBackend(TTSBackendBase):
    """本地 Style-Bert-VITS2 (CUDA) 推理服务后端。"""

    backend_name: str = "voice"
    backend_description: str = "本地 Style-Bert-VITS2 (CUDA) 推理服务"
    default_audio_format: str = "wav"

    async def execute(
        self,
        text: str,
        voice: Optional[VoiceProfile] = None,
        **kwargs: Any,
    ) -> TTSResult:
        """
        执行 Style-Bert-VITS2 语音合成。

        Args:
            text: 待合成文本（日文）。
            voice: 音色档案；为 None 时用配置中的默认档案。
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
        api_url = self.get_config(ConfigKeys.VOICE_API_URL, "http://127.0.0.1:5000/voice")
        language = self.get_config(ConfigKeys.VOICE_LANGUAGE, "JP")
        length = self.get_config(ConfigKeys.VOICE_LENGTH, 1.0)
        timeout = self.get_config(ConfigKeys.GENERAL_TIMEOUT, 60)

        # 确定实际使用的音色档案
        profile = voice if isinstance(voice, VoiceProfile) else self._get_default_profile()

        logger.info(
            "%s /voice 请求: text='%s...', model=%s, speaker=%s, style=%s",
            self.log_prefix, text[:50], profile.model, profile.speaker, profile.style,
        )

        # Style-Bert-VITS2 全部走 query 参数
        params: dict[str, Any] = {
            "text": text,
            "model_name": profile.model,
            "speaker_name": profile.speaker,
            "style": profile.style,
            "language": language,
            "length": length,
        }

        session_manager = await TTSSessionManager.get_instance()
        try:
            async with session_manager.post(
                api_url,
                params=params,
                backend_name=self.backend_name,
                timeout=timeout,
            ) as response:
                if response.status == 200:
                    audio_data = await response.read()

                    is_valid, error_msg = TTSFileManager.validate_audio_data(audio_data)
                    if not is_valid:
                        return TTSResult(
                            False,
                            f"音频数据无效: {error_msg}",
                            backend_name=self.backend_name,
                        )

                    # 粗校验 WAV 魔数（RIFF）
                    if audio_data[:4] != b"RIFF":
                        return TTSResult(
                            False,
                            "返回数据不是有效的 WAV（缺少 RIFF 头）",
                            backend_name=self.backend_name,
                        )

                    return await self.send_audio(
                        audio_data=audio_data,
                        voice_info=profile.name,
                    )

                if response.status == 422:
                    # 参数错误：解析 detail[0].msg 暴露给用户
                    error_text = await response.text()
                    error_detail = self._extract_422_message(error_text)
                    logger.error("%s /voice 参数错误: %s", self.log_prefix, error_detail)
                    return TTSResult(
                        False,
                        f"参数错误: {error_detail}",
                        backend_name=self.backend_name,
                    )

                # 其他非 2xx 状态码
                error_text = await response.text()
                logger.error(
                    "%s /voice 调用失败[%s]: %s",
                    self.log_prefix, response.status, error_text[:200],
                )
                return TTSResult(
                    False,
                    f"Style-Bert-VITS2 API调用失败: HTTP {response.status}",
                    backend_name=self.backend_name,
                )

        except asyncio.TimeoutError:
            logger.error("%s /voice 调用超时 (>%ss)", self.log_prefix, timeout)
            return TTSResult(
                False, "Style-Bert-VITS2 调用超时", backend_name=self.backend_name
            )
        except aiohttp.ClientError as exc:
            logger.error(
                "%s /voice 网络错误: %s: %s",
                self.log_prefix, type(exc).__name__, exc,
            )
            return TTSResult(
                False,
                f"网络错误: {type(exc).__name__}",
                backend_name=self.backend_name,
            )

    def _get_default_profile(self) -> VoiceProfile:
        """从配置构造默认音色档案。"""

        default_voice = str(self.get_config(ConfigKeys.VOICE_DEFAULT_VOICE, "") or "")
        voices = self.get_config(ConfigKeys.VOICE_VOICES, []) or []
        for entry in voices:
            name = str(getattr(entry, "name", "") or "")
            if name and name == default_voice:
                return VoiceProfile(
                    name=name,
                    model=str(getattr(entry, "model", "") or ""),
                    speaker=str(getattr(entry, "speaker", "") or ""),
                    style=str(getattr(entry, "style", "Neutral") or "Neutral"),
                )
        # 配置缺失时退到第一个档案；列表为空则用空串（服务端会 422）
        if voices:
            entry = voices[0]
            return VoiceProfile(
                name=str(getattr(entry, "name", default_voice or "unknown") or default_voice or "unknown"),
                model=str(getattr(entry, "model", "") or ""),
                speaker=str(getattr(entry, "speaker", "") or ""),
                style=str(getattr(entry, "style", "Neutral") or "Neutral"),
            )
        return VoiceProfile(name=default_voice or "unknown", model="", speaker="", style="Neutral")

    @staticmethod
    def _extract_422_message(error_text: str) -> str:
        """从 422 响应体提取可读错误消息。

        Style-Bert-VITS2 的 422 响应形如：
        ``{"detail":[{"type":"invalid_params","msg":"...","loc":["query","xxx"]}]}``
        """

        if not error_text:
            return "未知参数错误"
        try:
            payload = json.loads(error_text)
        except (json.JSONDecodeError, ValueError):
            return error_text[:200]
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict):
                return str(first.get("msg") or first.get("type") or "参数错误")
        return error_text[:200]
