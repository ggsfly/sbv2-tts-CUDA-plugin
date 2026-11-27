"""
Style-Bert-VITS2 (CUDA) 后端实现。

调用本地 Style-Bert-VITS2 FastAPI 服务的 ``/voice`` 接口：

- 请求：POST，**全部参数走 query string**（FastAPI ``Query``，即使 POST 也不读 body）
- 参数：``text``、``model_name``、``speaker_name``、``style``、``language``（固定 JP）、``length`` 等
- 响应：200 -> WAV 音频二进制（RIFF 头）；422 -> JSON ``{"detail":[{"msg":...}]}``

合成与投递分离：
- :meth:`VoiceBackend.synthesize` 只请求 ``/voice`` 并校验响应，返回 WAV 字节，**不投递**；
- :meth:`VoiceBackend.execute` 调 :meth:`VoiceBackend.synthesize` 后再走
  :meth:`TTSBackendBase.send_audio` 投递一次，供命令路线使用。
"""

from typing import Any, Optional, Tuple

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

    作为后端运行时的音色描述对象。

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


# 内置默认音色档案映射
DEFAULT_VOICE_PROFILES: dict[str, VoiceProfile] = {
    "Ling v2": VoiceProfile(name="Ling v2", model="Ling-v2", speaker="Ling v2", style="Neutral"),
    "Fusetsu_v1.5": VoiceProfile(name="Fusetsu_v1.5", model="Fusetsu-v1.5", speaker="Fusetsu_v1.5", style="Neutral"),
}


class VoiceBackend(TTSBackendBase):
    """本地 Style-Bert-VITS2 (CUDA) 推理服务后端。"""

    backend_name: str = "voice"
    backend_description: str = "本地 Style-Bert-VITS2 (CUDA) 推理服务"
    default_audio_format: str = "wav"

    async def synthesize(
        self,
        text: str,
        voice: VoiceProfile,
    ) -> Tuple[Optional[bytes], str]:
        """
        请求 Style-Bert-VITS2 合成音频：只返回数据，不投递。

        Args:
            text: 待合成文本（日文）。
            voice: 已解析的音色档案。

        Returns:
            ``(wav_bytes, "")`` 成功；``(None, 错误文案)`` 失败。
        """

        # 文本校验
        if not text or not text.strip():
            return None, "待合成的文本为空"

        # 配置读取
        api_url = self.get_config(ConfigKeys.VOICE_API_URL, "http://127.0.0.1:5000/voice")
        length = self.get_config(ConfigKeys.VOICE_LENGTH, 1.0)
        timeout = self.get_config(ConfigKeys.GENERAL_TIMEOUT, 60)

        logger.info(
            "%s /voice 请求: text='%s...', model=%s, speaker=%s, style=%s",
            self.log_prefix, text[:50], voice.model, voice.speaker, voice.style,
        )

        # Style-Bert-VITS2 全部走 query 参数
        # language 固定 "JP"：当前模型仅支持日文输入，中文等语种会胡言乱语，
        # 该项无配置意义（v1.2.1 起移除配置字段），保留参数以对齐服务端契约
        params: dict[str, Any] = {
            "text": text,
            "model_name": voice.model,
            "speaker_name": voice.speaker,
            "style": voice.style,
            "language": "JP",
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
                        return None, f"音频数据无效: {error_msg}"

                    # 粗校验 WAV 魔数（RIFF）
                    if audio_data[:4] != b"RIFF":
                        return None, "返回数据不是有效的 WAV（缺少 RIFF 头）"

                    return audio_data, ""

                if response.status == 422:
                    # 参数错误：解析 detail[0].msg 暴露给用户
                    error_text = await response.text()
                    error_detail = self._extract_422_message(error_text)
                    logger.error("%s /voice 参数错误: %s", self.log_prefix, error_detail)
                    return None, f"参数错误: {error_detail}"

                # 其他非 2xx 状态码
                error_text = await response.text()
                logger.error(
                    "%s /voice 调用失败[%s]: %s",
                    self.log_prefix, response.status, error_text[:200],
                )
                return None, f"Style-Bert-VITS2 API调用失败: HTTP {response.status}"

        except asyncio.TimeoutError:
            logger.error("%s /voice 调用超时 (>%ss)", self.log_prefix, timeout)
            return None, "Style-Bert-VITS2 调用超时"
        except aiohttp.ClientError as exc:
            logger.error(
                "%s /voice 网络错误: %s: %s",
                self.log_prefix, type(exc).__name__, exc,
            )
            return None, f"网络错误: {type(exc).__name__}"

    async def execute(
        self,
        text: str,
        voice: Optional[VoiceProfile] = None,
        **kwargs: Any,
    ) -> TTSResult:
        """
        合成并投递 Style-Bert-VITS2 语音（命令路线使用）。

        Args:
            text: 待合成文本（日文）。
            voice: 音色档案；为 None 时用配置中的默认档案。
            **kwargs: 预留扩展参数（当前未使用）。

        Returns:
            :class:`TTSResult`。
        """

        del kwargs
        profile = voice if isinstance(voice, VoiceProfile) else self._get_default_profile()

        # 投递阶段的超时/网络异常仍按 /voice 的同一套文案映射，
        # 避免冒到命令 handler 被另一套文案改写。
        try:
            audio_data, error_message = await self.synthesize(text, profile)
            if audio_data is None:
                return TTSResult(
                    False, error_message, backend_name=self.backend_name
                )
            return await self.send_audio(
                audio_data=audio_data,
                voice_info=profile.name,
            )
        except asyncio.TimeoutError:
            logger.error("%s 语音投递超时: send_custom 未完成", self.log_prefix)
            return TTSResult(
                False, "Style-Bert-VITS2 调用超时", backend_name=self.backend_name
            )
        except aiohttp.ClientError as exc:
            logger.error(
                "%s 语音投递网络错误: %s: %s",
                self.log_prefix, type(exc).__name__, exc,
            )
            return TTSResult(
                False,
                f"网络错误: {type(exc).__name__}",
                backend_name=self.backend_name,
            )

    def _get_default_profile(self) -> VoiceProfile:
        """从配置构造默认音色档案。"""

        default_voice = str(self.get_config(ConfigKeys.VOICE_DEFAULT_VOICE, "Ling v2") or "Ling v2")
        voices = self.get_config(ConfigKeys.VOICE_VOICES, []) or []
        target = default_voice if (not voices or default_voice in voices) else str(voices[0])

        if target in DEFAULT_VOICE_PROFILES:
            return DEFAULT_VOICE_PROFILES[target]

        return VoiceProfile(
            name=target,
            model=target,
            speaker=target,
            style="Neutral",
        )

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
