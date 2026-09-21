"""文本处理工具类

提供 TTS 场景下的文本预处理能力：
- clean_text：去除首尾空白（保留原文以便上层决策）
- detect_language：按字符比例判定语言（zh / ja / en）

注：插件已移除“按标点分句 + 超长硬切”能力，改为整段一次性合成单条语音。
"""

import re


class TTSTextUtils:
    """TTS 文本处理工具类"""

    # 语言检测正则：中文 / 英文 / 日文（含平假名与片假名）
    CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
    ENGLISH_PATTERN = re.compile(r"[a-zA-Z]")
    JAPANESE_PATTERN = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")

    @classmethod
    def clean_text(cls, text: str, max_length: int = 500) -> str:
        """
        清理文本。

        本实现刻意简化：只去除首尾空白，保留原文。理由是 TTS 输入由 LLM
        生成或用户输入，原始内容应原样交给后端；硬截断 / 网络用语替换等
        行为会改变语音内容，由调用方在更高层按需启用。

        Args:
            text: 原始文本。
            max_length: 预留字段，不在本方法内使用，仅与旧实现保持签名一致。

        Returns:
            去除首尾空白后的文本；输入为空时返回空串。
        """
        if not text:
            return ""
        return text.strip()

    @classmethod
    def detect_language(cls, text: str) -> str:
        """
        检测文本语言。

        算法：统计中日英三类字符占比，占比最高的类别（且超过阈值）返回对应语言码。
        - 中文占比 > 0.3 → "zh"
        - 日文占比 > 0.3 → "ja"
        - 英文占比 > 0.8 → "en"
        - 其他情况默认 "zh"（含空串 / 数字 / 纯标点等退化输入）

        Args:
            text: 待检测文本。

        Returns:
            语言代码字符串，取值为 "zh" / "ja" / "en"。
        """
        if not text:
            return "zh"

        chinese_chars = len(cls.CHINESE_PATTERN.findall(text))
        english_chars = len(cls.ENGLISH_PATTERN.findall(text))
        japanese_chars = len(cls.JAPANESE_PATTERN.findall(text))
        total_chars = chinese_chars + english_chars + japanese_chars

        if total_chars == 0:
            return "zh"

        chinese_ratio = chinese_chars / total_chars
        japanese_ratio = japanese_chars / total_chars
        english_ratio = english_chars / total_chars

        if chinese_ratio > 0.3:
            return "zh"
        if japanese_ratio > 0.3:
            return "ja"
        if english_ratio > 0.8:
            return "en"
        return "zh"
