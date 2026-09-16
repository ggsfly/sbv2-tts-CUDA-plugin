"""文本处理工具类

提供 TTS 场景下的文本预处理能力：
- clean_text：去除首尾空白（保留原文以便上层决策）
- detect_language：按字符比例判定语言（zh / ja / en）
- split_sentences：按中英文句末标点切句，过短的句子并入前一句
- clamp_sentences：对超过服务端长度上限的段落做二次切分
"""

from typing import List

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

    @classmethod
    def split_sentences(cls, text: str, min_length: int = 2) -> List[str]:
        """
        按句末标点将文本切分为句子列表。

        支持的中英文标点：。 ！ ？ ; ； ！ ?
        切分后过短（长度 < min_length）的句子会并入前一句，避免把"啊""呢"
        之类的语气词单独发送给 TTS 引擎。

        Args:
            text: 待切分文本。
            min_length: 最小句子长度；设为 0 或负数则禁用合并。

        Returns:
            句子列表；输入为空时返回空列表。
        """
        if not text:
            return []

        # 用捕获组分割，保留分隔符以便拼接
        pattern = r"([。！？!?；;])"
        parts = re.split(pattern, text)

        sentences: List[str] = []
        current = ""

        for part in parts:
            if not part:
                continue

            # 当前 part 是标点时，附加到正在累积的句子末尾
            if re.match(pattern, part):
                current += part
                continue

            # 普通文本段：先把已有的 current 入栈，再用新段开启下一句
            if current.strip():
                sentences.append(current.strip())
            current = part

        # 收尾：把最后一段非空内容加入结果
        if current.strip():
            sentences.append(current.strip())

        # 合并过短句子到前一句
        if min_length > 0 and len(sentences) > 1:
            merged: List[str] = []
            for sent in sentences:
                if merged and len(sent) < min_length:
                    merged[-1] += sent
                else:
                    merged.append(sent)
            sentences = merged

        return sentences

    @classmethod
    def clamp_sentences(cls, sentences: List[str], max_length: int) -> List[str]:
        """
        对超过服务端长度上限的段落做二次切分。

        Style-Bert-VITS2 服务端对单次 ``text`` 有字符数上限（``limit``，默认 100），
        超过会返回 422。本方法在 ``split_sentences`` 之后兜底：对任何长度超过
        ``max_length`` 的句子，按 ``max_length`` 等长切分为多段，保证每段都不超限。

        Args:
            sentences: 经过句末标点切分后的句子列表。
            max_length: 单段最大字符数；<= 0 时直接返回原列表。

        Returns:
            切分后的句子列表，每段长度 <= max_length。
        """
        if max_length <= 0:
            return list(sentences)

        result: List[str] = []
        for sentence in sentences:
            if len(sentence) <= max_length:
                result.append(sentence)
                continue
            # 等长硬切：不尝试在词/标点处断句，因为上游 split_sentences 已按标点切过，
            # 这里只剩"无标点的长段"，硬切是唯一可行做法。
            for i in range(0, len(sentence), max_length):
                chunk = sentence[i:i + max_length]
                if chunk:
                    result.append(chunk)
        return result
