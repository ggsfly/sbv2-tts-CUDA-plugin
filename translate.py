"""中文到日文的翻译层。

SBV2 是日文推理模型，直接传入中文会输出乱码，
因此在送入合成前需要先把中文原文翻译成自然日文。

设计要点
--------
本模块不依赖 ``maibot_sdk``，``llm_generate`` 由 plugin 注入（T6 在
``plugin.py`` 里把 ``self.ctx.llm.generate`` 作为回调传进来）。
这样做有两个好处：

1. ``translate.py`` 完全是标准库，便于单元测试时塞一个假函数即可跑通。
2. 上层 plugin 可以控制 LLM 客户端的生命周期（关闭、超时、模型选择），
   翻译层只关心"输入 prompt、拿到译文或错误"。

失败语义（重要）
----------------
- 翻译失败时 **绝不** 静默回退到中文原文。
  上层 plugin 必须能区分"成功拿到日文"和"翻译失败，
  不应该继续送入 SBV2"，避免用中文喂出乱码再让用户听到。
- 失败时通过 ``logger.error`` 暴露错误（log 中含"翻译"关键字，
  T8 验收脚本依赖这一字符串做指标统计）。
"""

from typing import Awaitable, Callable, Tuple

import asyncio
import logging

# 日志使用标准库 logger，遵循 AGENTS.md 的"插件层用 logging.getLogger"约定。
# 上层 plugin 可以在加载本模块时配置/接管 handler，不必耦合 maibot_sdk。
logger = logging.getLogger("plugin.sbv2_tts.translate")

# ``llm_generate`` 回调签名：
#   async def llm_generate(prompt: str, **kwargs) -> dict
# 返回值遵循 MaiBot LLM 客户端约定（``{"success": bool, ...}``），
# 实际注入的是 ``self.ctx.llm.generate``。
LLMGenerate = Callable[..., Awaitable[dict]]


class JPTranslator:
    """中文到日文的翻译器。

    单例无状态，实例化廉价；translate() 全部走 async，符合 plugin 异步上下文。
    """

    # prompt 模板：把"原文"和"最大长度"插值进来。
    # 显式告诉 LLM 输入不含 |||SPLIT|||，避免它学着输出分隔符污染下游分句逻辑。
    # 长度限制放在最后一句，利用 LLM 的"近因效应"提升约束遵从度。
    _PROMPT_TEMPLATE = (
        "你是一个日文翻译助手。请把以下中文内容翻译为自然、地道的日文，"
        "直接用于语音合成。\n"
        "只输出译文，不要解释、不要标点以外的符号、不要 markdown 格式。\n"
        "输入不含 `|||SPLIT|||` 标记，你也不要输出它。\n"
        "[硬性要求] 译文必须简洁，不超过 {max_length} 个字符。\n\n"
        "原文：{text}"
    )

    def __init__(self, max_length: int = 200) -> None:
        """初始化翻译器。

        :param max_length: 译文最大字符数限制，会写入 prompt 让 LLM 自约束。
        """
        self.max_length = max_length

    async def translate(
        self,
        text: str,
        log_prefix: str,
        llm_generate: LLMGenerate,
        translate_model: str = "",
    ) -> Tuple[bool, str]:
        """把中文原文翻译为日文。

        :param text: 中文原文，已剔除空白/分隔符，由调用方保证非空字符串。
        :param log_prefix: 日志前缀（通常是 ``[插件名][事件id]``），
            便于和 plugin 主流程的日志串起来。
        :param llm_generate: 异步 LLM 调用回调，签名见模块顶部 ``LLMGenerate``。
        :param translate_model: 指定模型名；空字符串表示让 Host 端用默认任务模型。
        :return: ``(success, payload)``：
            - 成功：``(True, 译文.strip())``
            - 失败：``(False, error_detail)``，**绝不返回原文**
        """
        if not text:
            # 空文本不是错误，但也不该走到 LLM；直接告诉调用方"没东西可翻译"。
            return False, ""

        prompt = self._PROMPT_TEMPLATE.format(max_length=self.max_length, text=text)

        try:
            response = await llm_generate(prompt, model=translate_model or "")
        except asyncio.TimeoutError as exc:
            # LLM 端超时，常见于 Host 卡死或任务模型未配置。
            logger.error("%s 日文翻译超时: %s", log_prefix, exc)
            return False, str(exc)
        except (ConnectionError, OSError) as exc:
            # 网络/底层 IO 类错误，单独分支便于排障。
            logger.error("%s 日文翻译网络错误: %s", log_prefix, exc)
            return False, str(exc)
        except Exception as exc:
            # 其他未预期异常：异常类型本身比消息更重要，一并记下来。
            logger.error(
                "%s 日文翻译异常: %s", log_prefix, exc, exc_info=True,
            )
            return False, str(exc)

        # MaiBot LLM 客户端约定：失败也返回 dict（success=False），所以正常解析即可。
        if not isinstance(response, dict) or not response.get("success"):
            error_detail = ""
            if isinstance(response, dict):
                # 优先取 ``error`` 字段（Host 约定的错误文案），退而取 ``response``。
                error_detail = str(
                    response.get("error") or response.get("response") or ""
                )
            logger.error("%s 日文翻译失败: %s", log_prefix, error_detail or "未知错误")
            return False, error_detail or "未知错误"

        # 兼容多种成功字段命名：content / response / text。
        # SBV2 翻译场景里 Host 通常用 ``response``，但 replyer 任务模型有时回 ``content``。
        translated = str(
            response.get("response") or response.get("content") or response.get("text") or ""
        ).strip()

        if translated:
            return True, translated

        # 极端情况：success=True 但正文为空，记下来并返回错误。
        # 不能返回原文——上层应该走错误分支，而不是把中文丢给 SBV2 出乱码。
        logger.error("%s 日文翻译返回空内容", log_prefix)
        return False, "LLM 返回空内容"