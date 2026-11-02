"""中文到日文的翻译层。

Style-Bert-VITS2 是日文推理模型，直接传入中文会输出乱码，
因此在送入合成前需要先把中文原文翻译成自然日文。

设计要点
--------
本模块不依赖 ``maibot_sdk``，``llm_generate`` 由 plugin 注入。
这样做有两个好处：

1. ``translate.py`` 完全是标准库，便于单元测试时塞一个假函数即可跑通。
2. 上层 plugin 可以控制 LLM 的生命周期与任务/模型选择，翻译层只关心
   "输入 prompt、拿到译文或错误"。

双引擎
------
v1.2.0 起支持两种翻译引擎，由 ``[translate].engine`` 配置切换：

- ``host_llm``：:class:`JPTranslator`，走宿主 LLM（``ctx.llm.generate``），
  依赖大模型 API；
- ``index_translate``：:class:`IndexTranslateTranslator`，直连本地
  Index-Translate（B 站 IndexTeam 开源翻译模型，llama.cpp / vLLM 提供
  OpenAI 兼容端点）的 ``POST {base_url}/chat/completions``。HTTP 依赖同样
  采用注入式（``post_factory``），本模块自身仍是标准库。

新版 SDK 调取方式
-----------------
SDK 2.x 的 ``ctx.llm.generate`` 同时接受 ``task_name``（任务名）与
``model_name``（具体模型名/标识符）。本模块把二者作为显式参数透传，
上层负责解析配置后注入：

- 留空 / 任务名 → 走任务路由；
- 模型名 / model_identifier → 直传 Host 解析；模型无效时 Host 报错，
  翻译失败如实暴露，**绝不静默回退到中文原文**。

失败语义（重要）
----------------
- 翻译失败时 **绝不** 静默回退到中文原文。
  上层 plugin 必须能区分"成功拿到日文"和"翻译失败"，
  不应该继续送入 Style-Bert-VITS2，避免用中文喂出乱码再让用户听到。
- 失败时通过 ``logger.error`` 暴露错误（log 中含"翻译"关键字，
  便于排障与指标统计）。
"""

from contextlib import AbstractAsyncContextManager
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import asyncio
import json
import logging
import re

# 日志使用标准库 logger，遵循 AGENTS.md 的"插件层用 logging.getLogger"约定。
logger = logging.getLogger("plugin.sbv2_tts.translate")

# ``llm_generate`` 回调签名：
#   async def llm_generate(prompt: str, *, task_name: str, model_name: str, **kwargs) -> dict
# 返回值遵循 MaiBot LLM 客户端约定（``{"success": bool, ...}``），
# 实际注入的是 ``self.ctx.llm.generate``。
LLMGenerate = Callable[..., Awaitable[dict]]

# ``post_factory`` 回调签名：
#   def post_factory(url: str, *, json: dict, timeout: int) -> AbstractAsyncContextManager
# 返回异步上下文管理器，yield 出的对象需提供 ``status: int`` 与 ``read() -> bytes``
# （与 aiohttp.ClientResponse 的最小接口一致）。实际注入的是
# ``functools.partial(session_manager.post, backend_name="index_translate")``。
PostFactory = Callable[..., AbstractAsyncContextManager[Any]]


class JPTranslator:
    """中文到日文的翻译器。

    单例无状态，实例化廉价；translate() 全部走 async，符合 plugin 异步上下文。
    """

    # prompt 模板：把"原文"和"最大长度"插值进来。
    # 长度限制放在最后一句，利用 LLM 的"近因效应"提升约束遵从度。
    _PROMPT_TEMPLATE = (
        "你是一个日文翻译助手。请把以下中文内容翻译为自然、地道的日文，"
        "直接用于语音合成。\n"
        "只输出译文，不要解释、不要标点以外的符号、不要 markdown 格式。\n"
        "[硬性要求] 译文必须简洁，不超过 {max_length} 个字符。\n\n"
        "原文：{text}"
    )

    def __init__(self, max_length: int = 100) -> None:
        """初始化翻译器。

        :param max_length: 译文最大字符数限制，会写入 prompt 让 LLM 自约束。
            默认 100，对齐 Style-Bert-VITS2 服务端 ``limit``。
        """

        self.max_length = max_length

    async def translate(
        self,
        text: str,
        log_prefix: str,
        llm_generate: LLMGenerate,
        task_name: str = "replyer",
        model_name: str = "",
    ) -> Tuple[bool, str]:
        """把中文原文翻译为日文。

        :param text: 中文原文，已剔除空白/分隔符，由调用方保证非空字符串。
        :param log_prefix: 日志前缀（通常是 ``[插件名][事件id]``）。
        :param llm_generate: 异步 LLM 调用回调，签名见模块顶部 ``LLMGenerate``。
        :param task_name: LLM 任务名（``replyer``/``utils`` 等）。留空使用
            ``replyer`` 任务。
        :param model_name: 具体模型名 / model_identifier；非空时由 Host 直连该模型，
            无效时 Host 报错并经本方法暴露。
        :return: ``(success, payload)``：
            - 成功：``(True, 译文.strip())``
            - 失败：``(False, error_detail)``，**绝不返回原文**
        """

        if not text:
            # 空文本不是错误，但也不该走到 LLM；直接告诉调用方"没东西可翻译"。
            return False, ""

        prompt = self._PROMPT_TEMPLATE.format(max_length=self.max_length, text=text)

        # 把任务名与模型名显式透传给 Host；两者语义不同：
        # task_name 决定任务路由与默认参数，model_name 钉死具体模型。
        request_kwargs: Dict[str, Any] = {"task_name": task_name or "replyer"}
        if model_name:
            request_kwargs["model_name"] = model_name

        try:
            response = await llm_generate(prompt, **request_kwargs)
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
        # Style-Bert-VITS2 翻译场景里 Host 通常用 ``response``，但部分任务模型回 ``content``。
        translated = str(
            response.get("response") or response.get("content") or response.get("text") or ""
        ).strip()

        if translated:
            return True, translated

        # 极端情况：success=True 但正文为空，记下来并返回错误。
        # 不能返回原文——上层应该走错误分支，而不是把中文丢给推理服务出乱码。
        logger.error("%s 日文翻译返回空内容", log_prefix)
        return False, "LLM 返回空内容"


class IndexTranslateTranslator:
    """走本地 Index-Translate（OpenAI 兼容端点）的中文到日文翻译器。

    Index-Translate 是 B 站 IndexTeam 基于 Qwen3.5 的翻译模型家族，
    本地以 llama.cpp ``llama-server``（或 vLLM）部署，暴露
    ``POST {base_url}/chat/completions``。

    调用要点（对齐官方推理文档）：
    - ``temperature=0``（贪心解码，翻译任务要求稳定输出）；
    - ``chat_template_kwargs.enable_thinking=false``，否则思维链可能替代
      译文出现在 ``content`` 里；
    - 输出仍剥 ``<think>...</think>`` 块兜底。

    Prompt 分两档（对齐官方 ``instTrans`` 约束格式）：
    - 无 style / glossary 时用无约束直译模板；
    - 有术语表或风格要求时用【源文】/【约束要求】结构，术语表用
      ``A→B、C→D`` 写法进【硬性要求】，风格进【注意】；
      长度上限复用 ``max_text_length``（对齐 Style-Bert-VITS2 服务端 limit）。

    HTTP 依赖注入
    -------------
    与 :class:`JPTranslator` 的 ``llm_generate`` 同构，``post_factory`` 由
    plugin 注入（``functools.partial(session_manager.post, backend_name=...)``），
    本模块不 import aiohttp，便于单测注入假响应。

    失败语义（与 JPTranslator 一致）
    --------------------------------
    翻译失败 **绝不** 静默回退中文原文，返回 ``(False, error_detail)``。
    """

    # 无约束直译模板（对齐官方 README 的最小 prompt）
    _PLAIN_PROMPT_TEMPLATE = (
        "请将以下文本翻译为日语，直接输出翻译结果，不要进行任何解释。\n\n{text}"
    )

    # instTrans 约束格式模板（对齐官方客户端的约束写法）
    _CONSTRAINED_PROMPT_TEMPLATE = (
        "请将以下文本翻译成日语，并且严格遵循所有约束要求。\n\n"
        "【源文】\n{text}\n\n"
        "【约束要求】\n{constraints}\n\n"
        "只输出译文，不要有任何额外说明。"
    )

    # 兜底剥离思维链块（enable_thinking 已在请求层关闭，此处防漏网）
    _THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

    def __init__(
        self,
        base_url: str,
        model: str,
        max_length: int = 100,
        style: str = "",
        glossary: Sequence[str] = (),
        timeout: int = 30,
        max_tokens: int = 512,
        post_factory: Optional[PostFactory] = None,
    ) -> None:
        """初始化翻译器。

        :param base_url: OpenAI 兼容端点根地址（如 ``http://127.0.0.1:8020/v1``）。
        :param model: 模型名（llama.cpp 用 ``--alias`` 声明的名称）。
        :param max_length: 译文最大字符数，进【硬性要求】约束；<=0 表示不约束。
        :param style: 风格软约束（进【注意】），空串表示无。
        :param glossary: 术语表条目列表，每条形如 ``中文→日本語``。
        :param timeout: 单次请求超时（秒），按次传给 post_factory。
        :param max_tokens: 生成 token 上限。
        :param post_factory: 异步 POST 上下文工厂；为 None 时在首次调用时报错暴露
            （本模块不内置 HTTP 实现，必须由上层注入）。
        """

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_length = max_length
        self.style = style.strip()
        self.glossary = tuple(item.strip() for item in glossary if item.strip())
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.post_factory = post_factory

    def _build_prompt(self, text: str) -> str:
        """按是否配置约束（术语表/风格/长度）选择 prompt 模板。"""

        constraints: List[str] = []
        if self.glossary:
            constraints.append(
                "【硬性要求】专名/术语对照: " + "、".join(self.glossary)
            )
        if self.max_length and self.max_length > 0:
            constraints.append(
                f"【硬性要求】译文必须简洁，不超过 {self.max_length} 个字符。"
            )
        if self.style:
            constraints.append(f"【注意】{self.style}")

        if not constraints:
            return self._PLAIN_PROMPT_TEMPLATE.format(text=text)

        # 约束按 1. 2. 3. 编号，对齐官方 instTrans 示例
        numbered = "\n".join(
            f"{idx}. {constraint}" for idx, constraint in enumerate(constraints, 1)
        )
        return self._CONSTRAINED_PROMPT_TEMPLATE.format(text=text, constraints=numbered)

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        """构造 OpenAI 兼容 chat/completions 请求体。"""

        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    @staticmethod
    def _strip_think(content: str) -> str:
        """剥掉可能混入正文的 <think>...</think> 块后 strip。"""

        return IndexTranslateTranslator._THINK_BLOCK_PATTERN.sub("", content).strip()

    async def translate(
        self,
        text: str,
        log_prefix: str,
    ) -> Tuple[bool, str]:
        """把中文原文翻译为日文。

        :param text: 中文原文，由调用方保证非空字符串。
        :param log_prefix: 日志前缀（通常是 ``[插件名][事件id]``）。
        :return: ``(success, payload)``：
            - 成功：``(True, 译文.strip())``
            - 失败：``(False, error_detail)``，**绝不返回原文**
        """

        if not text:
            # 空文本不是错误，但也不该发起请求；与 JPTranslator 语义一致。
            return False, ""

        if self.post_factory is None:
            # 未注入 HTTP 工厂属于装配错误，如实暴露而不是静默失败。
            logger.error("%s Index-Translate 翻译器未注入 post_factory", log_prefix)
            return False, "翻译器未注入 post_factory"

        url = f"{self.base_url}/chat/completions"
        payload = self._build_payload(self._build_prompt(text))

        try:
            async with self.post_factory(url, json=payload, timeout=self.timeout) as resp:
                body = await resp.read()
                status = int(getattr(resp, "status", 0))
        except asyncio.TimeoutError as exc:
            # 请求超时，常见于本地服务未启动或加载模型中。
            logger.error("%s Index-Translate 翻译超时: %s", log_prefix, exc)
            return False, str(exc)
        except (ConnectionError, OSError) as exc:
            # 连接被拒 / 网络不可达等 IO 类错误，单独分支便于排障。
            logger.error("%s Index-Translate 翻译网络错误: %s", log_prefix, exc)
            return False, str(exc)
        except Exception as exc:
            # 其他未预期异常（含 aiohttp.ClientError）：异常类型一并记日志。
            logger.error(
                "%s Index-Translate 翻译异常: %s", log_prefix, exc, exc_info=True,
            )
            return False, str(exc)

        if status != 200:
            # 非 200（含官方免费 API 的 504、llama.cpp 的 5xx）：带状态码如实暴露。
            body_text = body.decode("utf-8", errors="replace").strip()
            error_detail = f"HTTP {status}: {body_text[:200]}" if body_text else f"HTTP {status}"
            logger.error("%s Index-Translate 翻译失败: %s", log_prefix, error_detail)
            return False, error_detail

        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("%s Index-Translate 响应非 JSON: %s", log_prefix, exc)
            return False, f"响应非 JSON: {exc}"

        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices or not isinstance(choices, list):
            error_detail = str(data.get("error") or data)[:200] if isinstance(data, dict) else str(data)[:200]
            logger.error("%s Index-Translate 响应缺少 choices: %s", log_prefix, error_detail)
            return False, error_detail or "响应缺少 choices"

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = str((message or {}).get("content") or "")
        translated = self._strip_think(content)

        if translated:
            return True, translated

        # 兜底：enable_thinking 未生效导致整段是思维链、或服务返回空 content。
        # 不能返回原文——上层应该走错误分支，而不是把中文丢给推理服务出乱码。
        logger.error("%s Index-Translate 翻译返回空内容", log_prefix)
        return False, "翻译服务返回空内容"
