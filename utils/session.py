"""
HTTP Session 管理器

封装 aiohttp.ClientSession 的多后端连接池复用，避免每次请求都新建连接。
所有后端共用一个单例，按 backend_name 维度维护独立的 session 池。
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import asyncio
import logging

import aiohttp

logger = logging.getLogger("plugin.sbv2_tts.session")


class TTSSessionManager:
    """
    TTS HTTP Session 管理器。

    - 单例：通过 `await get_instance()` 获取。
    - 多后端：按 backend_name 维护独立的 ClientSession，关闭某个后端不影响其他。
    - 连接池：使用 TCPConnector，限制并发与每主机连接数，TTL DNS 缓存 5 分钟。
    - 异步上下文：post / get 用 `@asynccontextmanager` 暴露，自动 release response。
    """

    _instance: Optional["TTSSessionManager"] = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._default_timeout = 60

    @classmethod
    async def get_instance(cls) -> "TTSSessionManager":
        """
        获取单例实例。

        双重检查 + asyncio.Lock 保证并发安全。
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def get_session(
        self,
        backend_name: str = "default",
        timeout: Optional[int] = None,
    ) -> aiohttp.ClientSession:
        """
        获取或创建指定 backend_name 的 HTTP Session。

        Args:
            backend_name: 后端名称，用于隔离不同的连接池。
            timeout: 单次请求总超时（秒）；None 表示用默认 60 秒。

        Returns:
            可复用的 aiohttp.ClientSession 实例。
        """
        existing = self._sessions.get(backend_name)
        if existing is None or existing.closed:
            timeout_val = timeout if timeout is not None else self._default_timeout
            connector = aiohttp.TCPConnector(
                limit=10,  # 全局最大连接数
                limit_per_host=5,  # 单主机最大连接数
                ttl_dns_cache=300,  # DNS 缓存 5 分钟
                force_close=True,  # 禁用 keep-alive，规避部分 API 兼容性问题
            )
            self._sessions[backend_name] = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=timeout_val),
            )
            logger.debug("创建新的 HTTP Session: %s", backend_name)

        return self._sessions[backend_name]

    async def close_session(self, backend_name: Optional[str] = None) -> None:
        """
        优雅关闭指定后端或全部 Session。

        Args:
            backend_name: 为 None 时关闭所有 session。
        """
        if backend_name:
            session = self._sessions.get(backend_name)
            if session is not None:
                await session.close()
                self._sessions.pop(backend_name, None)
                logger.debug("关闭 HTTP Session: %s", backend_name)
            return

        for name, session in list(self._sessions.items()):
            if not session.closed:
                await session.close()
                logger.debug("关闭 HTTP Session: %s", name)
        self._sessions.clear()

    @asynccontextmanager
    async def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
        backend_name: str = "default",
        timeout: Optional[int] = None,
    ):
        """
        发送 POST 请求的异步上下文管理器。

        Usage::

            async with session_manager.post(url, json=payload) as response:
                body = await response.read()

        Args:
            url: 目标 URL。
            json: JSON 请求体（自动设置 Content-Type）。
            headers: 额外请求头。
            data: 表单 / 原始 body。
            backend_name: 后端名称，决定复用哪个 session。
            timeout: 本次请求独立的超时；None 时使用 session 默认值。

        Yields:
            aiohttp.ClientResponse；退出 with 块时自动 release。
        """
        session = await self.get_session(backend_name, timeout)

        # 单次调用指定了独立超时则覆盖
        req_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else None

        response = await session.post(
            url,
            json=json,
            headers=headers,
            data=data,
            timeout=req_timeout,
        )
        try:
            yield response
        finally:
            response.release()

    @asynccontextmanager
    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        backend_name: str = "default",
        timeout: Optional[int] = None,
    ):
        """
        发送 GET 请求的异步上下文管理器。

        Usage::

            async with session_manager.get(url, params={"k": "v"}) as response:
                payload = await response.json()

        Args:
            url: 目标 URL。
            headers: 额外请求头。
            params: URL 查询参数。
            backend_name: 后端名称，决定复用哪个 session。
            timeout: 本次请求独立的超时；None 时使用 session 默认值。

        Yields:
            aiohttp.ClientResponse；退出 with 块时自动 release。
        """
        session = await self.get_session(backend_name, timeout)

        req_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else None

        response = await session.get(
            url,
            headers=headers,
            params=params,
            timeout=req_timeout,
        )
        try:
            yield response
        finally:
            response.release()

    async def __aenter__(self) -> "TTSSessionManager":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close_session()