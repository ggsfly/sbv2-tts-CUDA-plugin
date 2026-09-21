"""
SBV2 TTS 插件工具层

存放与 TTS 后端无关的通用工具类：
- text：文本清理、语言检测
- file：异步文件 IO、临时路径生成、音频数据校验与 base64 编码
- session：aiohttp HTTP 会话单例与连接池
"""

import sys

sys.dont_write_bytecode = True