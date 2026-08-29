# SBV2 日文语音合成插件

MaiBot 的文本转语音插件，调用本地 SBV2 推理服务合成日文语音。SBV2 是日文推理模型，因此插件会先把输入文本翻译为日文，再送入本地推理服务合成语音，输出自然、地道的日文语音。

> **v1.0.0** — 基于 MaiBot SDK 2.x（`MaiBotPlugin` + `@Action` / `@Command` + Pydantic 配置）。

## 前置条件

本插件不内置、也不自动管理 SBV2 推理服务进程，需要你**手动启动**本地推理服务：

1. 启动 SBV2 API 服务：
   ```
   F:\lingchat-research-studio--SBV2-API\snapshots\master\sbv2api-DirectML\sbv2_api.exe
   ```
2. 等待日志出现 `Listening on 0.0.0.0:3000`，确认服务已监听 `3000` 端口。
3. 确认推理服务内已放入以下语音模型：
   - `Ling v2`（默认音色）
   - `Fusetsu_v1.5`

> 提示：若推理服务未启动或地址不通，合成会失败并触发错误提示，请先检查服务进程。

## 安装

插件随 MaiBot 一起内置，无需额外安装。依赖项为 `aiohttp`（`>=3.8.0`）。

## 配置

编辑 `plugins/sbv2-tts-plugin/config.toml`：

```toml
[plugin]
enabled = true                    # 是否启用插件
config_version = "1.0.0"          # 配置文件版本，勿改

[general]
timeout = 60                      # 请求超时（秒）
max_text_length = 200             # 单次合成的最大文本长度
use_base64_audio = false          # true=base64 走 IPC；false=文件路径 voiceurl（更稳）
split_sentences = true            # 长文本分句逐段发送
split_delay = 0.3                 # 句子之间延迟（秒）
send_error_messages = true        # 合成失败时是否给用户提示
translate_to_japanese = true      # 是否先把文本翻译为日文再合成（SBV2 为日文推理模型）
translate_model = ""              # 翻译用 LLM 模型名，留空用 Host 默认任务模型

[components]
tool_enabled = true               # LLM 自主触发的 Tool 组件（sbv2_tts_tool）
command_enabled = true            # 用户手动 /sbv2 命令

[sbv2]
api_url = "http://127.0.0.1:3000/synthesize"   # SBV2 推理服务地址
default_ident = "Ling v2"         # 默认说话人
idents = ["Ling v2", "Fusetsu_v1.5"]   # 可选说话人列表
```

### 配置字段说明

| 段 | 字段 | 说明 |
|----|------|------|
| `[plugin]` | `enabled` | 是否启用插件 |
| `[plugin]` | `config_version` | 配置文件版本号，勿改 |
| `[general]` | `timeout` | 请求 SBV2 推理服务的超时时间（秒） |
| `[general]` | `max_text_length` | 单次合成的最大文本长度，超长文本会被约束 |
| `[general]` | `use_base64_audio` | 音频回传方式，`true` 以 base64 走 IPC，`false` 用文件路径 voiceurl |
| `[general]` | `split_sentences` | 是否按句子拆分逐段合成发送 |
| `[general]` | `split_delay` | 分句之间的发送间隔（秒） |
| `[general]` | `send_error_messages` | 合成失败时是否向聊天流回显错误提示 |
| `[general]` | `translate_to_japanese` | 是否先翻译为日文再合成；SBV2 为日文推理模型，建议保持 `true` |
| `[general]` | `translate_model` | 翻译用 LLM 模型名，留空则用 Host 默认任务模型 |
| `[components]` | `tool_enabled` | 是否启用 LLM 自主触发的 `sbv2_tts_tool` Tool |
| `[components]` | `command_enabled` | 是否启用用户手动 `/sbv2` 命令 |
| `[sbv2]` | `api_url` | SBV2 推理服务地址，需与前置条件中的服务一致 |
| `[sbv2]` | `default_ident` | 默认说话人，未指定音色时使用 |
| `[sbv2]` | `idents` | 可选说话人列表，`-v` 指定时从该列表选择 |

## 使用方法

### 命令触发（用户手动）

```
/sbv2 你好世界                    # 使用默认音色（Ling v2）
/sbv2 こんにちは -v Fusetsu_v1.5  # 指定音色 Fusetsu_v1.5
/sbv2 help                        # 查看帮助
```

- 不带 `-v` 参数时使用配置中的 `default_ident`（默认 `Ling v2`）。
- `-v` 指定的音色需在配置的 `idents` 列表中，否则回退到默认音色。

### 自动触发（LLM 决定）

当 LLM 判断需要语音回复时，会调用插件注册的 `sbv2_tts_tool` Tool 自动合成语音，无需用户手动输入命令。可通过 `[components]` 段的 `tool_enabled` 开关控制。

## 智能分割

本插件支持智能分割：`|||SPLIT|||` 标记精确分段，长文本自动分句。

- **优先级**：`|||SPLIT|||` 标记优先切分 > 按标点自动切分 > 单句发送
- **示例**：`今天天气不错|||SPLIT|||适合出去玩|||SPLIT|||你觉得呢` → 三段语音依次发送
- `split_sentences = false` 时关闭标点自动切分，但仍会按 `|||SPLIT|||` 标记分段。

## 已知限制

- **日文模型须先翻译**：SBV2 是日文推理模型，直接传入中文会输出乱码。插件默认开启 `translate_to_japanese`，先由 LLM 翻译为日文再合成。
- **翻译失败不发语音**：翻译失败时插件**绝不**静默回退到中文原文，而是通过日志 `logger.error` 完整暴露错误，避免用中文喂出乱码再让用户听到；`send_error_messages` 决定是否向聊天流回显错误。
- **不自动管理 SBV2 服务进程**：插件不会拉起/监控 `sbv2_api.exe`，需手动启动推理服务并保证端口可达。

## 项目结构

```
sbv2-tts-plugin/
├── _manifest.json           # 插件 manifest v2
├── .gitignore               # 忽略规则
├── config.toml              # 用户配置
├── config_keys.py           # 配置 key 常量
├── plugin.py                # 插件入口（MaiBotPlugin 子类 + Pydantic 配置模型）
├── translate.py             # 中文到日文的翻译层（失败不发语音）
├── README.md                # 本文件
├── LICENSE                  # GPL-3.0-or-later
├── utils/                   # 工具层
│   ├── __init__.py
│   ├── text.py              # 文本清理/语言检测/标点分句
│   ├── file.py              # 异步文件 IO / 临时文件
│   └── session.py           # aiohttp 单例 session
└── tests/                   # 单元测试
    ├── test_utils.py
    └── tmp/
```

## 信息

- **版本**：1.0.0
- **作者**：ggsfly
- **许可**：GPL-3.0-or-later
- **插件 ID**：`ggsfly.sbv2-tts-plugin`
