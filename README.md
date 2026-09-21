# Style-Bert-VITS2 (CUDA) 日文语音合成插件

MaiBot 的文本转语音插件，调用本地 **Style-Bert-VITS2 (CUDA)** 推理服务合成日文语音。Style-Bert-VITS2 是日文推理模型，因此插件会先把输入文本翻译为自然日文，再送入本地推理服务合成语音，输出自然、地道的日文语音。

> **v1.1.0** — 基于 MaiBot SDK 2.x 构建：原生模型/任务直连、对接 Style-Bert-VITS2 (CUDA) API（`/voice`）、Profile 音色配置、全量 base64 语音投递。新增可选「语音前回显中文原文」（仅 @Tool，默认关闭），并把「智能分句」默认改为关闭。

---

## 前置条件

本插件不内置、也不自动管理 Style-Bert-VITS2 推理服务进程，需要你**自行部署并手动启动**本地推理服务：

### 1. 下载与解压模型包
请前往 ModelScope 官方模型仓库下载 Style-Bert-VITS2 (CUDA) 整合包：
👉 **[Style-Bert-VITS2-CUDA (ModelScope)](https://www.modelscope.cn/models/lingchat-research-studio/Style-Bert-VITS2-CUDA)**

下载后解压至本地任意目录，目录结构示例如下：
```
Style-Bert-VITS2-CUDA/
├── ..01 启动API服务.bat      # 启动 FastAPI 推理服务
├── config.yml               # 服务端配置（默认 limit=100）
├── server_fastapi.py        # API 服务入口
└── model_assets/            # 已加载的语音模型
    ├── Ling-v2/             # 对应音色 Ling v2
    └── Fusetsu-v1.5/        # 对应音色 Fusetsu_v1.5
```

### 2. 启动推理服务
1. 运行解压目录下的 `..01 启动API服务.bat`。
2. 等待控制台输出 `server listen: http://127.0.0.1:5000`，确认服务已监听 `5000` 端口。
3. 确认推理服务已就绪（可访问 `http://127.0.0.1:5000/docs` 查看交互式 API 文档）。

---

## 安装与依赖

1. 将本插件放置在 MaiBot 的 `plugins/ggsfly_sbv2-tts-CUDA-plugin` 目录下。
2. 依赖项为 `aiohttp`（`>=3.8.0`），MaiBot 运行时环境默认携带，无需额外安装。

---

## 配置说明

编辑插件目录下的 `config.toml`：

```toml
[plugin]
enabled = true
config_version = "1.1.0"

[general]
timeout = 60
# 单段合成文本的最大字符数。Style-Bert-VITS2 服务端 limit 默认 100，超过会返回 422。
# 插件会自动按该长度对长段做二次切分（clamp）。
max_text_length = 100
strip_voice_placeholder = true    # 剥离 replyer 正文中的 [语音消息] 占位回声
split_sentences = false           # 智能分句（默认关闭）；开启后按标点逐段合成
split_delay = 0.3                 # 句子之间的发送间隔（秒）
send_error_messages = true        # 合成失败时是否向聊天流回显错误提示
echo_original_text = false        # 仅 @Tool：发语音前回显一条整段中文原文（默认关闭）
translate_to_japanese = true      # 是否先把文本翻译为日文再合成
# 翻译用 LLM。支持两种填法：
#   1) 任务名：replyer / utils / planner / memory 等
#   2) 模型名 / 模型标识符：model_config.toml 中 [[models]] 的 name 或 model_identifier
# 留空 = replyer 任务。无效模型直传 Host 解析，报错即时暴露。
translate_model = ""

[components]
tool_enabled = true               # LLM 自主触发的 Tool 组件（sbv2_tts_tool）
command_enabled = true            # 用户手动 /sbv2 命令（sbv2_tts_command）

[voice]
# Style-Bert-VITS2 (CUDA) API 地址，需指向 /voice 端点
api_url = "http://127.0.0.1:5000/voice"
# 默认音色名（取自 voices 列表中某项）
default_voice = "Ling v2"
# 文本语言：JP / EN / ZH（默认 JP）
language = "JP"
# 语速，基准 1.0，越大越慢
length = 1.0
# 可选音色列表。-v 参数与 default_voice 从该列表匹配
voices = ["Ling v2", "Fusetsu_v1.5"]
```

### 配置字段说明

| 段 | 字段 | 说明 |
|---|---|---|
| `[plugin]` | `enabled` | 是否启用插件（默认 `true`） |
| `[plugin]` | `config_version` | 配置文件版本号，勿改（`"1.1.0"`） |
| `[general]` | `timeout` | 请求推理服务的超时时间（秒） |
| `[general]` | `max_text_length` | 单段合成的最大字符数，对齐服务端 `limit=100` |
| `[general]` | `strip_voice_placeholder` | 剥离 replyer 回复正文中的 `[语音消息]` 占位回声，默认 `true` |
| `[general]` | `split_sentences` | 是否按标点切分句子逐段合成发送（**默认关闭**） |
| `[general]` | `split_delay` | 分句发送间隔（秒） |
| `[general]` | `send_error_messages` | 合成或翻译失败时是否回显错误文本 |
| `[general]` | `echo_original_text` | **仅 @Tool**：发语音前回显一条整段中文原文（不分割、不回 planner；**默认关闭**，`translate_to_japanese=false` 时不生效） |
| `[general]` | `translate_to_japanese` | 是否先翻译为日文再合成；建议保持 `true` |
| `[general]` | `translate_model` | 翻译用 LLM：任务名或具体模型名。留空 = `replyer` 任务 |
| `[components]` | `tool_enabled` | 是否启用 `sbv2_tts_tool` Tool |
| `[components]` | `command_enabled` | 是否启用 `/sbv2` Command |
| `[voice]` | `api_url` | 推理服务端点，默认 `http://127.0.0.1:5000/voice` |
| `[voice]` | `default_voice` | 默认音色名称 |
| `[voice]` | `language` | 合成语种代码（`JP` / `EN` / `ZH`） |
| `[voice]` | `length` | 语速调节，默认 1.0 |
| `[voice]` | `voices` | 可选音色名称列表（字符串数组，完美适配 WebUI） |

---

## 使用方法

### 命令触发（用户手动）

```bash
/sbv2 你好世界                      # 使用默认音色（Ling v2）
/sbv2 こんにちは -v Fusetsu_v1.5    # 指定音色 Fusetsu_v1.5
/voice 今天天气不错                 # /sbv2 的等价别名
/sbv2 help                          # 查看帮助与当前所有可用音色
```

- 若传入未知音色，插件会明确返回错误信息并列出当前所有可用音色（不静默回退，防止音色偏离用户预期）。

### 自动触发（LLM Agent 规划）

当 Agent 决定进行日文语音回复时，会自主调用 `sbv2_tts_tool`。可通过 `[components].tool_enabled` 开启或关闭。

---

## 智能分句与防超限截断

1. **智能切分**：`|||SPLIT|||` 显式标记优先切分 > 标点符号自动切句 > 单句直接合成。
   > 注：**标点自动切句由 `split_sentences` 控制，默认关闭**；关闭时（无 `|||SPLIT|||`）整段作为单句合成。
2. **超限保护（Clamp）**：由于 Style-Bert-VITS2 服务端设置了 `limit=100` 字符上限，插件在分句后会对任何超过 `max_text_length` 的无标点长文本进行安全分块截断，彻底避免服务端返回 422 错误。

---

## （可选）语音前回显中文原文

针对"部分成员听不懂日文语音"的场景，插件提供 `echo_original_text` 开关（**默认关闭**）：

- **仅作用于 @Tool**（planner 自主调用 `sbv2_tts_tool` 发语音时）；`/sbv2`、`/voice` 手动命令**不会**回显，避免刷屏。
- 在**翻译成功、确有语音要发**之后、投递第一条语音之前，用 `send.text` 发出**一条整段中文原文**。
- **不分割**：回显内容是切分前的完整 `clean_text`（`|||SPLIT|||` 规整为换行），永远一条，不受 `split_sentences` 影响。
- **不返回 planner**：发送时置 `sync_to_maisaka_history=False`，该条中文不写入 maisaka 历史，planner 看不到、不会据此二次生成，避免重复刷屏。
- 依赖 `translate_to_japanese`：关闭翻译（中文语音）时该回显自动不生效。
- 开启方式：WebUI 勾选 `[general].echo_original_text`，或在 `config.toml` 设为 `true`。

---

## 权限与能力说明

本插件在 `_manifest.json` 中声明了以下必要权限（`capabilities`），各能力的使用场景如下：

| 能力标识 | 使用场景与说明 |
|---|---|
| `send.custom` | 核心语音投递能力：用于向聊天流投递 `voice` 类型 Base64 音频消息 |
| `send.text` | 文本投递：用于命令帮助、未知音色/合成失败提示，以及 `echo_original_text` 开启时在语音前回显中文原文（不回 planner） |
| `llm.generate` | 中译日核心能力：在合成前将中文文本翻译为自然日文（支持直选模型与任务分发） |
| `llm.get_available_models` | 任务名动态校验：用于识别 `translate_model` 所填项是模型任务名还是具体模型标识符 |
| `component.disable` | 组件动态管理：在插件加载时根据配置的 `tool_enabled` / `command_enabled` 开关按需禁用对应组件 |

---

## 故障排查

### 1. 启动失败或完全静默无响应
- **检查配置是否开启**：确认插件目录下生成的 `config.toml` 中 `[plugin].enabled = true`。
- **检查 SDK 依赖**：确认当前 MaiBot 运行环境的 `maibot-plugin-sdk` 版本满足 `>=2.8.1`。
- **检查组件日志**：查看 MaiBot 启动控制台输出，搜索 `[plugin.sbv2_tts]` 或 `ggsfly.sbv2-tts-cuda-plugin` 查找加载日志。

### 2. 语音合成失败：日文翻译失败
- **检查翻译模型配置**：若 `translate_model` 配置了特定模型名称，请确认宿主 `config/model_config.toml` 中已注册该模型且其 API 提供方连通；留空则默认回退至 `replyer` 任务。
- **排查错误文本**：插件会如实回显 Host 返回的错误信息（如额度耗尽、超时等），根据提示排查模型配置。

### 3. 语音合成网络错误 / 连接被拒
- **检查 API 服务**：确认已通过 `..01 启动API服务.bat` 成功启动本地 Style-Bert-VITS2 服务，且控制台输出 `server listen: http://127.0.0.1:5000`。
- **检查端口与地址**：确认 `config.toml` 中 `[voice].api_url` 指向正确的地址（默认 `http://127.0.0.1:5000/voice`）。
- **浏览器访问测试**：在浏览器打开 `http://127.0.0.1:5000/docs`，若无法打开则说明本地推理服务未正常运行。

### 4. 服务端返回 422 错误（参数错误）
- **音色不存在**：若使用 `/sbv2 -v <音色>` 指定了未知音色，请先使用 `/sbv2 help` 查看当前配置支持的音色名称。
- **单段文本过长**：Style-Bert-VITS2 默认对单次输入字符有硬性上限（默认 100 字符），插件已开启自动截断保护；若自行修改了服务端的 `limit`，请同步调整插件的 `max_text_length`。

---

## 测试方式

本项目测试精简合并至单文件 `tests/test.py`，完整覆盖 utils、翻译层、后端参数组装与 manifest 不变量校验：

```bash
uv run pytest plugins/ggsfly_sbv2-tts-CUDA-plugin/tests/test.py -q
```

---

## 项目结构

```
ggsfly_sbv2-tts-CUDA-plugin/
├── _manifest.json           # 插件 manifest v2 (id: ggsfly.sbv2-tts-cuda-plugin)
├── .gitignore               # 忽略规则（包含 /config.toml）
├── config_keys.py           # 配置键常量定义
├── plugin.py                # 插件入口（MaiBotPlugin + Tool / Command / Hook，内置配置模型）
├── translate.py             # 中译日翻译层（SDK 2.x task_name/model_name 驱动）
├── README.md                # 插件使用文档
├── LICENSE                  # GPL-3.0-or-later
├── backends/                # 后端适配层
│   ├── __init__.py
│   ├── base.py              # TTSResult 与基类定义（base64 投递）
│   └── voice.py             # Style-Bert-VITS2 /voice HTTP 客户端与 VoiceProfile
├── utils/                   # 通用工具
│   ├── __init__.py
│   ├── file.py              # base64 编解码与数据校验
│   ├── session.py           # aiohttp ClientSession 复用池
│   └── text.py              # 文本清理、语种检测、分句与 clamp 截断
└── tests/                   # 自动化测试
    └── test.py              # 全量自包含单元测试
```
