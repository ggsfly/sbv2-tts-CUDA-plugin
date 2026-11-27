# Style-Bert-VITS2 (CUDA) 日文语音合成插件

MaiBot 的文本转语音插件，调用本地 **Style-Bert-VITS2 (CUDA)** 推理服务合成日文语音。Style-Bert-VITS2 是日文推理模型，因此插件会先把文本翻译为自然日文，再送入本地推理服务合成语音，输出自然、地道的日文语音。

> **v1.3.0** — 自动回复改走 **ReplyExtension**：planner 在 `reply` 的 `plugin_options` 里选中 `sbv2_ja_voice` 时，本条回复正文（中文）照常发送，同时把它译为日文并追加一条语音；`/sbv2`、`/voice` 命令行为完全不变。删除了旧的 `sbv2_tts_tool` 工具与「语音前回显中文原文」开关（正文本身就是字幕）。最低版本要求提升为 MaiBot **1.4.8** / maibot-sdk **2.23.0**。
>
> **v1.2.0** — 基于 MaiBot SDK 2.x 构建：原生模型/任务直连、对接 Style-Bert-VITS2 (CUDA) API（`/voice`）、Profile 音色配置、全量 base64 语音投递。每条输入**一次性整段合成单条语音**（已移除智能分句 / `|||SPLIT|||` 分段 / 超长自动截断等分段子系统）。
>
> **v1.2.0 新增**：翻译引擎可切换（`[translate].engine`）——既可继续走宿主 LLM（`host_llm`），也可直连本地部署的 **Index-Translate** 翻译模型（`index_translate`，OpenAI 兼容端点，零 API 成本、低延迟，支持术语表硬约束与风格软约束）。
>
> **v1.2.1 调整**：翻译设置归拢至 `[translate]` 节（`translate_to_japanese`、`translate_model` 自 `[general]` 迁入）；配置节按数据流重排（通用 → 翻译 → 合成 → 组件开关），config.toml 生成顺序与 WebUI 显示顺序同步生效；`engine` 字段为 Literal 双值校验，WebUI 渲染为下拉选择。
>
> 各版本逐条变更见 [CHANGELOG.md](CHANGELOG.md)。

---

## 兼容版本

| 项 | 声明区间（manifest） | 本轮验证组合 |
|---|---|---|
| MaiBot Host | `host_application`: `1.4.8` ~ `1.4.99` | **1.4.8** |
| `maibot-plugin-sdk` | `sdk`: `2.23.0` ~ `2.23.99` | **2.23.0** |

- 下限是 ReplyExtension 路线的能力下限：`1.4.8` / `2.23.0` 起宿主才提供 reply 扩展（`plugin_options` 参数 schema、`prepare` / `before_send` 两阶段调用、组件 RPC 超时）。
- 上界钉在当前已验证的大版本族（Host `1.4.x` / SDK `2.23.x`）。宿主升到更新版本时不会被拒载，而是以兼容模式加载并给出「插件声明的最高支持版本」警告，提醒重新验证后再放宽声明，而不是静默放行到未知协议上。
- 加载后用 `/sbv2 help` 自检命令链路；自动回复链路的自检方式是让 planner 在 `reply` 里带上 `plugin_options`，日志里应出现扩展调用记录。

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

### 3.（可选）本地 Index-Translate 翻译服务（Linux）

若 `[translate].engine` 选 `index_translate`，需要自行部署 B 站 IndexTeam 开源的
**Index-Translate** 翻译模型（基于 Qwen3.5，专为中→日等翻译训练，支持术语表/
占位符等硬约束），以 llama.cpp 起 OpenAI 兼容端点。本插件同样**不自动管理**
该服务进程。

参考部署（GTX 1080 8GB 实测，可与 SBV2 服务共存）：

```bash
# 1) 获取 llama.cpp：直接用官方预编译 Vulkan 包（无需 CUDA Toolkit）
#    https://github.com/ggml-org/llama.cpp/releases 下载
#    llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz 解压即可

# 2) 下载模型（约 1.45GB，国内可用 hf-mirror.com）
curl -L -C - -o Index-Translate-2B.Q5_K_M.gguf \
  "https://hf-mirror.com/IndexTeam/Index-Translate-2B-GGUF/resolve/main/Index-Translate-2B.Q5_K_M.gguf"

# 3) 启动服务（端口用 8020，避开 MaiBot 主消息服务器的 8000）
llama-server -m Index-Translate-2B.Q5_K_M.gguf \
  --alias Index-Translate-2B -ngl 99 -c 16384 --jinja \
  --host 127.0.0.1 --port 8020
```

- 端点：`http://127.0.0.1:8020/v1`（OpenAI 兼容），模型名 `Index-Translate-2B`。
- 显存：整卡占用约 2.9GB（含桌面与 SBV2 进程），2B 模型稳态约 0.3~0.6 秒/条。
- 调用要点由插件自动处理：`temperature=0`、`chat_template_kwargs.enable_thinking=false`
  （否则思维链会混入译文）、输出剥掉思维链块。
- Windows 部署与官方免费 API（`https://index-translate.bilibili.com/v1`，同协议
  可直接填入 `[translate].base_url`）详见 Index-Translate 官方仓库：
  https://github.com/bilibili/Index-Translate

---

## 安装与依赖

1. 将本插件放置在 MaiBot 的 `plugins/ggsfly_sbv2-tts-CUDA-plugin` 目录下。
2. 依赖项为 `aiohttp`（`>=3.8.0`），MaiBot 运行时环境默认携带，无需额外安装。

---

## 两条路线

| 路线 | 触发方式 | 语音怎么发 |
|---|---|---|
| 命令 | 用户输入 `/sbv2` / `/voice` | 插件自己合成并通过 `ctx.send.custom("voice", base64)` 投递 |
| 自动回复 | planner 在 `reply` 的 `plugin_options` 里选中 `sbv2_ja_voice` | 插件把回复正文译为日文、合成语音后作为**追加消息**返回，由宿主在正文之后发送 |

**语音是按需的**：`[components].extension_enabled = true` 只表示「允许 planner 选这个扩展」，不代表每条回复都有语音。planner 没选 → 只发正文，**不翻译、不合成**。

---

## 配置说明

编辑插件目录下的 `config.toml`：

```toml
[plugin]
enabled = true
config_version = "1.3.0"

[general]
timeout = 60
# 译文长度上限（字符数），只注入翻译 prompt 约束日文译长。
# 它不改 Style-Bert-VITS2 服务端的 limit；超长不再自动截断，由后端如实返回 422。
max_text_length = 100
strip_voice_placeholder = true    # 剥离 replyer 正文中的 [语音消息] 占位回声
send_error_messages = true        # 命令路线合成失败时是否向聊天流回显错误提示

[translate]
# 命令路线是否免翻译：关闭后 /sbv2 把原文直接交给 language=JP 的后端合成；
# 自动回复的日语语音扩展始终走中译日引擎，不受本开关影响。
translate_to_japanese = true
# 翻译引擎：host_llm = 宿主 LLM（消耗大模型 API）；index_translate = 本地
# Index-Translate 服务（OpenAI 兼容端点，零 API 成本，部署见上文前置条件第 3 步）
engine = "host_llm"
# （仅 engine = host_llm）翻译用 LLM。支持两种填法：
#   1) 任务名：replyer / utils / planner / memory 等
#   2) 模型名：model_config.toml 中 [[models]] 的 name（须完全一致，宿主按 name 精确匹配，不认 model_identifier）
# 留空 = replyer 任务。无效模型直传 Host 解析，报错即时暴露。
translate_model = ""
base_url = "http://127.0.0.1:8020/v1"   # （仅 index_translate）OpenAI 兼容端点
model = "Index-Translate-2B"            # （仅 index_translate）模型名，与 llama-server --alias 一致
style = ""                              # （仅 index_translate）风格软约束，如"保持口语化、自然的直播说话风格，使用女性口吻"
glossary = []                           # （仅 index_translate）术语表硬约束，如 ["苦力怕→クリーパー"]
timeout = 30                            # （仅 index_translate）单次翻译超时（秒）
max_tokens = 512                        # （仅 index_translate）生成 token 上限

[voice]
# Style-Bert-VITS2 (CUDA) API 地址，需指向 /voice 端点
api_url = "http://127.0.0.1:5000/voice"
# 默认音色名（取自 voices 列表中某项）
default_voice = "Ling v2"
# 语速，基准 1.0，越大越慢
length = 1.0
# 可选音色列表。-v 参数与 default_voice 从该列表匹配
voices = ["Ling v2", "Fusetsu_v1.5"]

[components]
extension_enabled = true          # 允许 planner 在 reply.plugin_options 里选择 sbv2_ja_voice
command_enabled = true            # 用户手动 /sbv2 命令（sbv2_tts_command）
```

### 配置字段说明

| 段 | 字段 | 说明 |
|---|---|---|
| `[plugin]` | `enabled` | 是否启用插件（默认 `true`） |
| `[plugin]` | `config_version` | 配置文件版本号，勿改（`"1.3.0"`） |
| `[general]` | `timeout` | Style-Bert-VITS2 推理服务（`/voice`）请求超时（秒）；翻译请求超时见 `[translate].timeout` |
| `[general]` | `max_text_length` | 译文长度上限（只注入翻译 prompt，约束日文译长；不改服务端 `limit`，超长不再自动截断） |
| `[general]` | `strip_voice_placeholder` | 剥离 replyer 回复正文中的 `[语音消息]` 占位回声，默认 `true` |
| `[general]` | `send_error_messages` | **仅命令路线**：合成或翻译失败时是否回显错误文本（扩展路线的失败提示不受此开关影响） |
| `[translate]` | `translate_to_japanese` | **命令路线是否免翻译**；关闭后命令把清理后的原文交给 `language=JP` 的后端（扩展永远翻译） |
| `[translate]` | `engine` | 翻译引擎（**WebUI 下拉选择**）：`host_llm`（默认）/ `index_translate`；其他值在加载校验时直接报错 |
| `[translate]` | `translate_model` | 翻译用 LLM：任务名或具体模型名。留空 = `replyer` 任务（**仅 `host_llm` 引擎**） |
| `[translate]` | `base_url` | **仅 index_translate**：OpenAI 兼容端点，默认 `http://127.0.0.1:8020/v1` |
| `[translate]` | `model` | **仅 index_translate**：模型名，与 llama-server `--alias` 一致，默认 `Index-Translate-2B` |
| `[translate]` | `style` | **仅 index_translate**：风格软约束（prompt【注意】段），留空为无约束直译 |
| `[translate]` | `glossary` | **仅 index_translate**：术语表硬约束（prompt【硬性要求】），如 `["苦力怕→クリーパー"]` |
| `[translate]` | `timeout` | **仅 index_translate**：单次翻译请求超时（秒），默认 30 |
| `[translate]` | `max_tokens` | **仅 index_translate**：生成 token 上限，默认 512 |
| `[voice]` | `api_url` | 推理服务端点，默认 `http://127.0.0.1:5000/voice` |
| `[voice]` | `default_voice` | 默认音色名称 |
| `[voice]` | `length` | 语速调节，默认 1.0 |
| `[voice]` | `voices` | 可选音色名称列表（字符串数组，完美适配 WebUI） |
| `[components]` | `extension_enabled` | 允许 planner 选中 `sbv2_ja_voice`；关闭后扩展仍注册但被禁用，模型看不到该选项，命令不受影响 |
| `[components]` | `command_enabled` | 是否启用 `/sbv2` 命令组件 |

> 升级到 1.3.0 时由宿主按新骨架重建配置：自动补齐 `extension_enabled`（默认 `true`）、移除 `echo_original_text` 与 `tool_enabled`；旧 `tool_enabled = false` **不做迁移**，需要手动把 `extension_enabled` 设为 `false` 才能保持「没有自动语音」。

---

## 使用方法

### 路线一：命令触发（用户手动）

```bash
/sbv2 你好世界                      # 使用默认音色（Ling v2）
/sbv2 こんにちは -v Fusetsu_v1.5    # 指定音色 Fusetsu_v1.5
/voice 今天天气不错                 # /sbv2 的等价别名
/sbv2 help                          # 查看帮助与当前所有可用音色
```

- 若传入未知音色，插件会明确返回错误信息并列出当前所有可用音色（不静默回退，防止音色偏离用户预期）。
- 命令路线由插件自己投递语音，不经过 reply 扩展，也不受 `[components].extension_enabled` 影响。

### 路线二：自动回复（planner 选择扩展）

planner 调用官方 `reply` 工具时，把扩展名填进 `plugin_options`：

```json
{"plugin_options":{"ggsfly.sbv2-tts-cuda-plugin.sbv2_ja_voice":{}}}
```

想指定音色时带上 `voice`：

```json
{"plugin_options":{"ggsfly.sbv2-tts-cuda-plugin.sbv2_ja_voice":{"voice":"Fusetsu_v1.5"}}}
```

完整示例（`chat_anlyze` 是 `reply` 的必填项，不能为空）：

```json
{
  "chat_anlyze": "对方说想听一句日语晚安，适合一条短语音回应。",
  "reply_style": "简短表达",
  "plugin_options": {
    "ggsfly.sbv2-tts-cuda-plugin.sbv2_ja_voice": {}
  }
}
```

- **正文即字幕**：replyer 写出的中文正文照常发送，扩展把同一段正文译为日文后合成语音，**追加在已有消息之后**，不改动正文与引用设置。
- **朗读的是后处理前的正文**：扩展拿到的是过完 `maisaka.reply.before_post_process` 的 `text`，**不是**拆分 / 错字替换 / 附件编排之后的逐字字幕。
- 需要更短的台词时用 `reply` 自带的 `reply_style`（`简短表达` / `正常回复` / `长回复`），插件不往任何共享字段里塞写作指令。
- 扩展失败时只向当前聊天发一条错误提示，然后原样发送正文（**没有语音**）。提示文案形如 `本次未附加日语语音：日文翻译失败。`。

#### 与内置 `attach_voice` 的冲突（重要）

**不要同时选择本扩展和内置的 `attach_voice`。** 内置于 reply 的语音附件会在扩展已追加语音后检查「本次回复是否已包含语音附件」，命中即抛错（`本次回复已包含语音附件`），导致**整条回复在发送前失败**——不是「重复播报」，而是正文也发不出去。

---

## 合成方式：一次性整段合成

插件**不做任何分句 / 分段 / 超长自动截断**：每条输入整段翻译为日文后，一次性合成为**单条语音**。

- 译文长度由 `general.max_text_length`（注入翻译 prompt）约束，引导模型输出不超过约 100 字符；它不改服务端 `limit`。
- 若译文仍超过服务端 `limit`，**不再自动截断**，而是由后端如实返回 422 错误并回显（命令路线）或作为失败提示（扩展路线）。请缩短输入或调整服务端 `limit`。

---

## 权限与能力说明

本插件在 `_manifest.json` 中声明了以下必要权限（`capabilities`），各能力的使用场景如下：

| 能力标识 | 使用场景与说明 |
|---|---|
| `send.custom` | 命令路线的语音投递：向聊天流投递 `voice` 类型 Base64 音频消息（扩展路线由宿主发送，插件不调） |
| `send.text` | 文本投递：命令帮助、缺文本/未知音色提示；扩展路线的失败通知（`sync_to_maisaka_history=False`、`storage_message=False`） |
| `llm.generate` | 中译日核心能力：合成前把中文文本翻译为自然日文（支持直选模型与任务分发） |
| `llm.get_available_models` | 任务名动态校验：识别 `translate_model` 所填项是任务名还是具体模型名 |
| `component.enable` / `component.disable` | 组件动态管理：按 `extension_enabled` / `command_enabled` 启用或禁用 `sbv2_ja_voice`（`REPLY_EXTENSION`）与 `sbv2_tts_command`（`COMMAND`） |

---

## 故障排查

### 1. 启动失败或完全静默无响应
- **检查配置是否开启**：确认插件目录下生成的 `config.toml` 中 `[plugin].enabled = true`。
- **检查 SDK 依赖**：确认当前 MaiBot 运行环境的 `maibot-plugin-sdk` 版本满足 `>=2.23.0`、宿主 `>=1.4.8`。
- **检查组件日志**：查看 MaiBot 启动控制台输出，搜索 `[plugin.sbv2_tts]` 或 `ggsfly.sbv2-tts-cuda-plugin` 查找加载日志。

### 2. 语音合成失败：日文翻译失败
- **检查翻译引擎**：`[translate].engine = index_translate` 时，确认本地 llama-server 已启动（`curl http://127.0.0.1:8020/v1/models`）；服务未启动时插件加载日志会有「Index-Translate 翻译服务未连通」warning。
- **检查翻译模型配置**（host_llm 引擎）：若 `translate_model` 配置了特定模型名称，请确认宿主 `config/model_config.toml` 中已注册该模型且其 API 提供方连通；留空则默认回退至 `replyer` 任务。
- **排查错误文本**：插件会如实回显 Host / 翻译服务返回的错误信息（如额度耗尽、超时、HTTP 504 等），根据提示排查模型配置或翻译服务状态。

### 3. 语音合成网络错误 / 连接被拒
- **检查 API 服务**：确认已通过 `..01 启动API服务.bat` 成功启动本地 Style-Bert-VITS2 服务，且控制台输出 `server listen: http://127.0.0.1:5000`。
- **检查端口与地址**：确认 `config.toml` 中 `[voice].api_url` 指向正确的地址（默认 `http://127.0.0.1:5000/voice`）。
- **浏览器访问测试**：在浏览器打开 `http://127.0.0.1:5000/docs`，若无法打开则说明本地推理服务未正常运行。

### 4. 服务端返回 422 错误（参数错误）
- **音色不存在**：若使用 `/sbv2 -v <音色>` 或 `plugin_options.voice` 指定了未知音色，请先使用 `/sbv2 help` 查看当前配置支持的音色名称。
- **译文过长**：Style-Bert-VITS2 对单次输入有字符上限（默认 100）。插件不再自动截断，超长会如实返回 422；请缩短输入文本，或同步调整服务端 `limit` 与插件的 `max_text_length`（后者只约束译文生成长度）。

### 5. 自动回复只有正文、没有语音
- 确认 `[components].extension_enabled = true`（关闭时扩展被禁用，模型看不到 `plugin_options` 里的选项）。
- 确认 planner 本轮确实带了 `plugin_options`（没选就是没有语音，属预期）。客户端会先收中文正文，再收一条日文语音。
- 平台可能**部分成功**：前面的文字已发出、后面的语音才失败——此时会看到错误提示，但正文已经送达。

---

## 测试方式

本项目测试精简合并至单文件 `tests/test.py`，覆盖 utils、翻译层、后端合成/投递拆分、命令路线失败文案精确回归、ReplyExtension 路线、配置模型与重建、manifest 不变量：

```bash
uv run pytest plugins/ggsfly_sbv2-tts-CUDA-plugin/tests/test.py -q
```

---

## 项目结构

```
ggsfly_sbv2-tts-CUDA-plugin/
├── _manifest.json           # 插件 manifest v2 (id: ggsfly.sbv2-tts-cuda-plugin)
├── CHANGELOG.md             # 更新日志（manifest changelog 指向本文件）
├── .gitignore               # 忽略规则（包含 /config.toml）
├── config_keys.py           # 配置键常量定义
├── plugin.py                # 插件入口（MaiBotPlugin + ReplyExtension / Command / Hook，内置配置模型）
├── translate.py             # 中译日翻译层（JPTranslator 宿主 LLM / IndexTranslateTranslator 本地服务双引擎）
├── README.md                # 插件使用文档
├── LICENSE                  # MIT
├── backends/                # 后端适配层
│   ├── __init__.py
│   ├── base.py              # TTSResult 与基类定义（base64 投递）
│   └── voice.py             # Style-Bert-VITS2 /voice HTTP 客户端与 VoiceProfile（synthesize 只合成，execute 合成+投递）
├── utils/                   # 通用工具
│   ├── __init__.py
│   ├── file.py              # base64 编解码与数据校验
│   ├── session.py           # aiohttp ClientSession 复用池
│   └── text.py              # 文本清理、语种检测
└── tests/                   # 自动化测试
    └── test.py              # 全量自包含单元测试
```
