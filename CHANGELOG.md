# 更新日志

## v1.2.2

适配 MaiBot 1.4.8 / maibot-sdk 2.23.0。

### 主要功能与修复

- **[hook] 修复 `strip_voice_placeholder` 完全失效**：`maisaka.reply.before_post_process` 的 `modified_kwargs` 在宿主侧是**整体替换**而非增量合并（`src/plugin_runtime/host/hook_dispatcher.py`），原实现只回传 `response` 一个键，导致 `skip_post_process` / `enable_splitter` / `enable_chinese_typo` 丢失，宿主读取布尔值时抛错并降级为默认策略，剥离从未真正生效。现改为回传 `{**kwargs, "response": ...}`。
- **[send] 语音投递失败不再被吞掉**：`TTSBackendBase.send_audio` 原先把 `send.custom` 的返回布尔值丢弃，宿主拒发时仍返回成功并回显「成功发送语音」。现在检查返回值，失败时返回失败 `TTSResult` 并如实暴露给聊天流。
- **[llm] 修复任务名缓存写反**：`_resolve_translate_llm` 在 `get_available_models` 失败时写入空集合，导致此后进程内所有 `translate_model` 任务名都被当作模型名直传；现按注释保持未缓存状态，下次调用重试。
- **[component] 放宽组件 RPC 超时**：宿主组件调用默认 60s，而本插件最慢合法路径为「翻译 30s + 合成 60s」，`Tool` / `Command` 现显式声明 `timeout_ms=120000`。

### 插件 SDK / API 改动

- **[typing] 收窄 `config` 属性返回类型**：新增 `SBV2TTSPlugin.config -> SBV2TTSPluginConfig` 覆写，消除基类标注 `PluginConfigBase` 带来的 34 处静态类型错误。

### 元数据与文档

- **[manifest] 补齐现行字段**：新增 `display.icon`（lucide `audio-lines`）与 `changelog`；`host_application.min_version` 由 `1.0.0` 收敛为 `1.2.5`（宿主显式 `task_name` / `model_name` 路由与 `maisaka.reply.before_post_process` 钩子的可用下限）。
- **[docs] 新增本文件与 README「兼容版本」小节**，明确验证组合为 MaiBot 1.4.8 + SDK 2.23.0。
- **[test] 补充回归断言**：Hook 回传全量 kwargs、`send_audio` 失败传播、manifest 新字段与版本一致性。

## v1.2.1

- 翻译设置归拢至 `[translate]` 节（`translate_to_japanese`、`translate_model` 自 `[general]` 迁入）。
- 配置节按数据流重排（通用 → 翻译 → 合成 → 组件开关），`config.toml` 生成顺序与 WebUI 显示顺序同步生效。
- `translate.engine` 改为 `Literal["host_llm", "index_translate"]`，WebUI 渲染为下拉选择；非法值在加载期直接报错。
- 移除 `voice.language` 配置（当前模型仅支持日文输入，参数固定为 `JP`）。
- 协议由 GPL-3.0 改为 MIT。

## v1.2.0

- 新增本地 **Index-Translate** 翻译引擎（`[translate].engine = index_translate`），OpenAI 兼容端点，零 API 成本、低延迟，支持术语表硬约束与风格软约束。
- 翻译引擎可在宿主 LLM（`host_llm`）与本地服务之间切换。

## v1.1.1

- 移除智能分段子系统，改为一次性整段合成单条语音；超长不再自动截断，由后端如实返回 422。

## v1.1.0

- 语音前可选回显中文原文（仅 `@Tool`，不回 planner）；智能分句默认关闭。

## v1.0.0

- 首个正式版本：`Tool` + `Command` 双触发面，宿主 LLM 中译日，`ctx.send.custom("voice", base64)` 投递。
