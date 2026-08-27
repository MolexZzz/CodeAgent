# MemCodeAgent

轻量级命令行编程助手，带有项目记忆和代码检索功能。

MemCodeAgent 是为软件工程推免考核项目开发的：它与大语言模型交互，调用本地工具，读写文件，执行命令，观察结果，并迭代完成编程任务。核心的 agent 循环、工具执行、上下文管理、输出解析和记忆层均在本仓库中自行实现，未使用任何 agent 框架。

## 目标

- 提供清晰的命令行编程助手工作流
- 保持本地文件和命令执行的显式可检查性
- 添加基于关键词的项目记忆功能
- 实现简洁的代码检索机制
- 保持实现足够小，便于在面试中解释

## 快速开始

```powershell
python -m pip install -e .[dev]
$env:OPENAI_API_KEY="your-key"
$env:MEMCODE_MODEL="gpt-4o-mini"
mca run "阅读这个项目并总结下一步实现方向"
```

可选环境变量：

```text
OPENAI_API_KEY      调用真实模型所需
OPENAI_BASE_URL     可选的 OpenAI 兼容端点
MEMCODE_MODEL       模型名称，默认为 gpt-4o-mini
```

## 架构

```text
用户任务
  -> 检索项目记忆（关键词匹配）
  -> 询问 LLM 下一步操作（OpenAI 原生 tool calling）
  -> 执行本地工具（工作区沙盒化）
  -> 将观察结果追加到上下文
  -> 重复直到任务完成或达到最大步数
  -> 将任务摘要和元数据持久化到记忆
```

**已实现功能：**
- **原生 LLM 工具调用**：使用 OpenAI function-calling API，支持 6 个工具（list_files、read_file、search_text、write_file、apply_patch、run_command）
- **自行实现的 agent 循环**：对话管理、工具分发、终止条件、错误处理
- **本地工具执行**：所有文件/命令操作在本地运行，带工作区路径验证和危险命令拦截
- **轻量级记忆**：任务历史持久化到 `.memcode/memory.json`，通过 token 重叠评分检索（无向量数据库）
- **多编辑补丁**：`apply_patch` 支持多个顺序替换，输出统一 diff
- **Typer CLI**：`mca run <任务>` 命令，支持工作区选择、步数限制和 dry-run 模式
- **全面测试**：29 个单元测试覆盖工具执行、记忆持久化/检索和 LLM 消息解析

## 安全说明

API 密钥必须通过环境变量或本地未入库的配置文件提供。请勿提交凭据、包含可见密钥的终端录屏或生成的密钥文件。

## 许可证

MIT
