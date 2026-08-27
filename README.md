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

安装：

```bash
python -m pip install -e .[dev]
```

配置环境变量（根据你使用的平台选择）：

**PowerShell:**
```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选，默认为 OpenAI 官方
$env:MEMCODE_MODEL="gpt-4o-mini"  # 可选，默认为 gpt-4o-mini
```

**cmd (Windows):**
```cmd
set OPENAI_API_KEY=your-key
set OPENAI_BASE_URL=https://api.openai.com/v1
set MEMCODE_MODEL=gpt-4o-mini
```

**bash/zsh (Linux/macOS):**
```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export MEMCODE_MODEL="gpt-4o-mini"
```

运行：

```bash
# 单次任务执行
mca run "阅读这个项目并总结下一步实现方向"

# 交互式 REPL
mca chat
```

### 环境变量说明

| 变量 | 必需 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | 是 | API 密钥（OpenAI 或兼容服务） |
| `OPENAI_BASE_URL` | 否 | API 端点地址，默认为 `https://api.openai.com/v1` |
| `MEMCODE_MODEL` | 否 | 模型名称，默认为 `gpt-4o-mini` |

### 使用其他服务商

本项目支持任何 OpenAI 兼容的 API 服务，包括但不限于：

**DeepSeek:**
```bash
export OPENAI_API_KEY="your-deepseek-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export MEMCODE_MODEL="deepseek-chat"
```

**月之暗面 Kimi:**
```bash
export OPENAI_API_KEY="your-kimi-key"
export OPENAI_BASE_URL="https://api.moonshot.cn/v1"
export MEMCODE_MODEL="moonshot-v1-8k"
```

**Azure OpenAI:**
```bash
export OPENAI_API_KEY="your-azure-key"
export OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
export MEMCODE_MODEL="gpt-4"
```

**本地部署 (Ollama/vLLM 等):**
```bash
export OPENAI_API_KEY="dummy"  # 本地服务通常不需要真实密钥
export OPENAI_BASE_URL="http://localhost:8000/v1"
export MEMCODE_MODEL="qwen2.5-coder"
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
