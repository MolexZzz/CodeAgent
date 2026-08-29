# MemCodeAgent

MemCodeAgent 是一个面向本地仓库的轻量级命令行编程智能体。它通过大语言模型的原生工具调用能力检查代码、修改文件、执行命令，并在指定工作区中迭代验证变更。

项目中的智能体运行时、上下文管理、工具执行、安全检查和本地项目记忆均由本仓库实现。模型调用使用标准的 OpenAI 兼容 Chat Completions API。

## 功能特性

- 面向软件工程任务的交互式命令行界面。
- 使用原生 LLM 工具调用完成文件查看、搜索、编辑、补丁和命令执行。
- 校验工作区边界，避免文件操作越出指定项目。
- 默认在编辑文件和执行命令前请求确认。
- 提供只读的 `/plan` 与 `/explain` 命令，用于设计和代码理解任务。
- 使用 ReAct 风格循环：记录工具结果后请求模型给出下一步操作。
- 代码变更后自动检测并运行测试命令。
- 默认保护任务开始前已经存在的测试文件。
- 在 `.memcode/` 下持久化会话历史和轻量级项目记忆。
- 支持配置上下文窗口、重试预算和工具调用预算。
- 通过环境变量支持 OpenAI 兼容的模型服务。

## 环境要求

- Python 3.11 或更高版本
- 已配置的模型服务 API Key

## 安装

克隆仓库后，以可编辑模式安装项目：

```bash
python -m pip install -e ".[dev]"
```

该命令会安装 CLI 命令 `mca` 以及开发依赖，其中包括 `pytest`。

## 配置

复制示例配置文件，并填写模型服务的凭据：

```bash
copy .env.example .env
```

在 macOS 或 Linux 上：

```bash
cp .env.example .env
```

至少在 `.env` 中配置 API Key 和模型名称：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
MEMCODE_MODEL=gpt-4o-mini
```

`OPENAI_BASE_URL` 为可选项。使用 OpenAI 兼容接口时，将其设置为对应服务的 API 地址。`.env.example` 中提供了其他服务商的配置示例。

也可以不创建 `.env`，而是通过环境变量配置：

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:MEMCODE_MODEL = "gpt-4o-mini"
```

请勿提交包含真实凭据的 `.env` 文件；该文件默认已被 Git 忽略。

## 使用方式

在当前目录启动交互式会话：

```bash
mca
```

为指定仓库启动会话：

```bash
mca chat --workspace D:\path\to\project
```

执行单次任务：

```bash
mca run "为用户名为空的情况添加校验，并运行相关测试。" --workspace .
```

构建或刷新本地项目记忆索引：

```bash
mca index --workspace .
```

常用选项：

```bash
# 仅生成计划，不修改文件或执行命令。
mca chat --dry-run

# 对已信任的工作区关闭逐工具确认。
mca chat --no-approve

# 允许智能体修改任务开始前已有的测试文件。
mca chat --no-protect-tests

# 提高任务的错误重试次数。
mca chat --max-error-retries 15
```

## 交互命令

在 `mca chat` 会话中可使用以下命令：

| 命令 | 说明 |
| --- | --- |
| `/plan <任务>` | 生成只读的实现计划。 |
| `/explain <问题>` | 基于工作区给出只读说明。 |
| `/models` | 列出已知模型及其凭据状态。 |
| `/model [名称]` | 选择一个已配置的模型。 |
| `/workspace [路径]` | 显示或切换工作区。 |
| `/context` | 显示上下文窗口统计信息。 |
| `/tokens` | 显示当前会话的 Token 用量。 |
| `/save` | 持久化当前会话。 |
| `/clear` | 清除已持久化的会话记录。 |
| `/help` | 列出可用命令。 |

普通的实现请求会进入智能体工作流：模型选择下一步操作，CLI 在策略检查后于本地执行该操作，将结果追加到对话中，并持续循环，直到任务完成、暂停或达到预算限制。

## 工作原理

对于编程任务，MemCodeAgent 在本地执行以下循环：

```text
用户请求
  -> 创建任务上下文并检索相关项目记忆
  -> 请求模型给出下一步操作
  -> 校验并执行请求的本地工具
  -> 将工具结果追加到对话
  -> 重复执行，直至完成、暂停、得到验证结果或达到预算上限
```

智能体可使用的本地工具包括：

- 列出文件和读取文件内容
- 在工作区内搜索文本
- 写入文件和应用补丁
- 执行经确认的 Shell 命令
- 检查相关仓库变更

所有工具操作均受所选工作区限制。策略层会识别可能删除文件、访问网络或改变环境的命令，以便执行前进行审查。

## 本地数据

MemCodeAgent 会在选定工作区下写入运行状态：

```text
.memcode/
  memory.json
  session.json
```

这些文件保存任务摘要、检索元数据和会话状态，属于本地运行产物，默认被 Git 忽略。

## 开发

运行完整测试：

```bash
pytest
```

运行指定测试模块：

```bash
pytest tests/test_agent_workflow.py
```

## 安全说明

- 将模型服务凭据保存到环境变量或未跟踪的 `.env` 文件中。
- 批准命令前请确认其用途，尤其是安装依赖、访问网络、删除文件或修改版本控制状态的命令。
- 仅在信任工作区和任务内容时使用 `--no-approve`。
- 智能体会操作本地文件和命令，提交前应审阅其生成的变更。

## 许可证

本项目采用 [MIT License](LICENSE) 发布。
