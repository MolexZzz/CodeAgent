# CodeAgent

CodeAgent 是一个面向本地项目的命令行编程助手。你给它任务，它会自己查代码、动文件、执行命令，并在当前工作区里把变更验证到位。

## 主要能力

- 命令行交互式工作流
- 本地文件读取、搜索、写入和补丁应用
- 工作区边界校验，避免越界读写
- 命令执行前的安全审查与确认
- 任务历史、项目规则、代码索引三层本地记忆
- 代码修改后的自动验证流程
- 会话可恢复，支持中断后继续

## 安装

```bash
python -m pip install -e ".[dev]"
```

安装后可使用 `mca` 命令。

## 配置

复制示例配置并填写模型凭据：

```bash
copy .env.example .env
```

至少配置模型 Key 和模型名，例如：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
MEMCODE_MODEL=gpt-4o-mini
```

如果使用其他兼容服务，改成对应的 `*_API_KEY` 和 `*_BASE_URL` 即可。

## 使用方式

启动默认交互会话：

```bash
mca
```

执行单次任务：

```bash
mca run "修复这个测试失败并补充测试" --workspace .
```

重建本地代码索引：

```bash
mca index --workspace .
```

`mca index --workspace` 的作用是扫描工作区里的代码文件。目前 Java 和 Python 会进入本地索引，享受 BM25、向量检索和 graph
  扩展带来的加速；其他语言不进入索引，仍通过常规的本地读取和搜索工具按需处理。它只更新 .memcode/ 里的索引文件。通常在大
  范围改动代码后、或者首次进入新仓库时运行一次就够了。

## 工作原理

CodeAgent 的执行链路：

```text
用户任务
  -> 解析任务并检索相关代码/历史/项目规则
  -> 请求模型决定下一步动作
  -> 本地执行工具调用
  -> 把工具结果写回上下文
  -> 继续循环，直到完成、暂停、失败或达到预算上限
```

主要由这几层组成：

- `cli.py`：提供 `mca`、`run`、`chat`、`index`
- `agent.py`：组织任务循环、审批、验证和结果收尾
- `controller.py`：控制模型步进、状态机和工具回放
- `llm.py`：封装 OpenAI 兼容 Chat Completions 调用
- `tools.py`：本地工具执行器
- `workspace.py`：工作区边界和安全校验
- `memory_manager.py`：统一管理上下文、任务记忆和项目记忆

## 本地数据

CodeAgent 会在工作区下写入 `.memcode/`：

```text
.memcode/
  memory.json
  project_memory.json
  session.json
  transcript.jsonl
  code_index.json
  code_vectors.npy
```

- `memory.json`：任务历史、修改文件、失败命令、经验摘要
- `project_memory.json`：项目级规则或长期提示
- `session.json`：可恢复的会话快照
- `transcript.jsonl`：追加式对话记录
- `code_index.json` / `code_vectors.npy`：本地代码索引和向量缓存

## 安全说明

- 模型凭据放在环境变量或未跟踪的 `.env` 中
- 命令执行前请确认用途，尤其是删除、网络、安装和版本控制操作
- 只在信任工作区时关闭批准流程

## 许可证

MIT
