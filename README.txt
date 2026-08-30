Git 仓库地址：https://github.com/MolexZzz/CodeAgent.git

项目简介：
CodeAgent 是一个轻量级命令行编程智能体。它通过大语言模型决定下一步动作，但文件读写、文本搜索、命令执行、上下文管理、模型输出解析、循环终止和错误处理等关键逻辑都由本项目自行实现，不依赖 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等框架。

常用方式：
1. 安装 Python 3.11+。
2. 在项目根目录执行：python -m pip install -e .[dev]
3. 设置环境变量 OPENAI_API_KEY，可选设置 OPENAI_BASE_URL 和 MEMCODE_MODEL。
4. 运行：mca
5. 执行单次任务：mca run "修复一个测试失败并补充测试"

工作原理：
1. 先从本地代码索引、任务历史和项目规则中检索上下文。
2. 再请求模型决定下一步动作。
3. 本地执行工具调用，并把结果写回上下文。
4. 持续循环，直到任务完成、暂停、失败或达到预算上限。

本地数据：
1. .memcode/memory.json：任务历史、修改文件、失败命令、经验摘要。
2. .memcode/project_memory.json：项目级规则。
3. .memcode/session.json：会话快照。
4. .memcode/transcript.jsonl：追加式对话记录。
5. .memcode/code_index.json 和 .memcode/code_vectors.npy：本地代码索引。

说明：
1. mca index --workspace 用来重建本地代码索引。
2. 最常用的入口是 mca。
