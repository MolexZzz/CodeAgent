Git 仓库地址：https://github.com/MolexZzz/MemCodeAgent.git

运行方式：
1. 安装 Python 3.11+。
2. 在项目根目录执行：python -m pip install -e .[dev]
3. 设置环境变量 OPENAI_API_KEY，可选设置 OPENAI_BASE_URL 和 MEMCODE_MODEL。
4. 运行：mca run "修复一个测试失败并补充测试"

项目简介：
MemCodeAgent 是一个轻量命令行编程智能体。它通过大语言模型决定下一步动作，但文件读写、文本搜索、命令执行、上下文管理、模型输出解析、循环终止和错误处理等关键逻辑由本项目自行实现，不依赖 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等 agent 框架。

特色功能规划：
1. 安全本地工具层：限制工作区路径、命令超时、危险命令拦截、输出截断。
2. 项目记忆：记录任务总结、修改文件、失败命令和修复经验。
3. 结构化代码检索：参考分层记忆和实体分组检索思想，按文件/符号/任务召回相关上下文。
4. 可解释执行日志：展示每轮工具选择、参数、观察结果和检索依据。
