Git 仓库地址：https://github.com/MolexZzz/MemCodeAgent.git

运行方式：
1. 安装 Python 3.11+。
2. 在项目根目录执行：python -m pip install -e .[dev]
3. 设置环境变量 OPENAI_API_KEY，可选设置 OPENAI_BASE_URL 和 MEMCODE_MODEL。
4. 运行：mca run "修复一个测试失败并补充测试"

项目简介：
MemCodeAgent 是一个轻量命令行编程智能体。它通过大语言模型决定下一步动作，但文件读写、文本搜索、命令执行、上下文管理、模型输出解析、循环终止和错误处理等关键逻辑由本项目自行实现，不依赖 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等 agent 框架。

已实现功能：
1. 原生工具调用：使用 OpenAI function-calling 接口，模型可选择 list_files、read_file、search_text、write_file、apply_patch、run_command 六个工具，自行解析 tool_calls 并分发执行。
2. 安全本地工具层：路径限制在工作区内、危险命令拦截、命令超时、输出截断，防止越界读写和误操作。
3. 多编辑补丁：apply_patch 支持一次提交多个精确字符串替换，返回统一 diff，便于核对改动。
4. 轻量项目记忆：任务结束后将任务描述、结果摘要、修改文件、失败命令写入 .memcode/memory.json，下次执行时按关键词重叠打分召回相关历史，无需向量数据库。
5. 命令行入口：mca run 支持指定工作区、最大步数、dry-run 预览模式。
6. 单元测试：29 个测试覆盖工具执行、记忆读写检索与模型输出解析，均可通过 pytest 验证。

安全提示：API 密钥仅通过环境变量提供，不会出现在仓库、README 或视频中。
