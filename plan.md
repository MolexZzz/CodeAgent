# MemCodeAgent 单 Agent Runtime 改进计划

更新时间：2026-08-28

## 目标和边界

目标是把 MemCodeAgent 做成一个可用于真实代码仓库的、单 Agent coding agent：

```text
用户请求
   ↓
意图识别
   ↓
Runtime Controller
   ├── 当前阶段
   ├── 阶段转换
   ├── 工具权限
   ├── 用户确认
   ├── 进度监控
   ├── 预算和中断
   └── 完成条件
   ↓
LLM 决策
   ↓
本地工具执行
```

核心原则：

> LLM 决定“如何解决问题”，Runtime 决定“现在是否允许这样做”。

本项目只实现一个 Agent，不实现 Subagent、多 Agent 协作或托管式代码执行。必须继续满足课程要求：自行实现对话历史、上下文管理、工具定义与本地执行、模型输出解析、循环终止和错误处理；不使用 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 等 Agent 框架。

状态说明：

- `[ ]` 待处理
- `[/]` 进行中
- `[x]` 已完成
- `[!]` 暂停或需要重新设计

## 当前基线

- [x] 本地文件读取、搜索、写入、补丁和命令执行
- [x] DeepSeek 官方兼容接口配置
- [x] 会话历史和 `.memcode/session.json` 持久化
- [x] 只读工具自动执行，写入/补丁/命令默认确认
- [x] `Ctrl-C` 中断和 `max_steps` 步数上限
- [x] `/plan`、`/history`、`/context`、`/cache` 基础命令
- [x] 基础阶段状态、测试保护和 diff 检查
- [x] `summarize_tree`、`diff_summary` 基础仓库工具
- [x] 已移除针对某个测试仓库的外部验收逻辑
- [x] AST + BM25 + Graph 记忆/检索基础设施

说明：记忆和检索属于“找到相关信息”的一层；Runtime Controller 属于“控制 Agent 能做什么、何时停止”的一层，两者保留并解耦。

---

## 模块职责边界

这张表用于约束后续实现，避免把所有逻辑重新塞回一个巨大的工具循环里。

| 模块 | 负责什么 | 不负责什么 |
|---|---|---|
| `IntentRouter` | 判断用户请求属于 `ANSWER`、`PLAN` 还是 `MODIFY` | 不执行工具，不修改任务状态 |
| `AgentController` | 驱动整个 Agent Loop，串联阶段、权限、预算、确认、进度和完成检查 | 不决定具体技术方案，不直接读写文件 |
| `StateMachine` | 维护当前阶段，并根据事件执行合法阶段转换 | 不调用模型，不执行工具 |
| `TransitionGuard` | 判断阶段转换是否合法 | 不替模型选择下一步技术动作 |
| `ToolPolicy` | 判断当前阶段下某个工具调用应该允许、拒绝还是确认 | 不执行工具，不生成工具参数 |
| `ToolExecutor` | 调用本地工具并返回结构化结果 | 不决定是否允许执行，不修改 Runtime 状态 |
| `ProgressMonitor` | 检测重复、无进展、循环和预算风险 | 不直接修改 phase，不自行宣布任务完成 |
| `CompletionGuard` | 判断当前任务是否允许进入完成状态 | 不判断代码业务逻辑是否“主观正确” |
| `ContextManager` | 管理历史消息、压缩和上下文窗口 | 不决定任务阶段或工具权限 |
| `Memory/Retrieval` | 用 AST、BM25、Graph 帮助找到相关代码和历史信息 | 不控制 Agent Loop，不替代 Controller |
| `LLM` | 推理、解释、提出计划、选择技术动作和生成工具请求 | 不能直接修改 Runtime 状态，不能绕过权限 |
| `CLI/UI` | 展示进度、收集确认、展示结果 | 不直接控制 Agent 状态，不绕过 Controller |

总规则：

> 任何模块都不能绕过 `AgentController` 直接改变 Runtime 状态。

---

## P0：Agent Runtime 基础

P0 的目的不是增加很多工具，而是先保证 Agent 的行为可控、可解释、可恢复。

### P0.1 任务意图路由

**目的：** 防止用户只是问代码问题时，Agent 却进入修改流程；也防止用户明确要求修改时，Agent 只给建议不动手。

- [x] 定义三种任务模式：`ANSWER`、`PLAN`、`MODIFY`。
- [x] `ANSWER`：允许读取、搜索、摘要和必要的只读命令，直接回答，不修改文件、不要求修改确认。
- [x] `PLAN`：允许只读探索，输出具体实施方案、影响文件、风险和测试建议，绝不修改文件。
- [x] `MODIFY`：进入完整的仓库任务闭环。
- [x] 显式修改词优先：修复、修改、实现、添加、删除、重构、补测试、改代码等必须进入 `MODIFY`。
- [x] 解释、分析、比较、为什么、是什么、是否对齐等问题默认进入 `ANSWER`。
- [x] “怎么改、如何重构、给方案”等问题默认进入 `PLAN`。
- [x] 低置信度时向用户澄清“只分析、先制定计划还是直接修改”，不能让模型自行猜测。

**完成标准：** 用户输入“这个函数为什么错”不会触发写入；输入“修复这个 bug 并测试”一定进入修改流程。

**验证方式：** 增加意图路由单元测试和三类端到端工作流测试。

### P0.2 Agent Controller 核心对象

**目的：** 把阶段、权限、预算、确认和中断统一放在一个 Runtime 控制层，避免逻辑散落在 prompt 和工具循环中。

- [x] 增加 `AgentController`，统一持有任务模式、当前阶段、计划、步数、工具调用记录和任务结果。
- [/] 将现有 `_run_loop_interactive` 中的控制逻辑逐步迁移到 Controller。
- [x] Controller 负责每一步循环：准备上下文 → 请求模型 → 解析决策 → 执行工具 → 记录消息。
- [x] Controller 只负责执行 Runtime 策略，不负责决定具体技术方案。
- [x] 模型不能直接修改 phase、完成状态或预算。

**完成标准：** 所有工具调用和任务结束都经过 Controller；CLI 只负责输入输出，不再自行决定 Agent 状态。

**验证方式：** 对 Controller 使用假的 LLM 响应和假的工具执行器进行确定性测试。

### P0.3 阶段和 Transition Guard

**目的：** 防止模型在分析阶段偷偷修改，或在测试失败后无规则地重复尝试。

阶段定义：

```text
UNDERSTAND → PLAN → CONFIRM → IMPLEMENT → DIFF_CHECK → TEST → VERIFY → DONE
```

允许的受控回退：

```text
TEST 失败 → IMPLEMENT
发现原假设错误 → UNDERSTAND
需要重新设计 → PLAN
```

- [x] 定义明确的 `Phase` 枚举。
- [x] 定义明确的事件，如 `exploration_complete`、`plan_ready`、`user_approved`、`implementation_done`、`test_failed`。
- [x] 通过 `transition(current_phase, event)` 决定下一阶段。
- [ ] 禁止模型直接写入阶段字段。
- [ ] 非法转换记录 Runtime 错误并要求模型重新决策。

**完成标准：** `PLAN` 阶段调用写入工具会被 Runtime 拒绝；测试失败可以回到实现，但不能跳过验证直接完成。

**验证方式：** 覆盖合法转换、非法转换、测试失败回退和重新规划场景。

### P0.4 Tool Policy 和分级确认

**目的：** 在安全性和交互效率之间取得平衡，避免每次读取都打断用户，也避免修改和危险命令无确认执行。

- [ ] 用 `ToolPolicy.evaluate(phase, tool, args)` 返回 `ALLOW`、`DENY` 或 `CONFIRM`。
- [ ] `list_files`、`read_file`、`read_file_range`、`search_text`、摘要工具默认 `ALLOW`。
- [ ] `write_file`、`apply_patch` 默认 `CONFIRM`。
- [ ] `run_command` 默认 `CONFIRM`；对删除、安装依赖、覆盖文件、网络访问等命令显示更明确的风险提示。
- [ ] 当前阶段禁止的工具直接 `DENY`，不通过确认绕过。
- [ ] 用户拒绝后把结果反馈给模型，由模型改用其他方案或向用户说明。

**完成标准：** 只读探索流畅，所有实际修改和高风险命令有明确确认点。

**验证方式：** 工具权限矩阵测试、拒绝路径测试和 CLI 交互测试。

### P0.5 Progress Monitor、预算和中断

**目的：** 解决重复读取、无进展循环、token 无限增长和无法停止的问题。

- [ ] 记录每次工具的名称、规范化参数、结果摘要和产生的进展。
- [ ] 检测相同工具+相同参数的重复调用。
- [ ] 检测连续只读但没有新文件、符号、假设或测试结果的步骤。
- [ ] 检测阶段反复切换、工具单一化和重复最终回答。
- [ ] 将预算拆成 `max_steps`、`max_tool_calls`、`max_read_bytes`、`max_context_tokens`、`max_test_attempts` 和 `max_replan_count`。
- [ ] 达到预算时显示“任务尚未完成，已暂停”，并询问是否继续。
- [ ] `Ctrl-C` 在模型请求、工具执行和确认提示阶段都能终止当前任务并保存会话。

**完成标准：** Agent 卡住时能自动暂停并说明原因；用户可以继续或结束；中断后重新打开仍能恢复任务状态。

**验证方式：** 重复调用、无进展、步数耗尽、连续测试失败和 Ctrl-C 测试。

### P0.6 修改—Diff—测试—验证闭环

**目的：** 禁止 Agent 改完文件就直接宣布完成，保证每个修改任务都有可检查的结果。

- [ ] 修改完成后强制进入 `DIFF_CHECK`。
- [ ] 显示修改文件、增删行和关键 diff 摘要。
- [ ] 自动选择并运行项目已有的测试命令；命令不确定时向用户说明。
- [ ] 将测试结果分类为通过、断言失败、编译/语法错误、运行时错误、命令错误和环境错误。
- [ ] 代码错误回到 `IMPLEMENT`；环境问题停止盲目修复并向用户报告。
- [ ] `VERIFY` 阶段检查任务目标、修改范围和测试结果。
- [ ] 最终回答必须包含修改文件、验证命令、结果和未解决事项。

**完成标准：** 没有 diff 或验证结果时，Controller 不能进入 `DONE`。

**验证方式：** 修改成功、测试失败、环境失败、无测试项目和无关 diff 场景测试。

### P0.7 Completion Guard

**目的：** 把“能否完成任务”从模型自由声明升级为 Runtime 检查，避免 Agent 在没有验证、没有 diff 或目标未满足时直接说完成。

- [ ] 增加 `CompletionGuard`。
- [ ] `ANSWER` 任务：必须已经生成面向用户的回答，且没有发生写入。
- [ ] `PLAN` 任务：必须已经生成实施方案，且没有发生写入。
- [ ] `MODIFY` 任务：必须经过 diff 检查、测试/验证步骤和目标一致性检查。
- [ ] 如果存在未解决错误、用户拒绝的关键工具调用或环境阻塞，不能进入 `DONE`，只能进入暂停/报告状态。
- [ ] LLM 的“已完成”只能作为完成请求，最终是否完成由 `CompletionGuard` 决定。

**完成标准：** `DONE` 状态只能由 `CompletionGuard` 放行；模型不能凭一句总结结束修改任务。

**验证方式：** 覆盖无 diff、无测试结果、用户拒绝关键修改、环境错误和正常完成场景。

### P0.8 Agent Progress UI

**目的：** 让用户看到“Agent 正在做什么”，而不是只看到连续的工具调用和大段文件内容。

- [ ] 将底层 `ToolEvent` 与面向用户的 `AgentEvent`、Runtime 事件分离。
- [ ] 默认显示阶段、当前目标、简短进度和失败原因。
- [ ] 工具调用只显示简短目标，如 `read_file: src/service.py:80-130`。
- [ ] 大型参数和文件内容默认折叠或截断。
- [ ] 统一中文阶段、成功、失败、暂停和确认提示。
- [ ] 清理模型输出中的伪工具调用协议标记，避免直接打印到终端。

**完成标准：** 一次真实仓库任务中，用户能看懂探索、计划、修改、测试和验证的阶段变化。

**验证方式：** CLI 快照/手工演示，并检查失败命令不再只显示一个空的 `✗`。

### P0.9 Agent Runtime 测试集

**目的：** 用固定行为测试防止后续改动把 Agent 重新变回不可控的工具循环。

- [ ] T1：解释问题 → `ANSWER`，无修改。
- [ ] T2：重构建议 → `PLAN`，无修改。
- [ ] T3：修复 bug → `MODIFY`。
- [ ] T4：PLAN 阶段调用编辑工具 → `DENY`。
- [ ] T5：重复读取同一范围 → 触发无进展保护。
- [ ] T6：达到步数预算 → 暂停并询问继续。
- [ ] T7：测试断言失败 → 回到实现。
- [ ] T8：测试环境错误 → 停止盲目修改并报告。
- [ ] T9：Ctrl-C → 中断并持久化状态。
- [ ] T10：修改完成 → 必须经过 diff、test、verify。

**完成标准：** 这组测试成为每次 Runtime 修改后的回归门槛。

---

## P0 实际开发顺序

文档中的编号表示架构模块，不表示必须完全串行完成。实际编码时按下面顺序推进，每一步都补测试，避免最后才发现 Runtime 行为不稳定：

1. `IntentRouter`
2. `Phase`、`StateMachine`、`TransitionGuard`
3. `ToolPolicy`
4. `AgentController`
5. `CompletionGuard`
6. `ProgressMonitor`、预算和中断
7. 修改 → Diff → 测试 → 验证闭环
8. Agent Runtime 回归测试集
9. Agent Progress UI

这意味着 UI 最后打磨，但 Runtime 测试要随着每个模块同步建立。

---

## P1：Code Intelligence 和上下文效率

P1 的目的，是让 Agent 在较大的真实仓库中“少读、读对、读得有结构”，而不是继续依赖大量 `read_file`。

### P1.1 分段读取

- [ ] 增加 `read_file_range`，支持行号范围和字符上限。
- [ ] 记录已读取范围，重复请求时复用或提醒。
- [ ] 大文件默认先返回摘要，不直接把全文送入模型。

### P1.2 符号摘要

- [ ] 增加 `summarize_symbols`。
- [ ] Python 使用 AST 提取类、函数、方法和行号。
- [ ] Cangjie、JavaScript、TypeScript、Java、Go、Rust、C/C++ 提供通用正则降级。
- [ ] 摘要结果包含文件、符号类型、名称和位置，不包含大段源码。

### P1.3 定义、引用和依赖

- [ ] 增加 `find_definition`。
- [ ] 增加 `find_references`。
- [ ] 优先复用现有 AST、BM25、Graph 索引。
- [ ] 找不到精确结果时明确返回“不确定”，不能伪造关系。

### P1.4 仓库摘要和变更摘要

- [ ] 改进 `summarize_tree`，过滤缓存、构建产物和超大目录。
- [ ] 扩展 `diff_summary`，按文件和目录汇总变更。
- [ ] 为测试文件、配置文件和入口文件提供高优先级提示。

### P1.5 Context 和 Memory 优化

- [ ] 上下文压缩前提示用户。
- [ ] 压缩时优先保留目标、计划、当前假设、变更摘要和失败测试。
- [ ] 使用 AST + BM25 + Graph 进行相关代码检索，避免替换 Runtime。
- [ ] 对重复工具结果做缓存，减少 token 和磁盘读取。
- [ ] 增加上下文增长和缓存命中回归测试。

**P1 完成标准：** 在中型仓库中，Agent 能先通过树摘要、符号和相关检索定位文件，再读取必要片段，而不是盲目扫描全文。

---

## P2：成熟度增强

这些能力有价值，但不阻塞一个合格的单 Agent coding agent。

- [ ] 支持项目级 `AGENTS.md`，按目录继承规则。
- [ ] 支持实现过程中受控重新规划。
- [ ] 增加自适应探索深度和仓库摘要缓存。
- [ ] 优化不同语言的符号和引用分析。
- [ ] 改进终端展示，未来可复用到 TUI 或 Web UI。

不在当前范围内：

- 不实现 Subagent 或多 Agent 协作。
- 不引入现成 Agent SDK。
- 不为单个 demo 仓库添加专用验收逻辑。

---

## 每次修改记录

| 日期 | 修改内容 | 状态 | 提交 | 验证 |
|---|---|---|---|---|
| 2026-08-28 | 创建初版路线图 | 已调整 | 未单独提交 | 已被本版本替换 |
| 2026-08-28 | 按 Runtime Controller 方向重写计划，补充目的、完成标准和验证方式 | 已完成 | 待提交 | `git diff --check` 通过 |
| 2026-08-28 | 补充模块职责边界、`CompletionGuard` 和 P0 实际开发顺序 | 已完成 | `0488696` | `git diff --check` 通过 |
| 2026-08-28 | 实现 P0.1 IntentRouter 初版：`ANSWER`/`PLAN` 只读分流，`MODIFY` 进入现有修改闭环 | 已完成 | `5a6754c` | 相关 pytest：31 passed |
| 2026-08-28 | 补齐 P0.1：单次 `run()` 接入路由、低置信度澄清、ANSWER/PLAN 只读端到端测试 | 已完成 | 待提交 | `python -m py_compile src\\memcodeagent\\agent.py` 通过；相关 pytest：34 passed |
| 2026-08-28 | 完成 P0.3 第一小步：独立 `Phase`、`RuntimeEvent`、`TransitionGuard`、`StateMachine` 及转换测试 | 已完成 | 待提交 | `python -m py_compile ...` 通过；相关 pytest：36 passed |
| 2026-08-28 | P0.3 后续：把状态机接入 AgentController，并拒绝非法工具阶段 | 待开始 | 待提交 | 待验证 |
| 2026-08-28 | 开始实现 P0.2 AgentController 核心循环 | 进行中 | 待提交 | 待验证 |
| 2026-08-28 | 完成 P0.2 第一小步：独立 Controller、单步执行、状态持久化和确定性测试 | 已完成 | `81a502f` | 相关 pytest：38 passed |
| 2026-08-28 | 完成 P0.2 第二小步：交互循环通过 Controller 获取决策并执行工具 | 部分完成 | 待提交 | `python -m py_compile ...` 通过；相关 pytest：38 passed |

## 每次代码修改的固定流程

1. 在本文件把相关条目标为 `[/]`。
2. 修改代码和测试。
3. 运行与改动相关的测试。
4. 更新条目为 `[x]`，补充提交编号和验证结果。
5. 再开始下一项，不把多个未验证的大改动混在一起。

基础验证命令：

```text
python -m py_compile src\memcodeagent\*.py
pytest -q tests/test_agent_workflow.py tests/test_tools.py tests/test_retry_and_verification.py tests/test_context_manager.py tests/test_workspace.py
```

如果测试受到 Windows 临时目录权限、网络或模型下载影响，必须单独记录为环境问题，不能当作代码断言通过。
