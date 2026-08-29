# ReAct Agent 重构分析

## 一、总体判断

当前 Agent 的核心问题不是关键词不够多，而是整体架构仍然是“先分类，再决定模型能做什么”。

当前流程大致是：

```text
用户输入
  -> 关键词分类
  -> 决定允许哪些工具
  -> 模型开始工作
```

更适合代码 Agent 的流程应该是：

```text
用户输入
  -> 模型决定下一步
  -> 运行时做安全检查
  -> 执行工具
  -> 模型继续决定
```

核心原则：

> 模型负责决定下一步做什么，运行时负责判断这样做是否安全。

## 二、重构原则与边界

本次重构采用“模型驱动、运行时守护”的 ReAct 架构，但不引入任何 Agent 框架或 SDK。

允许继续使用：

- 模型厂商 API 客户端
- OpenAI 兼容网关
- 模型原生 tool calling

必须继续由本项目自行实现：

- 对话历史与上下文管理
- 工具定义、解析和本地执行
- workspace 路径边界
- 用户审批与命令安全检查
- ReAct 循环
- 错误处理、重试和终止条件
- 测试验证与会话持久化

普通自然语言请求不再通过关键词判断 `ANSWER`、`PLAN` 或 `MODIFY`。只有显式斜杠命令才进入特殊模式：

```text
/plan      -> 只读计划模式
/explain   -> 只读回答模式
普通输入   -> 普通 ReAct Agent Loop
```

## 三、需要重构的代码

### 1. `src/memcodeagent/agent.py`

这是改动最大的部分。

当前问题：

- `run()` 先调用 `IntentRouter.resolve()`，再决定进入只读循环还是修改循环。
- `chat()` 也先调用 `IntentRouter.resolve()`。
- `_run_loop_interactive()` 又通过 `_should_plan()` 强制进入 `_prepare_task()`。
- 同时存在 `_run_loop()`、`_run_loop_interactive()`、`_run_loop_legacy()` 三套执行逻辑。
- `_run_read_only_interactive()` 与普通 Agent Loop 使用不同的决策机制。
- 计划确认、普通任务、恢复任务之间没有统一的任务生命周期。

建议重构为一个统一循环：

```text
普通请求
  -> 创建任务上下文
  -> LLM next_action()
  -> 执行工具
  -> 将工具结果追加到 messages
  -> LLM next_action()
  -> 直到模型完成、测试通过、暂停或达到预算
```

保留两个显式特殊入口：

```text
/plan      -> 强制只读计划
/explain   -> 强制只读回答
普通输入   -> 统一 ReAct Loop
```

`IntentRouter` 不应再负责识别 `MODIFY`、`PLAN`、`ANSWER`，最多只负责识别 `/plan`、`/explain` 这种明确命令。

复杂任务的计划应成为模型行为，而不是固定的前置人工阶段。系统提示可以要求模型在涉及多个文件或多个验证步骤时维护分步计划，但不应强制每个普通任务先停下来等待确认。

第一阶段不建议自动创建 `todo.md`：

- 可能污染用户仓库。
- 可能产生额外文件修改。
- 计划文件可能与实际代码状态不一致。

计划可以先保存在对话上下文或 session metadata 中，后续再考虑增加内存型 todo 工具。

### 2. `src/memcodeagent/policy.py`

当前 `ToolPolicy` 同时承担了两件事：

- 安全控制
- 任务阶段控制

例如：

```python
if phase in {"EXPLORING", "INSPECTING"} and tool_name not in self.READ_ONLY:
    return DENY
```

这会导致 Agent 在“探索阶段”不能修改文件，即使模型已经判断探索完成，也会被阶段字符串拦截。

建议将 Policy 的职责限制为：

- 是否越出 workspace
- 是否修改受保护测试
- 是否属于危险命令
- 是否需要用户批准
- 是否重复调用
- 是否超过预算

不要再根据 `INSPECTING`、`PLANNING` 等展示阶段限制工具类型。

阶段只能描述当前主要活动，不能决定模型是否有权调用读取、修改或命令工具。真正的硬限制应来自安全条件和用户授权。

阶段可以继续存在，但主要用于：

- UI 显示
- 日志记录
- 统计
- 会话恢复
- 判断是否完成

阶段不应该成为模型能力的硬闸门。

### 3. `src/memcodeagent/controller.py`

`AgentController.step()` 的方向是正确的，它已经具备：

```text
调用模型
  -> 处理 tool calls
  -> 执行 ToolPolicy
  -> 执行工具
  -> 追加 observation
  -> 返回下一轮所需上下文
```

还需要进一步明确职责：

- Controller 负责执行一次 ReAct step。
- ToolPolicy 负责判断工具是否安全。
- Agent Loop 负责决定什么时候继续、什么时候测试、什么时候结束。
- Controller 不应该通过状态机阻止模型从读取跳到修改。

`handle_user_request("MODIFY")`、`mark_implementation_started()`、`mark_implementation_done()` 等状态转换需要减少，避免出现“实际行为已经发生，但状态机仍不允许转换”的问题。

可以保留状态机，但应将它从“能力控制器”降级为“运行记录器和完成校验器”。

### 4. `src/memcodeagent/runtime.py`

当前状态机比较严格：

```text
UNDERSTAND
  -> PLAN
  -> CONFIRM
  -> IMPLEMENT
  -> DIFF_CHECK
  -> TEST
  -> VERIFY
  -> DONE
```

这适合固定流水线，但不适合 ReAct。真实任务可能是：

```text
读取文件
  -> 修改文件
  -> 运行测试
  -> 再读文件
  -> 再修改
  -> 再运行测试
```

或者：

```text
读取需求
  -> 运行测试
  -> 修改代码
  -> 读取错误位置
  -> 修改代码
```

建议将状态简化成：

```text
IDLE
RUNNING
WAITING_APPROVAL
VERIFYING
PAUSED
COMPLETED
FAILED
```

`RUNNING` 内部允许模型自由调用读取、修改和命令工具。具体行为由 ToolPolicy 和审批机制控制。

### 5. `src/memcodeagent/completion.py`

这一部分可以保留，但应从“任务模式完成”改成“事实完成”，并采用“硬约束 + 软提醒”的混合策略。

当前 `CompletionGuard` 依赖：

```python
mode == "ANSWER"
mode == "PLAN"
mode == "MODIFY"
```

重构后可以改为检查客观事实：

- 是否需要修改文件
- 是否产生了修改
- 是否检查了 diff
- 是否运行了验证命令
- 测试是否通过
- 是否还有未解决错误

例如：

```text
无代码修改的回答任务：
  answer_generated = True

只读计划任务：
  plan_generated = True
  files_changed = False

代码修改任务：
  files_changed = True
  diff_checked = True
  verification_passed = True
  unresolved_errors = False
```

运行时不应机械要求所有任务都必须运行测试。例如 README、文档或测试不存在的项目可能没有可运行的测试命令。

建议：

- 硬性阻止：未处理的工具失败、明确测试失败、未完成审批、修改后存在明显未解决错误、虚假报告完成。
- 软提醒：修改代码但尚未验证、没有检查 diff、项目没有测试命令等情况。

模型请求完成时，运行时可以将事实反馈追加到上下文，让模型决定是否补充验证，而不是无条件陷入完成检查循环。当前项目没有 `finish()` 或 `stop()` 工具，第一阶段继续使用现有的 `is_final` 机制，后续再单独评估是否需要引入显式完成工具。

### 6. 会话持久化

当前 `_save_session()` 保存了：

- messages
- phase
- plan
- controller state
- verification state

采用 ReAct 后，需要保存最小但可恢复的运行上下文：

- 当前任务是否正在执行
- 是否等待用户审批
- 当前审批对应的工具调用
- 当前任务的工作目标
- 当前预算和重试次数
- 最近一次工具调用
- 是否有待验证修改
- 是否允许继续执行

恢复时应以三类事实重新构造上下文：

```text
历史消息
当前 workspace 快照
最小运行时状态
```

恢复后应重新获取 `git status`、`git diff --stat` 和变更文件列表，而不是完全相信旧的阶段变量。

但审批和暂停状态不能只从消息中推断，至少需要明确保存：

- `pending_approval_tool`
- 是否处于暂停状态
- 当前任务是否仍可继续
- 预算和重试计数
- 是否存在未验证修改

## 四、上下文管理与观察结果压缩

ReAct 任务会持续累积文件内容、搜索结果、diff 和测试日志，因此上下文管理必须作为本次重构的一等能力。

当前项目已有：

- `ContextManager.trim()`：按轮次和 token 裁剪上下文。
- `ToolExecutor._limit_output()`：限制文件、搜索和命令输出。

重构时应继续保留并补强这些机制：

- 测试输出优先保留失败用例、错误类型、文件名、行号和关键摘要。
- 大型 diff、日志和搜索结果使用结构化截断，而不是简单丢弃全部上下文。
- 裁剪时不能破坏 assistant tool call 与 tool result 的对应关系。
- 持久化可以保留完整历史，但发送给模型的内容应是清洗和压缩后的上下文。
- 每轮工具观察结果应有独立大小上限，避免单个命令或文件读取耗尽整个上下文窗口。

## 五、重构后的单任务流程

以用户输入：

```text
请修复 ticketdesk 的排序问题，并运行测试。
```

为例。

### 1. 解析输入

只判断是否为特殊斜杠命令：

```text
是否以 / 开头？
```

如果不是，就直接创建普通任务，不再判断“这是计划、回答还是修改”。

### 2. 创建 Agent Context

系统创建任务上下文：

```text
TaskContext
- workspace
- messages
- current phase = RUNNING
- step budget
- approval policy
- protected files
- verification state
```

系统提示模型：

```text
你是一个代码 Agent。
请先检查相关文件，再根据结果决定是否修改。
需要修改时调用 apply_patch 或 write_file。
修改后运行测试并根据测试结果继续修复。
不要为了通过测试而删除或弱化测试。
```

### 3. 第一次模型决策

模型可能调用：

```text
list_files
read_file
search_text
```

运行时检查：

- 工具名称是否合法
- 路径是否在 workspace 内
- 是否为重复调用
- 是否需要审批

只读工具自动执行，结果加入对话。

### 4. 第二次模型决策

模型看到源码后，可能决定调用：

```text
apply_patch
```

运行时检查：

- 是否修改受保护测试
- 是否越出 workspace
- 是否需要用户确认
- patch 是否有效

如果需要审批：

```text
用户确认
  -> 执行 apply_patch
```

如果用户拒绝，则将拒绝结果返回给模型，模型可以继续分析、采用其他方案或暂停。

### 5. 修改后的验证

模型可能调用：

```text
run_command:
python -m pytest -q
```

运行时根据配置进行审批，然后把测试输出返回给模型。

### 6. 测试失败后的继续推理

模型看到失败结果后，可以继续调用：

```text
read_file
search_text
apply_patch
run_command
```

这个过程可以重复多轮，不需要重新经过关键词路由，也不需要人工重新确认“现在是否进入修改模式”。

### 7. 完成判断

模型输出最终回答前，运行时检查：

- 是否存在未处理的工具错误
- 如果改过代码，是否检查 diff
- 是否执行验证
- 验证是否通过
- 是否仍有未解决异常

满足条件后才允许完成，否则继续让模型工作或暂停。

### 8. 输出结果

最终输出：

- 修改了哪些文件
- 修复了什么问题
- 新增了什么功能
- 测试命令和结果
- 是否存在未完成事项

## 六、当前 Agent 的主要设计缺陷

### 1. 意图路由过早介入

当前流程：

```text
用户输入
  -> 关键词分类
  -> 决定允许哪些工具
  -> 模型开始工作
```

这会导致：

- “分析并修复”可能被判成只读。
- 长任务可能被判成计划任务。
- 中文表达稍微变化就无法识别。
- 模型无法根据源码情况临时改变策略。
- 用户确认“是”会被当成新的普通请求。

### 2. 规划被强制成前置人工阶段

当前 `_run_loop_interactive()` 中：

```python
if self._should_plan(messages) and not self._prepare_task(messages):
    return
```

这意味着很多普通任务会被强制执行：

```text
探索
  -> 生成计划
  -> 用户确认
  -> 修改
```

这会带来：

- 交互轮次增加
- 用户必须输入确认
- 确认状态容易丢失
- 规划阶段和执行阶段断开
- 模型在计划阶段不能使用必要工具
- 用户无法自然地说“先看看这个问题，能修就直接修”

### 3. 存在多套不一致的 Agent Loop

当前至少存在：

- `_run_loop()`
- `_run_loop_interactive()`
- `_run_loop_legacy()`
- `_run_read_only_interactive()`

它们对以下内容的处理并不一致：

- 是否先路由意图
- 是否强制计划
- 是否允许工具
- 如何运行测试
- 如何处理错误
- 如何判断完成
- 如何保存状态

这会导致同一个任务通过 `mca` 和单次 `run` 执行时行为不同。

建议最终只保留：

```text
run_react_loop()
```

交互模式和单次模式只负责提供不同的 UI、消息输入和暂停方式。

### 4. 阶段状态和实际行为耦合过深

当前阶段同时承担：

```text
UI 显示状态
工具权限状态
状态机转移依据
```

三种职责耦合后，只要某个阶段更新漏掉，就可能出现：

```text
模型想修改
  -> 策略认为还在探索
  -> 修改工具被拒绝
```

阶段应该描述“当前主要活动”，而不是描述“允许模型做什么”。

### 5. 状态机过于线性

代码任务不是固定流水线。测试失败后可能要重新读取源码，修改后也可能需要重新检查需求。当前状态机更像审批工作流，而不是自主编程循环。

应该允许：

```text
RUNNING
  <-> 工具调用
  <-> 测试
  <-> 修复
```

完成和暂停才是明确的终点。

### 6. Agent 过度依赖预定义工具类别

`_READ_ONLY_TOOLS` 和 `_CODE_EDIT_TOOLS` 被多个地方重复使用，并参与：

- 阶段切换
- 策略判断
- 测试触发
- 完成判断

工具分类可以保留用于统计和显示，但不能成为模型能力边界。

### 7. 会话恢复不完整

目前保存了大量消息，但没有完全保存“当前动作上下文”。因此恢复后可能出现：

- 任务已经修改了一部分，但 Agent 不知道是否需要验证。
- 正在等待确认，但恢复后不知道确认什么。
- 当前任务已经暂停，但普通输入又被当成新任务。
- 之前的计划存在，但执行状态没有恢复。

## 七、推荐的重构顺序

建议分阶段实施：

1. 先让普通请求绕过 `IntentRouter`，直接进入统一 ReAct Loop。
2. 保留 `/plan` 的只读计划功能。
3. 增加 `/explain` 的只读回答功能，或将其作为显式模式处理。
4. 去除 `_should_plan()` 对普通任务的强制影响。
5. 放宽 `ToolPolicy` 的阶段限制，只保留安全和审批限制。
6. 合并 `_run_loop()` 与 `_run_loop_interactive()` 的核心逻辑。
7. 简化 runtime 状态机。
8. 补充跨轮次 ReAct 测试：
   - 长任务直接修改
   - 模型先读后改
   - 测试失败后继续修复
   - 修改工具不被 `INSPECTING` 阶段误拒绝
   - 用户拒绝审批后模型继续工作
   - 会话恢复后继续未完成任务

## 八、目标架构

```text
特殊命令由 CLI 解析
普通请求直接进入 ReAct Loop

模型负责：
- 理解任务
- 选择工具
- 判断是否需要继续
- 根据结果修复问题
- 决定何时结束

运行时负责：
- 工具安全
- 用户审批
- workspace 边界
- 测试保护
- 预算限制
- 会话持久化
- 最终完成校验
```

最终目标是让 Agent 从“关键词驱动的有限状态流程”变成“模型驱动、运行时守护的自主执行循环”。
