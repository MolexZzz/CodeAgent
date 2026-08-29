# ReAct Agent 分阶段重构实施计划

## 一、实施目标

本计划用于落地 `react_refactor_analysis.md` 中的总体架构，并吸收上下文管理、动态计划、完成判断和会话恢复方面的修正意见。

核心目标：

```text
普通自然语言请求
  -> 不经过关键词意图路由
  -> 直接进入模型驱动的 ReAct Loop
```

模型负责决定下一步行动，运行时负责安全控制、工具执行、状态记录和完成校验。

本次重构不引入任何 Agent 框架或 SDK。对话历史、上下文管理、工具定义、本地执行、循环控制、错误处理和终止条件继续由本项目自行实现。

## 二、第一阶段：取消普通请求的关键词路由

重点修改：

```text
src/memcodeagent/agent.py
```

调整内容：

- `run()` 中普通任务不再调用 `IntentRouter.resolve()`。
- `chat()` 中普通输入直接进入统一 ReAct Loop。
- 保留 `/plan` 作为显式只读计划命令。
- 增加 `/explain` 作为显式只读回答命令。
- 删除或停用 `_should_plan()`、`_prepare_task()` 对普通任务的强制影响。
- “分析并修复”“先看看，不行再改”等混合任务直接交给模型判断。

目标行为：

```text
普通输入
  -> _run_react_loop()
```

## 三、第二阶段：统一执行循环

当前项目存在多套执行逻辑：

- `_run_loop()`
- `_run_loop_interactive()`
- `_run_loop_legacy()`
- `_run_read_only_interactive()`

建议抽出一个统一的核心方法：

```python
_run_react_loop(
    messages,
    interactive=False,
    mode="normal",
)
```

核心循环负责：

1. 调用 `llm.next_action()`。
2. 接收模型返回的 tool calls。
3. 经过 `ToolPolicy` 安全检查。
4. 必要时请求用户审批。
5. 执行本地工具。
6. 将工具结果追加到消息历史。
7. 压缩发送给模型的上下文。
8. 处理测试失败和重试。
9. 判断继续、暂停或完成。

`run()` 和 `chat()` 只负责：

- 初始化消息。
- 提供 UI 输出方式。
- 提供审批回调。
- 调用同一个 ReAct 核心循环。

## 四、第三阶段：放宽 ToolPolicy 的阶段限制

重点修改：

```text
src/memcodeagent/policy.py
src/memcodeagent/controller.py
```

删除这类基于展示阶段的硬限制：

```python
if phase in {"EXPLORING", "INSPECTING"} and tool_name not in self.READ_ONLY:
    return DENY
```

保留以下硬限制：

- 工具不存在。
- 路径越出 workspace。
- 修改受保护的既有测试。
- 用户拒绝审批。
- 危险命令未获批准。
- 重复调用。
- 超出工具或步骤预算。

模型应可以自然地执行：

```text
read_file
  -> apply_patch
  -> read_file
  -> run_command
  -> apply_patch
```

不能因为当前显示阶段是 `INSPECTING`，就拒绝模型调用修改工具。

## 五、第四阶段：保留并修正测试验证

修改后的代码应继续支持：

```text
修改文件
  -> 运行测试
  -> 读取测试结果
  -> 继续修复
```

测试失败时：

- 将清洗后的失败信息返回给模型。
- 允许模型重新读取源码。
- 允许模型继续修改和运行测试。
- 达到重试上限后暂停并报告原因。

不应因为测试失败就立即结束任务，也不应为了通过测试而删除或弱化测试。

## 六、第五阶段：简化 runtime 状态机

重点修改：

```text
src/memcodeagent/runtime.py
src/memcodeagent/controller.py
```

建议保留以下稳定状态：

```text
IDLE
RUNNING
WAITING_APPROVAL
VERIFYING
PAUSED
COMPLETED
FAILED
```

`RUNNING` 内部允许模型自由调用合法工具。工具是否执行由 ToolPolicy 和审批机制决定。

原有的：

```text
UNDERSTAND
  -> PLAN
  -> CONFIRM
  -> IMPLEMENT
  -> TEST
  -> VERIFY
```

不再作为强制执行路径，可以保留为日志、展示或统计信息。

## 七、第六阶段：改造完成判断

重点修改：

```text
src/memcodeagent/completion.py
src/memcodeagent/agent.py
```

采用“硬约束 + 软提醒”的混合策略。

硬性阻止：

- 工具调用失败且尚未处理。
- 测试明确失败。
- 审批尚未完成。
- 修改后存在明显未解决错误。
- 模型声称完成但实际没有对应结果。

软提醒：

- 修改代码但没有运行测试。
- 没有检查 diff。
- 项目没有测试命令。
- 只修改文档或配置文件。

暂时继续使用现有的 `is_final` 机制，不立即引入 `finish()` 或 `stop()` 工具。

运行时可以将检查事实追加给模型，让模型决定是否需要补充验证，而不是无条件进入死循环。

## 八、第七阶段：补强上下文和会话恢复

重点修改：

```text
src/memcodeagent/context_manager.py
src/memcodeagent/tools.py
src/memcodeagent/agent.py
src/memcodeagent/controller.py
```

### 上下文管理

当前项目已有：

- `ContextManager.trim()`：按轮次和 token 裁剪上下文。
- `ToolExecutor._limit_output()`：限制文件、搜索和命令输出。

重构时应补强：

- 测试输出优先保留失败用例、错误类型、文件名、行号和关键摘要。
- 大型 diff、日志和搜索结果使用结构化截断。
- 裁剪时不能破坏 assistant tool call 与 tool result 的对应关系。
- 持久化保留完整历史，发送给模型时使用裁剪后的上下文。
- 每轮观察结果设置独立大小上限。

### 会话恢复

恢复时以三类信息重新构造上下文：

```text
历史消息
当前 workspace 快照
最小运行时状态
```

恢复后重新获取：

- `git status`
- `git diff --stat`
- 变更文件列表
- 最近一次测试结果

至少明确保存：

- `pending_approval_tool`
- 是否处于暂停状态
- 当前任务是否仍可继续
- 预算和重试次数
- 是否存在未验证修改

不需要恢复所有旧的阶段变量，但审批状态和暂停状态不能只从普通文本中推断。

## 九、重构后的单任务流程

以用户输入：

```text
请修复 ticketdesk 的排序问题，并运行测试。
```

为例：

```text
用户请求
  -> 初始化 TaskContext
  -> LLM next_action()
  -> ToolPolicy 检查
  -> 必要时用户审批
  -> 本地执行工具
  -> 清洗并追加 observation
  -> ContextManager 压缩上下文
  -> LLM 再次决策
  -> 完成、暂停或继续修复
```

预期行为：

```text
读取需求和源码
  -> 模型形成动态计划
  -> 修改相关文件
  -> 运行 pytest
  -> 根据失败结果继续修复
  -> 检查 diff
  -> 输出总结
```

整个过程中不需要重新经过关键词路由，也不需要在模型从读取转向修改时重新确认任务类型。

## 十、测试计划

应新增或调整测试，覆盖：

- 长任务直接进入 ReAct Loop。
- 普通请求不再被 `IntentRouter` 路由为只读任务。
- 模型先读取再修改。
- `INSPECTING` 阶段不会误拒绝修改工具。
- 修改后能够运行测试。
- 测试失败后能够继续读取、修改和验证。
- 用户拒绝工具审批后，模型能收到拒绝结果。
- 修改代码但未测试时触发软提醒。
- 无测试项目或文档任务不会被强制要求运行测试。
- 会话恢复后能识别未验证修改和待审批操作。
- `/plan` 和 `/explain` 仍保持显式只读行为。

同时验证：

```text
python -m pytest -q
```

## 十一、实施顺序

建议按以下顺序实施：

```text
1. 修改普通请求路由
2. 增加统一 ReAct 核心循环
3. 放宽 ToolPolicy 阶段限制
4. 保留并修正测试验证
5. 简化 runtime 状态机
6. 改造完成判断
7. 补强上下文压缩和会话恢复
8. 补充测试并运行真实编程任务
```

第一步完成后，应先用真实任务验证：

```text
普通自然语言请求
  -> 不经过 IntentRouter
  -> 不经过 _should_plan
  -> 直接进入模型驱动的 ReAct Loop
```

## 十二、与总体分析文档的关系

两份文档职责不同：

- `react_refactor_analysis.md`：说明为什么重构、目标架构、当前缺陷和设计原则。
- `react_refactor_implementation_plan.md`：说明如何分阶段实施、每阶段修改范围、验证方式和风险控制。

后续重构应以两份文档共同作为依据：

```text
总体架构与原则
  -> react_refactor_analysis.md

具体实施步骤
  -> react_refactor_implementation_plan.md
```
