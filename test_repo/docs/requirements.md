# TicketDesk 维护需求

## 项目背景

TicketDesk 是一个供小型运营团队使用的内部工单服务。当前版本支持创建工单、
查看工单、关闭工单，以及查看用户自己的工单列表。

## 本次维护任务

请为 TicketDesk 准备下一版本：

1. 修复工单列表的状态筛选。`list_tickets(requester, status=...)` 必须按照
   `Ticket.status` 筛选，而不是按照优先级筛选。未传入 `status` 的现有调用方式
   必须保持兼容。
2. 为工单列表增加优先级排序和分页：
   `list_tickets(requester, status=None, sort_by="created_at", descending=False,
   page=1, page_size=20)`.
   - 合法的排序字段为 `created_at`、`priority` 和 `title`。
   - 优先级顺序为 `urgent > high > medium > low`。
   - 无效的页码、每页数量或排序字段必须抛出 `ValueError`。
3. 增加 `reopen_ticket(ticket_id, requester)`，并保持
   `get_ticket` 和 `close_ticket` 使用的工单所有者/管理员权限规则。
4. 检查 `service.py` 和 `cli.py` 中的校验代码。在不改变公开辅助函数的前提下，
   移除重复的校验逻辑。
5. 为筛选、排序、分页、重新打开工单、权限控制和无效输入增加回归测试及边界测试。
6. 在 `README.md` 中补充新的工单列表行为，并运行完整测试集。

## 验收标准

- 现有测试必须继续通过。
- 普通用户只能列出、查看、关闭或重新打开自己的工单；`admin` 可以管理所有工单。
- 不得修改与本次需求无关的文件。
- 最终测试命令必须成功退出。

## 当前版本的已知问题

当前版本为了用于维护实践，特意保留了以下未完成内容：

- 工单列表的状态筛选逻辑不正确；
- 尚未实现排序、分页和重新打开工单功能；
- 不同层中存在重复的校验逻辑；
- 尚未覆盖上述行为的回归测试。
