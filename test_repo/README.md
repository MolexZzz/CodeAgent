# TicketDesk

TicketDesk 是一个面向内部团队的小型工单服务。项目按照日常软件项目的
方式组织，包含数据访问层、业务服务层、命令行格式化工具、测试，以及产品
需求文档。

## 快速开始

```bash
python -m pytest -q
```

下一版本的维护需求位于 `docs/requirements.md`。

仓库外部验收脚本位于主项目的 `scripts/evaluate_ticketdesk.py`。它会在副本目录
中运行测试并执行额外的功能检查，避免只修改公开测试就得到全绿结果。
