"""
工单处理Agent — 工单CRUD与流转
负责创建、查询、更新工单，对接工单系统，处理退款/理赔/开户等业务办理类需求。
通过MCP工具协议调用外部工单系统。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call


class TicketStatus(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    PENDING_REVIEW = "pending_review"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


TICKET_SYSTEM_PROMPT = """你是一个面向电商客服场景的工单处理 Agent，只负责工单创建、查询和更新。

你不负责回答通用规则，也不负责查询订单物流状态。
- “怎么退款 / 退款流程 / 退款多久到账”属于规则咨询，不应该创建工单。
- “我的订单什么时候到 / 订单发货了吗”属于订单查询，不应该创建工单。
- 只有用户明确要求申请、办理、投诉、创建工单、查询工单、更新工单时，才进入工单处理。

你的任务：
1. 判断 action：create、query、update、clarify。
2. 提取 ticket_no、order_no、ticket_type、priority、description。
3. 创建工单必须尽量提取 order_no；如果用户没有提供订单号，返回 clarify。
4. 查询工单时，如果有工单号，提取 ticket_no；如果没有工单号但有订单号，提取 order_no。
5. 不要编造工单号、订单号、用户ID。

工单类型只能从以下值中选择：
- 退款
- 退货
- 换货
- 物流异常
- 商品质量
- 发票问题
- 优惠券问题
- 投诉
- 通用

优先级只能从以下值中选择：
- 低
- 中
- 高
- 紧急

状态只能从以下值中选择：
- 待处理
- 处理中
- 已完成
- 已关闭

请只返回合法 JSON，不要使用 markdown，不要使用代码块。

返回格式：
{
  "action": "create|query|update|clarify",
  "ticket_no": "",
  "order_no": "",
  "ticket_type": "退款",
  "priority": "中",
  "description": "用户希望申请退款",
  "status": "",
  "clarification_question": ""
}
"""


class TicketStore:
    """内存工单存储（生产环境应替换为数据库）"""

    def __init__(self):
        self._tickets: dict[str, dict] = {}

    def create(self, ticket_type: str, priority: str, summary: str, details: str, user_id: str) -> dict:
        ticket_id = f"TK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        ticket = {
            "ticket_id": ticket_id,
            "type": ticket_type,
            "priority": priority,
            "status": TicketStatus.CREATED.value,
            "summary": summary,
            "details": details,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._tickets[ticket_id] = ticket
        return ticket

    def query(self, ticket_id: str) -> dict | None:
        return self._tickets.get(ticket_id)

    def query_by_user(self, user_id: str) -> list[dict]:
        return [t for t in self._tickets.values() if t["user_id"] == user_id]

    def update_status(self, ticket_id: str, status: str) -> dict | None:
        ticket = self._tickets.get(ticket_id)
        if ticket:
            ticket["status"] = status
            ticket["updated_at"] = datetime.now().isoformat()
        return ticket


class TicketHandlerAgent:
    """工单处理 Agent：通过 MCP 工具操作 MySQL tickets 表"""

    def __init__(self, llm: ChatOpenAI, mcp_server: Any):
        self.llm = llm
        self.mcp_server = mcp_server

    @trace_agent_call("ticket_analyze")
    async def analyze_request(self, user_message: str) -> dict:
        """分析用户需求，提取工单信息"""
        messages = [
            SystemMessage(content=TICKET_SYSTEM_PROMPT),
            HumanMessage(content=f"用户消息: {user_message}"),
        ]

        response = await self.llm.ainvoke(messages)

        import json
        content = response.content.strip()

        # 兼容模型偶尔返回 ```json ... ```
        if content.startswith("```"):
            content = content.strip("`")
            content = content.replace("json", "", 1).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "action": "clarify",
                "ticket_no": "",
                "order_no": "",
                "ticket_type": "通用",
                "priority": "中",
                "description": user_message,
                "status": "",
                "clarification_question": "请补充订单号和需要处理的问题，我再帮您创建或查询工单。",
            }

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        call_result = await self.mcp_server.call_tool(tool_name, arguments)

        if not call_result.success:
            return {
                "success": False,
                "message": call_result.error or f"{tool_name} 调用失败",
                "result": None,
            }

        return {
            "success": True,
            "message": "",
            "result": call_result.result,
        }

    def _format_ticket(self, ticket: dict) -> str:
        return (
            f"📋 工单号: {ticket.get('ticket_no', '')}\n"
            f"🔗 关联订单: {ticket.get('order_no', '')}\n"
            f"📝 类型: {ticket.get('ticket_type', '')}\n"
            f"⚡ 优先级: {ticket.get('priority', '')}\n"
            f"📊 状态: {ticket.get('status', '')}\n"
            f"📄 问题描述: {ticket.get('description', '')}\n"
            f"🕐 创建时间: {ticket.get('created_at', '')}\n"
            f"🔄 更新时间: {ticket.get('updated_at', '')}"
        )

    @trace_agent_call("ticket_create")
    async def create_ticket(self, ticket_info: dict, user_id: str) -> str:
        order_no = ticket_info.get("order_no", "").strip()
        description = ticket_info.get("description", "").strip()

        if not order_no:
            return "请提供需要处理的订单号，我再帮您创建工单。"

        if not description:
            return "请补充需要处理的问题描述，例如退款原因、商品问题或物流异常情况。"

        tool_result = await self._call_tool(
            "ticket_create",
            {
                "user_id": user_id,
                "order_no": order_no,
                "ticket_type": ticket_info.get("ticket_type", "通用"),
                "priority": ticket_info.get("priority", "中"),
                "description": description,
            },
        )

        if not tool_result["success"]:
            return f"工单创建失败：{tool_result['message']}"

        result = tool_result["result"]
        if not result.get("found"):
            return result.get("message", "工单创建失败，请稍后重试。")

        ticket = result["ticket"]

        return (
            "工单已创建成功！\n\n"
            f"{self._format_ticket(ticket)}\n\n"
            "我们将尽快处理您的请求，请保存好工单号以便后续查询。"
        )

    @trace_agent_call("ticket_query")
    async def query_ticket(self, ticket_info: dict, user_id: str) -> str:
        """查询工单状态"""
        ticket_no = ticket_info.get("ticket_no", "").strip()
        order_no = ticket_info.get("order_no", "").strip()

        tool_result = await self._call_tool(
            "ticket_query",
            {
                "ticket_no": ticket_no,
                "user_id": user_id,
                "order_no": order_no,
                "limit": 5,
            },
        )

        if not tool_result["success"]:
            return f"工单查询失败：{tool_result['message']}"

        result = tool_result["result"]
        if not result.get("found"):
            return result.get("message", "未查询到符合条件的工单。")

        tickets = result.get("tickets", [])
        formatted = "\n\n".join(self._format_ticket(ticket) for ticket in tickets)

        return f"工单查询结果：\n\n{formatted}"

    @trace_agent_call("ticket_update")
    async def update_ticket(self, ticket_info: dict) -> str:
        ticket_no = ticket_info.get("ticket_no", "").strip()
        status = ticket_info.get("status", "").strip()

        if not ticket_no:
            return "请提供需要更新的工单号。"

        if not status:
            return "请提供需要更新的工单状态。"

        tool_result = await self._call_tool(
            "ticket_update",
            {
                "ticket_no": ticket_no,
                "status": status,
            },
        )

        if not tool_result["success"]:
            return f"工单更新失败：{tool_result['message']}"

        result = tool_result["result"]
        if not result.get("found"):
            return result.get("message", f"未找到工单号 {ticket_no}。")

        return (
            "工单状态已更新：\n\n"
            f"{self._format_ticket(result['ticket'])}"
        )

    @trace_agent_call("ticket_handler_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        作为Graph节点处理状态
        作为一个“工单助手（Ticket Handler）”分流器，自动识别用户的意图（创建、查询、更新或澄清工单），执行对应的操作，并将结果更新到全局状态中
        """
        messages = state.get("messages", [])
        user_id = state.get("user_id", "anonymous")

        if not messages:
            return state

        last_message = messages[-1].content
        ticket_info = await self.analyze_request(last_message)

        action = ticket_info.get("action", "clarify")

        if action == "create":
            result = await self.create_ticket(ticket_info, user_id)
        elif action == "query":
            result = await self.query_ticket(ticket_info, user_id)
        elif action == "update":
            result = await self.update_ticket(ticket_info)
        else:
            result = ticket_info.get("clarification_question") or "请补充订单号、工单号或需要处理的问题。"

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "ticket_handler": result,
            },
        }


