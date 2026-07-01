"""
工单处理Agent — 工单CRUD与流转
负责创建、查询、更新工单，对接工单系统，处理退款/换货/修改地址等办理类需求。
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
from utils.prompt_loader import get_prompt


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


TICKET_SYSTEM_PROMPT = get_prompt("ticket_handler", "system")

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


TICKET_HANDLE_TIME = {
    "退款": "通常 1-3 个工作日内处理，具体以平台审核结果为准。",
    "退货": "通常 1-3 个工作日内处理，具体以平台审核结果为准。",
    "换货": "通常 1-3 个工作日内处理，具体以平台审核结果为准。",
    "物流异常": "通常 1-3 个工作日内核实物流情况。",
    "地址修改": "通常会尽快处理；若订单已发货，是否能修改以物流和客服处理结果为准。",
    "商品质量": "通常 1-3 个工作日内核实处理。",
    "发票问题": "通常 1-2 个工作日内处理。",
    "优惠券问题": "通常 1-2 个工作日内核实处理。",
    "投诉": "会优先反馈给客服人员处理，具体进度以客服跟进结果为准。",
    "通用": "通常 1-3 个工作日内处理。",
}


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

    async def _call_tool(self, tool_name: str, arguments: dict, user_id: str | None = None) -> dict:
        call_result = await self.mcp_server.call_tool(tool_name, arguments, agent_name="ticket_handler", user_id=user_id or arguments.get("user_id"))

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

    def _format_ticket(self, ticket: dict, show_description: bool = True) -> str:

        ticket_type = ticket.get("ticket_type", "通用")
        priority = ticket.get("priority", "中")
        handle_time = self._get_handle_time(ticket_type, priority)

        lines = [
            f"📋 工单号：{ticket.get('ticket_no', '')}",
            f"🔗 关联订单：{ticket.get('order_no', '')}",
            f"📝 问题类型：{ticket_type}",
            f"📊 当前状态：{ticket.get('status', '')}",
            f"⚡ 参考处理时效：{handle_time}",
        ]

        if show_description:
            lines.append(f"📄 问题描述：{ticket.get('description', '')}")

        if ticket.get("created_at"):
            lines.append(f"🕐 创建时间：{ticket.get('created_at', '')}")

        if ticket.get("updated_at"):
            lines.append(f"🔄 更新时间：{ticket.get('updated_at', '')}")

        return "\n".join(lines)
        # return (
        #     f"📋 工单号: {ticket.get('ticket_no', '')}\n"
        #     f"🔗 关联订单: {ticket.get('order_no', '')}\n"
        #     f"📝 类型: {ticket.get('ticket_type', '')}\n"
        #     f"⚡ 优先级: {ticket.get('priority', '')}\n"
        #     f"📊 状态: {ticket.get('status', '')}\n"
        #     f"📄 问题描述: {ticket.get('description', '')}\n"
        #     f"🕐 创建时间: {ticket.get('created_at', '')}\n"
        #     f"🔄 更新时间: {ticket.get('updated_at', '')}"
        # )

    def _get_handle_time(self, ticket_type: str, priority: str = "中") -> str:
        """
        根据工单类型和优先级返回参考处理时效。
        注意：这里只给参考时效，不承诺一定完成。
        """
        if priority in {"紧急", "高"} and ticket_type in {"投诉", "物流异常", "商品质量"}:
            return "会优先反馈给客服人员处理，具体进度以平台核实和客服跟进结果为准。"

        return TICKET_HANDLE_TIME.get(
            ticket_type,
            "通常 1-3 个工作日内处理，具体以平台审核和客服跟进结果为准。",
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
            user_id=user_id,
        )

        if not tool_result["success"]:
            return f"工单创建失败：{tool_result['message']}"

        result = tool_result["result"]
        if not result.get("found"):
            return result.get("message", "工单创建失败，请稍后重试。")

        ticket = result["ticket"]

        ticket_type = ticket.get("ticket_type", ticket_info.get("ticket_type", "通用"))
        priority = ticket.get("priority", ticket_info.get("priority", "中"))
        handle_time = self._get_handle_time(ticket_type, priority)

        return (
            "已为您提交处理申请。\n\n"
            f"{self._format_ticket(ticket, show_description=False)}\n\n"
            "您可以后续通过工单号查询处理进度。涉及退款、赔付、换货或地址修改等事项，最终以平台审核和客服处理结果为准。"
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
            user_id=user_id,
        )

        if not tool_result["success"]:
            return f"工单查询失败：{tool_result['message']}"

        result = tool_result["result"]
        if not result.get("found"):
            return result.get("message", "未查询到符合条件的工单。")

        tickets = result.get("tickets", [])
        formatted = "\n\n".join(
            self._format_ticket(ticket, show_description=True)
            for ticket in tickets
        )

        return (
            "为您查询到以下处理记录：\n\n"
            f"{formatted}\n\n"
            "处理进度以客服人员实际跟进结果为准。"
        )


    @trace_agent_call("ticket_update")
    async def update_ticket(self, ticket_info: dict, user_id: str) -> str:
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
            user_id=user_id,
        )

        if not tool_result["success"]:
            return f"工单更新失败：{tool_result['message']}"

        result = tool_result["result"]
        if not result.get("found"):
            return result.get("message", f"未找到工单号 {ticket_no}。")

        return (
            "处理记录状态已更新：\n\n"
            f"{self._format_ticket(result['ticket'])}\n\n"
            "后续处理结果以客服人员跟进和平台审核结果为准。"
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
            result = await self.update_ticket(ticket_info, user_id)
        else:
            result = ticket_info.get("clarification_question") or "请补充订单号、工单号或需要处理的问题。"

        return {
            "sub_results": {
            "ticket_handler": result,
            },
        }


