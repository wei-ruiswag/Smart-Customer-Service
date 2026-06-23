# backend/agents/order_agent.py
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call
from utils.prompt_loader import get_prompt


class OrderAgent:
    """订单 Agent：负责当前用户订单详情、订单状态、物流状态、签收时间和七天无理由起算判断。"""

    def __init__(self, llm: ChatOpenAI, mcp_server: Any):
        self.llm = llm
        self.mcp_server = mcp_server

    def _parse_json(self, content: str, fallback_message: str) -> dict[str, Any]:
        content = content.strip()

        if content.startswith("```"):
            content = content.strip("`")
            content = content.replace("json", "", 1).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    pass

        return {
            "action": "clarify",
            "order_no": "",
            "need_detail": True,
            "need_logistics": False,
            "need_product": False,
            "need_refund_window": False,
            "clarification_question": fallback_message,
        }

    @trace_agent_call("order_analyze")
    async def analyze_request(self, user_message: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=get_prompt("order_agent", "analyze")),
            HumanMessage(content=f"用户消息：{user_message}"),
        ]
        response = await self.llm.ainvoke(messages)
        return self._parse_json(
            response.content,
            fallback_message="请提供订单号，或说明是否需要查询最近订单。",
        )

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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

    @trace_agent_call("order_answer")
    async def generate_answer(
        self,
        user_message: str,
        order_info: dict[str, Any],
        parsed: dict[str, Any],
    ) -> str:
        messages = [
            SystemMessage(content=get_prompt("order_agent", "answer")),
            HumanMessage(
                content=(
                    f"用户问题：{user_message}\n\n"
                    f"解析结果：{json.dumps(parsed, ensure_ascii=False)}\n\n"
                    f"订单工具返回：{json.dumps(order_info, ensure_ascii=False, default=str)}"
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    @trace_agent_call("order_agent_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_id = state.get("user_id", "")

        if not messages:
            return state

        user_message = messages[-1].content
        parsed = await self.analyze_request(user_message)

        action = parsed.get("action", "query")
        if action == "clarify":
            answer = parsed.get("clarification_question") or "请提供订单号，或说明是否需要查询最近订单。"
            return {
                **state,
                "sub_results": {
                    **state.get("sub_results", {}),
                    "order_agent": answer,
                },
            }

        order_no = parsed.get("order_no", "") or ""
        tool_result = await self._call_tool(
            "order_query",
            {
                "order_no": order_no,
                "user_id": user_id,
                "limit": 5,
            },
        )

        if not tool_result["success"]:
            answer = f"订单查询失败：{tool_result['message']}"
        else:
            answer = await self.generate_answer(
                user_message=user_message,
                order_info=tool_result["result"],
                parsed=parsed,
            )

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "order_agent": answer,
            },
        }