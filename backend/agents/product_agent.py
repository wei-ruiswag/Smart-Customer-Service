# backend/agents/product_agent.py
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call
from utils.prompt_loader import get_prompt


class ProductAgent:
    """商品 Agent：负责商品价格、库存、分类、描述、推荐、商品售后字段查询。"""

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
            "keyword": "",
            "category": "",
            "min_price": None,
            "max_price": None,
            "only_in_stock": True,
            "limit": 5,
            "need_after_sale_policy": False,
            "clarification_question": fallback_message,
        }

    @trace_agent_call("product_analyze")
    async def analyze_request(self, user_message: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=get_prompt("product_agent", "analyze")),
            HumanMessage(content=f"用户消息：{user_message}"),
        ]
        response = await self.llm.ainvoke(messages)
        return self._parse_json(
            response.content,
            fallback_message="请补充您想查询的商品名称、分类、预算或使用需求。",
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

    @trace_agent_call("product_answer")
    async def generate_answer(
        self,
        user_message: str,
        product_info: dict[str, Any],
        parsed: dict[str, Any],
    ) -> str:
        messages = [
            SystemMessage(content=get_prompt("product_agent", "answer")),
            HumanMessage(
                content=(
                    f"用户问题：{user_message}\n\n"
                    f"解析结果：{json.dumps(parsed, ensure_ascii=False)}\n\n"
                    f"商品工具返回：{json.dumps(product_info, ensure_ascii=False, default=str)}"
                )
            ),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    @trace_agent_call("product_agent_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return state

        user_message = messages[-1].content
        parsed = await self.analyze_request(user_message)

        action = parsed.get("action", "query")
        if action == "clarify":
            answer = parsed.get("clarification_question") or "请补充您想查询的商品名称、分类、预算或使用需求。"
            return {
                **state,
                "sub_results": {
                    **state.get("sub_results", {}),
                    "product_agent": answer,
                },
            }

        tool_result = await self._call_tool(
            "product_query",
            {
                "keyword": parsed.get("keyword", "") or "",
                "category": parsed.get("category", "") or "",
                "min_price": parsed.get("min_price"),
                "max_price": parsed.get("max_price"),
                "only_in_stock": bool(parsed.get("only_in_stock", True)),
                "limit": int(parsed.get("limit") or 5),
            },
        )

        if not tool_result["success"]:
            answer = f"商品查询失败：{tool_result['message']}"
        else:
            answer = await self.generate_answer(
                user_message=user_message,
                product_info=tool_result["result"],
                parsed=parsed,
            )

        return {
            **state,
            "sub_results": {
                **state.get("sub_results", {}),
                "product_agent": answer,
            },
        }