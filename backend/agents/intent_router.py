# backend/agents/intent_router.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tracing.otel_config import trace_agent_call
from utils.prompt_loader import get_prompt


class IntentCategory(str, Enum):
    CONSULTATION = "consultation"
    ORDER_QUERY = "order_query"
    PRODUCT_QUERY = "product_query"
    TICKET_OPERATION = "ticket_operation"
    COMPLAINT = "complaint"
    COMPLIANCE = "compliance"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    primary_intent: str
    secondary_intent: str
    confidence: float
    entities: dict[str, Any]
    suggested_agent: str
    reason: str = ""


VALID_AGENTS = {
    "knowledge_rag",
    "product_agent",
    "order_agent",
    "ticket_handler",
}


class IntentRouterAgent:
    """
    意图路由 Agent。
    负责语义理解、实体抽取、初步路由建议。
    不直接生成最终回复。
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _safe_parse_json(self, content: str) -> dict[str, Any]:
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
            "primary_intent": "unknown",
            "secondary_intent": "unknown",
            "confidence": 0.0,
            "entities": {},
            "suggested_agent": "knowledge_rag",
            "reason": "意图识别结果不是合法 JSON，回退到知识库问答。",
        }

    def _normalize_result(self, result: dict[str, Any]) -> IntentResult:
        suggested_agent = str(result.get("suggested_agent", "knowledge_rag")).strip()

        if suggested_agent not in VALID_AGENTS:
            suggested_agent = "knowledge_rag"

        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(confidence, 1.0))

        entities = result.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}

        return IntentResult(
            primary_intent=str(result.get("primary_intent", "unknown")),
            secondary_intent=str(result.get("secondary_intent", "unknown")),
            confidence=confidence,
            entities=entities,
            suggested_agent=suggested_agent,
            reason=str(result.get("reason", "")),
        )

    @trace_agent_call("intent_router_classify")
    async def classify(self, user_message: str) -> IntentResult:
        messages = [
            SystemMessage(content=get_prompt("intent_router", "system")),
            HumanMessage(content=f"用户消息：{user_message}"),
        ]

        response = await self.llm.ainvoke(messages)
        parsed = self._safe_parse_json(response.content)
        return self._normalize_result(parsed)

    @trace_agent_call("intent_router_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return state

        user_message = messages[-1].content
        intent_result = await self.classify(user_message)
        intent_dict = asdict(intent_result)

        return {
            **state,
            # 初步路由建议，Supervisor 后面可以继续覆盖
            "intent": intent_result.suggested_agent,
            # 结构化意图识别结果，给 Supervisor 和后续 Agent 参考
            "intent_result": intent_dict,
            "current_agent": "intent_router",
        }