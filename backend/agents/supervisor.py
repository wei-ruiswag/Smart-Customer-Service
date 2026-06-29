"""
Supervisor编排Agent — 中央协调者
负责接收用户请求，根据意图路由到对应子Agent，汇总结果返回。
采用LangGraph StateGraph实现，支持并行调度和Human-in-the-Loop断点。
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from agents.intent_router import IntentRouterAgent
from agents.knowledge_rag import KnowledgeRAGAgent
from agents.ticket_handler import TicketHandlerAgent
from agents.compliance_checker import ComplianceCheckerAgent
from agents.product_agent import ProductAgent
from agents.order_agent import OrderAgent
from utils.prompt_loader import get_prompt
from memory.working_memory import WorkingMemory
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from mcp.mcp_server import MCPToolServer, create_default_tools
from tracing.otel_config import trace_agent_call


# ─── 状态定义 ───

class AgentState(TypedDict):
    """Supervisor编排的全局状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    intent: str
    intent_result: dict[str, Any]
    sub_results: dict[str, Any]             # 只保存业务 Agent 的自然语言回复
    compliance_passed: bool
    compliance_report: dict[str, Any]       # 合规审查结果单独保存
    final_response: str
    current_agent: str
    retry_count: int


# ─── Supervisor节点 ───

SUPERVISOR_SYSTEM_PROMPT = get_prompt("supervisor", "system")
# """你是一个智能客服系统的Supervisor（主管编排Agent）。
# 你的职责是：
# 1. 分析用户意图，决定分发给哪个子Agent处理
# 2. 汇总子Agent的处理结果，生成最终回复
# 3. 确保所有回复都经过合规审查
#
# 可用的子Agent：
# - intent_router: 意图识别和分类
# - knowledge_rag: 知识库检索和回答
# - ticket_handler: 工单创建和查询
# - compliance_checker: 合规审查和敏感词检测
#
# 根据用户消息，决定下一步路由到哪个Agent。
# """


class SupervisorNode:
    """
    Supervisor决策节点：
    IntentRouter 高置信度 → 直接采纳
    IntentRouter 低置信度 → Supervisor 再判断一次
    Supervisor 判断异常 → 回退到 IntentRouter 建议或 knowledge_rag
    """

    def __init__(self, llm: ChatOpenAI, working_memory: WorkingMemory):
        self.llm = llm
        self.working_memory = working_memory

    @trace_agent_call("supervisor")
    async def route_decision(self, state: AgentState) -> AgentState:
        """根据 IntentRouter 结果和上下文，决定最终路由"""
        messages = state["messages"]
        session_id = state.get("session_id", "default")
        context = self.working_memory.get_context(session_id)

        intent_result = state.get("intent_result", {})
        suggested_agent = str(intent_result.get("suggested_agent", "")).strip()
        confidence = float(intent_result.get("confidence", 0.0) or 0.0)

        valid_intents = {
            "knowledge_rag",
            "product_agent",
            "order_agent",
            "ticket_handler",
        }

        # 高置信度时，Supervisor 可以直接采纳 IntentRouter 建议。
        # 这样减少一次 LLM 路由判断，提高速度。
        if confidence >= 0.85 and suggested_agent in valid_intents:
            intent = suggested_agent
        else:
            routing_prompt = [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                SystemMessage(content=f"当前工作记忆上下文: {context}"),
                SystemMessage(content=f"IntentRouter识别结果: {intent_result}"),
                *messages,
                HumanMessage(
                    content=(
                        "请分析用户的最新消息，并结合 IntentRouter 识别结果，"
                        "返回最终应该路由到的Agent名称。"
                        "只返回以下之一: knowledge_rag, product_agent, order_agent, ticket_handler"
                    )
                ),
            ]

            response = await self.llm.ainvoke(routing_prompt)
            intent = response.content.strip().lower()

            if intent not in valid_intents:
                intent = suggested_agent if suggested_agent in valid_intents else "knowledge_rag"

        self.working_memory.update(
            session_id,
            {
                "last_intent": intent,
                "last_intent_result": intent_result,
            },
        )

        return {
            **state,
            "intent": intent,
            "current_agent": "supervisor",
        }

    @trace_agent_call("supervisor_synthesize")
    async def synthesize_response(self, state: AgentState) -> AgentState:
        """汇总子Agent结果，生成最终回复"""

        sub_results = state.get("sub_results", {})
        compliance_report = state.get("compliance_report", {})
        compliance_passed = state.get(
            "compliance_passed",
            compliance_report.get("passed", True),
        )

        # 合规不通过时，不把原业务回复直接返回给用户
        if not compliance_passed:
            final_response = (
                "抱歉，当前回复可能涉及隐私信息、交易风险或不合规承诺，"
                "已为您转人工客服处理。"
            )
        else:
            result_parts = []

            # 只拼接字符串类型的业务回复，跳过 dict/list 等结构化数据
            for agent_name, result in sub_results.items():
                if isinstance(result, str) and result.strip():
                    result_parts.append(result.strip())

            if result_parts:
                final_response = "\n\n".join(result_parts)
            else:
                final_response = "抱歉，暂时无法处理您的请求，请稍后重试。"

        return {
            **state,
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)],
        }

    # @trace_agent_call("supervisor_synthesize")
    # async def synthesize_response(self, state: AgentState) -> AgentState:
    #     """汇总子Agent结果，生成最终回复"""
    #     sub_results = state.get("sub_results", {})
    #     compliance_passed = state.get("compliance_passed", True)
    #
    #     if not compliance_passed:
    #         final_response = (
    #             "抱歉，您的请求涉及敏感内容，已转交人工客服处理。"
    #             "工单编号已自动生成，请留意后续通知。"
    #         )
    #     else:
    #         result_parts = []
    #         for agent_name, result in sub_results.items():
    #             if result:
    #                 result_parts.append(result)
    #         final_response = "\n\n".join(result_parts) if result_parts else "抱歉，暂时无法处理您的请求，请稍后重试。"
    #
    #     return {
    #         **state,
    #         "final_response": final_response,
    #         "messages": [AIMessage(content=final_response)],
    #     }


# ─── 路由函数 ───

def route_to_agent(state: AgentState) -> str:
    """根据意图路由到对应Agent节点"""
    intent = state.get("intent", "knowledge_rag")
    route_map = {
        "knowledge_rag": "knowledge_rag",
        "product_agent": "product_agent",
        "order_agent": "order_agent",
        "ticket_handler": "ticket_handler",
        # "compliance_checker": "compliance_check",
    }
    return route_map.get(intent, "knowledge_rag")


def should_check_compliance(state: AgentState) -> str:
    """所有回复都需经过合规审查"""
    return "compliance_check"


# ─── 构建Graph ───

def create_supervisor_graph(
    llm: ChatOpenAI | None = None,
    working_memory: WorkingMemory | None = None,
    short_term_memory: ShortTermMemory | None = None,
    long_term_memory: LongTermMemory | None = None,
    enable_checkpointing: bool = True,
    mcp_server: MCPToolServer | None = None,
) -> StateGraph:
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME"),
            temperature=0,
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    if working_memory is None:
        working_memory = WorkingMemory()

    if mcp_server is None:
        mcp_server = create_default_tools(MCPToolServer())

    supervisor = SupervisorNode(llm, working_memory)

    intent_router = IntentRouterAgent(llm)
    knowledge_agent = KnowledgeRAGAgent(llm, long_term_memory)
    product_agent = ProductAgent(llm, mcp_server=mcp_server)
    order_agent = OrderAgent(llm, mcp_server=mcp_server)
    ticket_agent = TicketHandlerAgent(llm, mcp_server=mcp_server)
    compliance_agent = ComplianceCheckerAgent(llm, mcp_server=mcp_server)

    graph = StateGraph(AgentState)

    graph.add_node("intent_router", intent_router.process)
    graph.add_node("supervisor_route", supervisor.route_decision)
    graph.add_node("knowledge_rag", knowledge_agent.process)
    graph.add_node("product_agent", product_agent.process)
    graph.add_node("order_agent", order_agent.process)
    graph.add_node("ticket_handler", ticket_agent.process)
    graph.add_node("compliance_check", compliance_agent.process)
    graph.add_node("synthesize", supervisor.synthesize_response)

    # 先做意图识别，再交给 Supervisor 做最终路由
    graph.set_entry_point("intent_router")
    graph.add_edge("intent_router", "supervisor_route")

    graph.add_conditional_edges(
        "supervisor_route",
        route_to_agent,
        {
            "knowledge_rag": "knowledge_rag",
            "product_agent": "product_agent",
            "order_agent": "order_agent",
            "ticket_handler": "ticket_handler",
        },
    )

    # 所有业务 Agent 结束后都进入合规审查
    graph.add_edge("knowledge_rag", "compliance_check")
    graph.add_edge("product_agent", "compliance_check")
    graph.add_edge("order_agent", "compliance_check")
    graph.add_edge("ticket_handler", "compliance_check")

    graph.add_edge("compliance_check", "synthesize")
    graph.add_edge("synthesize", END)

    checkpointer = MemorySaver() if enable_checkpointing else None
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled