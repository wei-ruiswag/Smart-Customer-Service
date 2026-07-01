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

# ———— reducer函数 ————
def merge_dict(
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> dict[str, Any]:
    """合并多个并行 Agent 写入的 sub_results"""
    return {
        **(old or {}),
        **(new or {}),
    }


def merge_unique_list(
    old: list[str] | None,
    new: list[str] | None,
) -> list[str]:
    """合并多个目标 Agent，去重保序"""
    result = []

    for item in (old or []) + (new or []):
        if item not in result:
            result.append(item)

    return result

# ─── 状态定义 ───

class AgentState(TypedDict):
    """Supervisor编排的全局状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    intent: str
    intent_result: dict[str, Any]

    # 新增：本轮需要执行的多个业务 Agent
    target_agents: Annotated[list[str], merge_unique_list]
    sub_results: Annotated[dict[str, Any], merge_dict]  # 关键：并行 Agent 都会写 sub_results，所以必须有 reducer

    # sub_results: dict[str, Any]             # 只保存业务 Agent 的自然语言回复
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

    def _normalize_target_agents(self, raw_agents: Any) -> list[str]:
        """
        把大模型（或前端）传过来的五花八门的“下一个要执行的 Agent”的数据格式
        统一清洗成一个干净、合法、去重且有保底机制的字符串列表
        """
        valid_agents = {
            "knowledge_rag",
            "product_agent",
            "order_agent",
            "ticket_handler",
        }

        if isinstance(raw_agents, str):
            raw_agents = [
                item.strip()
                for item in raw_agents.replace("，", ",").split(",")
                if item.strip()
            ]

        if not isinstance(raw_agents, list):
            raw_agents = []

        result = []
        for agent in raw_agents:
            agent = str(agent).strip()
            if agent in valid_agents and agent not in result:
                result.append(agent)

        return result or ["knowledge_rag"]

    @trace_agent_call("supervisor")
    async def route_decision(self, state: AgentState) -> AgentState:
        """根据 IntentRouter 结果和上下文，决定最终路由，支持多 Agent 并行"""
        messages = state["messages"]
        session_id = state.get("session_id", "default")
        context = self.working_memory.get_context(session_id)

        intent_result = state.get("intent_result", {})
        suggested_agent = str(intent_result.get("suggested_agent", "")).strip()
        confidence = float(intent_result.get("confidence", 0.0) or 0.0)

        valid_agents = {
            "knowledge_rag",
            "product_agent",
            "order_agent",
            "ticket_handler",
        }

        user_message = messages[-1].content if messages else ""

        entities = intent_result.get("entities", {}) or {}
        if not isinstance(entities, dict):
            entities = {}

        has_product_entity = bool(
            entities.get("product_name")
            or entities.get("product")
            or entities.get("category")
        )

        def has_any(keywords: list[str]) -> bool:
            return any(word in user_message for word in keywords)

        # 注意：这里不要用泛泛的“订单”“订单号”来触发 order_agent。
        # 因为用户办理退款/退货时也会提供订单号，这种不应该并行 order_agent。
        order_query_keywords = [
            "订单状态", "物流状态", "快递单号", "发货了吗", "到哪了",
            "签收时间", "支付状态", "订单买了什么", "查物流",
            "我的订单到哪", "订单到哪",
        ]

        policy_keywords = [
            "退货政策", "退款规则", "退款多久到账", "怎么退款", "如何退款",
            "退货规则", "怎么退货", "如何退货", "售后政策",
            "换货流程", "七天无理由", "物流说明", "优惠券规则",
            "会员规则", "发票规则", "发票怎么开",
        ]

        product_keywords = [
            "商品", "价格", "库存", "推荐", "型号", "参数", "分类",
            "耳机", "手机", "笔记本", "键盘", "鼠标", "鼠标垫",
        ]

        product_after_sale_keywords = [
            "售后政策", "售后说明", "支持七天无理由吗",
            "能不能退", "能退吗", "可以退吗",
        ]

        product_reference_keywords = [
            "这款", "这个商品", "该商品", "这台", "这部",
            "这个手机", "这个耳机",
        ]

        # 售后办理 / 后台工单类诉求
        # 不要只写“退款 / 退货 / 换货”，否则“退款规则是什么”会误判。
        ticket_action_keywords = [
            "我要退款", "想退款", "申请退款", "帮我退款", "办理退款",
            "我要退货", "想退货", "申请退货", "帮我退货", "办理退货",
            "我要换货", "想换货", "申请换货", "帮我换货", "办理换货",
            "我要投诉", "投诉", "人工客服", "转人工", "人工处理",
            "修改地址", "改地址", "取消订单",
            "赔付", "补偿",
            "商品坏了", "商品有问题", "少发", "漏发", "发错",
            "没收到", "物流异常", "一直没送到",
        ]

        ticket_query_keywords = [
            "工单进度", "查询工单", "工单状态", "工单编号", "售后进度",
        ]

        # 条件式办理：这种才需要先 order_agent
        dependent_ticket_keywords = [
            "如果", "符合", "能不能退", "是否可以退", "满足条件",
            "先查", "确认后", "能退的话", "可以退的话",
        ]

        has_order_query = has_any(order_query_keywords)
        has_policy_query = has_any(policy_keywords)
        has_product_query = has_any(product_keywords)

        has_product_after_sale_query = (
            has_any(product_after_sale_keywords)
            and (
                has_product_entity
                or has_any(product_reference_keywords)
            )
        )

        has_ticket_action = has_any(ticket_action_keywords)
        has_ticket_query = has_any(ticket_query_keywords)
        has_dependent_ticket_action = (
            has_ticket_action
            and has_any(dependent_ticket_keywords)
        )

        # 重新生成规则路由结果，不直接被 suggested_agent 绑死
        target_agents: list[str] = []

        # 1. 工单/售后进度查询
        if has_ticket_query:
            target_agents = ["ticket_handler"]

        # 2. 条件式售后办理：先查订单事实，不直接创建工单
        elif has_dependent_ticket_action:
            target_agents = ["order_agent"]

            if has_policy_query:
                target_agents.append("knowledge_rag")

        # 3. 明确售后办理/投诉/人工介入：直接 ticket_handler
        # 注意：即使用户提供了订单号，也不并行 order_agent。
        elif has_ticket_action:
            target_agents = ["ticket_handler"]

            # 如果用户同时问通用规则，可以并行 knowledge_rag
            if has_policy_query:
                target_agents.append("knowledge_rag")

            # 如果用户同时问具体商品售后，可以并行 product_agent
            if has_product_after_sale_query or has_product_query:
                target_agents.append("product_agent")

        # 4. 非办理类请求：按查询任务并行
        else:
            if has_order_query:
                target_agents.append("order_agent")

            if has_product_query or has_product_after_sale_query:
                target_agents.append("product_agent")

            if has_policy_query:
                target_agents.append("knowledge_rag")

        # 5. 如果规则没有命中，再采纳 IntentRouter 的 suggested_agent
        if not target_agents:
            if suggested_agent in valid_agents:
                target_agents = [suggested_agent]
            else:
                target_agents = ["knowledge_rag"]

        # 6. 低置信度时，再交给 Supervisor LLM 兜底判断
        # 注意：这段必须和上面的 if/elif 平级，不能缩进到 has_ticket_action 里面。
        if confidence < 0.85:
            routing_prompt = [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                SystemMessage(content=f"当前工作记忆上下文: {context}"),
                SystemMessage(content=f"IntentRouter识别结果: {intent_result}"),
                *messages,
                HumanMessage(
                    content=(
                        "请分析用户的最新消息，并判断需要调用哪些业务Agent。\n"
                        "可选Agent如下：\n"
                        "- knowledge_rag: 平台通用规则、退款规则、退货流程、物流说明、优惠券、会员、发票等知识库问答\n"
                        "- product_agent: 商品价格、库存、分类、推荐、商品参数，以及 products.after_sale_policy 中的商品级售后说明\n"
                        "- order_agent: 当前用户订单事实查询，例如订单状态、物流状态、签收时间、订单七天无理由时间窗口\n"
                        "- ticket_handler: 售后办理、异常处理、投诉处理和人工介入类请求；"
                        "系统会将退款、退货、换货、投诉、修改地址、赔付、补偿等诉求转成后台客服工单，"
                        "用户不一定会说“创建工单”。\n\n"
                        "重要规则：\n"
                        "1. 如果用户明确要退款、退货、换货、投诉、修改地址或人工处理，返回 ticket_handler。\n"
                        "2. 工单创建时不要为了订单校验并行返回 order_agent；订单号和用户身份由 ticket_handler/ticket_create 工具处理。\n"
                        "3. 只有用户表达条件式办理时，例如“如果能退就帮我退”“先看能不能退，能退的话再处理”，才优先返回 order_agent。\n"
                        "4. 如果只是问订单状态、物流状态、签收时间，返回 order_agent。\n"
                        "5. 如果只是问平台规则，返回 knowledge_rag。\n"
                        "6. 如果问具体商品或商品级售后，返回 product_agent。\n"
                        "7. 多个相互独立的查询任务可以返回多个 Agent。\n\n"
                        "请只返回JSON，不要输出解释：\n"
                        "{\n"
                        '  "target_agents": ["order_agent", "knowledge_rag"],\n'
                        '  "reason": "用户同时查询订单状态和退货政策，两个任务相互独立"\n'
                        "}"
                    )
                ),
            ]

            response = await self.llm.ainvoke(routing_prompt)

            try:
                import json
                parsed = json.loads(response.content.strip())
                llm_agents = self._normalize_target_agents(
                    parsed.get("target_agents", [])
                )

                # 对 LLM 结果再做一次保护：
                # 明确售后办理时，不因为出现订单号就并行 order_agent。
                if has_ticket_action and not has_dependent_ticket_action:
                    llm_agents = [
                        agent for agent in llm_agents
                        if agent != "order_agent"
                    ]
                    if "ticket_handler" not in llm_agents:
                        llm_agents.insert(0, "ticket_handler")

                target_agents = llm_agents

            except Exception:
                target_agents = self._normalize_target_agents(target_agents)

        target_agents = self._normalize_target_agents(target_agents)
        intent = target_agents[0]

        self.working_memory.update(
            session_id,
            {
                "last_intent": intent,
                "last_target_agents": target_agents,
                "last_intent_result": intent_result,
            },
        )

        return {
            **state,
            "intent": intent,
            "target_agents": target_agents,
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

# def route_to_agent(state: AgentState) -> str:
#     """根据意图路由到对应Agent节点"""
#     intent = state.get("intent", "knowledge_rag")
#     route_map = {
#         "knowledge_rag": "knowledge_rag",
#         "product_agent": "product_agent",
#         "order_agent": "order_agent",
#         "ticket_handler": "ticket_handler",
#         # "compliance_checker": "compliance_check",
#     }
#     return route_map.get(intent, "knowledge_rag")
def route_to_agents(state: AgentState) -> list[str]:
    """
    根据 target_agents 路由到一个或多个业务 Agent。
    返回 list 时，LangGraph 会并行执行这些节点。
    """
    valid_agents = {
        "knowledge_rag",
        "product_agent",
        "order_agent",
        "ticket_handler",
    }

    target_agents = state.get("target_agents") or []

    # 兼容旧逻辑：如果没有 target_agents，就使用 intent
    if not target_agents:
        target_agents = [state.get("intent", "knowledge_rag")]

    normalized = []
    for agent in target_agents:
        if agent in valid_agents and agent not in normalized:
            normalized.append(agent)

    return normalized or ["knowledge_rag"]


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
        route_to_agents,
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