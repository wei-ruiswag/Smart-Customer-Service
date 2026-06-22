"""
MCP工具协议服务端 — Model Context Protocol实现
遵循Anthropic MCP标准，通过JSON-RPC 2.0提供工具注册/发现/调用能力。
支持动态工具扩展，Agent通过统一接口调用外部系统。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from datetime import datetime

from mcp.tools.knowledge_tool import knowledge_search
from mcp.tools.product_tool import product_query
from mcp.tools.ticket_tool import ticket_create, ticket_query, ticket_update

@dataclass
class ToolDefinition:
    """MCP工具定义"""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    category: str = "general"
    requires_auth: bool = False


@dataclass
class ToolCallResult:
    """工具调用结果"""
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MCPToolServer:
    """
    MCP工具服务端。

    实现 Model Context Protocol 的核心功能：
    1. 工具注册 (Tool Registration)
    2. 工具发现 (Tool Discovery) - Agent可查询可用工具列表
    3. 工具调用 (Tool Invocation) - 通过JSON-RPC 2.0协议调用
    4. 结果返回 (Result Delivery)

    遵循MCP规范：
    - 使用JSON-RPC 2.0消息格式
    - 支持工具的inputSchema声明
    - 提供标准化的错误码
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._call_log: list[ToolCallResult] = []

    def register_tool(self, tool: ToolDefinition) -> None:
        """注册一个MCP工具"""
        self._tools[tool.name] = tool

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        category: str = "general",
        requires_auth: bool = False,
    ) -> Callable:
        """工具注册装饰器"""
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable:
            tool = ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=func,
                category=category,
                requires_auth=requires_auth,
            )
            self._tools[name] = tool
            return func
        return decorator

    def list_tools(self, category: str | None = None) -> list[dict]:
        """
        工具发现：列出所有可用工具。
        对应MCP的 tools/list 方法。
        """
        tools = []
        for tool in self._tools.values():
            if category and tool.category != category:
                continue
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "category": tool.category,
            })
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """
        工具调用：执行指定工具。
        对应MCP的 tools/call 方法。
        """
        import time

        tool = self._tools.get(name)
        if tool is None:
            result = ToolCallResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' not found. Available: {list(self._tools.keys())}",
            )
            self._call_log.append(result)
            return result

        start = time.time()
        try:
            output = await tool.handler(**arguments)
            duration_ms = (time.time() - start) * 1000

            result = ToolCallResult(
                tool_name=name,
                success=True,
                result=output,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            result = ToolCallResult(
                tool_name=name,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

        self._call_log.append(result)
        return result

    async def handle_jsonrpc(self, request: dict) -> dict:
        """
        处理JSON-RPC 2.0请求。
        MCP协议传输层实现。
        """
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id", 1)

        try:
            if method == "tools/list":
                result = self.list_tools(category=params.get("category"))
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                call_result = await self.call_tool(tool_name, arguments)
                result = {
                    "success": call_result.success,
                    "result": call_result.result,
                    "error": call_result.error,
                }
            elif method == "ping":
                result = {"status": "ok"}
            else:
                return {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": req_id,
                }

            return {"jsonrpc": "2.0", "result": result, "id": req_id}

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": req_id,
            }

    def get_call_log(self, last_n: int = 100) -> list[dict]:
        """获取最近的工具调用日志"""
        return [
            {
                "tool": r.tool_name,
                "success": r.success,
                "duration_ms": r.duration_ms,
                "timestamp": r.timestamp,
                "error": r.error,
            }
            for r in self._call_log[-last_n:]
        ]


def create_default_tools(server: MCPToolServer) -> MCPToolServer:
    """注册默认的MCP工具集"""

    @server.register(
        name="order_query",
        description="查询订单信息，支持按订单号或用户ID查询",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
                "user_id": {"type": "string", "description": "用户ID"},
            },
        },
        category="order",
    )
    async def order_query(order_id: str = "", user_id: str = "") -> dict:
        return {
            "order_id": order_id or "ORD-20260401-001",
            "status": "shipped",
            "amount": 299.00,
            "product": "智能理财产品A",
            "created_at": "2026-04-01T10:00:00",
        }

    server.register(
        name="knowledge_search",
        description="搜索企业知识库，返回相关文档片段",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "top_k": {"type": "integer", "description": "返回数量", "default": 3},
            },
            "required": ["query"],
        },
        category="knowledge",
    )(knowledge_search)

    # @server.register(
    #     name="knowledge_search",
    #     description="搜索企业知识库，返回相关文档片段",
    #     input_schema={
    #         "type": "object",
    #         "properties": {
    #             "query": {"type": "string", "description": "搜索查询"},
    #             "top_k": {"type": "integer", "description": "返回数量", "default": 3},
    #         },
    #         "required": ["query"],
    #     },
    #     category="knowledge",
    # )
    # async def knowledge_search(query: str, top_k: int = 3) -> list[dict]:
    #     return KnowledgeService.search(query=query, top_k=top_k)
    #     # return [
    #     #     {"content": f"关于'{query}'的知识库文档片段", "source": "FAQ.md", "score": 0.95},
    #     # ]

    server.register(
        name="ticket_create",
        description="创建客服工单，写入 MySQL tickets 表",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户ID"},
                "order_no": {"type": "string", "description": "关联订单号"},
                "ticket_type": {
                    "type": "string",
                    "description": "工单类型，例如：退款、退货、换货、物流异常、商品质量、发票问题、优惠券问题、投诉、通用",
                },
                "priority": {
                    "type": "string",
                    "enum": ["低", "中", "高", "紧急"],
                    "description": "工单优先级",
                    "default": "中",
                },
                "description": {"type": "string", "description": "工单问题描述"},
            },
            "required": ["user_id", "order_no", "ticket_type", "description"],
        },
        category="ticket",
    )(ticket_create)

    server.register(
        name="ticket_query",
        description="查询客服工单，支持按工单号、订单号或用户ID查询",
        input_schema={
            "type": "object",
            "properties": {
                "ticket_no": {"type": "string", "description": "工单号"},
                "user_id": {"type": "string", "description": "用户ID"},
                "order_no": {"type": "string", "description": "订单号"},
                "limit": {"type": "integer", "description": "返回数量", "default": 5},
            },
        },
        category="ticket",
    )(ticket_query)

    server.register(
        name="ticket_update",
        description="更新客服工单状态",
        input_schema={
            "type": "object",
            "properties": {
                "ticket_no": {"type": "string", "description": "工单号"},
                "status": {
                    "type": "string",
                    "enum": ["待处理", "处理中", "已完成", "已关闭"],
                    "description": "新的工单状态",
                },
            },
            "required": ["ticket_no", "status"],
        },
        category="ticket",
    )(ticket_update)

    # @server.register(
    #     name="ticket_create",
    #     description="创建客服工单",
    #     input_schema={
    #         "type": "object",
    #         "properties": {
    #             "title": {"type": "string"},
    #             "description": {"type": "string"},
    #             "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
    #             "category": {"type": "string"},
    #         },
    #         "required": ["title", "description"],
    #     },
    #     category="ticket",
    # )
    # async def ticket_create(title: str, description: str, priority: str = "medium", category: str = "general") -> dict:
    #     import uuid
    #     return {
    #         "ticket_id": f"TK-{uuid.uuid4().hex[:8].upper()}",
    #         "title": title,
    #         "status": "created",
    #         "priority": priority,
    #     }

    @server.register(
        name="risk_check",
        description="风控接口 — 检查交易/操作的风险等级",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "action": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["user_id", "action"],
        },
        category="compliance",
    )
    async def risk_check(user_id: str, action: str, amount: float = 0.0) -> dict:
        risk_level = "low"
        if amount > 50000:
            risk_level = "high"
        elif amount > 10000:
            risk_level = "medium"

        return {
            "user_id": user_id,
            "action": action,
            "risk_level": risk_level,
            "requires_manual_review": risk_level == "high",
        }

    server.register(
        name="product_query",
        description="查询商品信息，支持按关键词、分类、价格范围和库存状态查询",
        input_schema={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "商品关键词，例如：耳机、手机、键盘"},
                "category": {"type": "string", "description": "商品分类，例如：耳机、手机、家电"},
                "min_price": {"type": "number", "description": "最低价格"},
                "max_price": {"type": "number", "description": "最高价格"},
                "only_in_stock": {"type": "boolean", "description": "是否只查询有库存商品", "default": True},
                "limit": {"type": "integer", "description": "返回数量", "default": 5},
            },
        },
        category="product",
    )(product_query)

    return server
