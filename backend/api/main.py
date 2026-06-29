"""
FastAPI入口 — 提供REST API + SSE流式响应
"""

from __future__ import annotations

import os
import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.supervisor import create_supervisor_graph
from memory.working_memory import WorkingMemory
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from mcp.mcp_server import MCPToolServer, create_default_tools
from tracing.otel_config import init_tracer, AgentMetrics




load_dotenv()


working_memory = WorkingMemory()
short_term_memory = ShortTermMemory(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
long_term_memory = LongTermMemory()
# long_term_memory = LongTermMemory(index_path=os.getenv("FAISS_INDEX_PATH", "./vector_store/faiss_index"))
mcp_server = create_default_tools(MCPToolServer())
metrics = AgentMetrics()
graph = None


import logging
from pathlib import Path


def setup_compliance_logger():
    """初始化并配置合规/风控审计日志记录器 (Compliance Logger)。

    该日志用于持久化记录 Agent 生成内容的安全风控、合规检查结果。
    """
    # 1. 确保日志存储目录存在
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. 获取或创建名为 "compliance" 的全局单例日志对象
    logger = logging.getLogger("compliance")
    logger.setLevel(logging.INFO)

    # 3. 【核心防坑设计】防止 Uvicorn 等框架热重载(Reload)时重复添加 Handler
    # 如果不加这个判断，每次代码热更新或多次调用该函数，都会往 logger 里重复 addHandler，
    # 导致最终 compliance.log 文件里同一条日志被重复打印 N 次。
    if logger.handlers:
        return

    # 4. 创建文件处理器 (FileHandler)，指定日志写入路径及 UTF-8 编码（防止中文乱码）
    file_handler = logging.FileHandler(
        log_dir / "compliance.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)

    # 5. 定义日志输出格式，标准样式如: 2026-06-18 17:05:00,123 [INFO] 日志内容
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)

    # 6. 将配置好的文件处理器绑定到 logger 对象上
    logger.addHandler(file_handler)

    # 7. 允许日志向上传递 (Propagate)
    # 这样除了写入 compliance.log 外，如果根日志 (Root Logger) 配置了控制台输出，
    # 终端屏幕上也能同时同步看到合规日志。
    logger.propagate = True

# 执行初始化配置
setup_compliance_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global graph

    init_tracer()

    # init_tracer(
    #     service_name=os.getenv("OTEL_SERVICE_NAME", "smart-cs-multi-agent"),
    #     otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    # )

    graph = create_supervisor_graph(
        working_memory=working_memory,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
        mcp_server=mcp_server,
    )

    long_term_memory.add_document(
        content="我们的理财产品A年化收益率为3.5%-5.2%，投资期限为6个月至3年，最低投资金额10000元。注意：理财非存款，产品有风险，投资须谨慎。",
        source="product_faq.md",
    )
    long_term_memory.add_document(
        content="退款政策：用户在购买后7天内可申请无理由退款，超过7天需提供合理原因。退款将在3-5个工作日内原路退回。",
        source="refund_policy.md",
    )
    long_term_memory.add_document(
        content="开户流程：1.准备身份证原件 2.填写开户申请表 3.进行视频认证 4.设置交易密码 5.完成风险评估问卷。整个流程约需15-30分钟。",
        source="account_guide.md",
    )

    yield


app = FastAPI(
    title="智能客服多Agent系统",
    description="基于LangGraph的Supervisor编排多Agent智能客服系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    # intent: str
    # compliance_passed: bool


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """主聊天接口"""
    if graph is None:
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())

    await short_term_memory.add_message(session_id, "user", request.message)

    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "compliance_report": {},
        "final_response": "",
        "current_agent": "",
        "retry_count": 0,
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

    final_response = result.get("final_response", "系统处理异常，请稍后重试")

    await short_term_memory.add_message(session_id, "assistant", final_response)

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        # intent=result.get("intent", "unknown"),
        # compliance_passed=result.get("compliance_passed", True),
    )


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """获取对话历史"""
    history = await short_term_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/api/tools")
async def list_tools():
    """MCP工具发现接口"""
    return {"tools": mcp_server.list_tools()}


# @app.post("/api/tools/call")
# async def call_tool(request: dict):
#     arguments = request.get("arguments", {}) or {}
#     """MCP工具调用接口"""
#     result = await mcp_server.call_tool(
#         name=request.get("name", ""),
#         arguments=arguments,
#         user_id=request.get("user_id") or arguments.get("user_id") or "debug_user",
#     )
#     return {
#         "success": result.success,
#         "result": result.result,
#         "error": result.error,
#         "duration_ms": result.duration_ms,
#     }

@app.post("/api/tools/call")
async def call_tool(
    request: dict,
    x_tool_key: str | None = Header(default=None),
):
    """
    MCP工具调用调试接口。

    注意：
    1. 默认生产环境关闭
    2. 只有 ENABLE_TOOL_DEBUG_API=true 时可用
    3. 需要请求头 X-Tool-Key
    """
    if os.getenv("ENABLE_TOOL_DEBUG_API", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="接口不存在")

    expected_key = os.getenv("TOOL_DEBUG_API_KEY", "")
    if not expected_key or x_tool_key != expected_key:
        raise HTTPException(status_code=403, detail="无权调用工具调试接口")

    arguments = request.get("arguments", {}) or {}

    result = await mcp_server.call_tool(
        name=request.get("name", ""),
        arguments=arguments,
        agent_name="api_debug",
        user_id=request.get("user_id") or arguments.get("user_id") or "debug_user",
    )

    return {
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }

@app.get("/api/metrics")
async def get_metrics():
    """获取系统指标"""
    return {
        "agent_metrics": metrics.get_summary(),
        "tool_call_log": mcp_server.get_call_log(last_n=20),
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
