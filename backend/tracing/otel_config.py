"""
全链路追踪 — OpenTelemetry 集成
为每个 Agent 调用创建 Span，记录延迟、路由决策、成功失败等信息。
支持 console / OTLP(Jaeger) 两种导出方式。
"""

from __future__ import annotations

import functools
import inspect
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


_tracer = None
_initialized = False


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def init_tracer() -> None:
    """
    初始化 OpenTelemetry 追踪器。

    .env 示例：
    ENABLE_OTEL=true
    OTEL_EXPORTER=otlp
    OTEL_SERVICE_NAME=smart-cs-multi-agent
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    OTEL_TRACES_SAMPLER=always_on
    """
    global _tracer, _initialized

    if _initialized:
        return

    _initialized = True

    if not _HAS_OTEL:
        print("[OTEL] opentelemetry 未安装，跳过 tracing 初始化")
        return

    enable_otel = _env_bool("ENABLE_OTEL", default=False)
    if not enable_otel:
        print("[OTEL] ENABLE_OTEL=false，tracing 已关闭")
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "smart-cs-multi-agent")
    exporter_type = os.getenv("OTEL_EXPORTER", "console").lower()
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": os.getenv("SERVICE_VERSION", "dev"),
            "deployment.environment": os.getenv("APP_ENV", "local"),
        }
    )

    provider = TracerProvider(resource=resource)

    if exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True,
                timeout=1,
            )
            print(f"[OTEL] tracing 导出到 OTLP: {otlp_endpoint}")

        except ImportError:
            print("[OTEL] 未安装 opentelemetry-exporter-otlp，回退到 ConsoleSpanExporter")
            exporter = ConsoleSpanExporter()

    else:
        print("[OTEL] tracing 使用 ConsoleSpanExporter")
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(service_name)


def get_tracer():
    """
    获取全局 Tracer。
    如果还没有初始化，会尝试自动初始化。
    """
    global _tracer

    if _tracer is None:
        init_tracer()

    return _tracer


def trace_agent_call(agent_name: str) -> Callable:
    """
    Agent 调用追踪装饰器。
    支持 async / sync 函数。
    """
    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                tracer = get_tracer()

                if tracer is None:
                    return await func(*args, **kwargs)

                span_name = f"agent.{agent_name}.{func.__name__}"

                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute("agent.name", agent_name)
                    span.set_attribute("agent.method", func.__name__)

                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        duration_ms = (time.time() - start_time) * 1000

                        span.set_attribute("agent.duration_ms", duration_ms)
                        span.set_attribute("agent.success", True)

                        if isinstance(result, dict):
                            span.set_attribute("agent.result_keys", str(list(result.keys())))

                        return result

                    except Exception as e:
                        duration_ms = (time.time() - start_time) * 1000

                        span.set_attribute("agent.duration_ms", duration_ms)
                        span.set_attribute("agent.success", False)
                        span.set_attribute("agent.error", str(e))
                        span.record_exception(e)
                        raise

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            tracer = get_tracer()

            if tracer is None:
                return func(*args, **kwargs)

            span_name = f"agent.{agent_name}.{func.__name__}"

            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("agent.name", agent_name)
                span.set_attribute("agent.method", func.__name__)

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    span.set_attribute("agent.duration_ms", duration_ms)
                    span.set_attribute("agent.success", True)

                    if isinstance(result, dict):
                        span.set_attribute("agent.result_keys", str(list(result.keys())))

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000

                    span.set_attribute("agent.duration_ms", duration_ms)
                    span.set_attribute("agent.success", False)
                    span.set_attribute("agent.error", str(e))
                    span.record_exception(e)
                    raise

        return sync_wrapper

    return decorator


class AgentMetrics:
    """Agent 调用指标收集器"""

    def __init__(self):
        self._call_counts: dict[str, int] = {}
        self._total_duration: dict[str, float] = {}
        self._error_counts: dict[str, int] = {}

    def record_call(self, agent_name: str, duration_ms: float, success: bool):
        self._call_counts[agent_name] = self._call_counts.get(agent_name, 0) + 1
        self._total_duration[agent_name] = self._total_duration.get(agent_name, 0.0) + duration_ms
        if not success:
            self._error_counts[agent_name] = self._error_counts.get(agent_name, 0) + 1

    def get_summary(self) -> dict[str, Any]:
        summary = {}
        for agent_name in self._call_counts:
            calls = self._call_counts[agent_name]
            total_ms = self._total_duration[agent_name]
            errors = self._error_counts.get(agent_name, 0)
            summary[agent_name] = {
                "total_calls": calls,
                "avg_duration_ms": total_ms / calls if calls > 0 else 0,
                "error_rate": errors / calls if calls > 0 else 0,
            }
        return summary