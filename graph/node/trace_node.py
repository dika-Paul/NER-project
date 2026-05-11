from functools import wraps
from inspect import signature
from typing import Any, Callable

from langgraph.runtime import Runtime

from graph.graph_state import GraphContext, GraphState


def is_trace_enabled(runtime: Runtime[GraphContext] | None) -> bool:
    if runtime is None:
        return True

    context = getattr(runtime, "context", None)
    if context is None:
        return True

    if isinstance(context, dict):
        return bool(context.get("trace_enabled", True))

    return bool(getattr(context, "trace_enabled", True))


def _node_accepts_runtime(node_fn: Callable[..., Any]) -> bool:
    parameters = signature(node_fn).parameters
    return "runtime" in parameters or len(parameters) >= 2


def _iteration_label(graph_state: GraphState) -> str:
    iteration = getattr(graph_state, "iteration", 0) + 1
    iterations = getattr(graph_state, "iterations", "?")
    return f"{iteration}/{iterations}"


def _batch_size(graph_state: GraphState) -> int:
    current_batch = getattr(graph_state, "current_batch", {})
    if isinstance(current_batch, dict):
        return len(current_batch)
    return 0


def _update_keys(result: Any) -> list[str]:
    if isinstance(result, dict):
        return list(result.keys())
    return [type(result).__name__]


def trace_node(node_name: str, node_fn: Callable[..., Any]) -> Callable[..., Any]:
    accepts_runtime = _node_accepts_runtime(node_fn)

    @wraps(node_fn)
    def traced_node(
        graph_state: GraphState,
        runtime: Runtime[GraphContext] | None = None,
    ) -> Any:
        trace_enabled = is_trace_enabled(runtime)

        if trace_enabled:
            print(
                f"[GraphTrace] -> {node_name} | "
                f"iter={_iteration_label(graph_state)} | "
                f"batch={_batch_size(graph_state)}",
                flush=True,
            )

        try:
            if accepts_runtime:
                result = node_fn(graph_state, runtime)
            else:
                result = node_fn(graph_state)
        except Exception as exc:
            if trace_enabled:
                print(
                    f"[GraphTrace] !! {node_name} | "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            raise

        if trace_enabled:
            print(
                f"[GraphTrace] <- {node_name} | "
                f"updates={_update_keys(result)}",
                flush=True,
            )

        return result

    return traced_node
