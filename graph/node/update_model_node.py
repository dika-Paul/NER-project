from graph.graph_state import GraphState


PRECISION_STABLE_DROP = 0.02


def _require_metric(metrics: dict[str, float], metric_name: str, source_name: str) -> float:
    if metric_name not in metrics:
        raise ValueError(f"{source_name} must contain {metric_name}.")
    return float(metrics[metric_name])


def _decrease_threshold(graph_state: GraphState) -> float:
    return round(max(
        graph_state.min_distance_ratio_threshold,
        graph_state.distance_ratio_threshold - graph_state.threshold_step,
    ), 10)


def _increase_threshold(graph_state: GraphState) -> float:
    return round(min(
        graph_state.max_distance_ratio_threshold,
        graph_state.distance_ratio_threshold + graph_state.threshold_step,
    ), 10)


def update_model_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        根据 current_metrics 与 best_metrics 来判断是否更新差异率阈值：
        1.  若 F1 提升且 Precision 稳定：
            更新 best_model，适当提高阈值，放宽伪标注筛选

            若 F1 不提升或 Precision 明显下降：
            不更新 best_model，适当降低阈值，收紧伪标注筛选

        2.  如果更新 best_model, 直接替换 best_model_path 的地址，而不要创建新的文件
    """
    current_f1 = _require_metric(
        graph_state.current_metrics,
        "f1",
        "current_metrics")
    current_precision = _require_metric(
        graph_state.current_metrics,
        "precision",
        "current_metrics",
    )

    is_first_update = (
        not graph_state.best_metrics
        or float(graph_state.best_metrics.get("f1", -1.0)) < 0
        or not graph_state.best_model_path
    )

    if is_first_update:
        should_update_best_model = True
    else:
        best_f1 = _require_metric(
            graph_state.best_metrics,
            "f1",
            "best_metrics")
        best_precision = _require_metric(
            graph_state.best_metrics,
            "precision",
            "best_metrics",
        )
        f1_improved = current_f1 > best_f1
        precision_stable = current_precision >= best_precision - PRECISION_STABLE_DROP
        should_update_best_model = f1_improved and precision_stable

    if should_update_best_model:
        if not graph_state.ner_model_path:
            raise ValueError("ner_model_path cannot be empty when updating best_model_path.")

        return {
            "best_model_path": graph_state.ner_model_path,
            "best_metrics": graph_state.current_metrics,
            "distance_ratio_threshold": _increase_threshold(graph_state),
        }

    return {
        "distance_ratio_threshold": _decrease_threshold(graph_state),
    }
