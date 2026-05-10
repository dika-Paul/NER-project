

from graph.graph_state import GraphState


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decision_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        计算每个样本最终选择：
            ner_llm <= distance_ratio_threshold
                -> 接受 NER 结果

            ner_llm > distance_ratio_threshold
                -> 看 NER 与 judge 结果

            ner_judge <= model_distance_ratio_threshold
                -> 接受 NER 结果

            ner_llm 与 ner_judge 都超过阈值时：
                llm_judge <= model_distance_ratio_threshold
                    -> 两个大模型一致，接受 LLM 共识

            否则
                -> reject
    """
    decision_records = dict(graph_state.decision_records)

    for sample_id in graph_state.current_batch.keys():
        distance_record = graph_state.distance_ratio_records.get(sample_id, {})
        if not isinstance(distance_record, dict):
            decision_records[sample_id] = "reject"
            continue

        ner_llm_distance_ratio = _safe_float(distance_record.get("ner_llm"))
        if ner_llm_distance_ratio is None:
            decision_records[sample_id] = "reject"
            continue

        if ner_llm_distance_ratio <= graph_state.distance_ratio_threshold:
            decision_records[sample_id] = "accept_ner"
            continue

        ner_judge_distance_ratio = _safe_float(distance_record.get("ner_judge"))
        if (
            ner_judge_distance_ratio is not None
            and ner_judge_distance_ratio <= graph_state.model_distance_ratio_threshold
        ):
            decision_records[sample_id] = "accept_ner"
            continue

        llm_judge_distance_ratio = _safe_float(distance_record.get("llm_judge"))
        if (
            llm_judge_distance_ratio is not None
            and llm_judge_distance_ratio <= graph_state.model_distance_ratio_threshold
        ):
            decision_records[sample_id] = "accept_llm_consensus"
        else:
            decision_records[sample_id] = "reject"

    return {"decision_records": decision_records}
