
from typing import Any

from rapidfuzz.distance import Levenshtein

from graph.graph_state import GraphState


def _normalize_entity_dict(entity_dict: dict[str, Any] | None) -> str:
    if not isinstance(entity_dict, dict):
        return ""

    normalized_parts = []
    for entity_type in sorted(str(key) for key in entity_dict.keys()):
        values = entity_dict.get(entity_type, [])
        if not isinstance(values, list):
            values = [values]

        normalized_values = sorted(
            {
                str(value).strip()
                for value in values
                if str(value).strip()
            }
        )
        normalized_parts.extend(normalized_values)

    return "".join(normalized_parts)


def _distance_ratio(reference_text: str, candidate_text: str) -> float:
    if not reference_text:
        return 0.0 if not candidate_text else 1.0

    edit_distance = Levenshtein.distance(reference_text, candidate_text)
    return float(edit_distance) / len(reference_text)


def labeled_res_distance_compute_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        用来计算 ner 与 llm 的 distance ratio：
        1. 对每一个样本计算 distance ratio
        2. 将计算结果放入 distance_ratio_records 的 ner_llm 中
    """
    distance_ratio_records = {
        sample_id: dict(record)
        for sample_id, record in graph_state.distance_ratio_records.items()
    }

    for sample_id in graph_state.current_batch.keys():
        ner_entity_dict = graph_state.ner_entity_dicts.get(sample_id, {})
        primary_entity_dict = (
            graph_state.model_entity_dicts
            .get(sample_id, {})
            .get("primary", {})
        )

        ner_text = _normalize_entity_dict(ner_entity_dict)
        primary_text = _normalize_entity_dict(primary_entity_dict)

        sample_record = dict(distance_ratio_records.get(sample_id, {}))
        sample_record["ner_llm"] = _distance_ratio(ner_text, primary_text)
        distance_ratio_records[sample_id] = sample_record

    return {"distance_ratio_records": distance_ratio_records}


def judge_res_distance_compute_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        用来计算引入 judge model 后的 distance ratio：
        1. 计算 NER 与 judge model 的 distance ratio，写入 ner_judge
        2. 计算 primary LLM 与 judge model 的 distance ratio，写入 llm_judge
    """
    distance_ratio_records = {
        sample_id: dict(record)
        for sample_id, record in graph_state.distance_ratio_records.items()
    }

    for sample_id in graph_state.current_batch.keys():
        model_entity_dict = graph_state.model_entity_dicts.get(sample_id, {})
        if not isinstance(model_entity_dict, dict) or "judge" not in model_entity_dict:
            continue

        ner_entity_dict = graph_state.ner_entity_dicts.get(sample_id, {})
        primary_entity_dict = model_entity_dict.get("primary", {})
        judge_entity_dict = model_entity_dict.get("judge", {})

        ner_text = _normalize_entity_dict(ner_entity_dict)
        primary_text = _normalize_entity_dict(primary_entity_dict)
        judge_text = _normalize_entity_dict(judge_entity_dict)

        sample_record = dict(distance_ratio_records.get(sample_id, {}))
        sample_record["ner_judge"] = _distance_ratio(ner_text, judge_text)
        sample_record["llm_judge"] = _distance_ratio(primary_text, judge_text)
        distance_ratio_records[sample_id] = sample_record

    return {"distance_ratio_records": distance_ratio_records}
