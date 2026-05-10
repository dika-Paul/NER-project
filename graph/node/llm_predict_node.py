from typing import Any

from langgraph.runtime import Runtime

from graph.graph_state import GraphContext, GraphState


LLM_MAX_CONCURRENCY = 5


def _get_context_value(runtime: Runtime[GraphContext], key: str) -> Any:
    context = getattr(runtime, "context", None)
    if context is None:
        raise ValueError(
            "runtime.context must provide prompt, llm, output_parser, and entity_schema."
        )

    if isinstance(context, dict):
        value = context.get(key)
    else:
        value = getattr(context, key, None)

    if value is None:
        raise ValueError(f"runtime.context must provide {key}.")
    return value


def _to_plain_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        dumped_value = value.model_dump()
        if isinstance(dumped_value, dict):
            return dumped_value

    if hasattr(value, "dict"):
        dumped_value = value.dict()
        if isinstance(dumped_value, dict):
            return dumped_value

    return None


def _entities_from_parsed_output(output: Any) -> list[dict[str, Any]]:
    payload = _to_plain_dict(output)
    if payload is not None:
        entities = payload.get("entities", [])
    elif hasattr(output, "entities"):
        entities = getattr(output, "entities")
    else:
        raise ValueError("Parsed LLM output must provide an entities list.")

    if not isinstance(entities, list):
        raise ValueError("Parsed LLM output entities must be a list.")

    normalized_entities = []
    for entity in entities:
        entity_dict = _to_plain_dict(entity)
        if entity_dict is None:
            continue
        normalized_entities.append(entity_dict)

    return normalized_entities


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _locate_text(text: str, entity_text: str) -> tuple[int, int] | None:
    start = text.find(entity_text)
    if start < 0:
        return None
    return start, start + len(entity_text)


def _normalize_entity(
    entity: Any,
    text: str,
    allowed_types: set[str],
) -> dict[str, Any] | None:
    if not isinstance(entity, dict):
        return None

    entity_type = str(entity.get("type", "")).strip()
    if entity_type not in allowed_types:
        return None

    entity_text = str(entity.get("text", "")).strip()
    if not entity_text:
        return None

    start = _coerce_int(entity.get("start"))
    end = _coerce_int(entity.get("end"))

    if start is not None and end is not None:
        span_is_valid = 0 <= start < end <= len(text)
        if span_is_valid and text[start:end] == entity_text:
            return {
                "type": entity_type,
                "text": entity_text,
                "start": start,
                "end": end,
            }

    located_span = _locate_text(text, entity_text)
    if located_span is None:
        return None

    corrected_start, corrected_end = located_span
    return {
        "type": entity_type,
        "text": entity_text,
        "start": corrected_start,
        "end": corrected_end,
    }


def _normalize_entities(
    raw_entities: list[Any],
    text: str,
    entity_schema: list[str],
) -> list[dict[str, Any]]:
    allowed_types = {str(entity_type) for entity_type in entity_schema}
    normalized_entities = []
    seen_entities = set()

    for raw_entity in raw_entities:
        entity = _normalize_entity(raw_entity, text, allowed_types)
        if entity is None:
            continue

        entity_key = (
            entity["type"],
            entity["text"],
            entity["start"],
            entity["end"],
        )
        if entity_key in seen_entities:
            continue

        seen_entities.add(entity_key)
        normalized_entities.append(entity)

    return normalized_entities


def _entity_dict_from_spans(entities: list[dict[str, Any]]) -> dict[str, list[str]]:
    entity_dict: dict[str, list[str]] = {}
    for entity in entities:
        entity_type = str(entity.get("type", "")).strip()
        entity_text = str(entity.get("text", "")).strip()
        if not entity_type or not entity_text:
            continue

        values = entity_dict.setdefault(entity_type, [])
        if entity_text not in values:
            values.append(entity_text)

    return entity_dict


def _copy_llm_outputs(
    llm_outputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        sample_id: dict(sample_outputs)
        for sample_id, sample_outputs in llm_outputs.items()
    }


def _copy_model_entity_dicts(
    model_entity_dicts: dict[str, dict[str, dict[str, list[str]]]],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    return {
        sample_id: {
            model_name: dict(entity_dict)
            for model_name, entity_dict in sample_dicts.items()
        }
        for sample_id, sample_dicts in model_entity_dicts.items()
    }


def _validate_entity_schema(entity_schema: Any) -> list[str]:
    if not isinstance(entity_schema, list) or not entity_schema:
        raise ValueError("runtime.context entity_schema must be a non-empty list.")
    return [str(entity_type) for entity_type in entity_schema]


def _sample_needs_judge(graph_state: GraphState, sample_id: str) -> bool:
    distance_ratio_record = graph_state.distance_ratio_records.get(sample_id)
    if not isinstance(distance_ratio_record, dict) or "ner_llm" not in distance_ratio_record:
        return False

    try:
        ner_llm_distance_ratio = float(distance_ratio_record["ner_llm"])
    except (TypeError, ValueError):
        return False

    return ner_llm_distance_ratio > graph_state.distance_ratio_threshold


def _predict_spans(
    graph_state: GraphState,
    runtime: Runtime[GraphContext],
    *,
    llm_context_key: str,
    output_key: str,
    sample_filter,
) -> dict:
    if not graph_state.current_batch:
        return {"llm_outputs": {}}

    prompt = _get_context_value(runtime, "prompt")
    llm = _get_context_value(runtime, llm_context_key)
    output_parser = _get_context_value(runtime, "output_parser")
    entity_schema = _validate_entity_schema(_get_context_value(runtime, "entity_schema"))

    chain = prompt | llm | output_parser
    format_instructions = output_parser.get_format_instructions()
    llm_outputs = _copy_llm_outputs(graph_state.llm_outputs)
    prediction_requests = []

    for sample_id, sample in graph_state.current_batch.items():
        if not sample_filter(sample_id):
            continue

        text = str(sample.get("text", "")).strip()
        tokens = sample.get("tokens", [])
        if not isinstance(tokens, list):
            raise ValueError(f"current_batch sample {sample_id} tokens must be a list.")

        prediction_requests.append(
            (
                sample_id,
                text,
                {
                    "sample_id": sample_id,
                    "text": text,
                    "tokens": [str(token) for token in tokens],
                    "entity_schema": entity_schema,
                    "format_instructions": format_instructions,
                },
            )
        )

    if not prediction_requests:
        return {"llm_outputs": llm_outputs}

    responses = chain.batch(
        [request_input for _, _, request_input in prediction_requests],
        config={"max_concurrency": LLM_MAX_CONCURRENCY},
    )

    for (sample_id, text, _request_input), response in zip(
        prediction_requests,
        responses,
    ):
        raw_entities = _entities_from_parsed_output(response)
        entities = _normalize_entities(raw_entities, text, entity_schema)

        sample_outputs = dict(llm_outputs.get(sample_id, {}))
        sample_outputs[output_key] = {"entities": entities}
        llm_outputs[sample_id] = sample_outputs

    return {"llm_outputs": llm_outputs}


def _span2dict(graph_state: GraphState, output_key: str) -> dict:
    if not graph_state.llm_outputs:
        return {"model_entity_dicts": {}}

    model_entity_dicts = _copy_model_entity_dicts(graph_state.model_entity_dicts)

    for sample_id, sample_outputs in graph_state.llm_outputs.items():
        model_output = sample_outputs.get(output_key)
        if not isinstance(model_output, dict):
            continue

        entities = model_output.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        sample_dicts = dict(model_entity_dicts.get(sample_id, {}))
        sample_dicts[output_key] = _entity_dict_from_spans(entities)
        model_entity_dicts[sample_id] = sample_dicts

    return {"model_entity_dicts": model_entity_dicts}


def llm_predict_node(
    graph_state: GraphState,
    runtime: Runtime[GraphContext],
) -> dict:
    """
    节点功能：
        对抽取的未标注样本进行span格式预测：
        1. 处理抽取好的样本
        2. 使用 langchain 来调用大模型，提示词和大模型都来自 context
        3. 对大模型输出的 span 内容进行规范与正确性处理
        4. 处理结果放入 llm_outputs 中的 primary
    """
    return _predict_spans(
        graph_state,
        runtime,
        llm_context_key="llm",
        output_key="primary",
        sample_filter=lambda _sample_id: True,
    )


def primary_span2dict_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        根据 llm_outputs 的 primary 结果构造 model_entity_dicts
    """
    return _span2dict(graph_state, "primary")


def judge_llm_predict_node(
    graph_state: GraphState,
    runtime: Runtime[GraphContext],
) -> dict:
    """
    节点功能：
        对 NER 与 primary LLM 差异率过高的样本进行 judge LLM span 格式预测。
    """
    return _predict_spans(
        graph_state,
        runtime,
        llm_context_key="judge_llm",
        output_key="judge",
        sample_filter=lambda sample_id: _sample_needs_judge(graph_state, sample_id),
    )


def judge_span2dict_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        根据 llm_outputs 的 judge 结果构造 model_entity_dicts 中的 judge 字典。
    """
    return _span2dict(graph_state, "judge")
