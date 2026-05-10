
from pathlib import Path
from typing import Any

from graph.graph_state import GraphState


ACCEPT_NER = "accept_ner"
ACCEPT_LLM_CONSENSUS = "accept_llm_consensus"


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _token_char_spans(text: str, tokens: list[str]) -> list[tuple[int, int] | None]:
    token_spans = []
    search_start = 0

    for token in tokens:
        start = text.find(token, search_start)
        if start < 0:
            token_spans.append(None)
            continue

        end = start + len(token)
        token_spans.append((start, end))
        search_start = end

    return token_spans


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _llm_entities_to_bio(
    text: str,
    tokens: list[str],
    entities: list[Any],
) -> list[tuple[str, str]] | None:
    if not tokens:
        return None

    token_spans = _token_char_spans(text, tokens)
    labels = ["O"] * len(tokens)
    applied_entity_count = 0

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        entity_type = str(entity.get("type", "")).strip()
        start = _coerce_int(entity.get("start"))
        end = _coerce_int(entity.get("end"))
        if not entity_type or start is None or end is None:
            continue
        if start < 0 or start >= end or end > len(text):
            continue

        matched_token_indexes = []
        entity_span = (start, end)
        for token_index, token_span in enumerate(token_spans):
            if token_span is None or labels[token_index] != "O":
                continue
            if _spans_overlap(token_span, entity_span):
                matched_token_indexes.append(token_index)

        if not matched_token_indexes:
            continue

        applied_entity_count += 1
        for position, token_index in enumerate(matched_token_indexes):
            prefix = "B" if position == 0 else "I"
            labels[token_index] = f"{prefix}-{entity_type}"

    if applied_entity_count == 0:
        return None

    return list(zip(tokens, labels))


def _get_primary_entities(graph_state: GraphState, sample_id: str) -> list[Any] | None:
    sample_outputs = graph_state.llm_outputs.get(sample_id)
    if not isinstance(sample_outputs, dict):
        return None

    primary_output = sample_outputs.get("primary")
    if not isinstance(primary_output, dict):
        return None

    entities = primary_output.get("entities")
    if not isinstance(entities, list):
        return None

    return entities


def _get_accepted_bio_sequences(
    graph_state: GraphState,
) -> list[list[tuple[str, str]]]:
    accepted_sequences = []

    for sample_id, sample in graph_state.current_batch.items():
        decision = graph_state.decision_records.get(sample_id)

        if decision == ACCEPT_NER:
            bio_sequence = graph_state.ner_bio_results.get(sample_id)
            if bio_sequence:
                accepted_sequences.append(
                    [(str(token), str(label)) for token, label in bio_sequence]
                )
            continue

        if decision != ACCEPT_LLM_CONSENSUS:
            continue

        tokens = sample.get("tokens", [])
        if not isinstance(tokens, list):
            continue

        entities = _get_primary_entities(graph_state, sample_id)
        if entities is None:
            continue

        bio_sequence = _llm_entities_to_bio(
            text=str(sample.get("text", "")),
            tokens=[str(token) for token in tokens],
            entities=entities,
        )
        if bio_sequence:
            accepted_sequences.append(bio_sequence)

    return accepted_sequences


def _append_separator(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""

    with path.open("rb") as file:
        file.seek(max(path.stat().st_size - 8, 0))
        tail = file.read().decode("utf-8", errors="ignore").replace("\r\n", "\n")

    if tail.endswith("\n\n"):
        return ""
    if tail.endswith("\n"):
        return "\n"
    return "\n\n"


def _append_bio_sequences(
    train_path: str,
    bio_sequences: list[list[tuple[str, str]]],
) -> None:
    path = Path(train_path).expanduser()
    if not path.parent.exists():
        raise FileNotFoundError(f"train_path parent directory does not exist: {path.parent}")

    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(_append_separator(path))
        for bio_sequence in bio_sequences:
            for token, label in bio_sequence:
                file.write(f"{token} {label}\n")
            file.write("\n")


def add_train_data_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        把接受的样本转换为 BIO 伪标注并追加进训练集：
        1. 接受ner输出的直接添加BIO数据
        2. 接受llm输出的先处理span转化为BIO再添加
    """
    accepted_bio_sequences = _get_accepted_bio_sequences(graph_state)
    if accepted_bio_sequences:
        _append_bio_sequences(graph_state.train_path, accepted_bio_sequences)

    return {"train_path": graph_state.train_path}
