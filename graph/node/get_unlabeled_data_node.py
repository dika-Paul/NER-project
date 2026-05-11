import json
import math
import random
from pathlib import Path
from typing import Any

from graph.graph_state import GraphState


REQUIRED_SAMPLE_FIELDS = ("sample_id", "text", "tokens")


def _load_unlabeled_pool(unlabeled_pool_path: str) -> list[dict[str, Any]]:
    if not unlabeled_pool_path:
        raise ValueError("unlabeled_pool_path cannot be empty.")

    path = Path(unlabeled_pool_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"unlabeled_pool_path does not exist or is not a file: {path}")

    samples = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in unlabeled_pool_path at line {line_number}: {exc}"
                ) from exc

            if not isinstance(sample, dict):
                raise ValueError(
                    f"Sample at line {line_number} must be a JSON object."
                )

            for field_name in REQUIRED_SAMPLE_FIELDS:
                if field_name not in sample:
                    raise ValueError(
                        f"Sample at line {line_number} must contain {field_name}."
                    )

            samples.append(sample)

    return samples


def _get_sample_id(sample: dict[str, Any]) -> str:
    sample_id = sample["sample_id"]
    if sample_id is None or str(sample_id) == "":
        raise ValueError("sample_id cannot be empty.")
    return str(sample_id)


def get_unlabeled_data_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        从文本池中获取一定量的未标注样本：
        1. 根据抽取百分比从文本池中随机获取样本
        2. 通过 processed_sample_ids 筛去已经成功加入训练集的样本
        3. 将本次提取样本整理成 current_batch 形式
    """
    if not 0 < graph_state.batch_size <= 1:
        raise ValueError("batch_size must be in the range (0, 1].")

    samples = _load_unlabeled_pool(graph_state.unlabeled_pool_path)
    if not samples:
        return {"current_batch": {}}

    # processed_sample_ids means samples already appended to the train set.
    processed_ids = list(graph_state.processed_sample_ids)
    processed_id_set = set(processed_ids)

    available_samples = [
        sample for sample in samples if _get_sample_id(sample) not in processed_id_set
    ]

    if not available_samples:
        return {"current_batch": {}}

    sample_count = max(1, math.ceil(len(samples) * graph_state.batch_size))
    sample_count = min(sample_count, len(available_samples))

    random_generator = random.Random(graph_state.iteration)
    selected_samples = random_generator.sample(available_samples, sample_count)

    current_batch = {}
    for sample in selected_samples:
        sample_id = _get_sample_id(sample)
        current_batch[sample_id] = {
            key: value for key, value in sample.items() if key != "sample_id"
        }

    return {"current_batch": current_batch}
