from pathlib import Path
import shutil

from graph.graph_state import GraphState


INITIAL_TRAIN_POOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "GPT_data_source"
    / "matscholar_data"
    / "train_labeled.txt"
)


def _resolve_required_file(path_value: str, field_name: str) -> None:
    if not path_value:
        raise ValueError(f"{field_name} cannot be empty.")

    path = Path(path_value).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist or is not a file: {path}")

    return


def _reset_train_pool(train_path: str) -> None:
    if not train_path:
        raise ValueError("train_path cannot be empty.")

    if not INITIAL_TRAIN_POOL_PATH.exists() or not INITIAL_TRAIN_POOL_PATH.is_file():
        raise FileNotFoundError(
            "initial train pool source does not exist or is not a file: "
            f"{INITIAL_TRAIN_POOL_PATH}"
        )

    target_path = Path(train_path).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(INITIAL_TRAIN_POOL_PATH, target_path)


def initialize_node(graph_state: GraphState) -> dict:
    """
    节点功能：
        进行图数据的初始化：
        1. 判断训练集，验证集，测试集以及文本池的存在性
        2. 初始化迭代轮数
        3. 初始化 NER-LLM 差异率阈值为 0.5，双 LLM 差异率阈值为 0.3
        4. 设置 processed_sample_ids，previous_metrics 为空列表
        5. 设置 best_metrics 为最低至，保证初始第一次训练的值能够被保存
    """
    _reset_train_pool(graph_state.train_path)
    _resolve_required_file(graph_state.train_path, "train_path")
    _resolve_required_file(graph_state.valid_path, "valid_path")
    _resolve_required_file(graph_state.unlabeled_pool_path,"unlabeled_pool_path",)

    return {
        "iteration": 0,
        "distance_ratio_threshold": 0.50,
        "min_distance_ratio_threshold": 0.30,
        "max_distance_ratio_threshold": 0.80,
        "model_distance_ratio_threshold": 0.30,
        "threshold_step": 0.05,
        "processed_sample_ids": [],
        "previous_metrics": [],
        "best_metrics": {
            "loss": float("inf"),
            "precision": -1.0,
            "recall": -1.0,
            "f1": -1.0,
        },
    }
