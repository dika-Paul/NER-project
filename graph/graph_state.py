from dataclasses import dataclass, field
from typing import Any, TypedDict, Annotated

from langchain.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from torch import nn


def add_metrics(old: Any, new: Any) -> Any:
    if old is None:
        old = []
    if new is None:
        new = []
    if not isinstance(old, list):
        old = [old]
    if not isinstance(new, list):
        new = [new]
    return old + new

@dataclass
class InputState:
    train_path: str = field(default_factory=str)
    valid_path: str = field(default_factory=str)
    test_path: str = field(default_factory=str)
    unlabeled_pool_path: str = field(default_factory=str)
    ner_model_path: str = field(default_factory=str)


@dataclass
class OutputState:
    best_model_path: str = field(default_factory=str)
    previous_metrics: list[dict[str, float]] = field(default_factory=list)
    best_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class GraphState:
    # 训练集路径。
    train_path: str = field(default_factory=str)

    # 验证集路径。
    valid_path: str = field(default_factory=str)

    # 测试集路径。
    test_path: str = field(default_factory=str)

    # 文本池路径。
    unlabeled_pool_path: str = field(default_factory=str)

    # 当前使用的 NER 模型权重路径。
    ner_model_path: str = field(default_factory=str)

    # 最佳 NER 模型权重路径
    best_model_path: str = field(default_factory=str)

    # 当前迭代轮次，初始为 0。
    iteration: int = 0

    # 迭代轮数，防止伪标注循环无限运行。
    iterations: int = 5

    # 每轮从未标注文本池中抽取的样本百分比。
    batch_size: float = 0.2

    # 当前相似度阈值，样本相似度达到该值才视为高置信。
    similarity_threshold: float = 0.60

    # 动态阈值允许下降到的最低值，避免引入过多噪声样本。
    min_similarity_threshold: float = 0.50

    # 动态阈值允许上升到的最高值，避免筛选过严导致无样本可加。
    max_similarity_threshold: float = 0.90

    # 每次根据验证集指标调整阈值时的变化步长。
    threshold_step: float = 0.05

    # 模型相似度阈值
    model_similarity_threshold: float = 0.90

    # 当前轮从未标注池中抽取的样本列表
    current_batch: dict[str, dict[str, Any]] = field(default_factory=dict)

    # 已经被抽取或处理过的未标注样本 id，避免重复标注。
    processed_sample_ids: list[str] = field(default_factory=list)

    # 当前轮 NER 模型输出的 BIO 预测结果。
    ner_bio_results: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    # 当前轮 NER 模型 BIO 结果转换后的实体字典。
    ner_entity_dicts: dict[str, dict[str, list]] = field(default_factory=dict)

    # 模型输出
    llm_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # 每条样本的相似度结果，例如 NER-LLM1、NER-LLM2、LLM1-LLM2 等。
    similarity_records: dict[str, dict[str, Any]] = field(default_factory=dict)

    # 每条样本的最终自动决策记录，例如 accept_ner、accept_llm_consensus、review、reject。
    decision_records: dict[str, str] = field(default_factory=dict)

    # 历史轮次模型在验证集上的指标，例如 precision、recall、f1、loss。
    previous_metrics: Annotated[list[dict[str, float]], add_metrics] = field(default_factory=list)

    # 当前轮模型在验证集上的指标，例如 precision、recall、f1、loss。
    current_metrics: dict[str, float] = field(default_factory=dict)

    # 历史最佳验证集指标，用于保存最佳模型和判断是否继续迭代。
    best_metrics: dict[str, float] = field(default_factory=dict)



class GraphContext(TypedDict, total=False):
    # 控制模型输出的提示词模板
    prompt: ChatPromptTemplate

    # NER模型
    ner_model: nn.Module

    # 实体分类
    entity_schema: list[str]

    # 第一大模型，默认用于所有样本的初始复核。
    llm: BaseChatModel

    # 第二大模型，仅在 NER 与第一大模型不一致时调用。
    judge_llm: BaseChatModel
