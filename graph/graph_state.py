from dataclasses import dataclass, field
from typing import Any, TypedDict, Annotated

from pydantic import BaseModel, Field
from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


class EntitySpan(BaseModel):
    type: str = Field(description="Entity type, must be one of entity_schema")
    text: str | None = Field(
        default=None,
        description="Exact entity text copied from original text",
    )
    start: int | None = Field(
        default=None,
        description="Start character offset in text",
    )
    end: int | None = Field(
        default=None,
        description="Exclusive end character offset in text",
    )


class SampleEntityExtractionOutput(BaseModel):
    sample_id: str = Field(description="The sample_id of the input sample")
    entities: list[EntitySpan] = Field(default_factory=list)


class BatchEntityExtractionOutput(BaseModel):
    samples: list[SampleEntityExtractionOutput] = Field(default_factory=list)


entity_output_parser = PydanticOutputParser(pydantic_object=BatchEntityExtractionOutput)


llm_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        """
        You are a strict named entity extraction assistant for materials science literature.

        Your task is to extract domain entities from each sample in the given sample list.
        Only extract entities whose type is included in the provided entity_schema.

        Rules:
        1. Return only valid JSON. Do not use Markdown. Do not add explanations.
        2. The output must be an object with exactly one key: "samples".
        3. Return one output item for every input sample_id.
        4. Each output item must have: "sample_id" and "entities".
        5. Each entity must have: "type", "text", "start", "end".
        6. "text" must be copied exactly from that sample's original text.
        7. "start" and "end" are character offsets in that sample's original text. "end" is exclusive.
        8. If no entity is found for a sample, return that sample with an empty "entities" list.
        9. Do not invent sample_id values or entities. If uncertain, omit the entity.
        10. Do not extract overlapping duplicate entities.
        11. Prefer complete entity spans over partial spans.
        12. Keep entities sorted by "start" within each sample.
        13. Never output partial entities such as {"type": "MAT"} or {"type": "DSC"}.
        14. If you cannot provide all four fields "type", "text", "start", and "end", omit that entity completely.
        15. If a span looks like an entity but its type is not clearly one of entity_schema, omit it.
        16. If a span is uncertain or cannot be aligned to exact character offsets in the original text, omit it.
        17. Do not output token-level "O" labels because this is span-based extraction.
        18. The model must not create new labels such as UNKNOWN, OTHER, MISC, or UNK.

        Common materials-domain label meanings, when applicable:
        - MAT: inorganic materials, material names, chemical formulas, compounds, composites, dopants.
        - PRO: material properties, physical/chemical properties, performance descriptors.
        - CMT: characterization or measurement methods, instruments, experimental techniques.
        - DSC: sample/material descriptors, morphology, structure, form, sample type.
        - APL: material applications, devices, use cases, functional application targets.
        - SMT: synthesis methods, preparation methods, processing routes.
        - SPL: symmetry labels, phase labels, crystal phases, structural phase names.

        Use only labels that appear in entity_schema.
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        entity_schema:
        {entity_schema}

        samples:
        {samples}

        You must format your output according to these instructions:
        {format_instructions}
        """
    )
])

entity_schema = ["MAT", "SPL", "DSC", "PRO", "APL", "SMT", "CMT"]


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
    unlabeled_pool_path: str = field(default_factory=str)


@dataclass
class OutputState:
    best_model_path: str = field(default_factory=str)
    previous_metrics: list[dict[str, float]] = field(default_factory=list)
    best_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class GraphState:
    # 训练集文件路径。
    # 内容通常为两列 CoNLL/BIO 格式：token label，空行分隔句子。
    # initialize_node 校验该路径；train_ner_node 读取该路径训练当前轮 NER 模型。
    train_path: str = field(default_factory=str)

    # 验证集文件路径。
    # 内容格式与 train_path 相同，用于 train_ner_node 在每轮训练中选择本轮最佳 checkpoint。
    valid_path: str = field(default_factory=str)

    # 未标注文本池路径。
    # 当前实现要求为 JSONL，每行一个样本，例如：
    # {"sample_id": str, "text": str, "tokens": list[str], ...}
    # get_unlabeled_data_node 从该文件中随机抽取本轮 current_batch。
    unlabeled_pool_path: str = field(default_factory=str)

    # 当前轮训练得到的 NER checkpoint 路径。
    # train_ner_node 写入，update_model_node 读取。
    # 路径通常形如 graph/model/<model_type>/<model_type>_iter_<iteration>.pt。
    ner_model_path: str = field(default_factory=str)

    # 历史最佳 NER checkpoint 路径。
    # update_model_node 在当前模型优于 best_metrics 时更新为 ner_model_path。
    # model_predict_node 使用该路径加载模型，对 current_batch 做 BIO 预测。
    best_model_path: str = field(default_factory=str)

    # 当前迭代轮次，initialize_node 重置为 0。
    # get_unlabeled_data_node 使用 iteration 作为随机种子，保证同一轮抽样可复现。
    # 后续迭代控制节点应在每轮结束后递增该值。
    iteration: int = 0

    # 最大迭代轮数。
    # 后续 should_continue/循环控制节点用它判断是否停止，防止伪标注训练无限循环。
    iterations: int = 6

    # 每轮从未标注池抽取的比例，取值范围为 (0, 1]。
    # get_unlabeled_data_node 按全量未标注池计算 ceil(total_pool_size * batch_size)。
    # 注意：这不是 DataLoader 的 batch size。
    batch_size: float = 0.25

    # NER 输出与 primary LLM 输出之间的编辑距离比例阈值。
    # 后续相似度节点按论文公式计算 distance_ratio = edit_distance / len(reference_string)。
    # distance_ratio <= distance_ratio_threshold 时，样本可视为高置信候选。
    # 阈值越大筛选越宽松，阈值越小筛选越严格。
    distance_ratio_threshold: float = 0.50

    # distance_ratio_threshold 动态调整时允许下降到的最低值。
    # update_model_node 降低阈值时不会低于该值，避免筛选过严导致长期无样本加入。
    min_distance_ratio_threshold: float = 0.30

    # distance_ratio_threshold 动态调整时允许上升到的最高值。
    # update_model_node 提高阈值时不会高于该值，避免引入过多噪声样本。
    max_distance_ratio_threshold: float = 0.80

    # 差异率阈值每次动态调整的步长。
    # update_model_node 使用 distance_ratio_threshold +/- threshold_step。
    threshold_step: float = 0.05

    # primary LLM 与 judge LLM 之间的编辑距离比例阈值。
    # 后续距离/决策节点可用 distance_ratio_records["llm_judge"] 与该值比较。
    # distance_ratio <= model_distance_ratio_threshold 时，认为两次大模型输出形成可接受共识。
    # 该阈值固定得更严格，用于要求两个大模型输出差异尽可能小。
    model_distance_ratio_threshold: float = 0.30

    # 当前轮从未标注池抽取出的样本。
    # get_unlabeled_data_node 写入；model_predict_node 和后续 LLM 节点读取。
    # 结构：
    # {
    #     sample_id: {"text": str, "tokens": list[str], ...}
    # }
    # 外层 key 是样本 id；内层不再保存 sample_id，方便用 id 直接索引样本内容。
    current_batch: dict[str, dict[str, Any]] = field(default_factory=dict)

    # 已经成功加入训练集的未标注样本 id 列表。
    # get_unlabeled_data_node 只读取该列表；add_train_data_node 在样本实际写入训练集后追加。
    # 结构：[sample_id, ...]
    processed_sample_ids: list[str] = field(default_factory=list)

    # 当前轮最佳 NER 模型对 current_batch 的 BIO 预测结果。
    # model_predict_node 写入；BIO2dict_node 读取。
    # 结构：
    # {
    #     sample_id: [(token, bio_label), ...]
    # }
    # 示例：{"s1": [("NiZn", "B-MAT"), ("ferrite", "I-MAT")]}
    ner_bio_results: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    # 由 ner_bio_results 转换得到的 NER 实体字典。
    # BIO2dict_node 写入；相似度计算节点读取。
    # 结构：
    # {
    #     sample_id: {
    #         entity_type: [entity_text, ...]
    #     }
    # }
    # 示例：{"s1": {"MAT": ["NiZn ferrite"], "CMT": ["EPR"]}}
    ner_entity_dicts: dict[str, dict[str, list]] = field(default_factory=dict)

    # 大模型对 current_batch 的实体抽取输出，按样本聚合。
    # 后续 primary LLM 节点写入 "primary"，judge LLM 节点可补充 "judge"。
    # 每个 LLM 输出使用 span 格式：
    # {
    #     sample_id: {
    #         "primary": {
    #             "entities": [
    #                 {"type": str, "text": str, "start": int, "end": int}
    #             ]
    #         },
    #         "judge": {
    #             "entities": [...]
    #         }
    #     }
    # }
    # start/end 建议表示实体在原始 text 中的字符起止位置，end 为右开区间。
    llm_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # 大模型 span 输出转换得到的实体字典。
    # primary_span2dict_node 写入 "primary"，后续 judge 节点可补充 "judge"。
    # 结构：
    # {
    #     sample_id: {
    #         "primary": {entity_type: [entity_text, ...]},
    #         "judge": {entity_type: [entity_text, ...]}
    #     }
    # }
    model_entity_dicts: dict[str, dict[str, dict[str, list[str]]]] = field(default_factory=dict)

    # 每条样本的编辑距离比例与阈值判断记录。
    # 距离计算节点写入；决策节点读取。
    # 结构约定：
    # {
    #     sample_id: {
    #         "ner_llm": float,
    #         "ner_judge": float | None,
    #         "llm_judge": float | None,
    #     }
    # }
    # ner_llm 通常与 distance_ratio_threshold 比较；llm_judge 通常与 model_distance_ratio_threshold 比较。
    distance_ratio_records: dict[str, dict[str, Any]] = field(default_factory=dict)

    # 每条样本的最终自动决策结果。
    # 后续决策节点根据 ner_entity_dicts、llm_outputs、distance_ratio_records 写入。
    # 结构：
    # {
    #     sample_id: "accept_ner" | "accept_llm_consensus" | "reject"
    # }
    # accept_* 表示可进入伪标注训练集；review/reject 表示需要人工复核或丢弃。
    decision_records: dict[str, str] = field(default_factory=dict)

    # 历史轮次模型在验证集上的指标列表。
    # train_ner_node 每轮返回当前指标并通过 add_metrics 追加。
    # 结构：
    # [
    #     {"loss": float, "precision": float, "recall": float, "f1": float},
    #     ...
    # ]
    previous_metrics: Annotated[list[dict[str, float]], add_metrics] = field(default_factory=list)

    # 当前轮训练出的模型在验证集上的最佳指标。
    # train_ner_node 写入；update_model_node 读取。
    # 结构：{"loss": float, "precision": float, "recall": float, "f1": float}
    current_metrics: dict[str, float] = field(default_factory=dict)

    # 历史最佳模型对应的验证集指标。
    # initialize_node 初始化为 loss=inf、precision/recall/f1=-1.0；
    # update_model_node 在接受当前模型为最佳模型时更新。
    # 结构：{"loss": float, "precision": float, "recall": float, "f1": float}
    best_metrics: dict[str, float] = field(default_factory=dict)



class GraphContext(TypedDict, total=False):
    # 控制模型输出的提示词模板
    prompt: ChatPromptTemplate

    # 输出解释器
    output_parser: PydanticOutputParser

    # NER 模型类型。
    # 可选值："bilstm_crf"、"bert_bilstm_crf"、"bert_softmax"、"matscibert_softmax"。
    ner_model: str

    # 实体分类
    entity_schema: list[str]

    # 第一大模型，默认用于所有样本的初始复核。
    llm: BaseChatModel

    # 第二大模型，仅在 NER 与第一大模型不一致时调用。
    judge_llm: BaseChatModel

    # 是否打印 graph 节点运行跟踪日志，默认 True。
    trace_enabled: bool
