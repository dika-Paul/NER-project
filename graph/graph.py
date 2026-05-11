from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from graph.graph_state import GraphContext, GraphState, InputState, OutputState
from graph.node.add_train_data_node import add_train_data_node
from graph.node.decision_node import decision_node
from graph.node.distance_compute_node import (
    judge_res_distance_compute_node,
    labeled_res_distance_compute_node,
)
from graph.node.get_unlabeled_data_node import get_unlabeled_data_node
from graph.node.initialize_node import initialize_node
from graph.node.iteration_update import iteration_update_node
from graph.node.llm_predict_node import (
    judge_llm_predict_node,
    judge_span2dict_node,
    llm_predict_node,
    primary_span2dict_node,
)
from graph.node.model_predict_node import BIO2dict_node, model_predict_node
from graph.node.train_ner_node import train_ner_node
from graph.node.trace_node import is_trace_enabled, trace_node
from graph.node.update_model_node import update_model_node


def should_judge(
    graph_state: GraphState,
    runtime: Runtime[GraphContext],
) -> Literal["judge_llm_predict", "decision"]:
    for sample_id in graph_state.current_batch.keys():
        distance_record = graph_state.distance_ratio_records.get(sample_id, {})
        if not isinstance(distance_record, dict):
            continue

        try:
            ner_llm_distance_ratio = float(distance_record.get("ner_llm"))
        except (TypeError, ValueError):
            continue

        if ner_llm_distance_ratio > graph_state.distance_ratio_threshold:
            if is_trace_enabled(runtime):
                print("[GraphRoute] should_judge -> judge_llm_predict", flush=True)
            return "judge_llm_predict"

    if is_trace_enabled(runtime):
        print("[GraphRoute] should_judge -> decision", flush=True)
    return "decision"


def should_continue_training(
    graph_state: GraphState,
    runtime: Runtime[GraphContext],
) -> Literal["continue", "end"]:
    if graph_state.iteration < graph_state.iterations:
        if is_trace_enabled(runtime):
            print("[GraphRoute] should_continue_training -> continue", flush=True)
        return "continue"

    if is_trace_enabled(runtime):
        print("[GraphRoute] should_continue_training -> end", flush=True)
    return "end"


def build_train_graph():
    graph_builder = StateGraph(
        GraphState,
        context_schema=GraphContext,
        input_schema=InputState,
        output_schema=OutputState
    )

    graph_builder.add_node("initialize", trace_node("initialize", initialize_node))
    graph_builder.add_node("train_ner", trace_node("train_ner", train_ner_node))
    graph_builder.add_node("update_model", trace_node("update_model", update_model_node))
    graph_builder.add_node(
        "get_unlabeled_data",
        trace_node("get_unlabeled_data", get_unlabeled_data_node),
    )
    graph_builder.add_node("model_predict", trace_node("model_predict", model_predict_node))
    graph_builder.add_node("bio_to_dict", trace_node("bio_to_dict", BIO2dict_node))
    graph_builder.add_node("llm_predict", trace_node("llm_predict", llm_predict_node))
    graph_builder.add_node(
        "primary_span_to_dict",
        trace_node("primary_span_to_dict", primary_span2dict_node),
    )
    graph_builder.add_node(
        "compute_ner_llm_distance",
        trace_node("compute_ner_llm_distance", labeled_res_distance_compute_node),
    )
    graph_builder.add_node(
        "judge_llm_predict",
        trace_node("judge_llm_predict", judge_llm_predict_node),
    )
    graph_builder.add_node(
        "judge_span_to_dict",
        trace_node("judge_span_to_dict", judge_span2dict_node),
    )
    graph_builder.add_node(
        "compute_judge_distance",
        trace_node("compute_judge_distance", judge_res_distance_compute_node),
    )
    graph_builder.add_node("decision", trace_node("decision", decision_node))
    graph_builder.add_node("add_train_data", trace_node("add_train_data", add_train_data_node))
    graph_builder.add_node(
        "iteration_update",
        trace_node("iteration_update", iteration_update_node),
    )

    graph_builder.add_edge(START, "initialize")
    graph_builder.add_edge("initialize", "train_ner")
    graph_builder.add_edge("train_ner", "update_model")
    graph_builder.add_edge("update_model", "get_unlabeled_data")
    graph_builder.add_edge("get_unlabeled_data", "model_predict")
    graph_builder.add_edge("model_predict", "bio_to_dict")
    graph_builder.add_edge("get_unlabeled_data", "llm_predict")
    graph_builder.add_edge("llm_predict", "primary_span_to_dict")
    graph_builder.add_edge(["bio_to_dict", "primary_span_to_dict"], "compute_ner_llm_distance")
    graph_builder.add_conditional_edges(
        "compute_ner_llm_distance",
        should_judge,
        {
            "judge_llm_predict": "judge_llm_predict",
            "decision": "decision",
        },
    )
    graph_builder.add_edge("judge_llm_predict", "judge_span_to_dict")
    graph_builder.add_edge("judge_span_to_dict", "compute_judge_distance")
    graph_builder.add_edge("compute_judge_distance", "decision")
    graph_builder.add_edge("decision", "add_train_data")
    graph_builder.add_edge("add_train_data", "iteration_update")
    graph_builder.add_conditional_edges(
        "iteration_update",
        should_continue_training,
        {
            "continue": "train_ner",
            "end": END,
        },
    )

    return graph_builder.compile()
