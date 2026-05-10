from graph.graph_state import GraphState


def iteration_update_node(graph_state: GraphState) -> dict:
    return {
        "iteration": graph_state.iteration+1
    }