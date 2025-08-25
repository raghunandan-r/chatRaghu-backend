from __future__ import annotations

from typing import Dict

from graph.engine import GraphEngine
from graph.config import GraphConfig
from graph.adapters import (
    RouterAdapter,
    GenerateSimpleResponseAdapter,
    GenerateAnswerWithHistoryAdapter,
    GenerateAnswerWithRagAdapter,
)


def build_graph() -> tuple[Dict[str, object], Dict[str, Dict[str, str]], str]:
    nodes = {
        # TODO remove the whole thing all the way to retrieval.py? - "deflection_categorizer": DeflectionCategorizerAdapter(),
        "router": RouterAdapter(),
        "generate_simple_response": GenerateSimpleResponseAdapter(),
        "generate_answer_with_history": GenerateAnswerWithHistoryAdapter(),
        "generate_answer_with_rag": GenerateAnswerWithRagAdapter(),
    }

    edges = {
        "router": {
            "answer_with_history": "generate_answer_with_history",
            "retrieve_and_answer": "generate_answer_with_rag",
            "greeting": "generate_simple_response",
            "deflect": "generate_simple_response",
        },
        "generate_simple_response": {"default": "END"},
        "generate_answer_with_history": {"default": "END"},
        "generate_answer_with_rag": {"default": "END"},
    }

    entry_point = "router"
    return nodes, edges, entry_point


def create_engine(
    *, instructor_client, queue_manager=None, config: GraphConfig | None = None
) -> GraphEngine:
    nodes, edges, entry_point = build_graph()
    return GraphEngine(
        nodes=nodes,
        edges=edges,
        entry_point=entry_point,
        instructor_client=instructor_client,
        queue_manager=queue_manager,
        config=config or GraphConfig(),
    )
