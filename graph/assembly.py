from __future__ import annotations

from typing import Dict

from graph.engine import GraphEngine
from graph.config import GraphConfigImmi, GraphConfigDefault
from graph.adapters import (
    RouterAdapter,
    GenerateSimpleResponseAdapter,
    GenerateAnswerWithHistoryAdapter,
    GenerateAnswerWithRagAdapter,
)
from graph.retrieval import RetrieveTool, HybridRetrieveTool


def build_graph_default() -> tuple[Dict[str, object], Dict[str, Dict[str, str]], str]:
    nodes = {
        "router": RouterAdapter(),
        "generate_simple_response": GenerateSimpleResponseAdapter(),
        "generate_answer_with_history": GenerateAnswerWithHistoryAdapter(),
        "generate_answer_with_rag": GenerateAnswerWithRagAdapter(
            retriever=RetrieveTool()
        ),
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


def build_graph_immi() -> tuple[Dict[str, object], Dict[str, Dict[str, str]], str]:
    nodes = {
        "router": RouterAdapter(),
        "generate_simple_response": GenerateSimpleResponseAdapter(),
        "generate_answer_with_history": GenerateAnswerWithHistoryAdapter(),
        "generate_answer_with_rag": GenerateAnswerWithRagAdapter(
            retriever=HybridRetrieveTool()
        ),
    }

    edges = {
        "router": {
            "answer_with_history": "generate_answer_with_history",
            "retrieve_and_answer": "generate_answer_with_rag",
            "greeting": "generate_simple_response",
            "deflect": "generate_simple_response",
            "escalate": "generate_simple_response",
        },
        "generate_simple_response": {"default": "END"},
        "generate_answer_with_history": {"default": "END"},
        "generate_answer_with_rag": {"default": "END"},
    }

    entry_point = "router"
    return nodes, edges, entry_point


def create_engine_default(
    *, instructor_client, queue_manager=None, config: GraphConfigDefault | None = None
) -> GraphEngine:
    nodes, edges, entry_point = build_graph_default()
    return GraphEngine(
        nodes=nodes,
        edges=edges,
        entry_point=entry_point,
        instructor_client=instructor_client,
        queue_manager=queue_manager,
        config=config or GraphConfigDefault(),
    )


def create_engine_immi(
    *, instructor_client, queue_manager=None, config: GraphConfigImmi | None = None
) -> GraphEngine:
    nodes, edges, entry_point = build_graph_immi()
    return GraphEngine(
        nodes=nodes,
        edges=edges,
        entry_point=entry_point,
        instructor_client=instructor_client,
        queue_manager=queue_manager,
        config=config or GraphConfigImmi(),
    )
