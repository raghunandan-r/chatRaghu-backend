from __future__ import annotations

from typing import Dict

from graph.engine import GraphEngine
from graph.config import GraphConfig
from graph.adapters import (
    RelevanceCheckAdapter,
    QueryOrRespondAdapter,
    DeflectionCategorizerAdapter,
    GenerateAnswerAdapter,
)


def build_graph() -> tuple[Dict[str, object], Dict[str, Dict[str, str]], str]:
    nodes = {
        "relevance_check": RelevanceCheckAdapter(),
        "query_or_respond": QueryOrRespondAdapter(),
        "deflection_categorizer": DeflectionCategorizerAdapter(),
        "generate_answer": GenerateAnswerAdapter(),
    }

    edges = {
        "relevance_check": {"IRRELEVANT": "deflection_categorizer", "RELEVANT": "query_or_respond"},
        "deflection_categorizer": {"default": "generate_answer"},
        "query_or_respond": {"RETRIEVE": "generate_answer", "SUFFICIENT": "generate_answer"},
        "generate_answer": {"default": "END"},
    }

    entry_point = "relevance_check"
    return nodes, edges, entry_point


def create_engine(*, instructor_client, queue_manager=None, config: GraphConfig | None = None) -> GraphEngine:
    nodes, edges, entry_point = build_graph()
    return GraphEngine(
        nodes=nodes,
        edges=edges,
        entry_point=entry_point,
        instructor_client=instructor_client,
        queue_manager=queue_manager,
        config=config or GraphConfig(),
    )


