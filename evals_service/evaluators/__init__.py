"""
Evaluators package for assessing node outputs in the generation graph.

Each evaluator module should expose an async function that takes:
- node_execution: EnrichedNodeExecutionLog
- user_query: str
And returns a NodeEvaluation subclass instance.
"""

from typing import Callable, Dict, List

from .relevance_check import evaluate_router
from .rag_evaluator import evaluate_rag
from .history_evaluator import evaluate_history
from .simple_response_evaluator import evaluate_simple_response

# Registry mapping node names to their evaluator functions
EVALUATOR_REGISTRY: Dict[str, List[Callable]] = {
    "router": [evaluate_router],
    "generate_answer_with_rag": [evaluate_rag],
    "generate_answer_with_history": [evaluate_history],
    "generate_simple_response": [evaluate_simple_response],
}
