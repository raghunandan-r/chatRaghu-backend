"""
Evaluators package for assessing node outputs in the generation graph.

Each evaluator module should expose an async function that takes:
- node_execution: EnrichedNodeExecutionLog
- user_query: str
And returns a NodeEvaluation subclass instance.
"""

from typing import Callable, Dict, List

from .relevance_check import evaluate_relevance_check
from .generate_with_context import evaluate_generate_with_context
from .generate_with_persona import evaluate_generate_with_persona

# Registry mapping node names to their evaluator functions
EVALUATOR_REGISTRY: Dict[str, List[Callable]] = {
    "relevance_check": [evaluate_relevance_check],
    "generate_with_context": [evaluate_generate_with_context],
    "generate_with_persona": [evaluate_generate_with_persona],
}
