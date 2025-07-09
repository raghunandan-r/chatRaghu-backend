from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class NodeEvaluation(BaseModel):
    """Base class for node evaluation results"""

    node_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_success: bool
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class RelevanceCheckEval(NodeEvaluation):
    classification: Literal["IRRELEVANT", "RELEVANT"]
    format_valid: bool
    explanation: str


class QueryOrRespondEval(NodeEvaluation):
    classification: Literal["RETRIEVE", "SUFFICIENT"]
    format_valid: bool
    explanation: str


class FewShotSelectorEval(NodeEvaluation):
    category: Literal["OFFICIAL", "JEST", "HACK"]
    category_appropriate: bool
    style_appropriate: bool
    explanation: str


class GenerateWithContextEval(NodeEvaluation):
    overall_success: bool
    faithfulness: bool
    answer_relevance: bool
    includes_key_info: bool
    handles_irrelevance: bool
    context_relevance: bool
    explanation: str


class GenerateWithPersonaEval(NodeEvaluation):
    persona_adherence: bool
    follows_rules: bool
    faithfulness: bool
    answer_relevance: bool
    handles_irrelevance: bool
    context_relevance: bool
    explanation: str
