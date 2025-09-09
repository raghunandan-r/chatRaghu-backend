from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class NodeEvaluation(BaseModel):
    """Base class for node evaluation results"""

    node_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_success: bool
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class RouterEval(NodeEvaluation):
    classification: str
    routing_correct: bool
    explanation: str


class RAGEval(NodeEvaluation):
    """Evaluation for RAG adapter responses"""

    faithfulness: bool
    answer_relevance: bool
    includes_key_info: bool
    handles_irrelevance: bool
    document_relevance: bool
    is_safe: Optional[bool] = None
    is_clear: Optional[bool] = None
    explanation: str


class HistoryEval(NodeEvaluation):
    """Evaluation for History adapter responses"""

    faithfulness: bool
    answer_relevance: bool
    includes_key_info: bool
    handles_irrelevance: bool
    history_relevance: bool
    is_safe: Optional[bool] = None
    is_clear: Optional[bool] = None
    explanation: str


class SimpleResponseEval(NodeEvaluation):
    """Evaluation for SimpleResponse adapter responses"""

    handles_irrelevance: bool
    response_appropriateness: bool
    is_safe: Optional[bool] = None
    is_clear: Optional[bool] = None
    explanation: str
