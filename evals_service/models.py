from re import A
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal, Any
from datetime import datetime


class EnrichedNodeExecutionLog(BaseModel):
    """Enhanced node execution log with detailed metadata"""

    node_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    message_source: Literal[
        "human", "ai"
    ] = "ai"  # Track if message is human or AI generated
    # New fields for token tracking
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class ConversationFlow(BaseModel):
    """Tracks the complete flow of a conversation through the graph"""

    # New fields for batching and ordering
    run_id: Optional[str] = None  # For batching evaluations
    thread_id: str
    turn_index: int = 0  # To order turns within a thread
    user_query: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    graph_version: str = ""
    node_executions: List[EnrichedNodeExecutionLog] = Field(
        default_factory=list
    )  # Updated to use EnrichedNodeExecutionLog
    final_response: Optional[str] = None

    # New fields for timing and token tracking
    latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None


class EvaluationRequest(BaseModel):
    """Request model for evaluation"""

    thread_id: str
    query: str
    response: str
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    conversation_flow: ConversationFlow

    class Config:
        # Enable forward references
        from_attributes = True


class EvaluationResult(BaseModel):
    # Enhanced model to match our target schema for analytics

    # Identifiers from the graph flow
    run_id: str
    thread_id: str
    turn_index: int
    graph_version: str = ""

    # Timestamps & Latency
    timestamp_start: datetime  # Renamed from timestamp for clarity
    timestamp_end: datetime = Field(default_factory=datetime.utcnow)
    graph_latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    evaluation_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None  # Total end-to-end latency

    # Core conversation data
    query: str
    response: str
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    # Token Counts
    graph_total_prompt_tokens: Optional[int] = None
    graph_total_completion_tokens: Optional[int] = None
    eval_total_prompt_tokens: Optional[int] = None
    eval_total_completion_tokens: Optional[int] = None

    # The actual evaluation results - now a list for flattened structure
    evaluations: List[Dict[str, Any]] = Field(default_factory=list)

    # Overall metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    correctness_score: float
    relevance_score: float
    groundedness_score: float
    persona_consistency_score: float
    retrieval_quality_score: float


class RetryConfig(BaseModel):
    max_retries: int = 3
    delay_seconds: int = 5
    backoff_factor: float = 2.0
