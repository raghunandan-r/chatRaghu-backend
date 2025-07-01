from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal, Any
from datetime import datetime


class NodeMessage(BaseModel):
    """Represents a message with its associated node context"""

    content: str
    node_name: str
    message_type: Literal["input", "output"]
    message_source: Literal[
        "human", "ai"
    ] = "ai"  # Track if message is human or AI generated
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NodeExecution(BaseModel):
    """Tracks execution details of a single node"""

    node_name: str
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    messages: List[NodeMessage] = Field(default_factory=list)
    next_edge: Optional[str] = None
    execution_time: datetime = Field(default_factory=datetime.utcnow)


class EnrichedNodeExecutionLog(BaseModel):
    """Enhanced node execution log with detailed metadata"""

    node_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    graph_version: str = ""
    tags: List[str] = Field(default_factory=list)
    message_source: Literal[
        "human", "ai"
    ] = "ai"  # Track if message is human or AI generated


class ConversationFlow(BaseModel):
    """Tracks the complete flow of a conversation through the graph"""

    thread_id: str
    user_query: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    node_executions: List[EnrichedNodeExecutionLog] = Field(
        default_factory=list
    )  # Updated to use EnrichedNodeExecutionLog
    final_response: Optional[str] = None


class EvaluationRequest(BaseModel):
    """Request model for evaluation"""

    thread_id: str
    query: str
    response: str
    retrieved_docs: Optional[List[Dict[str, str]]] = None
    conversation_flow: ConversationFlow

    class Config:
        # Enable forward references
        from_attributes = True


class EvaluationResult(BaseModel):
    thread_id: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    query: Optional[str] = None
    response: Optional[str] = None
    retrieved_docs: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    evaluations: Optional[Dict[str, Any]] = Field(
        default_factory=dict
    )  # Store full evaluation objects
    metadata: Dict[str, Any]


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


class ResponseMessage(BaseModel):
    thread_id: str
    query: str
    response: str
    retrieved_docs: Optional[List[Dict]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3
    conversation_flow: ConversationFlow
