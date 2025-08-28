from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any
from datetime import datetime, timezone


@dataclass
class EnrichedNodeExecutionLog:
    """Enhanced node execution log with detailed metadata"""

    # Required fields first
    node_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    # Fields with defaults after
    message_source: Literal[
        "human", "ai"
    ] = "ai"  # Track if message is human or AI generated
    tags: List[str] = field(default_factory=list)
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    # New fields for token tracking
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "node_name": self.node_name,
            "input": self.input,
            "output": self.output,
            "message_source": self.message_source,
            "tags": self.tags,
            "retrieved_docs": self.retrieved_docs,
            "system_prompt": self.system_prompt,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class ConversationFlow:
    """Tracks the complete flow of a conversation through the graph"""

    thread_id: str
    user_query: str

    node_executions: List[EnrichedNodeExecutionLog] = field(default_factory=list)
    turn_index: int = 0  # To order turns within a thread
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    graph_version: str = ""

    run_id: Optional[str] = None  # For batching evaluations
    end_time: Optional[datetime] = None
    final_response: Optional[str] = None
    # New fields for timing and token tracking
    latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationFlow":
        if "start_time" in data and isinstance(data["start_time"], str):
            data["start_time"] = datetime.fromisoformat(
                data["start_time"].replace("Z", "+00:00")
            )
        return cls(**data)

    def to_dict(self) -> dict:
        """needed as model_dump was used to transfer between services"""
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "turn_index": self.turn_index,
            "user_query": self.user_query,  # Fixed: was 'str' instead of self.user_query
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "graph_version": self.graph_version,
            "node_executions": [
                node.to_dict() if hasattr(node, "to_dict") else node
                for node in self.node_executions
            ],
            "final_response": self.final_response,
            "latency_ms": self.latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


@dataclass
class EvaluationResult:
    """Enhanced model to match our target schema for analytics"""

    # Required fields first
    run_id: str
    thread_id: str
    turn_index: int
    timestamp_start: datetime  # Renamed from timestamp for clarity
    query: str
    response: str
    # Fields with defaults after
    graph_version: str = ""
    timestamp_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    graph_latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    evaluation_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None  # Total end-to-end latency
    retrieved_docs: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    # Token Counts
    graph_total_prompt_tokens: Optional[int] = None
    graph_total_completion_tokens: Optional[int] = None
    eval_total_prompt_tokens: Optional[int] = None
    eval_total_completion_tokens: Optional[int] = None
    # The actual evaluation results - now a list for flattened structure
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    # Overall metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "turn_index": self.turn_index,
            "timestamp_start": self.timestamp_start.isoformat()
            if self.timestamp_start
            else None,
            "query": self.query,
            "response": self.response,
            "graph_version": self.graph_version,
            "timestamp_end": self.timestamp_end.isoformat()
            if self.timestamp_end
            else None,
            "graph_latency_ms": self.graph_latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "evaluation_latency_ms": self.evaluation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "retrieved_docs": self.retrieved_docs,
            "graph_total_prompt_tokens": self.graph_total_prompt_tokens,
            "graph_total_completion_tokens": self.graph_total_completion_tokens,
            "eval_total_prompt_tokens": self.eval_total_prompt_tokens,
            "eval_total_completion_tokens": self.eval_total_completion_tokens,
            "evaluations": self.evaluations,
            "metadata": self.metadata,
        }


@dataclass
class RetryConfig:
    max_retries: int = 3
    delay_seconds: int = 5
    backoff_factor: float = 2.0
