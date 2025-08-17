"""
Simplified evaluation models for the main ChatRaghu backend service.

These models are used for communication with the evaluation service
and for internal tracking of conversation flows.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal, Any
from datetime import datetime, timezone


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
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    graph_version: Optional[str] = None  # Allows the caller to specify; defaults to None if not provided
    node_executions: List[EnrichedNodeExecutionLog] = Field(
        default_factory=list
    )  # Updated to use EnrichedNodeExecutionLog
    final_response: Optional[str] = None

    # New fields for timing and token tracking
    latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
