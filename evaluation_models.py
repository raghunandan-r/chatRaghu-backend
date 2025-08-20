"""
Simplified evaluation models for the main ChatRaghu backend service.

These models are used for communication with the evaluation service
and for internal tracking of conversation flows.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any
from datetime import datetime, timezone

@dataclass
class EnrichedNodeExecutionLog:
    """Enhanced node execution log with detailed metadata"""
    
    node_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    
    message_source: Literal["human", "ai"] = "ai"  # Track if message is human or AI generated
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None    
    # New fields for token tracking
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'EnrichedNodeExecutionLog':
        if 'start_time' in data and isinstance(data['start_time'], str):
            data['start_time'] = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
        if 'end_time' in data and isinstance(data['end_time'], str):
            data['end_time'] = datetime.fromisoformat(data['end_time'].replace('Z', '+00:00'))
        return cls(**data)
    
    def to_dict(self) -> dict:
        """needed as model_dump was used to transfer between services"""
        return {
            'node_name': self.node_name,
            'input': self.input,
            'output': self.output,
            'retrieved_docs': self.retrieved_docs,
            'system_prompt': self.system_prompt,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'tags': self.tags,
            'message_source': self.message_source,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens
        }


@dataclass
class ConversationFlow:
    """Tracks the complete flow of a conversation through the graph"""
    
    thread_id: str
    user_query: str
    
    turn_index: int = 0  # To order turns within a thread
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    node_executions: List[EnrichedNodeExecutionLog] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0    
    run_id: Optional[str] = None  # For batching evaluations
    end_time: Optional[datetime] = None
    graph_version: Optional[str] = None  # Allows the caller to specify; defaults to None if not provided
    final_response: Optional[str] = None
    
    latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationFlow':

        if 'start_time' in data and isinstance(data['start_time'], str):
            data['start_time'] = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
        return cls(**data)
    

    def to_dict(self) -> dict:
        """needed as model_dump was used to transfer between services"""
        return {
            'run_id': self.run_id,
            'thread_id': self.thread_id,
            'turn_index': self.turn_index,
            'user_query': self.user_query,  # Fixed: was 'str' instead of self.user_query
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'graph_version': self.graph_version,
            'node_executions': [node.to_dict() for node in self.node_executions],  # Added missing field
            'final_response': self.final_response,
            'latency_ms': self.latency_ms,
            'time_to_first_token_ms': self.time_to_first_token_ms,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens
        }

