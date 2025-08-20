from typing import List, Dict, Optional, Any, Literal, Union
from dataclasses import dataclass, field

# Message Models
@dataclass
class BaseMessage:
    content: str
    type: str


@dataclass
class HumanMessage:
    content: str
    type: Literal["human"] = "human"


@dataclass  
class AIMessage:
    content: str
    type: Literal["ai"] = "ai"


@dataclass
class SystemMessage:
    content: str
    type: Literal["system"] = "system"


@dataclass
class ToolMessage:
    content: str
    tool_name: str
    input: Dict[str, Any]
    output: Any
    type: Literal["tool"] = "tool"


# State Models
@dataclass
class MessagesState:
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
    thread_id: str
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def user_query(self):
        return next(
            (msg.content for msg in reversed(self.messages) if isinstance(msg, HumanMessage)),
            None,
        )

    def update_thread_store(self):
        """Update global message store with current state"""
        THREAD_MESSAGE_STORE[self.thread_id] = self.messages[-24:]  # Keep last 24 messages

    @classmethod
    def from_thread(cls, thread_id: str, new_message: Union[AIMessage, ToolMessage, HumanMessage, SystemMessage]) -> "MessagesState":
        """Create state from thread history + new message"""
        messages = THREAD_MESSAGE_STORE.get(thread_id, [])
        return cls(messages=[*messages, new_message], thread_id=thread_id)

@dataclass
class StreamingResponse:
    """Represents a chunk of a streaming response"""
    # All fields have defaults, so order doesn't matter, but group logically
    type: Literal["content", "function_call", "end", "usage"] = "content"
    content: Optional[str] = None
    function_name: Optional[str] = None
    function_args: Optional[Dict[str, Any]] = None
    usage_stats: Optional[Dict[str, Any]] = None  # For token usage in streaming


# Tool Models  
@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]

    async def execute(self, **kwargs) -> Any:
        raise NotImplementedError

@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'content' : self.content,
            'score' : self.score,
            'metadata' : self.metadata
        }

# Global thread message store
THREAD_MESSAGE_STORE: Dict[
    str, List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
] = {}

# Global embeddings cache for example selector
EXAMPLE_EMBEDDINGS: Optional[List[List[float]]] = None
QUERY_EMBEDDINGS_CACHE: Dict[str, List[float]] = {}
