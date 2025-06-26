from typing import List, Dict, Optional, Any, Literal, Union
from pydantic import BaseModel, Field


# Message Models
class BaseMessage(BaseModel):
    content: str
    type: str


class HumanMessage(BaseMessage):
    type: Literal["human"] = "human"


class AIMessage(BaseMessage):
    type: Literal["ai"] = "ai"


class SystemMessage(BaseMessage):
    type: Literal["system"] = "system"


class ToolMessage(BaseMessage):
    type: Literal["tool"] = "tool"
    tool_name: str
    input: Dict[str, Any]
    output: Any


# State Models
class MessagesState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
    thread_id: Optional[str] = None  # Adding thread_id but keeping it optional

    def update_thread_store(self):
        """Update global message store with current state"""
        if self.thread_id:
            THREAD_MESSAGE_STORE[self.thread_id] = self.messages[
                -24:
            ]  # Keep last 24 messages

    @classmethod
    def from_thread(cls, thread_id: str, new_message: HumanMessage) -> "MessagesState":
        """Create state from thread history + new message"""
        messages = THREAD_MESSAGE_STORE.get(thread_id, [])
        return cls(messages=[*messages, new_message], thread_id=thread_id)


class StreamingState(BaseModel):
    """Tracks the state of a streaming response"""

    buffer: str = ""
    is_function_call: bool = False
    function_name: Optional[str] = None
    function_args: Dict[str, Any] = Field(default_factory=dict)


class StreamingResponse(BaseModel):
    """Represents a chunk of a streaming response"""

    content: Optional[str] = None
    type: Literal["content", "function_call", "end"] = "content"
    function_name: Optional[str] = None
    function_args: Optional[Dict[str, Any]] = None


# Tool Models
class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

    async def execute(self, **kwargs) -> Any:
        raise NotImplementedError


class RetrievalResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Global thread message store
THREAD_MESSAGE_STORE: Dict[
    str, List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
] = {}

# Global embeddings cache for example selector
EXAMPLE_EMBEDDINGS: Optional[List[List[float]]] = None
QUERY_EMBEDDINGS_CACHE: Dict[str, List[float]] = {}
