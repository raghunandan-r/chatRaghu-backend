import os
import json
import logging
from dataclasses import dataclass, field, asdict

# import redis.asyncio as redis
from upstash_redis.asyncio import Redis
from typing import List, Dict, Optional, Any, Literal, Union, Type


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


logger = logging.getLogger(__name__)

# This remains for local development when SESSION_STORAGE_MODE is 'memory'
THREAD_MESSAGE_STORE: Dict[
    str, List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
] = {}

# A mapping from the message 'type' field to the appropriate dataclass for deserialization.
MESSAGE_TYPE_MAP: Dict[
    str, Type[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
] = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}

# At module level in graph/models.py
SESSION_STORAGE_MODE = os.getenv("SESSION_STORAGE_MODE", "memory")
# This will be the single client instance for sessions, set by the app on startup.
_redis_client: Optional[Redis] = None


def set_global_redis_client(client: Optional[Redis]):
    """Called once by app.py on startup to set the Redis client."""
    global _redis_client
    if SESSION_STORAGE_MODE == "redis" and client:
        _redis_client = client
        logger.info("Global Redis client has been set for session storage.")
    else:
        logger.info("Session storage is configured to use in-memory store.")


# State Models
@dataclass
class MessagesState:
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
    thread_id: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def user_query(self):
        return next(
            (
                msg.content
                for msg in reversed(self.messages)
                if isinstance(msg, HumanMessage)
            ),
            None,
        )

    async def update_thread_store(self):
        """Saves state to Redis if configured and available, otherwise uses in-memory."""
        if _redis_client:
            try:
                key = f"history:{self.thread_id}"
                messages_to_save = self.messages[-24:]
                payload = json.dumps([asdict(msg) for msg in messages_to_save])
                await _redis_client.set(key, payload, ex=86400)
                return
            except Exception as e:
                logger.error(
                    f"Redis write failed for session {self.thread_id}. Falling back to in-memory for this request. Error: {e}"
                )

        # Fallback for both "memory" mode and Redis failure
        THREAD_MESSAGE_STORE[self.thread_id] = self.messages[-24:]

    @classmethod
    async def from_thread(
        cls,
        thread_id: str,
        new_message: Union[AIMessage, ToolMessage, HumanMessage, SystemMessage],
    ) -> "MessagesState":
        """Loads state from Redis if configured and available, otherwise uses in-memory."""
        messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]] = []

        if _redis_client:
            try:
                key = f"history:{thread_id}"
                stored_history = await _redis_client.get(key)
                if stored_history:
                    history_data = json.loads(stored_history)
                    for msg_data in history_data:
                        message_class = MESSAGE_TYPE_MAP.get(msg_data.get("type"))
                        if message_class:
                            messages.append(message_class(**msg_data))
                # If history is loaded, we can return early
                return cls(messages=[*messages, new_message], thread_id=thread_id)
            except Exception as e:
                logger.error(
                    f"Redis read failed for session {thread_id}. Falling back to in-memory for this request. Error: {e}"
                )

        # Fallback for both "memory" mode and Redis failure
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
        return {"content": self.content, "score": self.score, "metadata": self.metadata}


# Global embeddings cache for example selector
EXAMPLE_EMBEDDINGS: Optional[List[List[float]]] = None
QUERY_EMBEDDINGS_CACHE: Dict[str, List[float]] = {}
