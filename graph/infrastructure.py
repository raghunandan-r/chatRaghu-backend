from abc import ABC
from typing import Dict, Optional, Any, AsyncGenerator, Tuple
from datetime import datetime
from pydantic import BaseModel
from .models import (
    MessagesState,
    StreamingResponse,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

"""
Infrastructure and mixins for graph nodes.

All reusable node mixins (e.g., ClassificationNodeMixin, RetrievalNodeMixin, SystemPromptNodeMixin, ConversationHistoryMixin) should be defined here for consistency and discoverability.
"""


# Node Base Classes
class Node(BaseModel):
    name: str

    async def process(self, state: MessagesState) -> MessagesState:
        raise NotImplementedError

    async def collect_metadata(
        self, state: MessagesState, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Base metadata collection - override in subclasses for specific metadata"""
        return {
            "retrieved_docs": None,
            "system_prompt": None,
            "tags": [],
            "custom_metadata": {},
        }


class StreamingNode(Node):
    """Base class for nodes that support streaming"""

    async def process_stream(
        self, state: MessagesState
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Process the state and yield streaming responses.
        Override this method in derived classes to implement streaming.
        """
        raise NotImplementedError


# Mixins for different node types
class ClassificationNodeMixin:
    """Mixin for nodes that perform classification tasks"""

    async def collect_metadata(
        self, state: MessagesState, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        # Get the classification result from the latest AI message
        latest_ai_message = next(
            (
                msg.content
                for msg in reversed(state.messages)
                if isinstance(msg, AIMessage)
            ),
            "",
        )

        return {
            "retrieved_docs": None,
            "system_prompt": None,
            "tags": ["classification"],
            "custom_metadata": {
                "classification_result": latest_ai_message,
                "processing_time": (end_time - start_time).total_seconds(),
            },
        }


class RetrievalNodeMixin:
    """Mixin for nodes that use retrieved documents"""

    async def collect_metadata(
        self, state: MessagesState, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        # Extract retrieved docs from ToolMessages
        tool_messages = [msg for msg in state.messages if isinstance(msg, ToolMessage)]
        retrieved_docs = []

        for msg in tool_messages:
            if msg.tool_name == "retrieve" and msg.output:
                if isinstance(msg.output, list):
                    for result in msg.output:
                        if hasattr(result, "dict"):
                            retrieved_docs.append(result.dict())
                        else:
                            retrieved_docs.append(
                                {"content": str(result), "metadata": {}}
                            )
                else:
                    retrieved_docs.append({"content": str(msg.output), "metadata": {}})

        return {
            "retrieved_docs": retrieved_docs,
            "system_prompt": None,
            "tags": ["retrieval"],
            "custom_metadata": {
                "docs_count": len(retrieved_docs),
                "processing_time": (end_time - start_time).total_seconds(),
            },
        }


class SystemPromptNodeMixin:
    """Mixin for nodes that use system prompts"""

    def _get_system_prompt(self, state: MessagesState) -> Optional[str]:
        """Override this method in subclasses to return the actual system prompt used"""
        return None

    async def collect_metadata(
        self, state: MessagesState, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        system_prompt = self._get_system_prompt(state)

        return {
            "retrieved_docs": None,
            "system_prompt": system_prompt,
            "tags": ["system_prompt"],
            "custom_metadata": {
                "processing_time": (end_time - start_time).total_seconds()
            },
        }


class ConversationHistoryMixin:
    """
    Mixin to provide reusable conversation history formatting for LLM input.
    Ensures tool call details are not exposed, but retrieved content is included as assistant messages.
    """

    def _format_retrieved_content(self, output):
        if isinstance(output, list) and all(
            hasattr(item, "content") and hasattr(item, "score") for item in output
        ):
            return "\n\n".join(
                f"Content: {item.content} (Score: {item.score:.2f})" for item in output
            )
        elif isinstance(output, str):
            return output
        return str(output)

    def _build_conversation_history(
        self, state, max_history=8, include_tool_content=True, exclude_tool_details=True
    ):
        recent_messages = (
            state.messages[-max_history * 2 :]
            if len(state.messages) > max_history * 2
            else state.messages
        )
        openai_messages = []
        last_human_message = ""
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
                last_human_message = msg.content
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif (
                include_tool_content
                and isinstance(msg, ToolMessage)
                and msg.tool_name == "retrieve"
            ):
                # Only include the retrieved content, not the tool call details
                retrieved_content = self._format_retrieved_content(msg.output)
                openai_messages.append(
                    {"role": "assistant", "content": retrieved_content}
                )
        return openai_messages, last_human_message


# Make StateGraph an abstract base class
class StateGraph(BaseModel, ABC):
    nodes: Dict[str, Node]
    edges: Dict[str, Dict[str, str]]
    entry_point: str
    queue_manager: Optional[Any] = None  # Add this field

    # not called, overriden. remove?
    async def execute(
        self, initial_state: MessagesState
    ) -> AsyncGenerator[Tuple[StreamingResponse, Dict], None]:
        """
        Execute the graph with the given initial state.
        This is an abstract method that must be implemented by subclasses.
        """
        pass


class StreamingStateGraph(StateGraph):
    """Extension of StateGraph that supports streaming responses"""

    def __init__(
        self,
        nodes: Dict[str, Node],
        edges: Dict[str, Dict[str, str]],
        entry_point: str,
        queue_manager: Optional[Any] = None,
    ):
        super().__init__(nodes=nodes, edges=edges, entry_point=entry_point)
        self.queue_manager = queue_manager

    # Note: The execute_stream method implementation is moved to nodes.py
    # to avoid circular import issues. The method will be added dynamically
    # after all imports are resolved.


# --- Lint check stub ---
# To enforce that all node mixins are defined in this file, add a linter or CI check that scans for classes ending with 'Mixin' outside infrastructure.py.
# Example (pseudo-code):
#   for file in graph/*:
#       if file != 'infrastructure.py':
#           assert not any(class_name.endswith('Mixin') for class_name in file)
