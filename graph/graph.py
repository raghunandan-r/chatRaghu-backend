from pinecone import Pinecone
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Literal, Union, AsyncGenerator, Tuple
from datetime import datetime
from utils.logger import logger
import json
import re
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, OpenAI
import numpy as np
from opik import track, opik_context
from abc import ABC
from evaluation_models import (
    ResponseMessage,
    ConversationFlow,
    EnrichedNodeExecutionLog,
)

if os.path.exists(".env"):
    load_dotenv(".env")
    load_dotenv(".env.development")

OPIK_API_KEY = os.getenv("OPIK_API_KEY")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE")
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME")

# Load prompt templates from the JSON file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_PATH = os.path.join(CURRENT_DIR, "prompt_templates.json")
with open(TEMPLATES_PATH, "r") as f:
    PROMPT_TEMPLATES = json.load(f)


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


# Global thread message store
THREAD_MESSAGE_STORE: Dict[
    str, List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
] = {}

# Global embeddings cache for example selector
EXAMPLE_EMBEDDINGS: Optional[List[List[float]]] = None
QUERY_EMBEDDINGS_CACHE: Dict[str, List[float]] = {}


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


# OpenAI Client Setup
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Precompile regex patterns for performance
_whitespace_pattern = re.compile(r"\s+")
_xml_tag_pattern = re.compile(r"<[^>]+>")
_special_section_pattern = re.compile(
    r"<(?:questions|tags)>.*?</(?:questions|tags)>", re.IGNORECASE | re.DOTALL
)


def preprocess_text(text: str) -> str:
    """Preprocess text by removing XML sections and normalizing whitespace"""
    text = _special_section_pattern.sub("", text)
    text = _xml_tag_pattern.sub("", text)
    return _whitespace_pattern.sub(" ", text).strip()


# Node Implementation
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


class VectorStore(BaseModel):
    """Vector store implementation using Pinecone"""

    index_name: str
    index: Optional[
        Any
    ] = None  # Change Index to Any since Pinecone's type isn't Pydantic compatible

    def __init__(self, **data):
        super().__init__(**data)
        self.index = pc.Index(self.index_name)

    async def similarity_search(
        self, query_embedding: List[float], k: int = 3
    ) -> List[tuple[Dict, float]]:
        try:
            # Query Pinecone directly with the provided embedding
            # No need to generate embeddings again
            query_response = self.index.query(
                vector=query_embedding, top_k=k, include_metadata=True
            )

            # Format results
            results = []
            for match in query_response.matches:
                doc = {
                    "page_content": match.metadata.get("text", ""),
                    "metadata": match.metadata,
                }
                results.append((doc, match.score))

            return results

        except Exception as e:
            logger.error("Vector store query failed", extra={"error": str(e)})
            raise


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

    @track(capture_output=False)
    async def execute_stream(
        self, initial_state: MessagesState
    ) -> AsyncGenerator[Tuple[StreamingResponse, Dict], None]:
        try:
            logger.info(
                "Starting graph execution", extra={"thread_id": initial_state.thread_id}
            )
            opik_context.update_current_trace(
                name="graph_execution", thread_id=initial_state.thread_id
            )

            # Initialize conversation flow tracking
            conversation_flow = ConversationFlow(
                thread_id=initial_state.thread_id,
                user_query=next(
                    (
                        msg.content
                        for msg in reversed(initial_state.messages)
                        if isinstance(msg, HumanMessage)
                    ),
                    "",
                ),
            )

            current_node = self.entry_point
            state = initial_state

            while current_node:
                logger.info(
                    "Processing node",
                    extra={
                        "node": current_node,
                        "thread_id": state.thread_id,
                        "messages_count": len(state.messages),
                    },
                )

                node = self.nodes[current_node]
                start_time = datetime.utcnow()

                # Create node execution record
                node_log = EnrichedNodeExecutionLog(
                    node_name=current_node,
                    input={
                        "messages": [msg.dict() for msg in state.messages],
                        "thread_id": state.thread_id,
                        "conversation_history": [
                            msg.dict() for msg in state.messages[:-1]
                        ]
                        if len(state.messages) > 1
                        else [],
                    },
                    output={},
                    retrieved_docs=None,
                    system_prompt=None,
                    start_time=start_time,
                    end_time=None,  # Will be set after processing
                    graph_version="v1",
                    tags=[],
                    message_source="ai",
                )

                logger.info(
                    f"Node {current_node} input captured",
                    extra={
                        "thread_id": state.thread_id,
                        "node": current_node,
                        "input_timestamp": start_time,
                    },
                )

                # Handle streaming nodes
                if isinstance(node, GenerateWithPersonaNode):
                    # For streaming nodes, we need to collect the complete response
                    complete_content = []

                    logger.info(
                        "Starting streaming node processing",
                        extra={"node": current_node, "thread_id": state.thread_id},
                    )

                    async for chunk in node.process_stream(state):
                        if chunk.type == "content" and chunk.content:
                            complete_content.append(chunk.content)

                        # Yield each chunk as it comes in
                        yield chunk, {"node": current_node}

                    # After streaming is complete, update the state with the complete response
                    if complete_content and current_node == "generate_with_persona":
                        full_response = "".join(complete_content)
                        opik_context.update_current_trace(
                            output={"full_response": full_response},
                            metadata={"response_length": len(full_response)},
                        )

                        # Update node log with output
                        node_log.output["response"] = full_response
                        node_log.end_time = datetime.utcnow()

                        logger.info(
                            "Completed streaming node processing",
                            extra={
                                "node": current_node,
                                "thread_id": state.thread_id,
                                "response": full_response,
                            },
                        )

                        state.messages.append(AIMessage(content=full_response))
                        state.update_thread_store()

                        # Queue for evaluation if queue manager is available
                        if self.queue_manager:
                            conversation_flow.end_time = datetime.utcnow()
                            conversation_flow.node_executions.append(node_log)

                            user_query = next(
                                (
                                    msg.content
                                    for msg in reversed(state.messages)
                                    if isinstance(msg, HumanMessage)
                                ),
                                None,
                            )

                            retrieved_docs = [
                                msg.output
                                for msg in state.messages
                                if isinstance(msg, ToolMessage)
                            ]

                            # Convert RetrievalResult objects to dictionaries for ResponseMessage
                            converted_docs = []
                            for doc in retrieved_docs:
                                if isinstance(doc, list):
                                    # Handle list of RetrievalResult objects
                                    for result in doc:
                                        if hasattr(result, "dict"):
                                            converted_docs.append(result.dict())
                                        else:
                                            converted_docs.append(
                                                {"content": str(result), "metadata": {}}
                                            )
                                elif hasattr(doc, "dict"):
                                    # Handle single RetrievalResult object
                                    converted_docs.append(doc.dict())
                                else:
                                    # Handle other types
                                    converted_docs.append(
                                        {"content": str(doc), "metadata": {}}
                                    )

                            logger.info(
                                "Queueing response for evaluation",
                                extra={
                                    "thread_id": state.thread_id,
                                    "node_executions_count": len(
                                        conversation_flow.node_executions
                                    ),
                                    "has_retrieved_docs": bool(converted_docs),
                                },
                            )

                            await self.queue_manager.enqueue_response(
                                ResponseMessage(
                                    thread_id=state.thread_id,
                                    query=user_query,
                                    response=full_response,
                                    retrieved_docs=converted_docs,
                                    conversation_flow=conversation_flow,
                                )
                            )

                else:
                    logger.info(
                        "Processing non-streaming node",
                        extra={"node": current_node, "thread_id": state.thread_id},
                    )

                    state = await node.process(state)

                    # Update node log with output
                    node_log.output["messages"] = [
                        msg.dict() for msg in state.messages[-1:]
                    ]
                    node_log.end_time = datetime.utcnow()

                    logger.info(
                        "Completed non-streaming node",
                        extra={
                            "node": current_node,
                            "thread_id": state.thread_id,
                            "processing_time": (
                                datetime.utcnow() - start_time
                            ).total_seconds(),
                        },
                    )

                # Collect node-specific metadata using the new approach
                try:
                    metadata = await node.collect_metadata(
                        state, start_time, node_log.end_time
                    )
                    node_log.retrieved_docs = metadata.get("retrieved_docs")
                    node_log.system_prompt = metadata.get("system_prompt")
                    node_log.tags = metadata.get("tags", [])
                    # Add any custom metadata to the output
                    if metadata.get("custom_metadata"):
                        node_log.output["custom_metadata"] = metadata["custom_metadata"]

                    logger.info(
                        f"Collected metadata for {current_node}",
                        extra={
                            "thread_id": state.thread_id,
                            "node": current_node,
                            "has_retrieved_docs": bool(node_log.retrieved_docs),
                            "has_system_prompt": bool(node_log.system_prompt),
                            "tags": node_log.tags,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to collect metadata for {current_node}",
                        extra={
                            "thread_id": state.thread_id,
                            "node": current_node,
                            "error": str(e),
                        },
                    )

                # Get next node and determine routing
                next_node = self.edges.get(current_node)
                if not next_node:
                    logger.info(
                        "No next node found",
                        extra={
                            "current_node": current_node,
                            "thread_id": state.thread_id,
                        },
                    )
                    break

                condition = "default"
                if current_node == "relevance_check":
                    condition = relevance_condition(state)
                    logger.info(
                        "Routing based on condition",
                        extra={
                            "node": current_node,
                            "condition": condition,
                            "thread_id": state.thread_id,
                        },
                    )
                elif current_node == "query_or_respond":
                    condition = query_or_respond_condition(state)
                    logger.info(
                        "Routing based on condition",
                        extra={
                            "node": current_node,
                            "condition": condition,
                            "thread_id": state.thread_id,
                        },
                    )

                # Record routing decision
                node_log.output["next_edge"] = condition
                conversation_flow.node_executions.append(node_log)

                if condition in next_node:
                    current_node = next_node.get(condition)
                else:
                    current_node = next_node.get("default")

                logger.info(
                    "Node transition",
                    extra={
                        "from_node": node_log.node_name,
                        "to_node": current_node,
                        "condition": condition,
                        "thread_id": state.thread_id,
                    },
                )

                if not current_node or current_node == "END":
                    break

            logger.info(
                "Completed graph execution",
                extra={
                    "thread_id": initial_state.thread_id,
                    "total_nodes_executed": len(conversation_flow.node_executions),
                    "execution_time": (
                        datetime.utcnow() - conversation_flow.start_time
                    ).total_seconds(),
                },
            )

        except Exception as e:
            logger.error(
                "Graph execution failed",
                extra={
                    "thread_id": initial_state.thread_id,
                    "error": str(e),
                    "node": current_node if "current_node" in locals() else None,
                    "conversation_flow": conversation_flow.model_dump(mode="json")
                    if "conversation_flow" in locals()
                    else None,
                },
            )
            raise


# Retrieval Tool Implementation
class RetrieveTool(Tool):
    name: str = "RETRIEVE"
    description: str = "Retrieve information related to a query"
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve information for",
                }
            },
            "required": ["query"],
        }
    )

    @track(capture_input=False)
    async def execute(self, query: str) -> tuple[str, List[RetrievalResult]]:
        try:
            logger.info(
                "Starting retrieval",
                extra={"action": "retrieval_start", "query": query},
            )

            # Get embeddings using OpenAI directly
            query_embedding = await self._get_embedding(query)

            # Get raw results from vector store
            doc_score_pairs = await vector_store.similarity_search(query_embedding)

            # Process results with threshold
            if doc_score_pairs:
                best_score = doc_score_pairs[0][1]
                threshold = max(0.7, best_score * 0.9)

                # Process and filter results in a single pass
                processed_results = []
                serialized_chunks = []

                for doc, score in doc_score_pairs:
                    if score >= threshold:
                        processed_content = preprocess_text(doc["page_content"])
                        serialized_chunks.append(
                            f"Content: {processed_content} (Score: {score:.2f})"
                        )
                        processed_results.append(
                            RetrievalResult(
                                content=processed_content,
                                score=score,
                                metadata=doc["metadata"],
                            )
                        )

                opik_context.update_current_span(
                    name="chunk_retrieval",
                    input={"query": query},
                    output={"docs": processed_results},
                )

                return "\n\n".join(serialized_chunks), processed_results

            return "", []

        except Exception as e:
            logger.error("Retrieval failed", extra={"error": str(e)})
            raise

    async def _get_embedding(self, text: str) -> List[float]:
        response = await client.embeddings.create(
            model="text-embedding-ada-002", input=text
        )
        return response.data[0].embedding


class RelevanceCheckNode(Node, ClassificationNodeMixin):
    name: str = "relevance_check"

    @track(capture_input=False)
    async def process(self, state: MessagesState) -> MessagesState:
        try:
            current_query = next(
                (
                    msg
                    for msg in reversed(state.messages)
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )

            if not current_query:
                raise ValueError("No user query found")

            logger.info(
                "Relevance check initiated",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "current_msg": current_query.content,
                },
            )

            system_message = PROMPT_TEMPLATES["relevance_check"]["system_message"]

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": current_query.content},
            ]

            response = await client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.1
            )
            content = response.choices[0].message.content
            logger.info(
                "Completed relevance check",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "response": content,
                },
            )
            opik_context.update_current_span(
                name="relevance_check",
                input={"query": current_query.content},
                output={"classification": content},
                metadata={"system_prompt": system_message},
            )

            return MessagesState(
                messages=[*state.messages, AIMessage(content=content)],
                thread_id=state.thread_id,
            )

        except Exception as e:
            logger.error(
                "Relevance check failed",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "error": str(e),
                },
            )
            raise


class QueryOrRespondNode(Node, ClassificationNodeMixin):
    name: str = "query_or_respond"

    @track(capture_input=False)
    async def process(self, state: MessagesState) -> MessagesState:
        try:
            # Convert your message history to OpenAI format
            openai_messages = [
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATES["query_or_respond"]["system_message"],
                }
            ]
            last_human_message = ""
            # Add conversation history with proper roles
            for msg in state.messages:
                if isinstance(msg, HumanMessage):
                    openai_messages.append({"role": "user", "content": msg.content})
                    last_human_message = msg.content
                elif isinstance(msg, AIMessage):
                    openai_messages.append(
                        {"role": "assistant", "content": msg.content}
                    )
                elif isinstance(msg, ToolMessage):
                    # For tool messages, show them as function calls and results
                    if msg.tool_name == "retrieve":
                        # First, add the assistant's decision to use the tool
                        openai_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"call_{len(openai_messages)}",
                                        "type": "function",
                                        "function": {
                                            "name": msg.tool_name,
                                            "arguments": json.dumps(msg.input),
                                        },
                                    }
                                ],
                            }
                        )

                        # Then add the tool's response
                        tool_content = ""
                        if isinstance(msg.output, list) and all(
                            isinstance(item, RetrievalResult) for item in msg.output
                        ):
                            tool_content = "\n\n".join(
                                [
                                    f"Content: {item.content} (Score: {item.score:.2f})"
                                    for item in msg.output
                                ]
                            )
                        elif isinstance(msg.output, str):
                            tool_content = msg.output

                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": f"call_{len(openai_messages)-1}",
                                "content": tool_content,
                            }
                        )

            openai_messages.append(
                {
                    "role": "user",
                    "content": f"focus on this latest query from the conversation: {last_human_message}",
                }
            )

            logger.info(
                "Query/Respond check with history",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "message_count": len(openai_messages),
                },
            )

            # Use modern tool calling API
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=openai_messages,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "retrieve",
                            "description": "Retrieve relevant information",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                    }
                ],
                tool_choice="auto",
                stream=False,
            )

            # Validate response structure
            if not hasattr(response, "choices") or not response.choices:
                logger.error(
                    "Invalid response structure - no choices",
                    extra={"thread_id": state.thread_id},
                )
                raise ValueError("Invalid response structure from OpenAI API")

            choice = response.choices[0]
            if not hasattr(choice, "message"):
                logger.error(
                    "Invalid response structure - no message",
                    extra={"thread_id": state.thread_id},
                )
                raise ValueError("Invalid response structure from OpenAI API")

            message = choice.message

            # Check for tool calls (modern API)
            if hasattr(message, "tool_calls") and message.tool_calls:
                # Handle tool call case
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "retrieve":
                    try:
                        query = json.loads(tool_call.function.arguments)["query"]
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(
                            "Failed to parse tool call arguments",
                            extra={
                                "error": str(e),
                                "arguments": tool_call.function.arguments,
                            },
                        )
                        query = last_human_message

                    # Create and use RetrieveTool
                    retriever = RetrieveTool()
                    serialized_chunks, processed_results = await retriever.execute(
                        query
                    )

                    opik_context.update_current_span(
                        name="query_or_respond_retrieve",
                        input={"query": query},
                        output={"response": serialized_chunks},
                        metadata={"docs": processed_results},
                    )

                    return MessagesState(
                        messages=[
                            *state.messages,
                            ToolMessage(
                                content="",
                                tool_name="retrieve",
                                input={"query": query},
                                output=processed_results,
                            ),
                        ],
                        thread_id=state.thread_id,
                    )

            # Check for function calls (legacy API fallback)
            elif hasattr(message, "function_call") and message.function_call:
                function_call = message.function_call
                try:
                    query = json.loads(function_call.arguments)["query"]
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(
                        "Failed to parse function call arguments",
                        extra={"error": str(e), "arguments": function_call.arguments},
                    )
                    query = last_human_message

                # Create and use RetrieveTool
                retriever = RetrieveTool()
                serialized_chunks, processed_results = await retriever.execute(query)

                opik_context.update_current_span(
                    name="query_or_respond_retrieve",
                    input={"query": query},
                    output={"response": serialized_chunks},
                    metadata={"docs": processed_results},
                )

                return MessagesState(
                    messages=[
                        *state.messages,
                        ToolMessage(
                            content="",
                            tool_name="retrieve",
                            input={"query": query},
                            output=processed_results,
                        ),
                    ],
                    thread_id=state.thread_id,
                )

            # Handle text response
            elif hasattr(message, "content") and message.content:
                response_content = message.content.strip()
                logger.info(
                    "Raw response from query_or_respond",
                    extra={"content": response_content, "thread_id": state.thread_id},
                )

                if "RETRIEVE" in response_content.upper():
                    # Text indicates retrieval is needed
                    query = last_human_message
                    logger.info(
                        "Text-based retrieval indication detected",
                        extra={"query": query, "thread_id": state.thread_id},
                    )

                    # Create and use RetrieveTool
                    retriever = RetrieveTool()
                    serialized_chunks, processed_results = await retriever.execute(
                        query
                    )

                    opik_context.update_current_span(
                        name="query_or_respond_retrieve",
                        input={"query": query},
                        output={"response": serialized_chunks},
                        metadata={"docs": processed_results},
                    )

                    return MessagesState(
                        messages=[
                            *state.messages,
                            ToolMessage(
                                content="",
                                tool_name="retrieve",
                                input={"query": query},
                                output=processed_results,
                            ),
                        ],
                        thread_id=state.thread_id,
                    )
                else:
                    # Anything else means sufficient context
                    logger.info(
                        "Sufficient context indicated",
                        extra={"thread_id": state.thread_id},
                    )

                    opik_context.update_current_span(
                        name="query_or_respond_direct",
                        input={"query": last_human_message},
                        output={"response": response_content},
                    )

                    return MessagesState(
                        messages=[*state.messages, AIMessage(content=response_content)],
                        thread_id=state.thread_id,
                    )
            else:
                # Fallback: no content and no tool calls - assume retrieval needed
                logger.warning(
                    "No content or tool calls in response, falling back to retrieval",
                    extra={"thread_id": state.thread_id},
                )
                query = last_human_message

                retriever = RetrieveTool()
                serialized_chunks, processed_results = await retriever.execute(query)

                opik_context.update_current_span(
                    name="query_or_respond_fallback",
                    input={"query": query},
                    output={"response": serialized_chunks},
                    metadata={"docs": processed_results},
                )

                return MessagesState(
                    messages=[
                        *state.messages,
                        ToolMessage(
                            content="",
                            tool_name="retrieve",
                            input={"query": query},
                            output=processed_results,
                        ),
                    ],
                    thread_id=state.thread_id,
                )

        except Exception as e:
            logger.error(
                "Query or respond check failed",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "error": str(e),
                },
            )
            raise


# Global cache for embeddings
EXAMPLE_EMBEDDINGS: List[List[float]] = []
QUERY_EMBEDDINGS_CACHE: Dict[str, List[float]] = {}


class ExampleSelector(BaseModel):
    examples: List[Dict[str, str]]

    @classmethod
    async def initialize_examples(cls, examples: List[Dict[str, str]]):
        """Initialize global example embeddings at server startup"""
        global EXAMPLE_EMBEDDINGS
        if not EXAMPLE_EMBEDDINGS and examples:
            embeddings_response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=[ex.get("user_query", ex.get("query", "")) for ex in examples],
            )
            EXAMPLE_EMBEDDINGS = [data.embedding for data in embeddings_response.data]

    async def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using global cache"""
        global QUERY_EMBEDDINGS_CACHE
        if query in QUERY_EMBEDDINGS_CACHE:
            return QUERY_EMBEDDINGS_CACHE[query]

        response = await client.embeddings.create(
            model="text-embedding-ada-002", input=[query]
        )
        embedding = response.data[0].embedding
        QUERY_EMBEDDINGS_CACHE[query] = embedding
        return embedding

    async def get_relevant_examples(
        self, query: str, k: int = 3
    ) -> List[Dict[str, str]]:
        """Get the most relevant examples using global embeddings"""
        # If no examples are available, return empty list
        if not self.examples or not EXAMPLE_EMBEDDINGS:
            logger.warning(
                "No examples available for few-shot selection",
                extra={
                    "examples_count": len(self.examples),
                    "embeddings_count": len(EXAMPLE_EMBEDDINGS)
                    if EXAMPLE_EMBEDDINGS
                    else 0,
                },
            )
            return []

        query_embedding = await self.get_query_embedding(query)

        # Calculate similarities
        similarities = [
            np.dot(query_embedding, ex_embedding)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(ex_embedding))
            for ex_embedding in EXAMPLE_EMBEDDINGS
        ]

        # Get top k examples, but ensure we don't exceed available examples
        k = min(k, len(self.examples), len(EXAMPLE_EMBEDDINGS))
        if k == 0:
            return []

        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_indices]


class FewShotSelectorNode(Node, SystemPromptNodeMixin):
    name: str = "few_shot_selector"
    example_selector: ExampleSelector = None

    async def init_selector(self):
        if not self.example_selector:
            examples = PROMPT_TEMPLATES.get("examples", [])
            self.example_selector = ExampleSelector(examples=examples)
            await self.example_selector.initialize_examples(examples)

    def _get_system_prompt(self, state: MessagesState) -> Optional[str]:
        """Return the system prompt used for few-shot selection"""
        return PROMPT_TEMPLATES["few_shot_selector"]["system_message"]

    @track(capture_input=False)
    async def process(self, state: MessagesState) -> MessagesState:
        try:
            await self.init_selector()

            current_query = next(
                (
                    msg
                    for msg in reversed(state.messages)
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )

            if not current_query:
                raise ValueError("No user query found")

            logger.info(
                "Few-shot selection initiated",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "current_msg": current_query.content,
                },
            )

            # Get relevant examples
            examples = await self.example_selector.get_relevant_examples(
                current_query.content, k=3
            )

            # Handle case where no examples are available
            if not examples:
                logger.warning(
                    "No examples available for few-shot selection, using fallback response",
                    extra={"node": self.name, "thread_id": state.thread_id},
                )
                # Return a fallback response when no examples are available
                fallback_response = "Category: OFFICIAL\nStyle: I'm here to help with your questions in a professional manner."

                return MessagesState(
                    messages=[*state.messages, AIMessage(content=fallback_response)],
                    thread_id=state.thread_id,
                )

            examples_text = "\n\n".join(
                [
                    f"Query: {ex.get('user_query', '')}\nCategory: {ex.get('potential_category', '')}\nResponse: {ex.get('response_style', '')}"
                    for ex in examples
                ]
            )

            system_message = (
                PROMPT_TEMPLATES["few_shot"]["prefix"]
                + "\n\n"
                + examples_text
                + "\n\n"
                + PROMPT_TEMPLATES["few_shot"]["suffix"].format(
                    query=current_query.content
                )
            )

            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Query: {current_query.content}\n\nExamples:\n{examples_text}",
                },
            ]

            response = await client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.1
            )
            content = response.choices[0].message.content
            logger.info(
                "Completed few-shot selection",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "response": content,
                },
            )
            opik_context.update_current_span(
                name="few_shot_selector",
                input={"query": current_query.content},
                output={"full_response": content},
                metadata={"examples": examples_text},
            )

            return MessagesState(
                messages=[*state.messages, AIMessage(content=content)],
                thread_id=state.thread_id,
            )

        except Exception as e:
            logger.error(
                "Few-shot selection failed",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "error": str(e),
                },
            )
            raise


class GenerateWithRetrievedContextNode(Node, RetrievalNodeMixin, SystemPromptNodeMixin):
    name: str = "generate_with_retrieved_context"

    def _get_system_prompt(self, state: MessagesState) -> Optional[str]:
        """Return the system prompt used for context generation"""
        user_query = next(
            (
                msg.content
                for msg in reversed(state.messages)
                if isinstance(msg, HumanMessage)
            ),
            "",
        )
        current_date = datetime.now().strftime("%B %d, %Y")

        # Process tool message outputs for docs content
        tool_messages = [msg for msg in state.messages if isinstance(msg, ToolMessage)]
        docs_content_parts = []
        for msg in tool_messages:
            if msg.tool_name == "retrieve" and msg.output:
                # Handle RetrievalResult objects
                if isinstance(msg.output, list) and all(
                    isinstance(item, RetrievalResult) for item in msg.output
                ):
                    for result in msg.output:
                        docs_content_parts.append(
                            f"Content: {result.content} (Score: {result.score:.2f})"
                        )
                # Handle the case where output is already a string
                elif isinstance(msg.output, str):
                    docs_content_parts.append(msg.output)

        docs_content = "\n\n".join(docs_content_parts)
        return PROMPT_TEMPLATES["generate_with_retrieved_context"][
            "system_message"
        ].format(
            current_date_str=current_date, query=user_query, docs_content=docs_content
        )

    async def collect_metadata(
        self, state: MessagesState, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Combine retrieval and system prompt metadata"""
        retrieval_metadata = await RetrievalNodeMixin.collect_metadata(
            self, state, start_time, end_time
        )
        system_metadata = await SystemPromptNodeMixin.collect_metadata(
            self, state, start_time, end_time
        )

        return {
            "retrieved_docs": retrieval_metadata["retrieved_docs"],
            "system_prompt": system_metadata["system_prompt"],
            "tags": ["retrieval", "system_prompt", "rag"],
            "custom_metadata": {
                **retrieval_metadata["custom_metadata"],
                **system_metadata["custom_metadata"],
            },
        }

    @track(capture_input=False)
    async def process(self, state: MessagesState) -> MessagesState:
        try:
            tool_messages = [
                msg for msg in state.messages if isinstance(msg, ToolMessage)
            ]
            user_query = next(
                (
                    msg
                    for msg in reversed(state.messages)
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )

            if not user_query:
                raise ValueError("No user query found")

            logger.info(
                "Context generation initiated",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "current_msg": user_query.content,
                },
            )

            current_date = datetime.now().strftime("%B %d, %Y")

            # Process tool message outputs
            docs_content_parts = []
            for msg in tool_messages:
                if msg.tool_name == "retrieve" and msg.output:
                    # Handle RetrievalResult objects
                    if isinstance(msg.output, list) and all(
                        isinstance(item, RetrievalResult) for item in msg.output
                    ):
                        for result in msg.output:
                            docs_content_parts.append(
                                f"Content: {result.content} (Score: {result.score:.2f})"
                            )
                    # Handle the case where output is already a string
                    elif isinstance(msg.output, str):
                        docs_content_parts.append(msg.output)
                    # Fallback for unexpected structures
                    else:
                        logger.warning(
                            "Unexpected output structure for retrieve tool",
                            extra={
                                "thread_id": state.thread_id,
                                "output_type": type(msg.output).__name__,
                                "output": str(msg.output)[:200] + "..."
                                if len(str(msg.output)) > 200
                                else str(msg.output),
                            },
                        )
                        docs_content_parts.append(f"Content: {str(msg.output)}")

            docs_content = "\n\n".join(docs_content_parts)

            system_message_content = PROMPT_TEMPLATES[
                "generate_with_retrieved_context"
            ]["system_message"].format(
                current_date_str=current_date,
                query=user_query.content,
                docs_content=docs_content,
            )

            logger.info(
                "Generated system message content", extra={"thread_id": state.thread_id}
            )

            messages = [
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_query.content},
            ]

            response = await client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.1
            )
            content = response.choices[0].message.content
            logger.info(
                "Completed context generation",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "response": content[:100] + ("..." if len(content) > 500 else ""),
                },
            )
            opik_context.update_current_span(
                name="generate_with_retrieved_context",
                input={"docs": docs_content},
                output={"full_response": content},
                metadata={"system_prompt": system_message_content},
            )

            # Add the generated content to state and return
            return MessagesState(
                messages=[*state.messages, AIMessage(content=content)],
                thread_id=state.thread_id,
            )

        except Exception as e:
            logger.error(
                "Streaming context generation failed",
                extra={
                    "action": "streaming_context_error",
                    "error": str(e),
                    "thread_id": state.thread_id,
                },
            )

            raise


class GenerateWithPersonaNode(StreamingNode, SystemPromptNodeMixin):
    name: str = "generate_with_persona"

    def _get_system_prompt(self, state: MessagesState) -> Optional[str]:
        """Return the system prompt used for persona generation"""

        last_ai_message = next(
            (
                msg.content
                for msg in reversed(state.messages)
                if isinstance(msg, AIMessage)
            ),
            None,
        )

        # Get the category from the few_shot_selector output
        category = "UNKNOWN"
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage) and any(
                cat in msg.content
                for cat in ["Category: JEST", "Category: HACK", "Category: OFFICIAL"]
            ):
                category = next(
                    (
                        cat
                        for cat in ["JEST", "HACK:MANIPULATION", "OFFICIAL"]
                        if f"Category: {cat}" in msg.content
                    ),
                    "UNKNOWN",
                )
                break

        # Format the system message
        return PROMPT_TEMPLATES["generate_with_persona"]["system_message"].format(
            last_ai_message=last_ai_message,
            category=category,
            suggest_email="Suggest 'you seem to be asking too many questions, why dont you reach out directly via email @ raghunandan092@gmail.com'"
            if len([m for m in state.messages if isinstance(m, HumanMessage)]) > 5
            else "",
        )

    async def collect_metadata(
        self, state: MessagesState, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Collect metadata for streaming persona generation"""
        system_metadata = await SystemPromptNodeMixin.collect_metadata(
            self, state, start_time, end_time
        )

        return {
            "retrieved_docs": None,
            "system_prompt": system_metadata["system_prompt"],
            "tags": ["system_prompt", "persona", "streaming"],
            "custom_metadata": {
                **system_metadata["custom_metadata"],
                "streaming_node": True,
            },
        }

    @track(capture_output=False, capture_input=False)
    async def process_stream(
        self, state: MessagesState
    ) -> AsyncGenerator[StreamingResponse, None]:
        try:
            logger.info(
                "Starting persona generation",
                extra={"node": self.name, "thread_id": state.thread_id},
            )

            query_count = (
                sum(
                    1 for message in state.messages if isinstance(message, HumanMessage)
                )
                > 5
            )
            last_ai_message = next(
                (
                    msg.content
                    for msg in reversed(state.messages)
                    if isinstance(msg, AIMessage)
                ),
                None,
            )
            user_query = next(
                (
                    msg.content
                    for msg in reversed(state.messages)
                    if isinstance(msg, HumanMessage)
                ),
                None,
            )

            if not user_query:
                raise ValueError("No user query found")

            logger.info(
                "User query found",
                extra={"query": user_query, "thread_id": state.thread_id},
            )

            # Get the category from the few_shot_selector output
            category = "UNKNOWN"
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage) and any(
                    cat in msg.content
                    for cat in [
                        "Category: JEST",
                        "Category: HACK",
                        "Category: OFFICIAL",
                    ]
                ):
                    category = next(
                        (
                            cat
                            for cat in ["JEST", "HACK:MANIPULATION", "OFFICIAL"]
                            if f"Category: {cat}" in msg.content
                        ),
                        "UNKNOWN",
                    )
                    break

            # Format the system message, replacing the placeholder
            system_message = PROMPT_TEMPLATES["generate_with_persona"][
                "system_message"
            ].format(
                last_ai_message=last_ai_message,
                category=category,
                suggest_email="Suggest 'you seem to be asking too many questions, why dont you reach out directly via email @ raghunandan092@gmail.com'"
                if query_count > 5
                else "",
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query},
            ]

            # Collect the complete response
            complete_response = []

            async for chunk in stream_chat_completion(messages, temperature=0.3):
                if chunk.type == "content" and chunk.content:
                    complete_response.append(chunk.content)
                yield chunk

            # Log the complete response
            full_response = "".join(complete_response)
            logger.info(
                "Completed persona generation",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "response": full_response[:500]
                    + ("..." if len(full_response) > 500 else ""),
                },
            )
            opik_context.update_current_span(
                name="generate_with_persona",
                input={"ai_message": last_ai_message},
                output={"full_response": full_response},
                metadata={"system_prompt": messages},
            )

        except Exception as e:
            logger.error(
                "Persona generation failed",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "error": str(e),
                },
            )
            raise


# Initialize single vector store instance
vector_store = VectorStore(index_name="langchain-chatraghu-embeddings")


def relevance_condition(state: MessagesState) -> str:
    """Route based on the relevance check response."""
    for message in reversed(state.messages):
        if isinstance(message, AIMessage):
            if "CONTEXTUAL" in message.content:
                return "CONTEXTUAL"
            elif "IRRELEVANT" in message.content:
                return "IRRELEVANT"
    return "RELEVANT"


def query_or_respond_condition(state: MessagesState) -> str:
    """Route based on the QueryOrRespondNode's decision."""
    # First check if there are any ToolMessages (indicating retrieval was performed)
    for message in reversed(state.messages):
        if isinstance(message, ToolMessage):
            if message.tool_name == "retrieve":
                return "tools"

    return "END"  # Default to END if no clear decision found


async def stream_chat_completion(
    messages: List[Dict[str, str]],
    functions: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.1,
) -> AsyncGenerator[StreamingResponse, None]:
    """
    Stream chat completion responses from OpenAI, handling both text and function calls.
    """
    try:
        # Prepare the API call parameters
        params = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if functions:
            params["functions"] = functions
            params["function_call"] = "auto"

        # Make the API call
        stream = await client.chat.completions.create(**params)

        state = StreamingState()

        async for chunk in stream:
            delta = chunk.choices[0].delta

            # Handle function calls
            if delta.function_call:
                state.is_function_call = True

                if delta.function_call.name:
                    state.function_name = delta.function_call.name

                if delta.function_call.arguments:
                    state.buffer += delta.function_call.arguments

                # If this is the last chunk
                if chunk.choices[0].finish_reason == "function_call":
                    try:
                        function_args = json.loads(state.buffer)
                        yield StreamingResponse(
                            type="function_call",
                            function_name=state.function_name,
                            function_args=function_args,
                        )
                    except json.JSONDecodeError as e:
                        logger.error(
                            "Failed to parse function arguments",
                            extra={"error": str(e), "buffer": state.buffer},
                        )
                        raise

            # Handle content streaming
            elif delta.content:
                if state.is_function_call:
                    state.buffer += delta.content
                else:
                    yield StreamingResponse(content=delta.content)

            # Handle end of stream
            if chunk.choices[0].finish_reason:
                yield StreamingResponse(type="end")

    except Exception as e:
        logger.error(
            "Streaming failed", extra={"action": "streaming_error", "error": str(e)}
        )
        raise


# Updated graph assembly with streaming support
streaming_graph = StreamingStateGraph(
    nodes={
        "relevance_check": RelevanceCheckNode(name="relevance_check"),
        "query_or_respond": QueryOrRespondNode(name="query_or_respond"),
        "few_shot_selector": FewShotSelectorNode(name="few_shot_selector"),
        "generate_with_context": GenerateWithRetrievedContextNode(
            name="generate_with_retrieved_context"
        ),
        "generate_with_persona": GenerateWithPersonaNode(name="generate_with_persona"),
    },
    edges={
        "relevance_check": {
            "CONTEXTUAL": "query_or_respond",
            "IRRELEVANT": "few_shot_selector",
            "RELEVANT": "query_or_respond",
        },
        "query_or_respond": {
            "tools": "generate_with_context",
            "END": "few_shot_selector",
        },
        "few_shot_selector": {"default": "generate_with_persona"},
        "generate_with_context": {"default": "generate_with_persona"},
        "generate_with_persona": {"default": "END"},
    },
    entry_point="relevance_check",
    queue_manager=None,  # Will be set after initialization
)


# Add a method to set queue manager
def set_queue_manager(queue_manager):
    global streaming_graph
    streaming_graph.queue_manager = queue_manager


# Initialize at server startup
async def init_example_selector():
    examples = PROMPT_TEMPLATES.get("examples", [])
    await ExampleSelector.initialize_examples(examples)
