import os
import json
import time
from typing import List, Dict, Optional, Any, AsyncGenerator, Tuple
from datetime import datetime

from opik import track, opik_context

from .models import (
    MessagesState,
    StreamingResponse,
    StreamingState,
    HumanMessage,
    AIMessage,
    ToolMessage,
    RetrievalResult,
)
from .infrastructure import (
    Node,
    StreamingNode,
    ClassificationNodeMixin,
    RetrievalNodeMixin,
    SystemPromptNodeMixin,
    StreamingStateGraph,
    ConversationHistoryMixin,
)
from .retrieval import RetrieveTool, ExampleSelector, client

from utils.logger import logger
from evaluation_models import (
    ResponseMessage,
    ConversationFlow,
    EnrichedNodeExecutionLog,
)

# Load prompt templates from the JSON file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_PATH = os.path.join(CURRENT_DIR, "prompt_templates.json")
with open(TEMPLATES_PATH, "r") as f:
    PROMPT_TEMPLATES = json.load(f)


# Processing Nodes
class RelevanceCheckNode(Node, ClassificationNodeMixin, ConversationHistoryMixin):
    name: str = "relevance_check"

    @track(capture_input=True)
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
            openai_messages, _ = self._build_conversation_history(
                state,
                max_history=8,
                include_tool_content=True,
                exclude_tool_details=True,
            )
            openai_messages.insert(
                0,
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATES["relevance_check"]["system_message"],
                },
            )
            openai_messages.append({"role": "user", "content": current_query.content})

            response = await client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=openai_messages,
                temperature=0.1,
            )
            content = response.choices[0].message.content

            # Extract token usage if available
            if hasattr(response, "usage") and response.usage:
                state._node_token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }

            logger.info(
                "Completed relevance check with history",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                    "response": content,
                },
            )
            opik_context.update_current_span(
                name="relevance_check",
                input={"query": current_query.content, "history": openai_messages},
                output={"classification": content},
            )
            new_state = MessagesState(
                messages=[*state.messages, AIMessage(content=content)],
                thread_id=state.thread_id,
            )
            if hasattr(state, "_node_token_usage"):
                new_state._node_token_usage = state._node_token_usage
            return new_state
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


class QueryOrRespondNode(Node, ClassificationNodeMixin, ConversationHistoryMixin):
    name: str = "query_or_respond"

    @track(capture_input=False)
    async def process(self, state: MessagesState) -> MessagesState:
        try:
            openai_messages, last_human_message = self._build_conversation_history(
                state,
                max_history=8,
                include_tool_content=True,
                exclude_tool_details=True,
            )
            openai_messages.insert(
                0,
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATES["query_or_respond"]["system_message"],
                },
            )
            openai_messages.append(
                {
                    "role": "user",
                    "content": f"focus on this latest query from the conversation: {last_human_message}",
                }
            )
            logger.info(
                "Query/Respond check with optimized history",
                extra={
                    "node": self.name,
                    "thread_id": state.thread_id,
                },
            )
            response = await client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
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

            # Extract token usage if available
            if hasattr(response, "usage") and response.usage:
                state._node_token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }

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


class FewShotSelectorNode(Node, SystemPromptNodeMixin):
    name: str = "few_shot_selector"
    example_selector: ExampleSelector = None

    async def init_selector(self):
        if not self.example_selector:
            examples = PROMPT_TEMPLATES.get("examples", [])
            self.example_selector = ExampleSelector(examples=examples)
            await self.example_selector.initialize_examples(examples)

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
                model="gpt-4.1-nano-2025-04-14", messages=messages, temperature=0.1
            )
            content = response.choices[0].message.content

            # Extract token usage if available
            if hasattr(response, "usage") and response.usage:
                state._node_token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }

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

            new_state = MessagesState(
                messages=[*state.messages, AIMessage(content=content)],
                thread_id=state.thread_id,
            )
            if hasattr(state, "_node_token_usage"):
                new_state._node_token_usage = state._node_token_usage
            return new_state

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
                model="gpt-4.1-nano-2025-04-14", messages=messages, temperature=0.1
            )
            content = response.choices[0].message.content

            # Extract token usage if available
            if hasattr(response, "usage") and response.usage:
                state._node_token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }

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
            new_state = MessagesState(
                messages=[*state.messages, AIMessage(content=content)],
                thread_id=state.thread_id,
            )
            if hasattr(state, "_node_token_usage"):
                new_state._node_token_usage = state._node_token_usage
            return new_state

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
                > 4
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

            # Collect the complete response and track token usage
            complete_response = []

            async for chunk in stream_chat_completion(messages, temperature=0.3):
                if chunk.type == "content" and chunk.content:
                    complete_response.append(chunk.content)

                # Yield all chunks including usage (execute_stream_impl will handle usage)
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


# Utility Functions
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
            "model": "gpt-4.1-nano-2025-04-14",
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
        usage_data = None

        async for chunk in stream:
            # Capture usage data if present in the chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage

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

        # Yield usage data at the end if captured
        if usage_data:
            yield StreamingResponse(
                type="usage",
                usage_stats={
                    "prompt_tokens": usage_data.prompt_tokens,
                    "completion_tokens": usage_data.completion_tokens,
                },
            )

    except Exception as e:
        logger.error(
            "Streaming failed", extra={"action": "streaming_error", "error": str(e)}
        )
        raise


# Graph Assembly
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


# Define the execute_stream implementation as a standalone function
@track(capture_input=False, capture_output=False)
async def execute_stream_impl(
    self, initial_state: MessagesState, run_id: str, turn_index: int
) -> AsyncGenerator[Tuple[StreamingResponse, Dict], None]:
    try:
        logger.info(
            "Starting graph execution", extra={"thread_id": initial_state.thread_id}
        )

        # Extract the user query once at the start
        user_query = next(
            (
                msg.content
                for msg in reversed(initial_state.messages)
                if isinstance(msg, HumanMessage)
            ),
            "",
        )

        opik_context.update_current_trace(
            name="graph_execution",
            thread_id=initial_state.thread_id,
            input={"user_query": user_query},
        )

        # Initialize conversation flow tracking
        conversation_flow = ConversationFlow(
            run_id=run_id,
            thread_id=initial_state.thread_id,
            turn_index=turn_index,
            user_query=user_query,
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
                    "messages": [msg.model_dump() for msg in state.messages],
                    "thread_id": state.thread_id,
                    "conversation_history": [
                        msg.model_dump() for msg in state.messages[:-1]
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
                first_token_received = False
                stream_start_time = time.monotonic()  # Use monotonic for intervals

                logger.info(
                    "Starting streaming node processing",
                    extra={"node": current_node, "thread_id": state.thread_id},
                )

                streaming_usage = None

                async for chunk in node.process_stream(state):
                    if chunk.type == "content" and chunk.content:
                        complete_content.append(chunk.content)

                        # Capture time to first token
                        if not first_token_received:
                            ttft_ms = (
                                datetime.utcnow() - conversation_flow.start_time
                            ).total_seconds() * 1000
                            conversation_flow.time_to_first_token_ms = ttft_ms
                            first_token_received = True
                            logger.debug(
                                "First token received",
                                extra={
                                    "node": current_node,
                                    "thread_id": state.thread_id,
                                    "ttft_ms": ttft_ms,
                                },
                            )
                    elif chunk.type == "usage" and chunk.usage_stats:
                        # Capture streaming usage data
                        streaming_usage = chunk.usage_stats
                        logger.debug(
                            "Captured streaming usage",
                            extra={
                                "node": current_node,
                                "thread_id": state.thread_id,
                                "usage": streaming_usage,
                            },
                        )
                        # Don't yield usage chunks to the client
                        continue

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

                    # Apply streaming usage to node log if captured
                    if streaming_usage:
                        node_log.prompt_tokens = streaming_usage.get("prompt_tokens")
                        node_log.completion_tokens = streaming_usage.get(
                            "completion_tokens"
                        )

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
                        conversation_flow.latency_ms = (
                            conversation_flow.end_time - conversation_flow.start_time
                        ).total_seconds() * 1000

                        # Aggregate token counts from all node executions
                        conversation_flow.total_prompt_tokens = sum(
                            log.prompt_tokens
                            for log in conversation_flow.node_executions
                            if log.prompt_tokens
                        )
                        conversation_flow.total_completion_tokens = sum(
                            log.completion_tokens
                            for log in conversation_flow.node_executions
                            if log.completion_tokens
                        )

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
                        # Evaluation service expects all values to be strings
                        converted_docs = []
                        for doc in retrieved_docs:
                            if isinstance(doc, list):
                                # Handle list of RetrievalResult objects
                                for result in doc:
                                    if hasattr(result, "model_dump"):
                                        raw_doc = result.model_dump()
                                        # Convert to string format expected by evaluation service
                                        converted_doc = {
                                            "content": str(raw_doc.get("content", "")),
                                            "score": str(raw_doc.get("score", "")),
                                            "metadata": str(
                                                raw_doc.get("metadata", {})
                                            ),
                                        }
                                        converted_docs.append(converted_doc)
                                    else:
                                        converted_docs.append(
                                            {"content": str(result), "metadata": "{}"}
                                        )
                            elif hasattr(doc, "model_dump"):
                                # Handle single RetrievalResult object
                                raw_doc = doc.model_dump()
                                # Convert to string format expected by evaluation service
                                converted_doc = {
                                    "content": str(raw_doc.get("content", "")),
                                    "score": str(raw_doc.get("score", "")),
                                    "metadata": str(raw_doc.get("metadata", {})),
                                }
                                converted_docs.append(converted_doc)
                            else:
                                # Handle other types
                                converted_docs.append(
                                    {"content": str(doc), "metadata": "{}"}
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
                    msg.model_dump() for msg in state.messages[-1:]
                ]
                node_log.end_time = datetime.utcnow()

                # Capture token usage if the node stored it in state
                if hasattr(state, "_node_token_usage") and state._node_token_usage:
                    node_log.prompt_tokens = state._node_token_usage.get(
                        "prompt_tokens"
                    )
                    node_log.completion_tokens = state._node_token_usage.get(
                        "completion_tokens"
                    )
                    # Clean up the temporary token usage data
                    delattr(state, "_node_token_usage")

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


# Add the execute_stream method to the StreamingStateGraph class
# This is done after all imports are resolved to avoid circular dependencies
StreamingStateGraph.execute_stream = execute_stream_impl


# Helper Functions
def set_queue_manager(queue_manager):
    global streaming_graph
    streaming_graph.queue_manager = queue_manager


async def init_example_selector():
    examples = PROMPT_TEMPLATES.get("examples", [])
    await ExampleSelector.initialize_examples(examples)
