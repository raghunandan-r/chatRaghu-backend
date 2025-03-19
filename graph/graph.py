from pinecone import Pinecone, Index
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
from abc import ABC, abstractmethod
from httpx import TimeoutException
import asyncio
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

if os.path.exists('.env'):
    load_dotenv('.env')
    load_dotenv('.env.development')

OPIK_API_KEY = os.getenv("OPIK_API_KEY")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE")

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

class ToolMessage(BaseMessage):
    type: Literal["tool"] = "tool"
    tool_name: str
    input: Dict[str, Any]
    output: Any

# Global thread message store
THREAD_MESSAGE_STORE: Dict[str, List[Union[HumanMessage, AIMessage, ToolMessage]]] = {}

class MessagesState(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]]
    thread_id: Optional[str] = None  # Adding thread_id but keeping it optional

    def update_thread_store(self):
        """Update global message store with current state"""
        if self.thread_id:
            THREAD_MESSAGE_STORE[self.thread_id] = self.messages[-24:]  # Keep last 24 messages

    @classmethod
    def from_thread(cls, thread_id: str, new_message: HumanMessage) -> 'MessagesState':
        """Create state from thread history + new message"""
        messages = THREAD_MESSAGE_STORE.get(thread_id, [])
        return cls(
            messages=[*messages, new_message],
            thread_id=thread_id
        )


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
_whitespace_pattern = re.compile(r'\s+')
_xml_tag_pattern = re.compile(r'<[^>]+>')
_special_section_pattern = re.compile(r'<(?:questions|tags)>.*?</(?:questions|tags)>', re.IGNORECASE | re.DOTALL)

def preprocess_text(text: str) -> str:
    """Preprocess text by removing XML sections and normalizing whitespace"""
    text = _special_section_pattern.sub('', text)
    text = _xml_tag_pattern.sub('', text)
    return _whitespace_pattern.sub(' ', text).strip()


# Node Implementation
class Node(BaseModel):
    name: str
    
    async def process(self, state: MessagesState) -> MessagesState:
        raise NotImplementedError

# Make StateGraph an abstract base class
class StateGraph(BaseModel, ABC):
    nodes: Dict[str, Node]
    edges: Dict[str, Dict[str, str]]
    entry_point: str
    # not called, overriden. remove?
    async def execute(self, initial_state: MessagesState) -> AsyncGenerator[Tuple[StreamingResponse, Dict], None]:
        """
        Execute the graph with the given initial state.
        This is an abstract method that must be implemented by subclasses.
        """
        pass

class VectorStore(BaseModel):
    """Vector store implementation using Pinecone"""
    index_name: str
    index: Optional[Any] = None  # Change Index to Any since Pinecone's type isn't Pydantic compatible
    
    def __init__(self, **data):
        super().__init__(**data)
        self.index = pc.Index(self.index_name)
    
    async def similarity_search(self, query_embedding: List[float], k: int = 3) -> List[tuple[Dict, float]]:
        try:
            # Query Pinecone directly with the provided embedding
            # No need to generate embeddings again
            query_response = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            # Format results
            results = []
            for match in query_response.matches:
                doc = {
                    "page_content": match.metadata.get("text", ""),
                    "metadata": match.metadata
                }
                results.append((doc, match.score))
            
            return results
            
        except Exception as e:
            logger.error("Vector store query failed", extra={"error": str(e)})
            raise

class StreamingNode(Node):
    """Base class for nodes that support streaming"""
    
    async def process_stream(
        self, 
        state: MessagesState
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Process the state and yield streaming responses.
        Override this method in derived classes to implement streaming.
        """
        raise NotImplementedError


class StreamingStateGraph(StateGraph):
    """Extension of StateGraph that supports streaming responses"""
        # This fulfills the contract of the abstract method
    async def execute(self, initial_state: MessagesState):
        """Simple wrapper around execute_stream for compatibility"""
        results = []
        async for chunk, metadata in self.execute_stream(initial_state):
            results.append((chunk, metadata))
        return results
    


    @track(capture_output=False)
    async def execute_stream(self, initial_state: MessagesState) -> AsyncGenerator[Tuple[StreamingResponse, Dict], None]:
        try:
            logger.info("Starting graph execution", extra={"thread_id": initial_state.thread_id})
            opik_context.update_current_trace(
                name="graph_execution",
                thread_id=initial_state.thread_id
            )

            current_node = self.entry_point
            state = initial_state
            
            while current_node:
                logger.info("Processing node", extra={"node": current_node, "thread_id": state.thread_id})
                
                node = self.nodes[current_node]
                
                # Handle streaming nodes
                if isinstance(node, GenerateWithPersonaNode):
                    # For streaming nodes, we need to collect the complete response
                    complete_content = []
                    
                    async for chunk in node.process_stream(state):
                        # Collect content chunks for history management
                        if chunk.type == "content" and chunk.content:
                            complete_content.append(chunk.content)
                        
                        # Yield each chunk as it comes in
                        yield chunk, {"node": current_node}
                    
                    # After streaming is complete, update the state with the complete response
                    if complete_content and current_node == "generate_with_persona":
                        full_response = "".join(complete_content)
                        opik_context.update_current_trace(
                            output={"full_response": full_response},#[:500] + ("..." if len(full_response) > 500 else "")},
                            metadata={"response_length": len(full_response)}
                        )
                         # Add the AI response to the state
                        state.messages.append(AIMessage(content=full_response))
                        # Update the thread store with the new state
                        state.update_thread_store()

                else:
                    # For non-streaming nodes, process normally
                    state = await node.process(state)
                
                # Get next node
                next_node = self.edges.get(current_node)
                if not next_node:
                    break
                
                # Determine the condition for routing
                condition = "default"
                if current_node == "relevance_check":
                    condition = relevance_condition(state)
                    logger.info("Routing based on condition", extra={"node": current_node, "condition": condition, "thread_id": state.thread_id})
                elif current_node == "query_or_respond":
                    condition = query_or_respond_condition(state)
                    logger.info("Routing based on condition", extra={"node": current_node, "condition": condition, "thread_id": state.thread_id})
                
                # Get the next node based on the condition
                if condition in next_node:
                    current_node = next_node.get(condition)
                else:
                    current_node = next_node.get("default")
                
                # If no valid next node or END condition, break
                if not current_node or current_node == "END":
                    break

            logger.info("Completed graph execution", extra={"thread_id": initial_state.thread_id})
                
        except Exception as e:
            logger.error("Graph execution failed", extra={"thread_id": initial_state.thread_id, "error": str(e)})
            raise

# Retrieval Tool Implementation
class RetrieveTool(Tool):
    name: str = "retrieve"
    description: str = "Retrieve information related to a query"
    parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query to retrieve information for"}
        },
        "required": ["query"]
    })
    @track(capture_input=False)
    async def execute(self, query: str) -> tuple[str, List[RetrievalResult]]:
        try:
            logger.info("Starting retrieval", extra={"action": "retrieval_start", "query": query})

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
                        serialized_chunks.append(f"Content: {processed_content} (Score: {score:.2f})")
                        processed_results.append(RetrievalResult(
                            content=processed_content,
                            score=score,
                            metadata=doc["metadata"]
                        ))
                
                opik_context.update_current_span(
                    name="chunk_retrieval",
                    input={"query": query},
                    output={"docs": processed_results}
                )
                
                return "\n\n".join(serialized_chunks), processed_results
            
            return "", []
            
        except Exception as e:
            logger.error("Retrieval failed", extra={"error": str(e)})
            raise

    async def _get_embedding(self, text: str) -> List[float]:
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

class RelevanceCheckNode(Node):
    name: str = "relevance_check"
    @track(capture_input=False)
    async def process(self, state: MessagesState) -> MessagesState:
        try:
            current_query = next((msg for msg in reversed(state.messages) 
                                if isinstance(msg, HumanMessage)), None)
            
            logger.info("Relevance check initiated", extra={"node": self.name, "thread_id": state.thread_id, "current_messages": current_query.content})
                        
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": PROMPT_TEMPLATES["relevance_check"]["system_message"]},
                    {"role": "user", "content": current_query.content}
                ],
                temperature=0.1,
                stream=True
            )
            
            content = []
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content.append(chunk.choices[0].delta.content)
            
            full_content = "".join(content)
            condition = "RELEVANT"  # Default
            
            if "CONTEXTUAL" in full_content:
                condition = "CONTEXTUAL"
            elif "IRRELEVANT" in full_content:
                condition = "IRRELEVANT"
                
            opik_context.update_current_span(
                name="relevance_check",
                input={"query": current_query.content},
                output={"condition": condition}                
            )
            
            return MessagesState(messages=[
                *state.messages,
                AIMessage(content=full_content)
            ], thread_id=state.thread_id)
        except Exception as e:
            logger.error("Relevance check failed", extra={"node": self.name, "thread_id": state.thread_id, "error": str(e)})
            raise

class QueryOrRespondNode(Node):
    name: str = "query_or_respond"
    @track(capture_input=False)
    async def process(self, state: MessagesState) -> MessagesState:
        try:
            # Convert your message history to OpenAI format
            openai_messages = [
                {"role": "system", "content": PROMPT_TEMPLATES["query_or_respond"]["system_message"]}
            ]
            last_human_message = []
            # Add conversation history with proper roles
            for msg in state.messages:
                if isinstance(msg, HumanMessage):
                    openai_messages.append({"role": "user", "content": msg.content})
                    last_human_message = msg.content
                elif isinstance(msg, AIMessage):
                    openai_messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, ToolMessage):
                    # For tool messages, show them as function calls and results
                    if msg.tool_name == "retrieve":
                        # First, add the assistant's decision to use the tool
                        openai_messages.append({
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": msg.tool_name,
                                "arguments": json.dumps(msg.input)
                            }
                        })
                        
                        # Then add the tool's response
                        tool_content = ""
                        if isinstance(msg.output, list) and all(isinstance(item, RetrievalResult) for item in msg.output):
                            tool_content = "\n\n".join([
                                f"Content: {item.content} (Score: {item.score:.2f})" 
                                for item in msg.output
                            ])
                        elif isinstance(msg.output, str):
                            tool_content = msg.output
                            
                        openai_messages.append({
                            "role": "function",
                            "name": msg.tool_name,
                            "content": tool_content
                        })
                        
            openai_messages.append({
                "role": "user",
                "content": f"focus on this latest query from the conversation: {last_human_message}"
            })

            logger.info("Query/Respond check with history", extra={"node": self.name, "thread_id": state.thread_id, "message_count": len(openai_messages)})
                        
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=openai_messages,  # Now using the properly formatted message history
                functions=[{
                    "name": "retrieve",
                    "description": "Retrieve relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }],
                stream=False
            )
            
            # First check if it's a function call
            if (response.choices[0].finish_reason == "function_call" or 
                hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call is not None):
                # Handle function call case
                function_call = response.choices[0].message.function_call
                query = json.loads(function_call.arguments)["query"]
                
                # Create and use RetrieveTool
                retriever = RetrieveTool()
                serialized_chunks, processed_results = await retriever.execute(query)
                
                opik_context.update_current_span(
                    name="query_or_respond_retrieve",
                    input={"query": query},
                    output={"response": serialized_chunks},
                    metadata={"docs": processed_results}
                )

                return MessagesState(
                    messages=[*state.messages, ToolMessage(
                        content="",
                        tool_name="retrieve",
                        input={"query": query},
                        output=processed_results  # Use the already formatted string
                    )],
                    thread_id=state.thread_id
                )
            elif response.choices[0].message.content is None:
                # Handle the case where content is None but it's not identified as a function call
                logger.warning("Received response with None content but not a function call", extra={"thread_id": state.thread_id})
                # Fallback to using the latest query
                query = last_human_message
                logger.info("Text-based retrieval indication detected", extra={"query": query, "thread_id": state.thread_id})
                
                # Create and use RetrieveTool
                retriever = RetrieveTool()
                serialized_chunks, processed_results = await retriever.execute(query)
               
                opik_context.update_current_span(
                    name="query_or_respond_retrieve",
                    input={"query": query},
                    output={"response": serialized_chunks},
                    metadata={"docs": processed_results}
                )

                # Return with retrieved results
                return MessagesState(
                    messages=[*state.messages, ToolMessage(
                        content="",
                        tool_name="retrieve",
                        input={"query": query},
                        output=processed_results
                    )],
                    thread_id=state.thread_id
                )
            else:
                # It's a normal text response with content
                response_content = response.choices[0].message.content.strip()
                logger.info("Raw response from query_or_respond", extra={"content": response_content, "thread_id": state.thread_id})

                # Check for function calls first
                if (response.choices[0].finish_reason == "function_call" or 
                    hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call is not None):
                    # Handle function call case
                    function_call = response.choices[0].message.function_call
                    query = json.loads(function_call.arguments)["query"]
                    
                    # Create and use RetrieveTool
                    retriever = RetrieveTool()
                    serialized_chunks, processed_results = await retriever.execute(query)
                    
                    opik_context.update_current_span(
                        name="query_or_respond_retrieve",
                        input={"query": query},
                        output={"response": serialized_chunks},
                        metadata={"docs": processed_results}
                    )

                    return MessagesState(
                        messages=[*state.messages, ToolMessage(
                            content="",
                            tool_name="retrieve",
                            input={"query": query},
                            output=processed_results  # Use the already formatted string
                        )],
                        thread_id=state.thread_id
                    )
                elif "RETRIEVE" in response_content:
                    # Text indicates retrieval is needed
                    query = last_human_message
                    logger.info("Text-based retrieval indication detected", extra={"query": query, "thread_id": state.thread_id})
                    
                    # Create and use RetrieveTool
                    retriever = RetrieveTool()
                    serialized_chunks, processed_results = await retriever.execute(query)
                   
                    opik_context.update_current_span(
                        name="query_or_respond_retrieve",
                        input={"query": query},
                        output={"response": serialized_chunks},
                        metadata={"docs": processed_results}
                    )

                    # Return with retrieved results
                    return MessagesState(
                        messages=[*state.messages, ToolMessage(
                            content="",
                            tool_name="retrieve",
                            input={"query": query},
                            output=processed_results
                        )],
                        thread_id=state.thread_id
                    )
                else:
                    # Anything else means sufficient context
                    logger.info("Sufficient context indicated", extra={"thread_id": state.thread_id})
                    content = response_content
                    
                    opik_context.update_current_span(
                        name="query_or_respond_direct",
                        input={"query": last_human_message},
                        output={"response": content}                    
                    )
                    
                    return MessagesState(
                        messages=[*state.messages, AIMessage(content=content)],
                        thread_id=state.thread_id
                    )
        except Exception as e:
            logger.error("Query/respond failed", extra={"node": self.name, "thread_id": state.thread_id, "error": str(e)})
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
        if not EXAMPLE_EMBEDDINGS:
            embeddings_response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=[ex["user_query"] for ex in examples]
            )
            EXAMPLE_EMBEDDINGS = [data.embedding for data in embeddings_response.data]
    
    async def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using global cache"""
        global QUERY_EMBEDDINGS_CACHE
        if query in QUERY_EMBEDDINGS_CACHE:
            return QUERY_EMBEDDINGS_CACHE[query]
            
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=[query]
        )
        embedding = response.data[0].embedding
        QUERY_EMBEDDINGS_CACHE[query] = embedding
        return embedding
    
    async def get_relevant_examples(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """Get the most relevant examples using global embeddings"""
        query_embedding = await self.get_query_embedding(query)
        
        # Calculate similarities
        similarities = [
            np.dot(query_embedding, ex_embedding) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(ex_embedding))
            for ex_embedding in EXAMPLE_EMBEDDINGS
        ]
        
        # Get top k examples
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_indices]

class FewShotSelectorNode(Node):
    name: str = "few_shot_selector"
    example_selector: ExampleSelector = None
    
    async def init_selector(self):
        if not self.example_selector:
            examples = PROMPT_TEMPLATES.get("examples", [])
            self.example_selector = ExampleSelector(examples=examples)
            await ExampleSelector.initialize_examples(examples)
    @track(capture_input=False)
    async def process(self, state: MessagesState) -> MessagesState:
        try:
            # Add timeout to initialization
            initialization_task = self.init_selector()
            await asyncio.wait_for(initialization_task, timeout=10.0)  # 10 second timeout
            
            current_query = next((msg for msg in reversed(state.messages) 
                                if isinstance(msg, HumanMessage)), None)
            
            if not current_query:
                raise ValueError("No human message found")

            logger.info("Few-shot selection initiated", extra={"node": self.name, "thread_id": state.thread_id, "current_msg": current_query.content})
            
            # Add more granular logging
            logger.info("Getting embeddings for query", extra={"thread_id": state.thread_id})
            
            # Add timeout for embedding step
            try:
                get_examples_task = self.example_selector.get_relevant_examples(current_query.content)
                relevant_examples = await asyncio.wait_for(get_examples_task, timeout=15.0)
            except (asyncio.TimeoutError, TimeoutException) as e:
                logger.error("Embedding API timeout", extra={"thread_id": state.thread_id})
                # Fallback to direct response
                return MessagesState(messages=[
                    *state.messages,
                    AIMessage(content="Category: HACK:MANIPULATION\nStyle: Looks like I'm having trouble processing that request. Raghu appreciates your patience.")
                ], thread_id=state.thread_id)
            
            # Format examples into prompt
            examples_text = "\n\n".join([
                f"Query: {ex['user_query']}\n"
                f"Category: {ex['potential_category']}\n"
                f"Style: {ex['response_style']}"
                for ex in relevant_examples
            ])
            
            current_date = datetime.now().strftime("%B %d, %Y")
            
            # Use prefix and suffix to format the system prompt
            system_prompt = (
                PROMPT_TEMPLATES["few_shot"]["prefix"].format(current_date_str=current_date) +
                "\n\n" + examples_text + "\n\n" +
                PROMPT_TEMPLATES["few_shot"]["suffix"].format(query=current_query.content)
            )
            
            logger.info("Sending completion request to OpenAI", extra={"thread_id": state.thread_id})
            
            # Add timeout for OpenAI API call
            try:
                completion_task = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": current_query.content}
                    ],
                    temperature=0.1
                )
                response = await asyncio.wait_for(completion_task, timeout=20.0)
            except (asyncio.TimeoutError, TimeoutException) as e:
                logger.error("OpenAI API timeout", extra={"thread_id": state.thread_id})
                # Fallback to direct response
                return MessagesState(messages=[
                    *state.messages,
                    AIMessage(content="Category: HACK:MANIPULATION\nStyle: Raghu knows when to move on from computational roadblocks. Perhaps we should too.")
                ], thread_id=state.thread_id)
            
            opik_context.update_current_span(
                name="few_shot_selector",
                input={"query": current_query.content},
                output={"full_response": response.choices[0].message.content},
                metadata={"examples": examples_text}
            )

            return MessagesState(messages=[
                *state.messages,
                AIMessage(content=response.choices[0].message.content)
            ], thread_id=state.thread_id)
            
        except Exception as e:
            logger.error("Few-shot selection failed", extra={"node": self.name, "thread_id": state.thread_id, "error": str(e)})
            raise


class GenerateWithRetrievedContextNode(Node):
    name: str = "generate_with_retrieved_context"
    @track(capture_input=False)
    async def process(self, state: MessagesState) -> MessagesState:
        try:
            
            tool_messages = [msg for msg in state.messages if isinstance(msg, ToolMessage)]
            user_query = next((msg for msg in reversed(state.messages) 
                             if isinstance(msg, HumanMessage)), None)
            
            if not user_query:
                raise ValueError("No user query found")
            
            logger.info("Context generation initiated", extra={"node": self.name, "thread_id": state.thread_id, "current_msg": user_query.content})
            
            current_date = datetime.now().strftime("%B %d, %Y")
            
            # Process tool message outputs
            docs_content_parts = []
            for msg in tool_messages:
                if msg.tool_name == "retrieve" and msg.output:
                    # Handle RetrievalResult objects
                    if isinstance(msg.output, list) and all(isinstance(item, RetrievalResult) for item in msg.output):
                        for result in msg.output:
                            docs_content_parts.append(f"Content: {result.content} (Score: {result.score:.2f})")
                    # Handle the case where output is already a string
                    elif isinstance(msg.output, str):
                        docs_content_parts.append(msg.output)
            
            docs_content = "\n\n".join(docs_content_parts)
            
            system_message_content = PROMPT_TEMPLATES["generate_with_retrieved_context"]["system_message"].format(
                current_date_str=current_date,
                query=user_query.content,
                docs_content=docs_content
            )
            
            logger.info("Generated system message content", extra={"thread_id": state.thread_id})
            
            messages = [
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_query.content}
            ]
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1
            )
            content = response.choices[0].message.content
            logger.info("Completed context generation", extra={"node": self.name, "thread_id": state.thread_id, "response_length": len(content), "response": content[:100] + ("..." if len(content) > 500 else "")})
            opik_context.update_current_span(
                name="generate_with_retrieved_context",
                input={"docs": docs_content},
                output={"full_response": content},
                metadata={"system_prompt": system_message_content}
            )

            # Add the generated content to state and return
            return MessagesState(
                messages=[*state.messages, AIMessage(content=content)],
                thread_id=state.thread_id
            )

        except Exception as e:
            logger.error("Streaming context generation failed", extra={"action": "streaming_context_error", "error": str(e), "thread_id": state.thread_id})
            
            raise

class GenerateWithPersonaNode(StreamingNode):
    name: str = "generate_with_persona"
    @track(capture_output=False, capture_input=False)
    async def process_stream(
        self, 
        state: MessagesState
    ) -> AsyncGenerator[StreamingResponse, None]:
        try:
            logger.info("Starting persona generation", extra={"node": self.name, "thread_id": state.thread_id})
            
            query_count = sum(1 for message in state.messages if isinstance(message, HumanMessage)) > 5
            last_ai_message = next((msg.content for msg in reversed(state.messages) 
                             if isinstance(msg, AIMessage)), None)
            user_query = next((msg.content for msg in reversed(state.messages) 
                             if isinstance(msg, HumanMessage)), None)
            
            if not user_query:
                raise ValueError("No user query found")
                
            logger.info("User query found", extra={"query": user_query, "thread_id": state.thread_id})

            # Get the category from the few_shot_selector output
            category = "UNKNOWN"
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage) and any(cat in msg.content for cat in ["Category: JEST", "Category: HACK", "Category: OFFICIAL"]):
                    category = next((cat for cat in ["JEST", "HACK:MANIPULATION", "OFFICIAL"] if f"Category: {cat}" in msg.content), "UNKNOWN")
                    break
            
            # Format the system message, replacing the placeholder
            system_message = PROMPT_TEMPLATES["generate_with_persona"]["system_message"].format(
              last_ai_message=last_ai_message,
              category=category,
              suggest_email="Suggest 'you seem to be asking too many questions, why dont you reach out directly via email @ raghunandan092@gmail.com'" if query_count > 5 else ""              
              )
              
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ]
            
            # Collect the complete response
            complete_response = []
            
            async for chunk in stream_chat_completion(messages, temperature=0.3):
                if chunk.type == "content" and chunk.content:
                    complete_response.append(chunk.content)
                yield chunk
            
            # Log the complete response
            full_response = "".join(complete_response)
            logger.info("Completed persona generation", extra={"node": self.name, "thread_id": state.thread_id, "response": full_response[:500] + ("..." if len(full_response) > 500 else "")})
            opik_context.update_current_span(
                name="generate_with_persona",
                input={"ai_message": last_ai_message},
                output={"full_response": full_response},
                metadata={"system_prompt": messages}
            )
                
        except Exception as e:
            logger.error("Persona generation failed", extra={"node": self.name, "thread_id": state.thread_id, "error": str(e)})
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
    """Route based on whether a tool was used in the QueryOrRespondNode."""
    for message in reversed(state.messages):
        if isinstance(message, ToolMessage) and message.tool_name == "retrieve":
            return "tools"
    return "END"

async def stream_chat_completion(
    messages: List[Dict[str, str]], 
    functions: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.1
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
            "stream": True
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
                            function_args=function_args
                        )
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse function arguments", extra={"error": str(e), "buffer": state.buffer})
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
        logger.error("Streaming failed", extra={"action": "streaming_error", "error": str(e)})
        raise


# Updated graph assembly with streaming support
streaming_graph = StreamingStateGraph(
    nodes={
        "relevance_check": RelevanceCheckNode(name="relevance_check"),
        "query_or_respond": QueryOrRespondNode(name="query_or_respond"),
        "few_shot_selector": FewShotSelectorNode(name="few_shot_selector"),
        "generate_with_context": GenerateWithRetrievedContextNode(name="generate_with_retrieved_context"),
        "generate_with_persona": GenerateWithPersonaNode(name="generate_with_persona"),
    },
    edges={
        "relevance_check": {
            "CONTEXTUAL": "query_or_respond",
            "IRRELEVANT": "few_shot_selector",
            "RELEVANT": "query_or_respond"
        },
        "query_or_respond": {
            "tools": "generate_with_context",
            "END": "few_shot_selector"
        },
        "few_shot_selector": {
            "default": "generate_with_persona"
        },
        "generate_with_context": {
            "default": "generate_with_persona"
        },
        "generate_with_persona": {
            "default": "END"
        }
    },
    entry_point="relevance_check"
)

# Initialize at server startup
async def init_example_selector():
    examples = PROMPT_TEMPLATES.get("examples", [])
    await ExampleSelector.initialize_examples(examples)



