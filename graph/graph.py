from langsmith import traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import ChatTogether
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages, AIMessage, ToolMessage, AIMessageChunk
from langchain_core.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,     
    PromptTemplate,
    FewShotPromptTemplate
)
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
from utils.logger import logger
import uuid
from sentry_sdk import capture_exception, capture_message
import json
import re


if os.path.exists('.env'):
    load_dotenv('.env')
    load_dotenv('.env.development')

# Load prompt templates from the JSON file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_PATH = os.path.join(CURRENT_DIR, "prompt_templates.json")
with open(TEMPLATES_PATH, "r") as f:
    PROMPT_TEMPLATES = json.load(f)




os.environ["LANGCHAIN_TRACING_V2"] = "true"
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
os.environ["USER_AGENT"] = "my-langchain-app/v0.1.1"


rate_limiter = InMemoryRateLimiter(
    requests_per_second=10,  
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

# Initialize OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", rate_limiter=rate_limiter, temperature=0.1)  # Use gpt-4
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize Pinecone

index_name = "langchain-chatraghu-embeddings"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
index = pc.Index(index_name)


trimmer = trim_messages(    
    max_tokens=24,
    strategy="last",
    token_counter=len,    
    start_on="human",
)

graph_builder = StateGraph(MessagesState)

# Global cache variable
_example_selector_cache: Optional[SemanticSimilarityExampleSelector] = None

def get_example_selector() -> SemanticSimilarityExampleSelector:
    """
    Get or create a cached example selector with FAISS index.
    Returns:
        SemanticSimilarityExampleSelector: Cached example selector instance
    """
    global _example_selector_cache

    try:
        # Access the examples stored in the JSON file
        examples_list = PROMPT_TEMPLATES.get("examples", [])
        
        if _example_selector_cache is None:
            logger.info("Initializing example selector cache", extra={
                "action": "cache_miss",
                "examples_count": len(examples_list)
            })
            _example_selector_cache = SemanticSimilarityExampleSelector.from_examples(
                examples_list,
                OpenAIEmbeddings(),
                FAISS,
                k=3,
            )
            logger.info("Example selector cache created", extra={
                "action": "cache_created",
                "cache_size": len(examples_list)
            })
        else:
            logger.debug("Using cached example selector", extra={
                "action": "cache_hit",
                "cache_size": len(examples_list)
            })

        return _example_selector_cache

    except Exception as e:
        logger.error("Failed to get example selector", extra={
            "action": "cache_error",
            "error": str(e),
            "error_type": e.__class__.__name__
        })
        capture_exception(e)
        raise


# Precompile regex patterns for performance
_whitespace_pattern = re.compile(r'\s+')
_xml_tag_pattern = re.compile(r'<[^>]+>')
# Pattern to remove the <questions>...</questions> and <tags>...</tags> sections (case insensitive, across multiple lines)
_special_section_pattern = re.compile(r'<(?:questions|tags)>.*?</(?:questions|tags)>', re.IGNORECASE | re.DOTALL)

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by performing the following steps:
    
    1. Remove entire sections along with their content for <questions>...</questions> and <tags>...</tags>.
    2. Remove any remaining XML tags.
    3. Normalize whitespace.
    
    This meets the requirement of cleaning the text for downstream processing.
    """
    # First, remove the special XML sections along with their content
    text = _special_section_pattern.sub('', text)
    # Remove all remaining XML tags
    text = _xml_tag_pattern.sub('', text)
    # Normalize whitespace
    return _whitespace_pattern.sub(' ', text).strip()


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    try:
        logger.info("Starting retrieval", extra={
            "action": "retrieval_start",
            "query": query
        })

        query_embedding = embeddings.embed_query(query)

        # Retrieve documents along with their scores.
        doc_score_pairs = vector_store.similarity_search_by_vector_with_score(
            embedding=query_embedding,
            k=3
            #fetch_k=4,
            #lambda_mult=0.6
        )
        
        logger.info("Completed retrieval", extra={
            "action": "retrieval_complete",
            "docs_retrieved": len(doc_score_pairs)
        })
        
        # Determine the cutoff threshold:
        # Since docs are in decreasing order, the first doc holds the highest score.
        # We filter for docs that are at or above 90% of that top score,
        # but we also enforce a minimum threshold of 0.7.
        if doc_score_pairs:
            best_score = doc_score_pairs[0][1]
            threshold = max(0.7, best_score * 0.9)
        else:
            threshold = 0.7
        
        # Combine the processing for both serialized content and retrieved docs into a single loop.
        lines = []
        retrieved_docs = []
        for doc, score in doc_score_pairs:
            if score >= threshold:
                processed_content = preprocess_text(doc.page_content)
                lines.append(f"Content: {processed_content} (Score: {score:.2f})")
                retrieved_docs.append(doc)
        
        serialized = "\n\n".join(lines)
        
        return serialized, retrieved_docs
         
    except Exception as e:
        logger.error("Retrieval failed", extra={
            "action": "retrieval_error",
            "error": str(e),
            "error_type": e.__class__.__name__
        })
        capture_exception(e)
        raise


def relevance_check(state: MessagesState):    
    try:
        current_query = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
        
        logger.info("Starting relevance check", extra={
            "action": "relevance_check_start",
            "query": current_query.content if current_query else ""
        })
        
        system_message_content = PROMPT_TEMPLATES["relevance_check"]["system_message"].format(query=current_query.content)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(template=system_message_content),
                HumanMessagePromptTemplate.from_template(template="{query}"),
            ]
        )
        messages_with_system_prompt = prompt_template.format_messages(query=current_query.content, messages=state["messages"])
        response = llm.invoke(messages_with_system_prompt)
        
        logger.info("Completed relevance check", extra={
            "action": "relevance_check_complete",
            "response_type": response.content
        })
        
        return {"messages": [response]}
        
    except Exception as e:
        logger.error("Relevance check failed", extra={
            "action": "relevance_check_error",
            "error": str(e),
            "error_type": e.__class__.__name__
        })
        capture_exception(e)
        raise

def relevance_condition(state: MessagesState) -> str:
    """Route based on the relevance check response."""
    # Get the last AI message which contains the relevance check result
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            # Check if the response contains either RELEVANT or IRRELEVANT
            if "CONTEXTUAL" in message.content:
                return "CONTEXTUAL"
            elif "IRRELEVANT" in message.content:
                return "IRRELEVANT"
    # Default case if no valid response is found
    return "RELEVANT"  # Default to relevant as a fallback


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""    

    current_query = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)

    logger.info("Processing user query for context check", extra={
        "action": "query_or_respond_start",
        "query": current_query.content if current_query else "No query found"
    })

    system_message_content = PROMPT_TEMPLATES["query_or_respond"]["system_message"].format(query=current_query.content)
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(template=system_message_content),
            HumanMessagePromptTemplate.from_template(template="{query}"),
        ]
    )

    messages_with_system_prompt = prompt_template.format_messages(query=current_query.content, messages=state["messages"])

    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(messages_with_system_prompt)

    logger.info("Completed processing user query", extra={
        "action": "query_or_respond_complete",
        "response": response
    })

    return {"messages": [response]}



# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# --- Global (or module-level) definition for a static few-shot prompt template ---
GLOBAL_FEW_SHOT_PREFIX = PROMPT_TEMPLATES["few_shot"]["prefix"]
GLOBAL_FEW_SHOT_SUFFIX = PROMPT_TEMPLATES["few_shot"]["suffix"]

# Pre-create the static FewShotPromptTemplate.
GLOBAL_FEW_SHOT_PROMPT = FewShotPromptTemplate(
    example_selector=get_example_selector(),  # No harm if this is cached and reused.
    example_prompt=PromptTemplate(
        input_variables=["user_query", "potential_category", "response_style"],
        template="user_query: {user_query}\npotential_category: {potential_category}\nresponse_style: {response_style}",
    ),
    prefix=GLOBAL_FEW_SHOT_PREFIX,
    suffix=GLOBAL_FEW_SHOT_SUFFIX,
    input_variables=["query"]
)


@traceable(run_type="chain")
async def few_shot_selector(state: MessagesState):
    """
    Identifies few shot prompt examples to the current query as HACK, FUN, or STANDARD based on similarity to examples.
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    current_query = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
    
    if not current_query:
        raise ValueError("No HumanMessage found in state['messages']")

    logger.info("Starting few shot selection process", extra={
        "action": "few_shot_selector_start",
        "query": current_query.content
    })

    # Format the static few-shot prompt with dynamic fields.
    prompt_with_examples = GLOBAL_FEW_SHOT_PROMPT.format(query=current_query.content, current_date_str=current_date)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompt_with_examples),
        MessagesPlaceholder(variable_name="chat_history")
    ])       
    final_prompt = prompt_template.format_messages(chat_history=state["messages"])
    response = await llm.ainvoke(final_prompt)

    logger.info("Completed few shot selection process", extra={
        "action": "few_shot_selector_complete",
        "response": response
    })

    return {"messages": [response]}



# Step 3: Generate a response using the retrieved content.
@traceable(run_type="chain")
async def generate_with_retrieved_context(state: MessagesState):
    """Generate answer with retrieved context."""
    try:
        tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
        user_query = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
                
        current_date = datetime.now().strftime("%B %d, %Y")  # Get today's date in a readable format
       
        docs_content = "\n\n".join(doc.content for doc in tool_messages)

        system_message_content = PROMPT_TEMPLATES["generate_with_retrieved_context"]["system_message"].format(
            current_date_str=current_date,
            query=user_query.content,
            docs_content=docs_content
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message_content),
            HumanMessagePromptTemplate.from_template("{query}")
        ])

        final_prompt = prompt_template.format_messages(query=user_query.content, current_date_str=current_date)
        response = await llm.bind(temperature=0.0).ainvoke(final_prompt)        
        
        logger.info("Completed context generation", extra={
            "action": "context_generation_complete",
            "response_length": len(response.content)
        })
        
        return {"messages": [response]}
        
    except Exception as e:
        logger.error("Context generation failed", extra={
            "action": "context_generation_error",
            "error": str(e),
            "error_type": e.__class__.__name__
        })
        capture_exception(e)
        raise



@traceable(run_type="chain", tags=["persona_response"])
async def generate_with_persona(state: MessagesState):
    """Generate response in persona."""
    try:
        query_count = sum(1 for message in state["messages"] if message.type == "human") > 5
        messages = state["messages"]
        last_ai_message = None
        filtered_messages = []

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and last_ai_message is None:
                last_ai_message = msg  # Capture the last AI message
                break

            filtered_messages.append(msg)  # Collect messages that are not ToolMessages

        persona_message_content = PROMPT_TEMPLATES["generate_with_persona"]["system_message"].format(
            last_ai_message=last_ai_message.content,
            suggest_email = (
                "Suggest 'you seem to be asking too many questions, why dont you reach out directly via email @ raghunandan092@gmail.com"
                if query_count else ""
            )
        )
       
        # Create the chat prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(persona_message_content)]
        ).format_messages()
        response = await llm.bind(temperature=0.6).ainvoke(prompt_template)
        state["messages"] = trimmer.invoke(state["messages"] + [response])
        
        logger.info("Completed persona generation", extra={
            "action": "persona_generation_complete",
            "response_length": len(response.content),
            "final_messages_count": len(state["messages"])
        })
        
        return {"messages": [response]}
        
    except Exception as e:
        logger.error("Persona generation failed", extra={
            "action": "persona_generation_error",
            "error": str(e),
            "error_type": e.__class__.__name__
        })
        capture_exception(e)
        raise


graph_builder.add_node(relevance_check)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate_with_retrieved_context)
graph_builder.add_node(generate_with_persona)
graph_builder.add_node(few_shot_selector)

graph_builder.set_entry_point("relevance_check")

graph_builder.add_conditional_edges(
    "relevance_check",
    relevance_condition,
    {
        "CONTEXTUAL": "query_or_respond",
        "IRRELEVANT": "few_shot_selector"
    }
)

graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {
        END: "few_shot_selector", 
        "tools": "tools"
    }
)
graph_builder.add_edge("tools", "generate_with_retrieved_context")
graph_builder.add_edge("few_shot_selector", "generate_with_persona")
graph_builder.add_edge("generate_with_retrieved_context", "generate_with_persona")
graph_builder.add_edge("generate_with_persona", END)


memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory
    )

async def get_graph():
    """
    Asynchronously builds and compiles the graph.
    Even though some nodes are async, the compile() method is synchronous.
    """
    memory = MemorySaver()
    # Remove await here, as graph_builder.compile is synchronous.
    compiled_graph = graph_builder.compile(checkpointer=memory)
    return compiled_graph

#Warm up the cache to load common responses on startup
# async def warm_up_cache():
#     """Pre-warm the cache with common responses on startup"""
#     doc_id = str(uuid.uuid4())
#     try:
#         set_llm_cache(InMemoryCache())
#         input_message = "Tell me about yourself?"
#         config = {"configurable": {"thread_id": doc_id}}
#         async for step in graph.astream(
#             {"messages": [{"role": "user", "content": input_message}]},
#             stream_mode="values",
#             config=config,
#         ):
#             step["messages"][-1].pretty_print()
#         logger.info("Cache warmup completed successfully")
#     except Exception as e:
#         logger.error(f"Cache warmup failed: {str(e)}")


# # If running directly, still allow manual testing
# if __name__ == "__main__":
#     asyncio.run(warm_up_cache())
