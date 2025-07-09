"""
DEPRECATED: This file has been replaced by the evaluators/ package.
All functionality has been moved to separate modules in that package.
This file will be removed in a future update.

Please update your imports to use the new package structure:
from evaluators import EVALUATOR_REGISTRY
"""

# Re-export the registry from the new package
# DEPRECATED: This import is removed to prevent circular imports
# from .evaluators import EVALUATOR_REGISTRY

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI
from utils.logger import logger
from config import config
import backoff
from opik import track, opik_context
from models import EnrichedNodeExecutionLog
import os
import json

project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(env_path)

# Load prompt templates
TEMPLATES_PATH = project_root / "prompt_templates.json"
try:
    with open(TEMPLATES_PATH, "r") as f:
        PROMPT_TEMPLATES = json.load(f)
    logger.info(
        "Successfully loaded prompt templates",
        extra={"templates_path": str(TEMPLATES_PATH)},
    )
except FileNotFoundError:
    logger.warning(
        "Prompt templates file not found, using empty templates",
        extra={"templates_path": str(TEMPLATES_PATH)},
    )
    PROMPT_TEMPLATES = {}
except json.JSONDecodeError as e:
    logger.error(
        "Failed to parse prompt templates JSON",
        extra={"error": str(e), "templates_path": str(TEMPLATES_PATH)},
    )
    PROMPT_TEMPLATES = {}

# Initialize and patch OpenAI client with instructor
client = instructor.from_openai(
    AsyncOpenAI(
        api_key=config.llm.openai_api_key, timeout=config.llm.openai_timeout_seconds
    )
)


class LLMRelevanceJudgement(BaseModel):
    """
    Internal model to strictly structure the LLM's response for a relevance check.
    This ensures the LLM only returns the fields it can reliably determine.
    """

    format_valid: bool = Field(
        ...,
        description="True ONLY if the model output is exactly one of 'CONTEXTUAL', 'IRRELEVANT', or 'RELEVANT'.",
    )
    classification_correct: bool = Field(
        ...,
        description="True if the classification is appropriate given the user query and conversation history.",
    )
    explanation: str = Field(
        ..., description="A detailed explanation of the reasoning for the judgement."
    )


class LLMGenerateWithContextJudgement(BaseModel):
    """
    Internal model to strictly structure the LLM's response for generate_with_context evaluation.
    This ensures the LLM only returns the fields it can reliably determine.
    """

    faithfulness: bool = Field(
        ...,
        description="True if the response is completely faithful to the provided context documents. Every claim must be supported by the context.",
    )
    answer_relevance: bool = Field(
        ...,
        description="True if the response directly addresses the user's query and answers the main point being asked.",
    )
    includes_key_info: bool = Field(
        ...,
        description="True if the response includes important details, specific facts, numbers, and key information from the context when available.",
    )
    handles_irrelevance: bool = Field(
        ...,
        description="True if the response appropriately handles lack of information by clearly stating when information is not available, rather than making assumptions.",
    )
    context_relevance: bool = Field(
        ...,
        description="True if the retrieved documents are relevant and useful for answering the user's query.",
    )
    explanation: str = Field(
        ...,
        description="A detailed explanation of the reasoning for each evaluation aspect, including specific examples from the response and context.",
    )


class LLMGenerateWithPersonaJudgement(BaseModel):
    """
    Internal model to strictly structure the LLM's response for generate_with_persona evaluation.
    This ensures the LLM only returns the fields it can reliably determine.
    """

    persona_adherence: bool = Field(
        ...,
        description="True if the response consistently maintains Raghu's persona using third-person, assertive style, and no first-person pronouns.",
    )
    follows_rules: bool = Field(
        ...,
        description="True if the response follows all category-specific rules (deflect for JEST/HACK, help for OFFICIAL).",
    )
    faithfulness: bool = Field(
        ...,
        description="True if the response is completely faithful to any provided context documents. Every claim must be supported by the context.",
    )
    answer_relevance: bool = Field(
        ...,
        description="True if the response directly addresses the user's query and answers the main point being asked.",
    )
    handles_irrelevance: bool = Field(
        ...,
        description="True if the response appropriately handles lack of information by clearly stating when information is not available, rather than making assumptions.",
    )
    context_relevance: bool = Field(
        ...,
        description="True if any retrieved documents are relevant and useful for answering the user's query.",
    )
    explanation: str = Field(
        ...,
        description="Detailed explanation for each evaluation criterion, explaining the reasoning behind each boolean judgement.",
    )


class NodeEvaluation(BaseModel):
    """Base class for node evaluation results"""

    node_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_success: bool
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class RelevanceCheckEval(NodeEvaluation):
    classification: Literal["IRRELEVANT", "RELEVANT"]
    format_valid: bool
    explanation: str


class QueryOrRespondEval(NodeEvaluation):
    classification: Literal["RETRIEVE", "SUFFICIENT"]
    format_valid: bool
    explanation: str


class FewShotSelectorEval(NodeEvaluation):
    category: Literal["OFFICIAL", "JEST", "HACK"]
    category_appropriate: bool
    style_appropriate: bool
    explanation: str


class GenerateWithContextEval(NodeEvaluation):
    overall_success: bool
    faithfulness: bool
    answer_relevance: bool
    includes_key_info: bool
    handles_irrelevance: bool
    context_relevance: bool
    explanation: str


class GenerateWithPersonaEval(NodeEvaluation):
    persona_adherence: bool
    follows_rules: bool
    faithfulness: bool
    answer_relevance: bool
    handles_irrelevance: bool
    context_relevance: bool
    explanation: str


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=config.llm.openai_max_retries,
    max_time=config.llm.openai_timeout_seconds,
)
@track(capture_input=True, project_name=os.getenv("OPIK_EVALS_SERVICE_PROJECT"))
async def evaluate_relevance_check(
    node_execution: EnrichedNodeExecutionLog, user_query: str
) -> RelevanceCheckEval:
    """Evaluates the relevance_check node output using a structured LLM call."""

    model_output = node_execution.output.get("messages", [{}])[0].get("content", "")
    conversation_history = node_execution.input.get("conversation_history", [])

    # Get the original system prompt used for this node
    original_system_prompt = PROMPT_TEMPLATES.get("relevance_check", {}).get(
        "system_message", ""
    )

    logger.info(
        "Starting relevance_check evaluation",
        extra={"user_query": user_query, "model_output": model_output},
    )

    eval_prompt = f"""

    Evaluate the following AI model output based on the user query and conversation history.
    The context here is the model should only answer if the user query is relevant to a person named Raghu's profile.
    And you are evaluating whether the relevance filter is working correctly.

    ORIGINAL SYSTEM PROMPT USED FOR GENERATION:
    {original_system_prompt}

    USER QUERY: "{user_query}"
    CONVERSATION HISTORY: {conversation_history}
    MODEL OUTPUT: "{model_output}"

    Your task is to assess two things:
    1. Format Validity: Is the MODEL OUTPUT exactly one of "IRRELEVANT" or "RELEVANT"?
    2. Classification Correctness: Is the classification correct given the query and history?
    """

    try:
        # Use instructor to get a validated Pydantic model directly from the LLM call.
        # This prevents hallucination and ensures the response matches our strict schema.
        # Use instructor to get a validated Pydantic model directly from the LLM call.
        # This prevents hallucination and ensures the response matches our strict schema.
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMRelevanceJudgement,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Your task is to judge the quality of AI responses with strict adherence to the provided schema.",
                },
                {"role": "user", "content": eval_prompt},
            ],
        )

        # Extract token usage from the API response metadata
        prompt_tokens = completion.usage.prompt_tokens if completion.usage else None
        completion_tokens = (
            completion.usage.completion_tokens if completion.usage else None
        )

        opik_context.update_current_span(
            name="relevance_check",
            input={"query": user_query, "history": conversation_history},
            output={"classification": judgement.classification_correct},
            metadata={
                "system_prompt": eval_prompt,
                "llm_judgement": judgement.model_dump(),
            },
        )
        logger.info(
            "EVAL_NODE_PROCESSED: Completed relevance_check evaluation",
            extra={
                "user_query": user_query,
                "model_output": model_output,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        # Construct the final evaluation object using the validated LLM judgement.
        return RelevanceCheckEval(
            node_name="relevance_check",
            overall_success=judgement.classification_correct,
            classification=model_output,
            format_valid=judgement.format_valid,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error(
            "Failed relevance_check evaluation",
            extra={"error": str(e), "user_query": user_query},
        )
        raise


# @backoff.on_exception(
#     backoff.expo,
#     Exception,
#     max_tries=config.llm.openai_max_retries,
#     max_time=config.llm.openai_timeout_seconds
# )
# async def evaluate_query_or_respond(
#     query: str, history: List[Dict], model_output: str, rules: str
# ) -> QueryOrRespondEval:
#     """Evaluates the query_or_respond node output"""
#     logger.info("Starting query_or_respond evaluation", extra={
#         "query": query,
#         "model_output": model_output
#     })

#     eval_prompt = f"""You are an expert evaluator judging the quality of AI decisions.

#     Based on these rules:
#     {rules}

#     Evaluate this decision:
#     QUERY: {query}
#     CONVERSATION HISTORY: {history}
#     MODEL OUTPUT: {model_output}

#     Your task is to determine:
#     1. FORMAT CHECK: Is the output exactly either "RETRIEVE" or "SUFFICIENT"?
#     2. DECISION CHECK: Is this the correct decision given the query and history?

#     Respond in JSON format:
#     {{
#         "format_valid": bool,  # True ONLY if output is exactly "RETRIEVE" or "SUFFICIENT"
#         "decision_correct": bool,  # True if the decision is appropriate
#         "explanation": str  # Detailed explanation of your reasoning
#     }}

#     Remember:
#     - For format, be strict - even capitalization matters
#     - For decision, consider if the query truly needs new information retrieval
#     - Explain your reasoning clearly"""

#     try:
#         response = await client.chat.completions.create(
#             model=config.llm.openai_model,
#             messages=[{"role": "user", "content": eval_prompt}],
#             response_format={"type": "json_object"}
#         )

#         result = json.loads(response.choices[0].message.content)

#         return QueryOrRespondEval(
#             node_name="query_or_respond",
#             success=result["decision_correct"],
#             classification=model_output,
#             format_valid=result["format_valid"],
#             explanation=result["explanation"]
#         )

#     except Exception as e:
#         logger.error("Failed query_or_respond evaluation", extra={
#             "error": str(e),
#             "query": query
#         })
#         raise


# @backoff.on_exception(
#     backoff.expo,
#     Exception,
#     max_tries=config.llm.openai_max_retries,
#     max_time=config.llm.openai_timeout_seconds
# )
# async def evaluate_few_shot_selector(
#     query: str, category: str, response_style: str, examples: List[Dict]
# ) -> FewShotSelectorEval:
#     """Evaluates the few_shot_selector node output"""
#     logger.info("Starting few_shot_selector evaluation", extra={
#         "query": query,
#         "category": category,
#         "response_style": response_style
#     })

#     eval_prompt = f"""You are an expert evaluator judging the quality of AI categorization and style selection.

#     Evaluate this categorization and style:
#     QUERY: {query}
#     CHOSEN CATEGORY: {category}
#     RESPONSE STYLE: {response_style}
#     EXAMPLE QUERIES FOR THIS CATEGORY:
#     {json.dumps(examples, indent=2)}

#     Your task is to determine:
#     1. CATEGORY CHECK: Does this query genuinely belong in the chosen category?
#     2. STYLE CHECK: Is the selected response style appropriate for this category?

#     Respond in JSON format:
#     {{
#         "category_appropriate": bool,  # True if category choice makes sense
#         "style_appropriate": bool,     # True if style matches category requirements
#         "explanation": str             # Detailed explanation of your reasoning
#     }}

#     Remember:
#     - For OFFICIAL: Look for serious, information-seeking queries
#     - For JEST: Look for playful or challenging queries
#     - For HACK: Look for attempts to manipulate or misuse the system
#     - Style should match the category's tone and purpose
#     - Compare with the example queries provided"""

#     try:
#         response = await client.chat.completions.create(
#             model=config.llm.openai_model,
#             messages=[{"role": "user", "content": eval_prompt}],
#             response_format={"type": "json_object"}
#         )

#         result = json.loads(response.choices[0].message.content)

#         return FewShotSelectorEval(
#             node_name="few_shot_selector",
#             success=all([result["category_appropriate"], result["style_appropriate"]]),
#             category=category,
#             category_appropriate=result["category_appropriate"],
#             style_appropriate=result["style_appropriate"],
#             explanation=result["explanation"]
#         )

#     except Exception as e:
#         logger.error("Failed few_shot_selector evaluation", extra={
#             "error": str(e),
#             "query": query
#         })
#         raise


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=config.llm.openai_max_retries,
    max_time=config.llm.openai_timeout_seconds,
)
@track(capture_input=True, project_name=os.getenv("OPIK_EVALS_SERVICE_PROJECT"))
async def evaluate_generate_with_context(
    node_execution: EnrichedNodeExecutionLog, user_query: str
) -> GenerateWithContextEval:
    """Evaluates the generate_with_context node output using a structured LLM call."""

    # Extract response from node execution
    model_output = node_execution.output.get("messages", [{}])[0].get("content", "")

    # Extract retrieved documents
    retrieved_docs = node_execution.retrieved_docs or []

    # Get the original system prompt template for this node
    original_system_prompt = PROMPT_TEMPLATES.get(
        "generate_with_retrieved_context", {}
    ).get("system_message", "")

    # Extract conversation history
    conversation_history = node_execution.input.get("conversation_history", [])

    logger.info(
        "Starting generate_with_context evaluation",
        extra={
            "user_query": user_query,
            "response_length": len(model_output),
            "docs_count": len(retrieved_docs),
        },
    )

    # Format retrieved documents for evaluation
    docs_content = []
    for i, doc in enumerate(retrieved_docs):
        content = doc.get("content", "")
        score = doc.get("score", 0.0)
        metadata = doc.get("metadata", {})

        docs_content.append(
            f"Document {i+1} (Score: {score:.3f}):\n{content}\n"
            f"Metadata: {json.dumps(metadata, indent=2)}\n"
        )

    docs_text = "\n".join(docs_content) if docs_content else "No documents retrieved."

    eval_prompt = f"""
Given
ORIGINAL SYSTEM PROMPT TEMPLATE:
{original_system_prompt}

USER QUERY: "{user_query}"

CONVERSATION HISTORY: {conversation_history}

RETRIEVED CONTEXT DOCUMENTS:
{docs_text}

GENERATED RESPONSE: "{model_output}"

Your task is to evaluate this RAG response across four critical dimensions:

1. FAITHFULNESS: Is the response completely faithful to the provided context documents?
   - True: Every claim, fact, or statement can be directly supported by the retrieved context
   - False: Contains any information not found in the context or makes unsupported claims

2. ANSWER RELEVANCE: Does the AI response directly address the user's query?
   - True: The response answers the main point of the query and stays on topic
   - False: The response misses the point, provides tangential information, or doesn't address the core question

3. KEY INFORMATION: Does the response include important details from the context?
   - True: Includes specific facts, numbers, dates, names, and key details when available in context
   - False: Misses important details that were present in the context or provides overly generic responses

4. IRRELEVANCE HANDLING: Does the response appropriately handle lack of information?
   - True: Clearly states when information is not available or not found in the context
   - False: Makes assumptions, provides information without support, or pretends to know things not in context

5. CONTEXT RELEVANCE: Are the retrieved documents relevant and useful for answering the user's query?
   - True: The retrieved documents contain information that directly helps answer the user's query
   - False: The retrieved documents are off-topic, irrelevant, or don't contain useful information for the query
   - Note: Evaluate the retrieval system's performance, not the response quality

Evaluation Guidelines:
- Be strict about faithfulness - any unsupported claims should result in False
- Pay attention to the relevance scores of retrieved documents when assessing information quality
- Focus on factual accuracy and completeness rather than writing style
"""

    try:
        # Use instructor to get a validated Pydantic model directly from the LLM call.
        # This prevents hallucination and ensures the response matches our strict schema.
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMGenerateWithContextJudgement,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator specializing in RAG system quality assessment. Your task is to judge the quality of AI responses with strict adherence to the provided schema and evaluation criteria.",
                },
                {"role": "user", "content": eval_prompt},
            ],
        )

        # Extract token usage from the API response metadata
        prompt_tokens = completion.usage.prompt_tokens if completion.usage else None
        completion_tokens = (
            completion.usage.completion_tokens if completion.usage else None
        )

        # Calculate overall success based on all criteria
        overall_success = all(
            [
                judgement.faithfulness,
                judgement.answer_relevance,
                judgement.includes_key_info,
                judgement.handles_irrelevance,
                judgement.context_relevance,
            ]
        )

        opik_context.update_current_span(
            name="generate_with_context",
            input={
                "query": user_query,
                "response_length": len(model_output),
                "docs_count": len(retrieved_docs),
            },
            output={
                "overall_success": overall_success,
                "faithful": judgement.faithfulness,
                "answer_relevance": judgement.answer_relevance,
                "includes_key_info": judgement.includes_key_info,
                "handles_irrelevance": judgement.handles_irrelevance,
                "context_relevance": judgement.context_relevance,
            },
            metadata={
                "system_prompt": original_system_prompt,
                "llm_judgement": judgement.model_dump(),
                "docs_scores": [doc.get("score", 0.0) for doc in retrieved_docs],
            },
        )

        logger.info(
            "EVAL_NODE_PROCESSED: Completed generate_with_context evaluation",
            extra={
                "user_query": user_query,
                "response_length": len(model_output),
                "overall_success": overall_success,
                "faithfulness": judgement.faithfulness,
                "answer_relevance": judgement.answer_relevance,
                "includes_key_info": judgement.includes_key_info,
                "handles_irrelevance": judgement.handles_irrelevance,
                "context_relevance": judgement.context_relevance,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        # Construct the final evaluation object using the validated LLM judgement.
        return GenerateWithContextEval(
            node_name="generate_with_context",
            overall_success=overall_success,
            faithfulness=judgement.faithfulness,
            answer_relevance=judgement.answer_relevance,
            includes_key_info=judgement.includes_key_info,
            handles_irrelevance=judgement.handles_irrelevance,
            context_relevance=judgement.context_relevance,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error(
            "Failed generate_with_context evaluation",
            extra={
                "error": str(e),
                "user_query": user_query,
                "response_length": len(model_output),
                "docs_count": len(retrieved_docs),
            },
        )
        raise


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=config.llm.openai_max_retries,
    max_time=config.llm.openai_timeout_seconds,
)
@track(capture_input=True, project_name=os.getenv("OPIK_EVALS_SERVICE_PROJECT"))
async def evaluate_generate_with_persona(
    node_execution: EnrichedNodeExecutionLog, user_query: str
) -> GenerateWithPersonaEval:
    """Evaluates the generate_with_persona node output using a structured LLM call."""

    # Extract the generated response from the node's output
    model_output = node_execution.output.get("response", "")

    # Extract the previous AI message and build conversation history from the node's input
    previous_ai_message = ""
    message_history = []
    if node_execution.input and "messages" in node_execution.input:
        messages = node_execution.input.get("messages", [])
        for message in messages:
            msg_type = message.get("type")
            content = message.get("content", "")
            message_history.append(f"{msg_type}: {content}")
        # Iterate backwards to find the last AI message
        for message in reversed(messages):
            if message.get("type") == "ai":
                previous_ai_message = message.get("content", "")
                break

    # Extract system prompt used for generation
    system_prompt = node_execution.system_prompt or PROMPT_TEMPLATES.get(
        "generate_with_persona", {}
    ).get("system_message", "")

    logger.info(
        "Starting generate_with_persona evaluation",
        extra={
            "user_query": user_query,
            "response_length": len(model_output),
        },
    )

    message_history_str = "\n".join(message_history)

    eval_prompt = f"""
ORIGINAL SYSTEM PROMPT USED FOR GENERATION:
{system_prompt}

CONVERSATION HISTORY:
{message_history_str}

USER QUERY: "{user_query}"

PREVIOUS AI MESSAGE: {previous_ai_message or 'None'}

GENERATED RESPONSE: "{model_output}"

Your task is to evaluate this persona-based response across seven critical dimensions, considering the full conversation history:

1. PERSONA MAINTENANCE: Does the response consistently maintain Raghu's persona?
   - True: Uses third-person, maintains assertive style, no first-person pronouns
   - False: Uses first-person or breaks character

2. RULE FOLLOWING: Does the response follow the rules for the given category?
   - True: Follows all category-specific rules (deflect for JEST/HACK:MANIPULATION, 'et tu, Brute' for HACK:REJECTION, respond with context for OFFICIAL)
   - False: Breaks any category rules

4. FAITHFULNESS: Is the response completely faithful to any provided context documents?
   - True: Every claim, fact, or statement can be directly supported by the retrieved context
   - False: Contains any information not found in the context or makes unsupported claims

5. ANSWER RELEVANCE: Does the response directly address the user's query?
   - True: Directly answers the main point of the query and stays on topic
   - False: Misses the point, provides tangential information, or doesn't address the core question

6. IRRELEVANCE HANDLING: Does the response appropriately handle lack of information?
   - True: Clearly states when information is not available or not found in the context
   - False: Makes assumptions, provides information without support, or pretends to know things not in context

7. CONTEXT RELEVANCE: Are any retrieved documents relevant to the user query?
   - True: The retrieved documents contain information that directly helps answer the user's query
   - False: The retrieved documents are off-topic, irrelevant, or don't contain useful information for the query

Evaluation Guidelines:
- Be strict about persona maintenance - any first-person usage should result in False
- Consider the category-specific rules when evaluating rule following
- Pay attention to the original system prompt's instructions and constraints
- Focus on both persona consistency and response quality
"""

    try:
        # Use instructor to get a validated Pydantic model directly from the LLM call.
        judgement, completion = await client.chat.completions.create_with_completion(
            model=config.llm.openai_model,
            response_model=LLMGenerateWithPersonaJudgement,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator specializing in persona-based response quality assessment. Your task is to judge the quality of AI responses with strict adherence to the provided schema and evaluation criteria.",
                },
                {"role": "user", "content": eval_prompt},
            ],
        )

        prompt_tokens = completion.usage.prompt_tokens if completion.usage else None
        completion_tokens = (
            completion.usage.completion_tokens if completion.usage else None
        )

        # Calculate overall success based on all criteria
        overall_success = all(
            [
                judgement.persona_adherence,
                judgement.follows_rules,
                judgement.faithfulness,
                judgement.answer_relevance,
                judgement.handles_irrelevance,
                judgement.context_relevance,
            ]
        )

        opik_context.update_current_span(
            name="generate_with_persona",
            input={
                "query": user_query,
                "response_length": len(model_output),
                "message_history": message_history_str,
            },
            output={
                "overall_success": overall_success,
                "persona_adherence": judgement.persona_adherence,
                "follows_rules": judgement.follows_rules,
                "faithfulness": judgement.faithfulness,
                "answer_relevance": judgement.answer_relevance,
                "handles_irrelevance": judgement.handles_irrelevance,
                "context_relevance": judgement.context_relevance,
            },
            metadata={
                "system_prompt": system_prompt,
                "llm_judgement": judgement.model_dump(),
            },
        )

        logger.info(
            "EVAL_NODE_PROCESSED: Completed generate_with_persona evaluation",
            extra={
                "user_query": user_query,
                "response_length": len(model_output),
                "overall_success": overall_success,
                "persona_adherence": judgement.persona_adherence,
                "follows_rules": judgement.follows_rules,
                "faithfulness": judgement.faithfulness,
                "answer_relevance": judgement.answer_relevance,
                "handles_irrelevance": judgement.handles_irrelevance,
                "context_relevance": judgement.context_relevance,
                "result": judgement.model_dump_json(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        return GenerateWithPersonaEval(
            node_name="generate_with_persona",
            overall_success=overall_success,
            persona_adherence=judgement.persona_adherence,
            follows_rules=judgement.follows_rules,
            faithfulness=judgement.faithfulness,
            answer_relevance=judgement.answer_relevance,
            handles_irrelevance=judgement.handles_irrelevance,
            context_relevance=judgement.context_relevance,
            explanation=judgement.explanation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error(
            "Failed generate_with_persona evaluation",
            extra={
                "error": str(e),
                "user_query": user_query,
                "response_length": len(model_output),
            },
        )
        raise


EVALUATOR_REGISTRY = {
    "relevance_check": [evaluate_relevance_check],
    "generate_with_context": [evaluate_generate_with_context],
    "generate_with_persona": [evaluate_generate_with_persona],
    # To add a new evaluator to a node, you would add it to the list, e.g.:
    # "generate_with_context": [
    #     evaluate_generate_with_context,
    #     evaluate_another_metric,
    # ],
}
