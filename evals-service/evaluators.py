from datetime import datetime
from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os


project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(env_path)

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class NodeEvaluation(BaseModel):
    """Base class for node evaluation results"""

    node_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool
    metrics: Dict[str, float]


class RelevanceCheckEval(NodeEvaluation):
    classification: Literal["CONTEXTUAL", "IRRELEVANT", "RELEVANT"]
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
    is_faithful: bool
    is_relevant: bool
    includes_key_info: bool
    handles_irrelevance: bool
    explanation: str


class GenerateWithPersonaEval(NodeEvaluation):
    maintains_persona: bool
    follows_rules: bool
    maintains_consistency: bool
    uses_correct_phrases: bool
    explanation: str


async def evaluate_relevance_check(
    user_query: str, conversation_history: List[Dict], model_output: str, rules: str
) -> RelevanceCheckEval:
    pass

    # """Evaluates the relevance_check node output"""
    # logger.info("Starting relevance_check evaluation", extra={
    #     "user_query": user_query,
    #     "model_output": model_output
    # })

    # eval_prompt = f"""You are an expert evaluator judging the quality of AI responses.

    # Based on these rules:
    # {rules}

    # Evaluate this query and model output:
    # USER QUERY: {user_query}
    # CONVERSATION HISTORY: {conversation_history}
    # MODEL OUTPUT: {model_output}

    # Your task is to determine:
    # 1. FORMAT CHECK: Is the output exactly one of: "CONTEXTUAL", "IRRELEVANT", or "RELEVANT"?
    # 2. CLASSIFICATION CHECK: Given the query and conversation history, is this the correct classification?

    # Respond in JSON format:
    # {{
    #     "format_valid": bool,  # True ONLY if output matches exactly one of the allowed values
    #     "classification_correct": bool,  # True if the classification makes sense given the context
    #     "explanation": str  # Detailed explanation of your reasoning
    # }}

    # Remember:
    # - Be strict about format - even slight variations should return false
    # - Consider conversation history when judging classification correctness
    # - Provide clear reasoning in your explanation"""

    # try:
    #     response = await client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[{"role": "user", "content": eval_prompt}],
    #         response_format={"type": "json_object"}
    #     )

    #     result = json.loads(response.choices[0].message.content)

    #     return RelevanceCheckEval(
    #         node_name="relevance_check",
    #         success=result["classification_correct"],
    #         classification=model_output,
    #         format_valid=result["format_valid"],
    #         explanation=result["explanation"],
    #         metrics={
    #             "format_validity": float(result["format_valid"]),
    #             "classification_accuracy": float(result["classification_correct"])
    #         }
    #     )

    # except Exception as e:
    #     logger.error("Failed relevance_check evaluation", extra={
    #         "error": str(e),
    #         "user_query": user_query
    #     })
    #     raise


async def evaluate_query_or_respond(
    query: str, history: List[Dict], model_output: str, rules: str
) -> QueryOrRespondEval:
    pass
    # """Evaluates the query_or_respond node output"""
    # logger.info("Starting query_or_respond evaluation", extra={
    #     "query": query,
    #     "model_output": model_output
    # })

    # eval_prompt = f"""You are an expert evaluator judging the quality of AI decisions.

    # Based on these rules:
    # {rules}

    # Evaluate this decision:
    # QUERY: {query}
    # CONVERSATION HISTORY: {history}
    # MODEL OUTPUT: {model_output}

    # Your task is to determine:
    # 1. FORMAT CHECK: Is the output exactly either "RETRIEVE" or "SUFFICIENT"?
    # 2. DECISION CHECK: Is this the correct decision given the query and history?

    # Respond in JSON format:
    # {{
    #     "format_valid": bool,  # True ONLY if output is exactly "RETRIEVE" or "SUFFICIENT"
    #     "decision_correct": bool,  # True if the decision is appropriate
    #     "explanation": str  # Detailed explanation of your reasoning
    # }}

    # Remember:
    # - For format, be strict - even capitalization matters
    # - For decision, consider if the query truly needs new information retrieval
    # - Explain your reasoning clearly"""

    # try:
    #     response = await client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[{"role": "user", "content": eval_prompt}],
    #         response_format={"type": "json_object"}
    #     )

    #     result = json.loads(response.choices[0].message.content)

    #     return QueryOrRespondEval(
    #         node_name="query_or_respond",
    #         success=result["decision_correct"],
    #         classification=model_output,
    #         format_valid=result["format_valid"],
    #         explanation=result["explanation"],
    #         metrics={
    #             "format_validity": float(result["format_valid"]),
    #             "decision_accuracy": float(result["decision_correct"])
    #         }
    #     )

    # except Exception as e:
    #     logger.error("Failed query_or_respond evaluation", extra={
    #         "error": str(e),
    #         "query": query
    #     })
    #     raise


async def evaluate_few_shot_selector(
    query: str, category: str, response_style: str, examples: List[Dict]
) -> FewShotSelectorEval:
    pass

    # """Evaluates the few_shot_selector node output"""
    # logger.info("Starting few_shot_selector evaluation", extra={
    #     "query": query,
    #     "category": category,
    #     "response_style": response_style
    # })

    # eval_prompt = f"""You are an expert evaluator judging the quality of AI categorization and style selection.

    # Evaluate this categorization and style:
    # QUERY: {query}
    # CHOSEN CATEGORY: {category}
    # RESPONSE STYLE: {response_style}
    # EXAMPLE QUERIES FOR THIS CATEGORY:
    # {json.dumps(examples, indent=2)}

    # Your task is to determine:
    # 1. CATEGORY CHECK: Does this query genuinely belong in the chosen category?
    # 2. STYLE CHECK: Is the selected response style appropriate for this category?

    # Respond in JSON format:
    # {{
    #     "category_appropriate": bool,  # True if category choice makes sense
    #     "style_appropriate": bool,     # True if style matches category requirements
    #     "explanation": str             # Detailed explanation of your reasoning
    # }}

    # Remember:
    # - For OFFICIAL: Look for serious, information-seeking queries
    # - For JEST: Look for playful or challenging queries
    # - For HACK: Look for attempts to manipulate or misuse the system
    # - Style should match the category's tone and purpose
    # - Compare with the example queries provided"""

    # try:
    #     response = await client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[{"role": "user", "content": eval_prompt}],
    #         response_format={"type": "json_object"}
    #     )

    #     result = json.loads(response.choices[0].message.content)

    #     return FewShotSelectorEval(
    #         node_name="few_shot_selector",
    #         success=all([result["category_appropriate"], result["style_appropriate"]]),
    #         category=category,
    #         category_appropriate=result["category_appropriate"],
    #         style_appropriate=result["style_appropriate"],
    #         explanation=result["explanation"],
    #         metrics={
    #             "category_accuracy": float(result["category_appropriate"]),
    #             "style_accuracy": float(result["style_appropriate"])
    #         }
    #     )

    # except Exception as e:
    #     logger.error("Failed few_shot_selector evaluation", extra={
    #         "error": str(e),
    #         "query": query
    #     })
    #     raise


async def evaluate_generate_with_context(
    query: str, response: str, docs_content: List[str]
) -> GenerateWithContextEval:
    pass

    # """Evaluates the generate_with_context node output"""
    # logger.info("Starting generate_with_context evaluation", extra={
    #     "query": query,
    #     "response_length": len(response)
    # })

    # eval_prompt = f"""You are an expert evaluator judging the quality of AI-generated responses.

    # Evaluate this RAG (Retrieval-Augmented Generation) response:
    # QUERY: {query}
    # CONTEXT DOCUMENTS: {' '.join(docs_content)}
    # RESPONSE: {response}

    # Your task is to determine four key aspects with strict criteria:

    # 1. FAITHFULNESS: Is the response completely faithful to the provided context?
    # - True: Every claim can be supported by the context
    # - False: Contains any information not found in context

    # 2. RELEVANCE: Does the response directly address the query?
    # - True: Directly answers the main point of the query
    # - False: Misses the point or provides tangential information

    # 3. KEY INFORMATION: Does it include the important details from context?
    # - True: Includes specific facts, numbers, and key details when available
    # - False: Misses important details that were present in context

    # 4. IRRELEVANCE HANDLING: Does it appropriately handle lack of information?
    # - True: Clearly states when information is not available
    # - False: Makes assumptions or provides information without support

    # Respond in JSON format:
    # {{
    #     "is_faithful": bool,
    #     "is_relevant": bool,
    #     "includes_key_info": bool,
    #     "handles_irrelevance": bool,
    #     "explanation": str  # Detailed explanation for each aspect
    # }}"""

    # try:
    #     response = await client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[{"role": "user", "content": eval_prompt}],
    #         response_format={"type": "json_object"}
    #     )

    #     result = json.loads(response.choices[0].message.content)

    #     return GenerateWithContextEval(
    #         node_name="generate_with_context",
    #         success=all([
    #             result["is_faithful"],
    #             result["is_relevant"],
    #             result["includes_key_info"],
    #             result["handles_irrelevance"]
    #         ]),
    #         is_faithful=result["is_faithful"],
    #         is_relevant=result["is_relevant"],
    #         includes_key_info=result["includes_key_info"],
    #         handles_irrelevance=result["handles_irrelevance"],
    #         explanation=result["explanation"],
    #         metrics={
    #             "faithfulness": float(result["is_faithful"]),
    #             "relevance": float(result["is_relevant"]),
    #             "key_info": float(result["includes_key_info"]),
    #             "irrelevance_handling": float(result["handles_irrelevance"])
    #         }
    #     )

    # except Exception as e:
    #     logger.error("Failed generate_with_context evaluation", extra={
    #         "error": str(e),
    #         "query": query
    #     })
    #     raise


async def evaluate_generate_with_persona(
    response: str, category: str, last_ai_message: Optional[str], rules: str
) -> GenerateWithPersonaEval:
    pass

    # """Evaluates the generate_with_persona node output"""
    # logger.info("Starting generate_with_persona evaluation", extra={
    #     "category": category,
    #     "response_length": len(response)
    # })

    # eval_prompt = f"""You are an expert evaluator judging the quality of AI persona-based responses.

    # Evaluate this response:
    # RESPONSE: {response}
    # CATEGORY: {category}
    # PREVIOUS AI MESSAGE: {last_ai_message or 'None'}
    # RULES: {rules}

    # Your task is to evaluate four critical aspects:

    # 1. PERSONA MAINTENANCE: Does it consistently maintain Raghu's persona?
    # - True: Uses third-person, maintains assertive style, no first-person pronouns
    # - False: Uses first-person or breaks character

    # 2. RULE FOLLOWING: Does it follow the rules for the given category?
    # - True: Follows all category-specific rules (deflect for JEST/HACK, help for OFFICIAL)
    # - False: Breaks any category rules

    # 3. CONSISTENCY: Does it maintain consistency with the previous message?
    # - True: Maintains same facts and stance as previous message
    # - False: Contradicts or ignores previous context

    # 4. PHRASE USAGE: Does it use required phrases correctly?
    # - True: Uses "Et tu, Brute?" when appropriate for JEST
    # - False: Misses required phrases or uses them incorrectly

    # Respond in JSON format:
    # {{
    #     "maintains_persona": bool,
    #     "follows_rules": bool,
    #     "maintains_consistency": bool,
    #     "uses_correct_phrases": bool,
    #     "explanation": str  # Detailed explanation for each aspect
    # }}"""

    # try:
    #     # Get LLM evaluation
    #     response = await client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=[{"role": "user", "content": eval_prompt}],
    #         response_format={"type": "json_object"}
    #     )

    #     result = json.loads(response.choices[0].message.content)

    #     return GenerateWithPersonaEval(
    #         node_name="generate_with_persona",
    #         success=all([
    #             result["maintains_persona"],
    #             result["follows_rules"],
    #             result["maintains_consistency"],
    #             result["uses_correct_phrases"]
    #         ]),
    #         maintains_persona=result["maintains_persona"],
    #         follows_rules=result["follows_rules"],
    #         maintains_consistency=result["maintains_consistency"],
    #         uses_correct_phrases=result["uses_correct_phrases"],
    #         explanation=result["explanation"],
    #         metrics={
    #             "persona_maintenance": float(result["maintains_persona"]),
    #             "rule_following": float(result["follows_rules"]),
    #             "consistency": float(result["maintains_consistency"]),
    #             "phrase_usage": float(result["uses_correct_phrases"])
    #         }
    #     )

    # except Exception as e:
    #     logger.error("Failed generate_with_persona evaluation", extra={
    #         "error": str(e),
    #         "category": category
    #     })
    #     raise
