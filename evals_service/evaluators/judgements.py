from pydantic import BaseModel, Field


class LLMRelevanceJudgement(BaseModel):
    """
    Internal model to strictly structure the LLM's response for a relevance check.
    This ensures the LLM only returns the fields it can reliably determine.
    """

    format_valid: bool = Field(
        ...,
        description="True ONLY if the model output is exactly one of 'IRRELEVANT', or 'RELEVANT'.",
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