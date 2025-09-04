"""
Pydantic schemas for LLM response validation.

These schemas are used ONLY for LLM outputs with the instructor library.
Internal data structures should use normal Python types (dataclasses/typing).
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class RoutingDecision(BaseModel):
    """Schema for routing decision LLM responses."""

    decision: Literal[
        "answer_with_history", "retrieve_and_answer", "deflect", "greeting", "escalate"
    ]
    query_for_retrieval: Optional[str] = Field(
        None, description="An improved query for retrieval, if needed"
    )
    language: Literal["english", "spanish"] = Field(
        default="english", description="The language of the query"
    )


class GenerationResponse(BaseModel):
    """Schema for final generation LLM responses.
    might extend this to include citations, mode, and safety flags in the future.
    """

    text: str = Field(
        ...,
        description="The response formatted in Markdown, including lists, bolding, hyperlinks, or other elements as appropriate.",
    )
    # citations: list[str] = []
    # mode: Literal["deflection","rag","history"]
    # safety_flags: list[str] | None = None
