"""
Pydantic schemas for LLM response validation.

These schemas are used ONLY for LLM outputs with the instructor library.
Internal data structures should use normal Python types (dataclasses/typing).
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class RelevanceDecision(BaseModel):
    """Schema for relevance check LLM responses."""
    decision: Literal["RELEVANT", "IRRELEVANT"]


class RoutingDecision(BaseModel):
    """Schema for routing decision LLM responses."""
    decision: Literal["RETRIEVE", "SUFFICIENT"]
    query_for_retrieval: Optional[str] = Field(None, description="Refined query")


class DeflectionCategoryDecision(BaseModel):
    """Schema for deflection category classification LLM responses."""
    decision: Literal["OFFICIAL", "JEST", "HACK"]


class GenerationResponse(BaseModel):
    """Schema for final generation LLM responses.
    might extend this to include citations, mode, and safety flags in the future.
    """
    text: str
    # citations: list[str] = []
    # mode: Literal["deflection","rag","history"]
    # safety_flags: list[str] | None = None
