"""
Data models: detected changes and LLM assessment schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field


@dataclass
class DetectedChange:
    """
    One detected change block suitable for LLM review.
    Stores both texts and a bit of surrounding context.
    """
    change_type: str  # insert|delete|replace
    base_pages: List[int]
    test_pages: List[int]
    base_text: str
    test_text: str
    base_context_before: str
    base_context_after: str
    test_context_before: str
    test_context_after: str


class LLMChangeAssessment(BaseModel):
    """LLM judgement about a change."""
    meaningful: bool = Field(..., description="True if the change is semantically meaningful.")
    change_type: str = Field(..., description="insert|delete|replace")
    summary: str = Field(..., description="Short summary of the change.")
    explanation: str = Field(..., description="Detailed explanation of what changed.")
    original_text: str = Field(..., description="Original text (from base). Empty if insert.")
    new_text: str = Field(..., description="New text (from test). Empty if delete.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in [0,1].")
