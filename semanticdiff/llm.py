"""
LLM review using OpenRouter (OpenAI SDK compatible) + Pydantic parsing.
"""

from __future__ import annotations

from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

from .models import DetectedChange, LLMChangeAssessment
from .utils import extract_first_json_object


class OpenRouterChangeReviewer:
    """
    Reviews changes using OpenRouter via OpenAI SDK.
    Output language is enforced via system prompt.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "",
        site_name: str = "",
        output_language: str = "Portuguese",
        timeout_sec: float = 90.0,
        max_retries: int = 2,
        verbose: bool = True,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec, max_retries=max_retries)
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.output_language = output_language
        self.verbose = verbose

    def review(self, change: DetectedChange, idx: int) -> LLMChangeAssessment:
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        system = (
            "You are a careful document-diff reviewer.\n"
            "You will receive a detected change between two versions of a PDF.\n"
            "Decide if it is semantically meaningful (not just reflow/formatting/whitespace).\n"
            f"Write summary and explanation in {self.output_language}.\n"
            "Return ONLY valid JSON matching the schema.\n"
        )

        user = (
            f"CHANGE #{idx}\n"
            f"TYPE: {change.change_type}\n\n"
            f"BASE pages: {change.base_pages}\n"
            f"TEST pages: {change.test_pages}\n\n"
            "BASE context BEFORE:\n"
            f"{change.base_context_before}\n\n"
            "BASE text:\n"
            f"{change.base_text}\n\n"
            "BASE context AFTER:\n"
            f"{change.base_context_after}\n\n"
            "TEST context BEFORE:\n"
            f"{change.test_context_before}\n\n"
            "TEST text:\n"
            f"{change.test_text}\n\n"
            "TEST context AFTER:\n"
            f"{change.test_context_after}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "meaningful": boolean,\n'
            '  "change_type": "insert"|"delete"|"replace",\n'
            '  "summary": string,\n'
            '  "explanation": string,\n'
            '  "original_text": string,\n'
            '  "new_text": string,\n'
            '  "confidence": number\n'
            "}\n"
        )

        if self.verbose:
            print(f"[LLM] Reviewing change {idx} ({change.change_type})...")

        completion = self.client.chat.completions.create(
            extra_headers=headers or None,
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        content = completion.choices[0].message.content or ""

        try:
            j = extract_first_json_object(content)
            return LLMChangeAssessment.model_validate_json(j)
        except (ValueError, ValidationError) as e:
            # Fallback: still return a valid object
            return LLMChangeAssessment(
                meaningful=True,
                change_type=change.change_type,
                summary=f"[Parse error] {type(e).__name__}",
                explanation=f"Failed to parse LLM JSON output. Raw preview:\n{content[:800]}",
                original_text=change.base_text or "",
                new_text=change.test_text or "",
                confidence=0.0,
            )
