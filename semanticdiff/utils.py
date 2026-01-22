"""
Utilities: normalization, safe JSON extraction, and small helpers.
"""

from __future__ import annotations

import json
import re

_SOFT_HYPHEN = "\u00ad"


def norm_text(s: str) -> str:
    """Light normalization for display."""
    s = s.replace(_SOFT_HYPHEN, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_key(s: str) -> str:
    """Stronger normalization for matching (keep punctuation, remove whitespace)."""
    s = norm_text(s)
    s = re.sub(r"\s+", "", s)
    return s


def norm_key_header_footer(s: str) -> str:
    """Normalization only for header/footer detection (digits -> #)."""
    s = norm_key(s)
    s = re.sub(r"\d+", "#", s)
    return s


def simplify_for_noise_check(s: str) -> str:
    """
    Aggressively simplify to detect reflow-only changes:
    - remove whitespace
    - drop most punctuation (keep alnum/underscore)
    """
    s = s.replace(_SOFT_HYPHEN, "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w]+", "", s, flags=re.UNICODE)
    return s


def extract_first_json_object(text: str) -> str:
    """
    Extract the first JSON object from an LLM response.
    Handles markdown code fences and extra text.
    """
    t = text.strip()
    t = t.replace("```json", "```")
    t = t.replace("```", "")
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    start = t.find("{")
    if start < 0:
        raise ValueError("No JSON object found.")

    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = t[start:i + 1].strip()
                json.loads(cand)
                return cand

    raise ValueError("Unbalanced JSON braces.")
