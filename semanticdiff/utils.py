"""
Utilities: normalization, safe JSON extraction, and small helpers.
"""

from __future__ import annotations

import json
import re

_SOFT_HYPHEN = "\u00ad"


def _build_fragment_pattern(fragment: str) -> str:
    """
    Build a permissive regex for watermark fragments.

    """
    fragment = norm_text(fragment)
    if not fragment:
        return ""

    if " " not in fragment and len(fragment) <= 40:
        return r"\s*".join(re.escape(ch) for ch in fragment)

    parts = [re.escape(p) for p in fragment.split() if p]
    return r"\s+".join(parts)


def norm_text(s: str) -> str:
    """Light normalization for display."""
    s = s.replace(_SOFT_HYPHEN, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_configured_fragments(
    s: str,
    fragments: list[str] | None,
    *,
    case_sensitive: bool = False,
) -> str:
    """
    Remove configured watermark fragments before diff/alignment.

    This is intentionally permissive with whitespace so that diagonal or
    letter-spaced watermark text is easier to suppress.
    """
    s = norm_text(s)
    if not s or not fragments:
        return s

    flags = 0 if case_sensitive else re.IGNORECASE
    out = s
    for fragment in fragments:
        pattern = _build_fragment_pattern(fragment)
        if not pattern:
            continue
        out = re.sub(pattern, " ", out, flags=flags)

    return norm_text(out)


def contains_configured_fragment(
    s: str,
    fragments: list[str] | None,
    *,
    case_sensitive: bool = False,
) -> bool:
    """Check whether text still contains any configured watermark fragment."""
    s = norm_text(s)
    if not s or not fragments:
        return False

    flags = 0 if case_sensitive else re.IGNORECASE
    for fragment in fragments:
        pattern = _build_fragment_pattern(fragment)
        if pattern and re.search(pattern, s, flags=flags):
            return True
    return False

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


def matches_any_regex(
    s: str,
    patterns: list[str] | None,
    *,
    case_sensitive: bool = False,
) -> bool:
    s = norm_text(s)
    if not s or not patterns:
        return False

    flags = 0 if case_sensitive else re.IGNORECASE
    for pat in patterns:
        if re.fullmatch(pat, s, flags=flags):
            return True
    return False