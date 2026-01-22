"""
YAML-driven configuration for SemanticDiff.

Design choice:
- Put all parameters in YAML, except secrets (API key), which should come from an env var.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import os
import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    base_pdf: str
    test_pdf: str
    output_dir: str = "comparison_output"


class LLMConfig(BaseModel):
    provider: Literal["openrouter"] = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    api_key: Optional[str] = None  # discouraged; prefer env
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash-lite"
    site_url: str = ""
    site_name: str = ""
    output_language: str = "Portuguese"
    timeout_sec: float = 90.0
    max_retries: int = 2

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise ValueError(
                f"Missing API key. Set env var '{self.api_key_env}' or provide llm.api_key in YAML."
            )
        return key


class DiffConfig(BaseModel):
    ignore_repeated_header_footer: bool = True
    header_pct: float = 0.10
    footer_pct: float = 0.10
    repeat_ratio: float = 0.30

    anchor_min_len: int = 10
    relocation_window: int = 40
    replace_sim_threshold: float = 0.995


class HighlightConfig(BaseModel):
    merge_rects: bool = True
    fill_opacity: float = 0.18
    stroke_opacity: float = 0.70
    border_width: float = 1.2


class ReportConfig(BaseModel):
    title: str = "LLM Diff Report"
    truncate_chars: int = 4000


class RuntimeConfig(BaseModel):
    verbose: bool = True


class SemanticDiffConfig(BaseModel):
    project: ProjectConfig
    llm: LLMConfig
    diff: DiffConfig = Field(default_factory=DiffConfig)
    highlight: HighlightConfig = Field(default_factory=HighlightConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SemanticDiffConfig":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
