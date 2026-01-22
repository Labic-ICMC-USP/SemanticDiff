"""
High-level pipeline:
- highlight & collect changes
- build side-by-side PDF
- call LLM for each change
- build report PDF
- append report to comparison PDF
"""

from __future__ import annotations

import os
from typing import List

from .config import SemanticDiffConfig
from .llm import OpenRouterChangeReviewer
from .models import LLMChangeAssessment
from .pdfdiff import (
    highlight_and_collect_changes,
    merge_side_by_side_vector,
    append_pdf,
)
from .report import save_llm_report_pdf


def run_from_config(cfg: SemanticDiffConfig) -> str:
    """Run the full pipeline and return the final PDF path."""
    os.makedirs(cfg.project.output_dir, exist_ok=True)

    highlighted_base = os.path.join(cfg.project.output_dir, "highlighted_base.pdf")
    highlighted_test = os.path.join(cfg.project.output_dir, "highlighted_test.pdf")
    side_by_side_pdf = os.path.join(cfg.project.output_dir, "side_by_side_comparison.pdf")
    report_pdf = os.path.join(cfg.project.output_dir, "llm_report.pdf")
    final_pdf = os.path.join(cfg.project.output_dir, "comparison_with_llm_report.pdf")

    changes = highlight_and_collect_changes(
        cfg.project.base_pdf,
        cfg.project.test_pdf,
        highlighted_base,
        highlighted_test,
        ignore_repeated_header_footer=cfg.diff.ignore_repeated_header_footer,
        header_pct=cfg.diff.header_pct,
        footer_pct=cfg.diff.footer_pct,
        repeat_ratio=cfg.diff.repeat_ratio,
        anchor_min_len=cfg.diff.anchor_min_len,
        relocation_window=cfg.diff.relocation_window,
        replace_sim_threshold=cfg.diff.replace_sim_threshold,
        merge_rects=cfg.highlight.merge_rects,
        fill_opacity=cfg.highlight.fill_opacity,
        stroke_opacity=cfg.highlight.stroke_opacity,
        border_width=cfg.highlight.border_width,
        verbose=cfg.runtime.verbose,
    )

    merge_side_by_side_vector(highlighted_base, highlighted_test, side_by_side_pdf)

    reviewer = OpenRouterChangeReviewer(
        api_key=cfg.llm.resolved_api_key(),
        model=cfg.llm.model,
        base_url=cfg.llm.base_url,
        site_url=cfg.llm.site_url,
        site_name=cfg.llm.site_name,
        output_language=cfg.llm.output_language,
        timeout_sec=cfg.llm.timeout_sec,
        max_retries=cfg.llm.max_retries,
        verbose=cfg.runtime.verbose,
    )

    assessments: List[LLMChangeAssessment] = []
    for i, ch in enumerate(changes, start=1):
        assessments.append(reviewer.review(ch, idx=i))

    save_llm_report_pdf(
        assessments,
        report_pdf,
        title=cfg.report.title,
        truncate_chars=cfg.report.truncate_chars,
    )

    append_pdf(side_by_side_pdf, report_pdf, final_pdf)

    if cfg.runtime.verbose:
        print(f"[DONE] Final PDF: {final_pdf}")

    return final_pdf
