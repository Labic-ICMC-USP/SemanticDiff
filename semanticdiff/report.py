"""
Generate an LLM report PDF and append it to the comparison PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

from .models import LLMChangeAssessment


def _escape_for_rl(s: str, truncate_chars: int = 4000) -> str:
    """Basic escaping for ReportLab Paragraph markup."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if truncate_chars and len(s) > truncate_chars:
        s = s[:truncate_chars] + "\n...[truncated]..."
    return s.replace("\n", "<br/>")


def save_llm_report_pdf(
    assessments: List[LLMChangeAssessment],
    out_pdf_path: str,
    *,
    title: str = "LLM Diff Report",
    truncate_chars: int = 4000,
) -> None:
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 0.5 * cm))

    meaningful = [a for a in assessments if a.meaningful]
    not_meaningful = [a for a in assessments if not a.meaningful]

    story.append(Paragraph(f"<b>Total changes reviewed:</b> {len(assessments)}", styles["Normal"]))
    story.append(Paragraph(f"<b>Meaningful:</b> {len(meaningful)}", styles["Normal"]))
    story.append(Paragraph(f"<b>Ignored:</b> {len(not_meaningful)}", styles["Normal"]))
    story.append(PageBreak())

    for i, a in enumerate(assessments, start=1):
        story.append(
            Paragraph(
                f"<b>Change #{i}</b> — <b>Type:</b> {a.change_type} — "
                f"<b>Meaningful:</b> {a.meaningful} — <b>Confidence:</b> {a.confidence:.2f}",
                styles["Heading3"],
            )
        )
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(f"<b>Summary:</b> {a.summary}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(f"<b>Explanation:</b> {a.explanation}", styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

        if a.original_text:
            story.append(Paragraph("<b>Original text:</b>", styles["Normal"]))
            story.append(
                Paragraph(f"<font name='Courier'>{_escape_for_rl(a.original_text, truncate_chars)}</font>", styles["BodyText"])
            )
            story.append(Spacer(1, 0.2 * cm))

        if a.new_text:
            story.append(Paragraph("<b>New text:</b>", styles["Normal"]))
            story.append(
                Paragraph(f"<font name='Courier'>{_escape_for_rl(a.new_text, truncate_chars)}</font>", styles["BodyText"])
            )
            story.append(Spacer(1, 0.2 * cm))

        story.append(PageBreak())

    doc = SimpleDocTemplate(
        out_pdf_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    doc.build(story)
