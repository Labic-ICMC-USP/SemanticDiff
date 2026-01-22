# SemanticDiff

SemanticDiff is a reflow-resistant PDF diff tool built for real-world documents where layout changes are common and naïve page-by-page comparison breaks down. It detects **insertions, deletions, and replacements** even when paragraphs shift across lines or pages, then writes **high-visibility highlights directly into the PDF content** so the marks are stable, fast, and suitable for downstream merging.

Beyond visual diffing, SemanticDiff produces a **vector, side-by-side comparison PDF** (no rasterization) and automatically packages each detected change—along with surrounding context—for **LLM-based semantic review**. The LLM confirms whether a change is meaningful, summarizes what changed, and explains the modification in a configurable language. The resulting report is rendered as a PDF and appended to the final artifact, yielding a single deliverable that combines evidence (highlights) with interpretation (LLM report).


## Use case

SemanticDiff was designed for comparing large academic and technical documents where “simple PDF diff” tools often fail due to layout reflow. Typical scenarios include comparing an **original** and a **revised** version of:

* journal and conference **papers** (before/after peer review)
* technical and administrative **reports**
* **theses and dissertations** (advisor revisions, committee corrections, final submission)

At the **Graduate Program in Computer Science and Computational Mathematics (PPG-CCMC), ICMC-USP  (PPG-CCMC, USP, Brazil)**, this is particularly useful when coordinating revisions across multiple iterations, ensuring that the final version reflects the requested changes, and producing a single artifact that is easy to audit and share with supervisors, committees, and administrative staff.

## Features

* **Reflow-resistant change detection**
  Detects insertions, deletions, and replacements even when paragraphs shift across lines or pages, reducing false positives caused by pagination, line wrapping, or minor layout adjustments.

* **Stable highlights embedded in the PDF content**
  Highlights are written into the PDF page content (not fragile annotations), producing marks that remain visible across viewers and survive downstream PDF merging.

* **Vector side-by-side comparison PDF**
  Generates a vector (non-rasterized) side-by-side PDF for fast navigation and visual inspection without losing text quality.

* **Change blocks with surrounding context**
  Each detected change is packaged with “before/after” context to improve interpretability and support reliable review.

* **LLM-based semantic validation and explanation**
  Each change can be sent to an LLM to verify whether it is **semantically meaningful** (not just formatting/reflow), and to generate a structured report with the original text, revised text, and an explanation in a configurable language.

* **Single deliverable for auditing**
  Produces a final PDF that combines the side-by-side comparison with an appended LLM report, enabling traceable review in one file.

## Why SemanticDiff

We developed SemanticDiff because we could not find an open-source solution that was both **robust for long PDFs** and **reflow-aware** in a way that consistently minimizes false differences. In real academic documents, small layout shifts can propagate across pages and make traditional diff outputs noisy and unreliable.

SemanticDiff addresses this by combining:

1. a reflow-resistant extraction and alignment strategy for PDF text, and
2. an optional LLM-based semantic layer that filters residual noise and produces clear explanations of what actually changed.

This approach makes the output substantially more useful for academic workflows, where the goal is not only to detect differences, but to **understand and validate them** at scale.


## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Configure

Create a YAML config (example: `semanticdiff.yaml`).

## Run

```bash
semanticdiff run --config semanticdiff.yaml
```

Outputs are written to `output_dir` from the YAML.

## Notes

- Works best on text-based PDFs (not scanned images).
- If your PDF has heavy headers/footers, keep `ignore_repeated_header_footer: true`.
