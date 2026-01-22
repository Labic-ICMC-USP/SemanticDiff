"""
PDF diff + highlighting.

Key idea:
- Extract "wrapped line" units (reflow-resistant).
- Align globally using unique anchors + LIS.
- Detect insert/delete/replace with relocation-aware noise filtering.
- Draw visible highlights directly into page content (shape rectangles).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter
from difflib import SequenceMatcher

import fitz  # PyMuPDF

from .models import DetectedChange
from .utils import norm_text, norm_key, norm_key_header_footer, simplify_for_noise_check


@dataclass
class Unit:
    """A text unit with a bbox (wrapped line / paragraph fragment)."""
    page: int
    rect: fitz.Rect
    text: str
    key: str


def _iter_text_lines_sorted(page: fitz.Page):
    """
    Extract lines from page.get_text("dict") with a stable reading order:
    - sort blocks by (y0, x0)
    - lines by (y0, x0)
    - spans by (y0, x0)
    """
    d = page.get_text("dict")
    blocks = [b for b in d.get("blocks", []) if b.get("type", 0) == 0]

    def block_key(b):
        x0, y0, x1, y1 = b["bbox"]
        return (y0, x0)

    blocks.sort(key=block_key)

    for b in blocks:
        lines = b.get("lines", [])

        def line_key(ln):
            x0, y0, x1, y1 = ln["bbox"]
            return (y0, x0)

        lines.sort(key=line_key)

        for ln in lines:
            spans = ln.get("spans", [])

            def span_key(sp):
                x0, y0, x1, y1 = sp["bbox"]
                return (y0, x0)

            spans.sort(key=span_key)

            text = "".join(sp.get("text", "") for sp in spans)
            text = norm_text(text)
            if not text:
                continue
            yield fitz.Rect(ln["bbox"]), text


def extract_units_wrapped(doc: fitz.Document) -> List[Unit]:
    """
    Build reflow-resistant 'wrapped line' units:
    - merges consecutive lines when they look like hard wraps (not paragraph breaks)
    - bbox = union of merged lines
    """
    units: List[Unit] = []

    for pno in range(len(doc)):
        page = doc[pno]

        buf_text_parts: List[str] = []
        buf_rect: Optional[fitz.Rect] = None

        def flush():
            nonlocal buf_text_parts, buf_rect
            if not buf_text_parts or buf_rect is None:
                buf_text_parts = []
                buf_rect = None
                return
            text = norm_text(" ".join(buf_text_parts))
            if text:
                units.append(Unit(page=pno, rect=buf_rect, text=text, key=norm_key(text)))
            buf_text_parts = []
            buf_rect = None

        prev_text = None

        for rect, text in _iter_text_lines_sorted(page):
            # Bullet/numbered lines: keep separate
            is_bullet = bool(re.match(r"^(\•|\-|\–|\—|\d+\.)\s+", text))

            if prev_text is None:
                buf_text_parts = [text]
                buf_rect = rect
                prev_text = text
                continue

            # Hyphenation across lines
            if prev_text.endswith("-") and not is_bullet:
                buf_text_parts[-1] = buf_text_parts[-1][:-1]
                buf_text_parts.append(text)
                buf_rect = buf_rect | rect
                prev_text = text
                continue

            ends_sentence = bool(re.search(r"[.!?:;)\]]$", prev_text))
            starts_upper = bool(re.match(r"^[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ]", text))

            # Merge if it looks like a hard wrap
            should_merge = (not ends_sentence) and (not is_bullet) and (not starts_upper)

            # Also merge if previous ends with comma (common wrap)
            if prev_text.endswith(",") and not is_bullet:
                should_merge = True

            if should_merge:
                buf_text_parts.append(text)
                buf_rect = buf_rect | rect
            else:
                flush()
                buf_text_parts = [text]
                buf_rect = rect

            prev_text = text

        flush()

    return units


def detect_repeated_header_footer_keys(
    units: List[Unit],
    doc: fitz.Document,
    header_pct: float = 0.10,
    footer_pct: float = 0.10,
    repeat_ratio: float = 0.30,
) -> set:
    """Identify boilerplate lines that repeat across many pages in header/footer region."""
    n_pages = len(doc)
    if n_pages == 0:
        return set()

    header_counts = Counter()
    footer_counts = Counter()

    header_seen = [set() for _ in range(n_pages)]
    footer_seen = [set() for _ in range(n_pages)]

    for u in units:
        page = doc[u.page]
        ph = page.rect.height

        hk = norm_key_header_footer(u.text)

        if u.rect.y1 <= ph * header_pct:
            if hk not in header_seen[u.page]:
                header_seen[u.page].add(hk)
                header_counts[hk] += 1

        if u.rect.y0 >= ph * (1.0 - footer_pct):
            if hk not in footer_seen[u.page]:
                footer_seen[u.page].add(hk)
                footer_counts[hk] += 1

    threshold = max(2, int(n_pages * repeat_ratio))
    repeated = {k for k, c in header_counts.items() if c >= threshold}
    repeated |= {k for k, c in footer_counts.items() if c >= threshold}
    return repeated


def filter_units_ignore_keys(units: List[Unit], ignore_keys: set) -> List[Unit]:
    out = []
    for u in units:
        hk = norm_key_header_footer(u.text)
        if hk in ignore_keys:
            continue
        out.append(u)
    return out


def _build_unique_anchors(a_keys: List[str], b_keys: List[str], min_len: int = 30):
    """
    Build anchor pairs (i, j) where a_keys[i] == b_keys[j] and key appears exactly once in both.
    """
    a_pos: Dict[str, List[int]] = {}
    b_pos: Dict[str, List[int]] = {}

    for i, k in enumerate(a_keys):
        if len(k) < min_len:
            continue
        a_pos.setdefault(k, []).append(i)

    for j, k in enumerate(b_keys):
        if len(k) < min_len:
            continue
        b_pos.setdefault(k, []).append(j)

    pairs = []
    for k, ai in a_pos.items():
        bj = b_pos.get(k)
        if bj and len(ai) == 1 and len(bj) == 1:
            pairs.append((ai[0], bj[0]))

    pairs.sort(key=lambda x: x[0])
    return pairs


def _lis_by_second(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Longest Increasing Subsequence on the second coordinate."""
    if not pairs:
        return []

    import bisect
    tails: List[int] = []
    tails_idx: List[int] = []
    prev = [-1] * len(pairs)

    for idx, (_, j) in enumerate(pairs):
        pos = bisect.bisect_left(tails, j)
        if pos == len(tails):
            tails.append(j)
            tails_idx.append(idx)
        else:
            tails[pos] = j
            tails_idx[pos] = idx
        prev[idx] = tails_idx[pos - 1] if pos > 0 else -1

    k = tails_idx[-1]
    out = []
    while k != -1:
        out.append(pairs[k])
        k = prev[k]
    out.reverse()
    return out


def _join_unit_text(units: List[Unit], start: int, end: int) -> str:
    return " ".join(u.text for u in units[start:end]).strip()


def _get_context(units: List[Unit], start: int, end: int, ctx: int = 2) -> Tuple[str, str]:
    b0 = max(0, start - ctx)
    b1 = start
    a0 = end
    a1 = min(len(units), end + ctx)
    before = _join_unit_text(units, b0, b1)
    after = _join_unit_text(units, a0, a1)
    return before, after


def _contains_nearby(needle_simpl: str, units: List[Unit], center: int, window: int = 40) -> bool:
    """Detect whether a simplified needle appears near center in units list (relocation detection)."""
    if not needle_simpl:
        return False

    a0 = max(0, center - window)
    a1 = min(len(units), center + window)
    hay = simplify_for_noise_check(_join_unit_text(units, a0, a1))

    if needle_simpl in hay:
        return True

    if len(needle_simpl) < 80:
        return False

    r = SequenceMatcher(None, needle_simpl, hay, autojunk=False).ratio()
    return r >= 0.98


def _should_ignore_change_as_reflow(
    tag: str,
    base_units: List[Unit],
    test_units: List[Unit],
    A1: int, A2: int, B1: int, B2: int,
    *,
    relocation_window: int = 40,
    replace_sim_threshold: float = 0.995,
) -> bool:
    """
    Reflow/noise filter:
    - Insert/delete ignored only if the same text appears nearby in the other doc (moved).
    - Replace ignored only if almost identical after simplification (format-only).
    """
    a_txt = _join_unit_text(base_units, A1, A2)
    b_txt = _join_unit_text(test_units, B1, B2)

    sa = simplify_for_noise_check(a_txt)
    sb = simplify_for_noise_check(b_txt)

    if sa == sb and sa:
        return True

    if tag == "insert":
        return _contains_nearby(sb, base_units, center=A1, window=relocation_window)

    if tag == "delete":
        return _contains_nearby(sa, test_units, center=B1, window=relocation_window)

    if tag == "replace":
        if sa and sb:
            r = SequenceMatcher(None, sa, sb, autojunk=False).ratio()
            if r >= replace_sim_threshold:
                return True

        moved_a = _contains_nearby(sa, test_units, center=B1, window=relocation_window) if sa else False
        moved_b = _contains_nearby(sb, base_units, center=A1, window=relocation_window) if sb else False
        return moved_a and moved_b

    return False


def _merge_rects(rects: List[fitz.Rect], y_tol: float = 2.0, x_tol: float = 4.0) -> List[fitz.Rect]:
    """Merge overlapping/near-overlapping rectangles to reduce drawing ops."""
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    merged = [rects[0]]
    for r in rects[1:]:
        last = merged[-1]
        close_y = (r.y0 <= last.y1 + y_tol) and (r.y1 >= last.y0 - y_tol)
        close_x = (r.x0 <= last.x1 + x_tol) and (r.x1 >= last.x0 - x_tol)
        if close_y and close_x:
            merged[-1] = fitz.Rect(
                min(last.x0, r.x0), min(last.y0, r.y0),
                max(last.x1, r.x1), max(last.y1, r.y1),
            )
        else:
            merged.append(r)
    return merged


def draw_highlights(
    doc: fitz.Document,
    rects_by_page: Dict[int, List[fitz.Rect]],
    color: Tuple[float, float, float],
    *,
    merge_rects: bool = True,
    fill_opacity: float = 0.18,
    stroke_opacity: float = 0.70,
    border_width: float = 1.2,
) -> None:
    """Draw semi-transparent rectangles into page content using a Shape."""
    for pno, rects in rects_by_page.items():
        page = doc[pno]
        if merge_rects:
            rects = _merge_rects(rects, y_tol=2.0, x_tol=4.0)

        if not rects:
            continue

        shape = page.new_shape()
        for r in rects:
            shape.draw_rect(r)

        try:
            shape.finish(
                color=color,
                fill=color,
                width=border_width,
                stroke_opacity=stroke_opacity,
                fill_opacity=fill_opacity,
            )
        except TypeError:
            shape.finish(color=color, fill=None, width=max(1.0, border_width))

        shape.commit()


def highlight_and_collect_changes(
    base_pdf_path: str,
    test_pdf_path: str,
    highlighted_base_out: str,
    highlighted_test_out: str,
    *,
    ignore_repeated_header_footer: bool = True,
    header_pct: float = 0.10,
    footer_pct: float = 0.10,
    repeat_ratio: float = 0.30,
    anchor_min_len: int = 10,
    relocation_window: int = 40,
    replace_sim_threshold: float = 0.995,
    # highlight tuning:
    merge_rects: bool = True,
    fill_opacity: float = 0.18,
    stroke_opacity: float = 0.70,
    border_width: float = 1.2,
    # colors:
    color_insert: Tuple[float, float, float] = (0.0, 0.6, 0.0),
    color_delete: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    color_replace: Tuple[float, float, float] = (1.0, 0.55, 0.0),
    verbose: bool = True,
) -> List[DetectedChange]:
    """Highlight both PDFs and return a list of detected changes for LLM review."""
    base_doc = fitz.open(base_pdf_path)
    test_doc = fitz.open(test_pdf_path)

    if verbose:
        print("[INFO] Extracting wrapped units (base)...")
    base_units_all = extract_units_wrapped(base_doc)

    if verbose:
        print("[INFO] Extracting wrapped units (test)...")
    test_units_all = extract_units_wrapped(test_doc)

    if ignore_repeated_header_footer:
        if verbose:
            print("[INFO] Detecting repeated header/footer...")
        ignore_base = detect_repeated_header_footer_keys(
            base_units_all, base_doc, header_pct=header_pct, footer_pct=footer_pct, repeat_ratio=repeat_ratio
        )
        ignore_test = detect_repeated_header_footer_keys(
            test_units_all, test_doc, header_pct=header_pct, footer_pct=footer_pct, repeat_ratio=repeat_ratio
        )
        ignore_keys = ignore_base | ignore_test
        base_units = filter_units_ignore_keys(base_units_all, ignore_keys)
        test_units = filter_units_ignore_keys(test_units_all, ignore_keys)
        if verbose:
            print(f"[INFO] Ignored {len(ignore_keys)} header/footer keys")
    else:
        base_units = base_units_all
        test_units = test_units_all

    base_keys = [u.key for u in base_units]
    test_keys = [u.key for u in test_units]

    if verbose:
        print(f"[INFO] Units: base={len(base_keys)} test={len(test_keys)}")

    raw_pairs = _build_unique_anchors(base_keys, test_keys, min_len=anchor_min_len)
    anchors = _lis_by_second(raw_pairs)

    if verbose:
        print(f"[INFO] Anchors: raw={len(raw_pairs)} kept(LIS)={len(anchors)}")

    anchors = [(-1, -1)] + anchors + [(len(base_keys), len(test_keys))]

    rects_base_del: Dict[int, List[fitz.Rect]] = {}
    rects_test_ins: Dict[int, List[fitz.Rect]] = {}
    rects_base_rep: Dict[int, List[fitz.Rect]] = {}
    rects_test_rep: Dict[int, List[fitz.Rect]] = {}

    changes: List[DetectedChange] = []
    total_ops = 0
    kept_ops = 0

    for (ai0, bj0), (ai1, bj1) in zip(anchors[:-1], anchors[1:]):
        a_start, a_end = ai0 + 1, ai1
        b_start, b_end = bj0 + 1, bj1

        a_seg = base_keys[a_start:a_end]
        b_seg = test_keys[b_start:b_end]
        if not a_seg and not b_seg:
            continue

        sm = SequenceMatcher(None, a_seg, b_seg, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue

            total_ops += 1

            A1, A2 = a_start + i1, a_start + i2
            B1, B2 = b_start + j1, b_start + j2

            if _should_ignore_change_as_reflow(
                tag,
                base_units, test_units,
                A1, A2, B1, B2,
                relocation_window=relocation_window,
                replace_sim_threshold=replace_sim_threshold,
            ):
                continue

            kept_ops += 1

            base_text = _join_unit_text(base_units, A1, A2) if tag in ("delete", "replace") else ""
            test_text = _join_unit_text(test_units, B1, B2) if tag in ("insert", "replace") else ""

            base_before, base_after = _get_context(base_units, A1, A2, ctx=2)
            test_before, test_after = _get_context(test_units, B1, B2, ctx=2)

            base_pages = sorted({u.page for u in base_units[A1:A2]}) if tag in ("delete", "replace") else []
            test_pages = sorted({u.page for u in test_units[B1:B2]}) if tag in ("insert", "replace") else []

            changes.append(
                DetectedChange(
                    change_type=tag,
                    base_pages=base_pages,
                    test_pages=test_pages,
                    base_text=base_text,
                    test_text=test_text,
                    base_context_before=base_before,
                    base_context_after=base_after,
                    test_context_before=test_before,
                    test_context_after=test_after,
                )
            )

            if tag == "insert":
                for u in test_units[B1:B2]:
                    rects_test_ins.setdefault(u.page, []).append(u.rect)
            elif tag == "delete":
                for u in base_units[A1:A2]:
                    rects_base_del.setdefault(u.page, []).append(u.rect)
            elif tag == "replace":
                for u in base_units[A1:A2]:
                    rects_base_rep.setdefault(u.page, []).append(u.rect)
                for u in test_units[B1:B2]:
                    rects_test_rep.setdefault(u.page, []).append(u.rect)

    if verbose:
        print(f"[INFO] Change ops: detected={total_ops} kept_after_validation={kept_ops}")
        print(f"[INFO] Changes collected for LLM: {len(changes)}")

    # Draw highlights
    draw_highlights(base_doc, rects_base_del, color_delete, merge_rects=merge_rects,
                    fill_opacity=fill_opacity, stroke_opacity=stroke_opacity, border_width=border_width)
    draw_highlights(base_doc, rects_base_rep, color_replace, merge_rects=merge_rects,
                    fill_opacity=fill_opacity, stroke_opacity=stroke_opacity, border_width=border_width)

    draw_highlights(test_doc, rects_test_ins, color_insert, merge_rects=merge_rects,
                    fill_opacity=fill_opacity, stroke_opacity=stroke_opacity, border_width=border_width)
    draw_highlights(test_doc, rects_test_rep, color_replace, merge_rects=merge_rects,
                    fill_opacity=fill_opacity, stroke_opacity=stroke_opacity, border_width=border_width)

    # Save
    base_doc.save(highlighted_base_out, garbage=4, deflate=True, incremental=False)
    test_doc.save(highlighted_test_out, garbage=4, deflate=True, incremental=False)

    base_doc.close()
    test_doc.close()

    return changes


def merge_side_by_side_vector(left_pdf_path: str, right_pdf_path: str, output_path: str) -> None:
    """Vector side-by-side merge using show_pdf_page (no rasterization)."""
    left = fitz.open(left_pdf_path)
    right = fitz.open(right_pdf_path)
    out = fitz.open()

    n = min(len(left), len(right))
    for pno in range(n):
        lp = left[pno]
        rp = right[pno]
        width = lp.rect.width + rp.rect.width
        height = max(lp.rect.height, rp.rect.height)
        page = out.new_page(width=width, height=height)
        page.show_pdf_page(fitz.Rect(0, 0, lp.rect.width, lp.rect.height), left, pno)
        page.show_pdf_page(
            fitz.Rect(lp.rect.width, 0, lp.rect.width + rp.rect.width, rp.rect.height),
            right,
            pno,
        )

    out.save(output_path, garbage=4, deflate=True, incremental=False)
    out.close()
    left.close()
    right.close()


def append_pdf(main_pdf: str, appendix_pdf: str, out_pdf: str) -> None:
    """Append appendix_pdf to main_pdf and save out_pdf."""
    main = fitz.open(main_pdf)
    app = fitz.open(appendix_pdf)
    out = fitz.open()
    out.insert_pdf(main)
    out.insert_pdf(app)
    out.save(out_pdf, garbage=4, deflate=True, incremental=False)
    out.close()
    main.close()
    app.close()
