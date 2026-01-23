"""
Chapter Headings Extraction Tool

Extract chapter subheadings from a PDF using:
1) PDF Bookmarks/TOC (if present)
2) Layout analysis fallback (font-size + bold + bbox heuristics)

Saves results into OUTPUT_DIR/<pdf_stem>/chapter_<n>_headings.json
and returns a small preview.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF

from ..base import MCPTool, ToolParameter, ExecutionError


# Configuration from environment (aligned with existing pdf tools)
PDF_DIR = Path(os.getenv("PDF_DIR", "/data/pdf"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _extract_from_bookmarks(doc: fitz.Document, chapter_number: int) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Try to extract headings from PDF TOC/Bookmarks.
    Returns (entries, has_toc).
    """
    toc = doc.get_toc(simple=False)
    if not toc:
        return [], False

    chap_pattern = re.compile(rf"Chapter\s*{chapter_number}\b", re.IGNORECASE)
    next_chap_pattern = re.compile(r"Chapter\s*\d+", re.IGNORECASE)

    chap_idx: Optional[int] = None
    chap_level: Optional[int] = None

    for i, row in enumerate(toc):
        level, title, _page = row[0], row[1], row[2]
        if chap_pattern.search(title or ""):
            chap_idx = i
            chap_level = level
            break

    if chap_idx is None or chap_level is None:
        return [], False

    entries: List[Dict[str, Any]] = []
    for row in toc[chap_idx + 1 :]:
        level, title, page = row[0], row[1], row[2]

        # Stop when next chapter begins at same or higher TOC level
        if level <= chap_level and next_chap_pattern.search(title or ""):
            break

        entries.append(
            {
                "level": max(1, level - chap_level),
                "title": (title or "").strip(),
                "page": int(page) if page is not None else None,  # PyMuPDF TOC page is typically 1-based
            }
        )

    entries = [e for e in entries if e["title"]]
    return entries, True


def _find_chapter_page_range_robust(doc: fitz.Document, chapter_number: int) -> Tuple[int, int]:
    """
    Robustly find chapter start/end pages by checking big-font spans containing "Chapter N".
    Returns (start_page_0_based, end_page_0_based_exclusive).
    """
    chap_pat = re.compile(rf"\bChapter\s*{chapter_number}\b", re.IGNORECASE)
    next_pat = re.compile(rf"\bChapter\s*{chapter_number + 1}\b", re.IGNORECASE)

    start_page: Optional[int] = None
    end_page: Optional[int] = None

    for i in range(len(doc)):
        try:
            page = doc.load_page(i)
            blocks = page.get_text("dict").get("blocks", [])

            found_current = False
            found_next = False
            max_size_on_page = 0.0

            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = (span.get("text") or "").strip()
                        size = float(span.get("size") or 0.0)
                        max_size_on_page = max(max_size_on_page, size)

                        if chap_pat.search(text) and size > 12.0:
                            found_current = True
                        if next_pat.search(text) and size > 12.0:
                            found_next = True

            if start_page is None and found_current and max_size_on_page > 10.0:
                start_page = i
                continue

            if start_page is not None and found_next:
                end_page = i
                break

        except Exception:
            continue

    if start_page is None:
        raise RuntimeError(f"Could not find start of Chapter {chapter_number}")

    if end_page is None:
        end_page = len(doc)

    return start_page, end_page


def _iter_spans(doc: fitz.Document, start_p: int, end_p: int) -> Iterable[Dict[str, Any]]:
    """Iterate all spans with bbox + font info within [start_p, end_p)."""
    for pno in range(start_p, end_p):
        page = doc.load_page(pno)
        page_height = float(page.rect.height)

        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text = (span.get("text") or "").strip()
                    if not text:
                        continue
                    yield {
                        "page": pno,
                        "text": text,
                        "size": round(float(span.get("size") or 0.0), 2),
                        "font": span.get("font") or "",
                        "flags": int(span.get("flags") or 0),
                        "bbox": tuple(span.get("bbox") or (0, 0, 0, 0)),
                        "page_height": page_height,
                    }


def _is_heading_candidate(span: Dict[str, Any], body_size: float) -> bool:
    """
    Filter out header/footer/captions/noise and keep heading-like spans.
    """
    text = span["text"]
    size = float(span["size"])
    x0, y0, x1, y1 = span["bbox"]
    h = float(span["page_height"])

    # Header/footer 영역 제거 (상하단 5%)
    if y0 < h * 0.05 or y1 > h * 0.95:
        return False

    # Figure/Table 캡션 제거
    if re.match(r"^(Fig|Figure|Table|Tab)\.?\s*\d+", text, re.IGNORECASE):
        return False

    # 숫자만 (페이지번호 등)
    if re.fullmatch(r"\d+", text):
        return False

    # 너무 긴 문장
    if len(text) > 100:
        return False

    # Bold/size heuristic
    font_lower = (span.get("font") or "").lower()
    flags = int(span.get("flags") or 0)
    is_bold = ("bold" in font_lower) or (flags & 2) or (flags & 16)
    is_larger = size >= (float(body_size) + 1.0)

    looks_like_sentence = text.strip().endswith(".")
    if (is_larger or is_bold) and not looks_like_sentence:
        return True

    return False


def _merge_heading_spans(
    candidates: List[Dict[str, Any]], y_tol: float = 3.0, x_gap_tol: float = 15.0
) -> List[Dict[str, Any]]:
    """Merge spans on the same line into a single heading."""
    sorted_spans = sorted(candidates, key=lambda s: (s["page"], s["bbox"][1], s["bbox"][0]))
    merged: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for span in sorted_spans:
        if current is None:
            current = dict(span)
            continue

        if span["page"] != current["page"]:
            merged.append(current)
            current = dict(span)
            continue

        y_diff = abs(span["bbox"][1] - current["bbox"][1])
        x_dist = span["bbox"][0] - current["bbox"][2]

        if y_diff <= y_tol and x_dist <= x_gap_tol:
            current["text"] = f"{current['text']} {span['text']}"
            current["bbox"] = (current["bbox"][0], current["bbox"][1], span["bbox"][2], span["bbox"][3])
        else:
            merged.append(current)
            current = dict(span)

    if current is not None:
        merged.append(current)
    return merged


def _cluster_sizes_to_levels(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cluster font sizes to assign hierarchy levels (1..n)."""
    if not headings:
        return headings

    sizes = sorted({round(float(h["size"]), 1) for h in headings}, reverse=True)
    tiers: List[List[float]] = []

    for s in sizes:
        if not tiers:
            tiers.append([s])
        else:
            if abs(tiers[-1][-1] - s) <= 0.3:
                tiers[-1].append(s)
            else:
                tiers.append([s])

    tier_map = {s: i + 1 for i, group in enumerate(tiers) for s in group}

    for h in headings:
        s_rounded = round(float(h["size"]), 1)
        h["level"] = int(tier_map.get(s_rounded, len(tiers)))

    return headings


class ExtractChapterHeadingsTool(MCPTool):
    """
    MCP Tool: Extract headings/subheadings of a given chapter from a PDF.
    """

    @property
    def name(self) -> str:
        return "extract_chapter_headings"

    @property
    def description(self) -> str:
        return (
            "Extract chapter headings/subheadings from a PDF using TOC/bookmarks if available, "
            "otherwise fallback to layout analysis (font size/bold/bbox). Saves JSON to output directory."
        )

    @property
    def category(self) -> str:
        return "pdf"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="filename",
                type="string",
                description="Name of the PDF file in PDF_DIR (e.g., 'Diagnostic Ultrasound - Physics and Equipment.pdf')",
                required=True,
            ),
            ToolParameter(
                name="chapter_number",
                type="integer",
                description="Target chapter number to extract (e.g., 5)",
                required=True,
            ),
            ToolParameter(
                name="prefer_toc",
                type="boolean",
                description="If true, use PDF bookmarks/TOC first when available (recommended).",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="max_preview",
                type="integer",
                description="Max number of heading items to include in response preview.",
                required=False,
                default=20,
            ),
        ]

    async def execute(
        self,
        filename: str,
        chapter_number: int,
        prefer_toc: bool = True,
        max_preview: int = 20,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            raise ExecutionError(f"File not found: {filename}", tool_name=self.name)

        pdf_name = pdf_path.stem
        out_dir = OUTPUT_DIR / pdf_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"chapter_{int(chapter_number)}_headings.json"

        try:
            doc = fitz.open(pdf_path)

            final_results: List[Dict[str, Any]] = []
            used_method = "layout"

            # 1) TOC/bookmarks
            if prefer_toc:
                bookmarks, has_toc = _extract_from_bookmarks(doc, int(chapter_number))
                if has_toc and bookmarks:
                    final_results = bookmarks
                    used_method = "toc"

            # 2) Layout fallback
            if not final_results:
                start_p, end_p = _find_chapter_page_range_robust(doc, int(chapter_number))
                raw_spans = list(_iter_spans(doc, start_p, end_p))
                if not raw_spans:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump([], f, ensure_ascii=False, indent=2)
                    return {
                        "status": "success",
                        "method": "layout",
                        "filename": filename,
                        "chapter_number": int(chapter_number),
                        "saved_to": str(out_path),
                        "total_headings": 0,
                        "preview": [],
                        "note": "No text spans found in detected chapter page range.",
                    }

                size_counts = Counter(round(float(s["size"]), 1) for s in raw_spans)
                body_size = float(size_counts.most_common(1)[0][0])

                candidates = [s for s in raw_spans if _is_heading_candidate(s, body_size)]
                merged = _merge_heading_spans(candidates)

                cleaned: List[Dict[str, Any]] = []
                seen = set()
                for m in merged:
                    t = re.sub(r"\s+", " ", m["text"]).strip()
                    if re.match(rf"^Chapter\s*{int(chapter_number)}\b", t, re.IGNORECASE):
                        continue
                    if len(t) < 3:
                        continue
                    key = (t, m["page"])
                    if key in seen:
                        continue
                    seen.add(key)
                    m["text"] = t
                    cleaned.append(m)

                structured = _cluster_sizes_to_levels(cleaned)
                final_results = [
                    {"level": int(it["level"]), "title": it["text"], "page": int(it["page"]) + 1}
                    for it in structured
                ]
                used_method = "layout"

            doc.close()

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)

            preview_items = final_results[: max(0, int(max_preview))]
            return {
                "status": "success",
                "method": used_method,
                "filename": filename,
                "chapter_number": int(chapter_number),
                "saved_to": str(out_path),
                "total_headings": len(final_results),
                "preview": preview_items,
            }

        except ExecutionError:
            raise
        except Exception as e:
            raise ExecutionError(f"Failed to extract chapter headings: {str(e)}", tool_name=self.name)
