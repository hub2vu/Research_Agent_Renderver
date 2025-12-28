import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..base import MCPTool, ToolParameter, ExecutionError


OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ExtractReferenceTitlesTool(MCPTool):
    @property
    def name(self) -> str:
        return "extract_reference_titles"

    @property
    def description(self) -> str:
        return (
            "Extract numbered references + paper titles from ./output/[paper]/extracted_text.txt "
            "with validation report"
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
                description="Name of the PDF file (with or without .pdf extension)",
                required=True,
            ),
            ToolParameter(
                name="save_files",
                type="boolean",
                description="Save outputs to reference_titles.json/txt and diagnostics files",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="include_raw_entries",
                type="boolean",
                description="Include raw reference entry text in reference_items.json (can be large)",
                required=False,
                default=False,
            ),
        ]

    async def execute(
        self,
        filename: str,
        save_files: bool = True,
        include_raw_entries: bool = False,
    ) -> Dict[str, Any]:
        paper = filename.replace(".pdf", "").strip()
        text_file = OUTPUT_DIR / paper / "extracted_text.txt"

        if not text_file.exists():
            raise ExecutionError(
                f"No extracted text found for '{paper}'. Expected: {text_file}",
                tool_name=self.name,
            )

        raw = text_file.read_text(encoding="utf-8", errors="ignore")

        refs_text = self._slice_references_section(raw)

        # ✅ 핵심: split 전에 레이아웃 노이즈 제거
        refs_text = self._strip_layout_noise(refs_text)

        ref_items = self._parse_numbered_references(refs_text)

        items_out: List[Dict[str, Any]] = []
        titles_with_no: List[Tuple[int, str]] = []

        for no, entry in ref_items:
            title = self._extract_title_from_entry(entry)
            ok = bool(title)

            if ok:
                titles_with_no.append((no, title))

            item: Dict[str, Any] = {"ref_no": no, "title": title, "title_extracted": ok}
            if include_raw_entries:
                item["raw_entry"] = entry
            items_out.append(item)

        diagnostics = self._build_diagnostics(ref_items, titles_with_no)
        titles_sorted = [t for _, t in sorted(titles_with_no, key=lambda x: x[0])]

        result: Dict[str, Any] = {
            "filename": paper,
            "text_file": str(text_file),
            "references_detected": len(ref_items),
            "titles_extracted": len(titles_sorted),
            "titles": titles_sorted,
            "diagnostics": diagnostics,
        }

        if save_files:
            self._save_outputs(paper, result, items_out)

        return result

    # -------------------------
    # Save
    # -------------------------
    def _save_outputs(self, paper: str, result: Dict[str, Any], items_out: List[Dict[str, Any]]) -> None:
        out_dir = OUTPUT_DIR / paper
        out_dir.mkdir(parents=True, exist_ok=True)

        titles_txt = out_dir / "reference_titles.txt"
        titles_json = out_dir / "reference_titles.json"
        items_json = out_dir / "reference_items.json"
        diag_json = out_dir / "reference_diagnostics.json"

        titles_txt.write_text("\n".join(result.get("titles", [])), encoding="utf-8")
        titles_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        items_json.write_text(json.dumps(items_out, ensure_ascii=False, indent=2), encoding="utf-8")
        diag_json.write_text(json.dumps(result.get("diagnostics", {}), ensure_ascii=False, indent=2), encoding="utf-8")

        result["output_files"] = {
            "titles_txt": str(titles_txt),
            "titles_json": str(titles_json),
            "items_json": str(items_json),
            "diagnostics_json": str(diag_json),
        }

    # -------------------------
    # References slicing
    # -------------------------
    def _slice_references_section(self, text: str) -> str:
        header_patterns = [
            r"\nReferences\s*\n",
            r"\nBibliography\s*\n",
            r"\nReference\s*\n",
        ]

        start_idx = None
        for p in header_patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                start_idx = m.end()
                break

        refs = text if start_idx is None else text[start_idx:]

        refs = re.split(
            r"\n(Appendix|Appendices|Acknowledg(e)?ments|Supplementary|Supplemental|Author Contributions|Funding)\b",
            refs,
            flags=re.IGNORECASE,
        )[0]

        return refs.strip()

    # ✅ NEW: remove page numbers + repeated running headers/footers
    def _strip_layout_noise(self, refs_text: str) -> str:
        lines = refs_text.splitlines()

        cleaned: List[str] = []
        for ln in lines:
            # remove "=== Page n ==="
            if re.match(r"^\s*===\s*Page\s*\d+\s*===\s*$", ln):
                continue
            # remove pure page-number lines like "20"
            if re.match(r"^\s*\d{1,4}\s*$", ln):
                continue
            cleaned.append(ln.rstrip())

        # remove repeated running headers/footers (same line repeated many times)
        cnt = Counter([ln.strip() for ln in cleaned if ln.strip()])
        # threshold: appears 3+ times and not a reference marker
        repeated = {
            s for s, c in cnt.items()
            if c >= 3 and not re.match(r"^\[\d+\]\s*$", s) and not re.match(r"^\d+\.\s*$", s)
        }

        cleaned2 = []
        for ln in cleaned:
            s = ln.strip()
            if s in repeated:
                continue
            cleaned2.append(ln)

        return "\n".join(cleaned2).strip()

    # -------------------------
    # Parse numbered references
    # -------------------------
    def _parse_numbered_references(self, refs_text: str) -> List[Tuple[int, str]]:
        refs_text = refs_text.strip()
        if not refs_text:
            return []

        pat_bracket = re.compile(r"(?m)^\s*\[(\d+)\]\s+(?!:)")
        ms = list(pat_bracket.finditer(refs_text))

        if not ms:
            pat_dot = re.compile(r"(?m)^\s*(\d+)\.\s+")
            ms = list(pat_dot.finditer(refs_text))

        if not ms:
            return []

        items: List[Tuple[int, str]] = []
        for i, m in enumerate(ms):
            no = int(m.group(1))
            start = m.end()
            end = ms[i + 1].start() if i + 1 < len(ms) else len(refs_text)
            entry = refs_text[start:end].strip()

            # normalize whitespace AFTER line noise removed
            entry = re.sub(r"\s+", " ", entry).strip()

            if len(entry) >= 10:
                items.append((no, entry))

        items.sort(key=lambda x: x[0])
        return items

    # -------------------------
    # Diagnostics
    # -------------------------
    def _build_diagnostics(
        self,
        ref_items: List[Tuple[int, str]],
        titles_with_no: List[Tuple[int, str]],
    ) -> Dict[str, Any]:
        nums = [n for n, _ in ref_items]
        titles_map = {n: t for n, t in titles_with_no}

        if not nums:
            return {}

        min_no = min(nums)
        max_no = max(nums)

        counts = Counter(nums)
        dup = sorted([n for n, c in counts.items() if c > 1])

        expected = list(range(min_no, max_no + 1))
        missing = [n for n in expected if n not in counts]
        missing_title_nos = [n for n in sorted(counts.keys()) if n not in titles_map]

        return {
            "detected_ref_nos": {"count": len(nums), "min": min_no, "max": max_no},
            "numbering_integrity": {
                "duplicates": dup,
                "missing_numbers_assuming_start_min": missing,
                "is_contiguous_if_start_min": (len(dup) == 0 and len(missing) == 0),
            },
            "title_extraction": {
                "titles_extracted": len(titles_with_no),
                "titles_missing_count": len(missing_title_nos),
                "titles_missing_ref_nos": missing_title_nos,
            },
        }

    # -------------------------
    # Title extraction (URL recovery included)
    # -------------------------
    def _extract_title_from_entry(self, entry: str) -> str:
        s = re.sub(r"\s+", " ", entry.strip())
        if not s:
            return ""

        # quoted title
        qm = re.search(r"[\"“”‘’']([^\"“”‘’']{6,})[\"“”‘’']", s)
        if qm:
            t = self._clean_title(qm.group(1))
            return t if self._is_plausible_title(t) else ""

        # title after first ". "
        m_start = re.search(r"\.\s+", s)
        if not m_start:
            cand = self._clean_title(self._cut_before_year_or_tail(s))
            return cand if self._is_plausible_title(cand) else ""

        rest = s[m_start.end():].strip()
        if not rest:
            return ""

        # end boundary: year first (simple + robust)
        end_pos = None
        mm_year = re.search(r"(?:,|\.)?\s*(19|20)\d{2}[a-z]?\b", rest)
        if mm_year:
            end_pos = mm_year.start()

        # fallback: next sentence
        if end_pos is None:
            mm_dot = re.search(r"\.\s+", rest)
            if mm_dot:
                end_pos = mm_dot.start()

        title = rest if end_pos is None else rest[:end_pos].strip()
        title = self._clean_title(title)

        # URL-ish recovery
        title = self._recover_title_if_urlish(title)

        return title if self._is_plausible_title(title) else ""

    def _recover_title_if_urlish(self, title: str) -> str:
        if not title:
            return ""

        if not re.search(r"https?://|www\.|arxiv\.org|openreview\.net|github\.com|semanticscholar\.org",
                         title, flags=re.IGNORECASE):
            return title

        cut = re.split(r"\bURL\b", title, flags=re.IGNORECASE)[0].strip()
        cut = re.split(r"https?://|www\.", cut, flags=re.IGNORECASE)[0].strip()
        cut = self._clean_title(cut)
        return cut if self._is_plausible_title(cut) else ""

    def _cut_before_year_or_tail(self, text: str) -> str:
        y = re.search(r"\b(19|20)\d{2}[a-z]?\b", text)
        if y:
            text = text[: y.start()].strip()
        text = re.sub(r"\s*URL\s+.*$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*doi:\s*.*$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*arXiv:\s*.*$", "", text, flags=re.IGNORECASE).strip()
        return text.strip()

    def _clean_title(self, title: str) -> str:
        t = re.sub(r"\s+", " ", title.strip())
        t = t.strip(" .,:;()[]{}")
        t = re.sub(r"\s*URL\s+.*$", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\s*doi:\s*.*$", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\s*arXiv:\s*.*$", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r",?\s*(19|20)\d{2}[a-z]?\b.*$", "", t).strip()
        return t

    def _is_plausible_title(self, t: str) -> bool:
        if not t or len(t) < 6:
            return False
        if any(t.lower().startswith(x) for x in ("et al", "arxiv", "doi", "url")):
            return False
        if not (re.search(r"[A-Za-z]", t) or re.search(r"[가-힣]", t)):
            return False
        return True
