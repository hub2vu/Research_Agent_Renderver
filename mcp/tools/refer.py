import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from ..base import MCPTool, ToolParameter, ExecutionError

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Shared helpers
# ----------------------------
REF_HEADERS = [
    "references",
    "bibliography",
    "works cited",
    "literature",
    "reference",
    "bibliographies",
]

# References 끝을 끊어줄 만한 마커들(텍스트 추출 방식마다 다양)
REF_END_MARKERS = [
    r"(?im)^\s*appendix\b",
    r"(?im)^\s*[A-Z]\s+appendix\b",          # "A Appendix"
    r"(?im)^\s*acknowledg(e)?ments?\b",
    r"(?im)^\s*author contributions?\b",
    r"(?im)^\s*supplementary\b",
    r"(?im)^\s*supplementary material\b",
    r"(?im)^\s*additional results\b",
    r"(?im)^\s*proofs?\b",
]


def _dehyphenate_linebreaks(text: str) -> str:
    # "capabil-\nities" -> "capabilities"
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _strip_pdf_page_markers(text: str) -> str:
    # "=== Page 14 ===" 제거
    return re.sub(r"(?m)^\s*===\s*Page\s*\d+\s*===\s*$", "", text)


def _remove_common_noise_lines(text: str) -> str:
    """
    너무 공격적으로 제거하면 References 헤더/엔트리까지 날아갈 수 있어서
    '반복 과다' 제거는 보수적으로.
    """
    lines = text.splitlines()
    cleaned: List[str] = []

    # 반복 라인(헤더/푸터) 후보 카운트
    c = Counter(L.strip() for L in lines if len(L.strip()) >= 8)

    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append("")
            continue

        # 페이지 번호만 있는 라인 제거
        if re.fullmatch(r"\d{1,4}", s):
            continue

        # 너무 반복되는 헤더/푸터 제거(단, references 아이템 시작처럼 보이면 보존)
        looks_like_ref_start = bool(re.match(r"^\s*(\[\d+\]|\(\d+\)|\d+\.)\s+", s))
        if c[s] > 6 and not looks_like_ref_start:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def _global_preprocess(text: str) -> str:
    text = _normalize_whitespace(text)
    text = _dehyphenate_linebreaks(text)
    text = _strip_pdf_page_markers(text)
    text = _remove_common_noise_lines(text)
    # 과도한 빈 줄 압축(단, 완전 제거 X)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def _find_references_block(text: str) -> Optional[str]:
    """
    References 헤더(REFERENCES/Bibliography 등) 기준으로 뒤를 잘라냄.
    문서에 references가 여러 번 등장할 수 있으므로 '마지막 섹션'을 우선.
    """
    # 라인 단위로 헤더 찾기
    header_regex = r"(?im)^\s*(%s)\s*$" % "|".join(re.escape(h) for h in REF_HEADERS)
    matches = list(re.finditer(header_regex, text))
    if not matches:
        return None

    # 마지막 references 헤더 사용
    start = matches[-1].end()
    block = text[start:].strip()

    # 끝 마커로 자르기
    end = len(block)
    for mpat in REF_END_MARKERS:
        m = re.search(mpat, block)
        if m:
            end = min(end, m.start())
    block = block[:end].strip()

    # 너무 짧으면 실패로 간주
    if len(block) < 200:
        return None
    return block


def _split_numbered_items(block: str) -> List[Tuple[int, str]]:
    """
    [12] / 12. / (12) 형태 지원
    """
    marker = re.compile(r"(?m)^\s*(\[(\d+)\]|\((\d+)\)|(\d+)\.)\s+")
    matches = list(marker.finditer(block))
    if not matches:
        return []

    items: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        no = m.group(2) or m.group(3) or m.group(4)
        no_int = int(no) if no else 0
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        content = block[start:end].strip()
        content = re.sub(r"\s+", " ", content).strip()
        if len(content) >= 20:
            items.append((no_int, content))
    return items


def _split_author_year_items(block: str) -> List[Tuple[int, str]]:
    """
    번호가 없을 때: author-year 스타일을 '새 엔트리 시작' 규칙으로 분할.
    - "Surname, ..." 또는 "Surname A." 처럼 시작
    - 혹은 "Surname et al." 같이 시작
    """
    lines = [L.rstrip() for L in block.splitlines()]
    # 빈 줄 기준으로 먼저 큰 덩어리 분할
    paras: List[str] = []
    buf: List[str] = []
    for L in lines:
        if not L.strip():
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(L)
    if buf:
        paras.append("\n".join(buf).strip())

    # paragraph가 충분히 참조처럼 보이면 그대로 사용
    if len(paras) >= 5:
        out = []
        for i, p in enumerate(paras, start=1):
            p1 = re.sub(r"\s+", " ", p).strip()
            if len(p1) >= 30:
                out.append((0, p1))
        if len(out) >= 3:
            return out

    # paragraph가 별로면 라인 기반 병합
    start_pat = re.compile(
        r"(?i)^(?:"
        r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+,\s"   # "Surname, "
        r"|[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+\s+[A-Z]\."  # "Surname A."
        r"|[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+\s+et\s+al\."  # "Surname et al."
        r")"
    )

    merged: List[str] = []
    cur: List[str] = []
    for L in lines:
        s = L.strip()
        if not s:
            continue
        is_start = bool(start_pat.match(s)) and (len(s) >= 15)
        if is_start and cur:
            merged.append(" ".join(cur).strip())
            cur = [s]
        else:
            cur.append(s)
    if cur:
        merged.append(" ".join(cur).strip())

    return [(0, m) for m in merged if len(m) >= 30]


def _identifier_mining(text: str) -> List[Tuple[int, str]]:
    """
    References 헤더가 아예 없거나, 섹션 분리가 실패하면 최후수단:
    DOI/arXiv/URL 라인을 뽑아 reference 후보로 반환
    """
    candidates = []
    seen = set()

    for pat in [
        r"\b10\.\d{4,9}/[^\s<>\"']+\b",
        r"(?i)\barXiv:\s*\d{4}\.\d{4,5}(v\d+)?\b",
        r"\bhttps?://[^\s]+",
    ]:
        for m in re.finditer(pat, text):
            v = m.group(0).rstrip(").,;")
            if v not in seen:
                seen.add(v)
                candidates.append(v)

    return [(0, c) for c in candidates]


def _extract_ids(entry: str) -> Dict[str, str]:
    ids: Dict[str, str] = {}

    doi = re.search(r"\b10\.\d{4,9}/[^\s<>\"']+\b", entry)
    if doi:
        ids["doi"] = doi.group(0).rstrip(").,;")

    arxiv = re.search(r"(?i)\barXiv:\s*(\d{4}\.\d{4,5})(v\d+)?\b", entry)
    if arxiv:
        ids["arxiv"] = arxiv.group(1) + (arxiv.group(2) or "")

    url = re.search(r"\bhttps?://[^\s]+", entry)
    if url:
        ids["url"] = url.group(0).rstrip(").,;")

    year = re.search(r"\b(19|20)\d{2}\b", entry)
    if year:
        ids["year"] = year.group(0)

    return ids


def _normalize_arxiv_to_doi_like(arxiv_id: str) -> str:
    """
    arXiv:2201.07207v2 -> 10.48550_arxiv.2201.07207
    (버전은 DOI 호환 노드에선 보통 제거하는 게 안정적)
    """
    base = re.sub(r"v\d+$", "", arxiv_id)
    return f"10.48550_arxiv.{base}"


def _clean_title_string(t: str) -> str:
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"(?i)arXiv:\s*\S+", "", t)
    t = t.strip(" .,[](){}")
    # "In Proceedings of ..." 같은 venue 시작 제거
    t = re.sub(r"(?i)^in\s+proceedings.*$", "", t).strip()
    return t.strip()


def _extract_title(entry: str) -> str:
    """
    참고문헌 포맷이 제각각이라 100%는 불가능.
    아래는 일반적으로 꽤 잘 맞는 휴리스틱 조합:
    1) "..." 또는 “...” 안
    2) 첫 번째/두 번째 문장 사이에서 title 후보 찾기
    3) year 앞 구간에서 마지막 문장 조각
    """
    clean = re.sub(r"\s+", " ", entry).strip()

    # 1) quotes
    m = re.search(r"[\"“](.+?)[\"”]", clean)
    if m and len(m.group(1)) >= 8:
        return _clean_title_string(m.group(1))

    # 2) ". " 기준 분할
    parts = clean.split(". ")
    if len(parts) >= 3:
        # 보통: Authors. Title. Venue...
        cand = parts[1]
        cand = _clean_title_string(cand)
        if 8 <= len(cand) <= 220:
            return cand

    # 3) year 기준
    my = re.search(r"\b(19|20)\d{2}\b", clean)
    if my:
        pre = clean[: my.start()]
        # 마지막 마침표 뒤를 title로 가정
        last_dot = pre.rfind(".")
        cand = pre[last_dot + 1 :] if last_dot != -1 else pre
        cand = _clean_title_string(cand)
        if 8 <= len(cand) <= 220:
            return cand

    return ""


# ----------------------------
# Tools
# ----------------------------
class ExtractReferenceTitlesTool(MCPTool):
    @property
    def name(self) -> str:
        return "extract_reference_titles"

    @property
    def description(self) -> str:
        return (
            "Robustly extract reference entries from extracted_text.txt and infer titles/ids. "
            "Also normalizes arXiv IDs to a DOI-like format (10.48550_arxiv.<id>) for graph compatibility."
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
                description="Name of the PDF file or paper folder (e.g., 2201.07207.pdf or 10.48550_arxiv.2201.07207)",
                required=True,
            ),
            ToolParameter(
                name="save_files",
                type="boolean",
                description="Save outputs to output directory",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="include_raw_entries",
                type="boolean",
                description="Include full raw reference entry text (can be large)",
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
        original_id = filename.replace(".pdf", "").strip()
        paper_id = self._normalize_paper_id(original_id)

        # prefer normalized folder, fallback original
        text_file = OUTPUT_DIR / paper_id / "extracted_text.txt"
        if not text_file.exists():
            text_file = OUTPUT_DIR / original_id / "extracted_text.txt"

        if not text_file.exists():
            raise ExecutionError(
                f"No extracted text found for '{paper_id}' or '{original_id}'. Expected at: {text_file}",
                tool_name=self.name,
            )

        raw_content = text_file.read_text(encoding="utf-8", errors="ignore")
        clean_text = _global_preprocess(raw_content)

        refs_block = _find_references_block(clean_text)
        if refs_block:
            items = _split_numbered_items(refs_block)
            if len(items) < 3:
                items = _split_author_year_items(refs_block)
        else:
            # 헤더가 없으면 최후수단
            items = _identifier_mining(clean_text)

        items_out: List[Dict[str, Any]] = []
        titles_found: List[str] = []

        for idx, (no, entry) in enumerate(items, start=1):
            ref_no = no if no > 0 else idx
            entry_one_line = re.sub(r"\s+", " ", entry).strip()

            ids = _extract_ids(entry_one_line)

            # arXiv가 있으면 그래프 호환 ID도 같이 제공
            graph_id = ""
            if "arxiv" in ids:
                graph_id = _normalize_arxiv_to_doi_like(ids["arxiv"])

            title = _extract_title(entry_one_line)
            if title:
                titles_found.append(title)

            items_out.append(
                {
                    "ref_no": ref_no,
                    "title": title,
                    "ids": ids,
                    "graph_id": graph_id,
                    "raw_text": entry_one_line if include_raw_entries else (entry_one_line[:200] + ("..." if len(entry_one_line) > 200 else "")),
                }
            )

        result: Dict[str, Any] = {
            "filename": paper_id,
            "original_filename": original_id,
            "references_detected": len(items),
            "titles_extracted": sum(1 for t in titles_found if t),
            "titles": titles_found,
            "diagnostics": {
                "has_references_header": refs_block is not None,
                "parsed_items_count": len(items),
                "is_numbered": any(no > 0 for no, _ in items),
                "id_normalized": paper_id != original_id,
            },
        }

        if save_files:
            self._save_outputs(paper_id, result, items_out)

        return result

    def _normalize_paper_id(self, paper_id: str) -> str:
        # 이미 변환된 경우
        if paper_id.startswith("10.48550_arxiv."):
            return paper_id

        # arXiv ID 패턴 (YYMM.NNNNN[vN])
        m = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", paper_id)
        if m:
            base = m.group(1)  # 버전 제거
            return f"10.48550_arxiv.{base}"

        return paper_id

    def _save_outputs(self, paper_id: str, result: Dict[str, Any], items: List[Dict[str, Any]]):
        out_dir = OUTPUT_DIR / paper_id
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "reference_titles.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "reference_titles.txt").write_text(
            "\n".join(result["titles"]), encoding="utf-8"
        )
        (out_dir / "reference_items.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class ExtractAllReferencesTool(MCPTool):
    """Extract references from all papers in /output/ directory."""

    @property
    def name(self) -> str:
        return "extract_all_references"

    @property
    def description(self) -> str:
        return (
            "Extract references from all papers in /output/ directory. "
            "Processes each paper folder that has extracted_text.txt."
        )

    @property
    def category(self) -> str:
        return "pdf"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="save_files",
                type="boolean",
                description="Save outputs to reference_titles.json/txt files",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="skip_existing",
                type="boolean",
                description="Skip papers that already have reference_titles.json",
                required=False,
                default=True,
            ),
        ]

    async def execute(
        self,
        save_files: bool = True,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        results = []
        processed = 0
        skipped = 0
        failed = 0

        extract_tool = ExtractReferenceTitlesTool()

        for folder in OUTPUT_DIR.iterdir():
            if not folder.is_dir():
                continue
            if folder.name == "graph":
                continue

            text_file = folder / "extracted_text.txt"
            refs_file = folder / "reference_titles.json"

            if not text_file.exists():
                continue

            if skip_existing and refs_file.exists():
                skipped += 1
                results.append(
                    {
                        "paper": folder.name,
                        "status": "skipped",
                        "reason": "reference_titles.json already exists",
                    }
                )
                continue

            try:
                result = await extract_tool.execute(
                    filename=folder.name,
                    save_files=save_files,
                    include_raw_entries=False,
                )
                processed += 1
                results.append(
                    {
                        "paper": folder.name,
                        "status": "success",
                        "references_detected": result.get("references_detected", 0),
                        "titles_extracted": result.get("titles_extracted", 0),
                    }
                )
            except Exception as e:
                failed += 1
                results.append({"paper": folder.name, "status": "failed", "error": str(e)})

        return {
            "total_folders": processed + skipped + failed,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "results": results,
        }
