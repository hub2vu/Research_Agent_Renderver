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
            "Robustly extract references and titles from extracted_text.txt, "
            "automatically normalizing arXiv IDs to DOI format for graph compatibility."
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
                description="Name of the PDF file (e.g., 2201.07207.pdf)",
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
                description="Include raw reference entry text",
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
        # 1. 파일명에서 paper_id 추출 및 정규화 (arXiv ID -> DOI 포맷)
        original_id = filename.replace(".pdf", "").strip()
        paper_id = self._normalize_paper_id(original_id)

        # 2. 텍스트 파일 경로 설정 (입력 디렉토리는 원본 ID나 정규화된 ID 모두 확인)
        # 우선 정규화된 ID 폴더 확인 -> 없으면 원본 ID 폴더 확인
        text_file = OUTPUT_DIR / paper_id / "extracted_text.txt"
        if not text_file.exists():
            text_file = OUTPUT_DIR / original_id / "extracted_text.txt"

        if not text_file.exists():
            raise ExecutionError(
                f"No extracted text found for '{paper_id}' or '{original_id}'. Expected at: {text_file}",
                tool_name=self.name,
            )

        raw_content = text_file.read_text(encoding="utf-8", errors="ignore")

        # 3. 전처리 및 추출 로직 실행
        clean_text = self._global_preprocess(raw_content)
        refs_block = self._isolate_references_block(clean_text)
        ref_items = self._parse_reference_items(refs_block)

        items_out: List[Dict[str, Any]] = []
        titles_found: List[Tuple[int, str]] = []

        for idx, (no, entry) in enumerate(ref_items):
            display_no = no if no > 0 else (idx + 1)
            title = self._extract_title_logic(entry)
            
            if title:
                titles_found.append((display_no, title))

            items_out.append({
                "ref_no": display_no,
                "title": title,
                "raw_text": entry if include_raw_entries else entry[:100] + "..."
            })

        titles_list = [t for _, t in titles_found]
        
        result: Dict[str, Any] = {
            "filename": paper_id,  # 정규화된 ID 사용 (그래프 생성 핵심)
            "original_filename": original_id,
            "references_detected": len(ref_items),
            "titles_extracted": len(titles_found),
            "titles": titles_list,
            "diagnostics": {
                "parsed_items_count": len(ref_items),
                "is_numbered": any(n > 0 for n, _ in ref_items),
                "id_normalized": paper_id != original_id
            }
        }

        if save_files:
            # 정규화된 paper_id 디렉토리에 저장해야 시스템이 인식함
            self._save_outputs(paper_id, result, items_out)

        return result

    def _normalize_paper_id(self, paper_id: str) -> str:
        """
        arXiv ID(예: 2201.07207)를 감지하여 
        시스템이 그래프를 생성할 수 있는 DOI 포맷(10.48550_arxiv.2201.07207)으로 변환합니다.
        """
        # 이미 변환된 경우 패스
        if paper_id.startswith("10.48550_arxiv."):
            return paper_id
            
        # arXiv ID 패턴 매칭 (YYMM.NNNNN)
        # 예: 2201.07207, 1706.03762v5
        arxiv_pattern = r'^\d{4}\.\d{4,5}(v\d+)?$'
        if re.match(arxiv_pattern, paper_id):
            return f"10.48550_arxiv.{paper_id}"
            
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

    # ... (이하 _global_preprocess, _isolate_references_block, _parse_reference_items, 
    #      _extract_title_logic 등 기존 로직 유지) ...
    
    def _global_preprocess(self, text: str) -> str:
        lines = text.splitlines()
        cleaned_lines = []
        line_counts = Counter(L.strip() for L in lines if len(L.strip()) > 5)
        
        for line in lines:
            s = line.strip()
            line = re.sub(r"\", "", line) # 소스 태그 제거
            s = line.strip()
            if not s: continue
            if re.match(r"^=== Page \d+ ===$", s): continue
            if re.match(r"^\d+$", s): continue
            if line_counts[s] > 3 and not re.match(r"^\[?\d+\]?\.?", s): continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _isolate_references_block(self, text: str) -> str:
        headers = ["REFERENCES", "BIBLIOGRAPHY", "References", "Bibliography"]
        start_idx = -1
        for h in headers:
            matches = list(re.finditer(r"\n\s*" + re.escape(h) + r"\s*\n", text, re.IGNORECASE))
            if matches:
                start_idx = matches[-1].end()
                break
        
        if start_idx == -1: start_idx = int(len(text) * 0.8)
        block = text[start_idx:]

        end_markers = [r"\n\s*APPENDIX", r"\n\s*Appendix", r"\n\s*SUPPLEMENTARY", r"\n\s*Supplementary", r"\n\s*[A-Z]\s+ADDITIONAL RESULTS"]
        end_idx = len(block)
        for marker in end_markers:
            m = re.search(marker, block)
            if m and m.start() < end_idx: end_idx = m.start()
        
        return block[:end_idx].strip()

    def _parse_reference_items(self, block: str) -> List[Tuple[int, str]]:
        numbered = self._try_parse_numbered(block)
        if len(numbered) >= 3: return numbered
        return self._try_parse_unnumbered(block)

    def _try_parse_numbered(self, text: str) -> List[Tuple[int, str]]:
        items = []
        matches = list(re.finditer(r"(?m)^\s*\[(\d+)\]\s+", text))
        if not matches: matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s+", text))
        
        for i, m in enumerate(matches):
            no = int(m.group(1))
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            content = re.sub(r"\s+", " ", text[start:end]).strip()
            if len(content) > 10: items.append((no, content))
        return items

    def _try_parse_unnumbered(self, text: str) -> List[Tuple[int, str]]:
        lines = [L.strip() for L in text.splitlines() if L.strip()]
        merged_items = []
        current_lines = []
        for line in lines:
            is_start = False
            if line and line[0].isupper() and "," in line[:50]: is_start = True
            if is_start and current_lines:
                merged_items.append(" ".join(current_lines))
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines: merged_items.append(" ".join(current_lines))
        return [(0, item) for item in merged_items if len(item) > 20]

    def _extract_title_logic(self, entry: str) -> str:
        clean = re.sub(r"\s+", " ", entry).strip()
        m_quote = re.search(r"[\"“](.+?)[\"”]", clean)
        if m_quote and len(m_quote.group(1)) > 10: return m_quote.group(1)
        
        parts = clean.split(". ")
        if len(parts) >= 2:
            candidate = parts[1]
            idx = 1
            while idx < len(parts) and len(parts[idx]) <= 2 and parts[idx].isupper(): idx += 1
            if idx < len(parts): candidate = parts[idx]
            return self._clean_title_string(candidate)
            
        m_year = re.search(r"\b(19|20)\d{2}\b", clean)
        if m_year:
            pre_year = clean[:m_year.start()]
            last_dot = pre_year.rfind(".")
            return self._clean_title_string(pre_year[last_dot+1:] if last_dot != -1 else pre_year)
        return ""

    def _clean_title_string(self, text: str) -> str:
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"arXiv:\S+", "", text, flags=re.IGNORECASE)
        text = text.strip(" .,[]()")
        text = re.sub(r"^In\s+.*", "", text, flags=re.IGNORECASE)
        return text.strip()