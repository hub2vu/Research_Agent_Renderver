import os
import logging
import json
import re
from typing import Any, Dict, List
from pathlib import Path
from openai import AsyncOpenAI

from ..base import MCPTool, ToolParameter, ExecutionError

# PDF 라이브러리
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# 로거 설정
logger = logging.getLogger("mcp.tools.page_analyzer")
logger.setLevel(logging.INFO)

# OpenAI 클라이언트
aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def interpret_paper_page(paper_id: str, page_num: int) -> Dict[str, Any]:
    """
    논문의 특정 페이지를 전체 맥락을 기반으로 '해설'합니다.
    """
    try:
        logger.info(f"🚀 [해석 요청] ID: {paper_id}, Page: {page_num}")

        base_path = Path(os.getenv("OUTPUT_DIR", "data/output"))
        paper_dir = find_paper_directory(base_path, paper_id)

        if not paper_dir:
            return {"error": f"Folder not found: {paper_id}"}

        # 텍스트 추출
        page_text = get_page_text_smart(paper_dir, page_num)

        if not page_text:
            return {
                "page": page_num,
                "original_text": "텍스트 없음",
                "interpretation": "해당 페이지의 텍스트를 찾을 수 없습니다.",
            }

        logger.info(f"✅ 텍스트 확보 완료 ({len(page_text)}자)")

        # 🔥 [핵심 변경] 프롬프트를 '해설가' 모드로 변경
        prompt = f"""
        당신은 노련한 AI 연구원으로서, 동료에게 논문 내용을 쉽게 설명해주는 역할을 맡았습니다.
        아래 제공된 논문의 한 페이지 내용을 읽고, 한국어로 명확하게 '해설'해 주세요.

        [분석할 페이지 내용 (Page {page_num})]:
        {page_text[:4000]}
        (내용이 너무 길면 잘릴 수 있음)

        **🚨 반드시 지켜야 할 지침:**
        1. **단순 번역 금지**: 영어를 한국어로 그대로 옮기지 마십시오. 내용을 완전히 소화한 뒤, 당신의 언어로 다시 서술하세요.
        2. **구조화된 출력**: 결과물은 반드시 아래 형식을 따르세요.
           - **💡 3줄 핵심 요약**: 이 페이지에서 가장 중요한 내용을 3문장으로 요약.
           - **📖 상세 해설**: 문단별 번역이 아니라, 논리적 흐름에 따라 이야기를 풀어서 설명. (예: "저자들은 여기서 ~라는 문제를 지적합니다. 그 이유는 ~이기 때문입니다.")
           - **🧠 주요 개념/용어**: 본문에 등장한 어려운 전문 용어나 개념이 있다면, 초보자도 이해할 수 있게 풀이.
        3. **톤앤매너**: 전문적이지만 이해하기 쉽게, 친절한 어조로 작성하세요.

        """

        # LLM 호출
        response = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and expert research assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        interpretation = response.choices[0].message.content

        return {
            "page": page_num,
            "original_text": (
                page_text[:200] + "..." if len(page_text) > 200 else page_text
            ),
            "interpretation": interpretation,
        }

    except Exception as e:
        logger.error(f"🔥 Server Error: {e}", exc_info=True)
        return {"error": str(e)}


def find_paper_directory(base_path: Path, paper_id: str) -> Path:
    target_dir = base_path / paper_id
    if target_dir.exists():
        return target_dir

    core_id = paper_id.split("arxiv.")[-1] if "arxiv." in paper_id else paper_id
    if not base_path.exists():
        return None

    for folder in base_path.iterdir():
        if folder.is_dir() and (core_id in folder.name or folder.name in paper_id):
            return folder
    return None


def get_page_text_smart(paper_dir: Path, page_num: int) -> str:
    """JSON 우선 확인 후 PDF 직접 읽기 (디버깅 강화 버전)"""
    json_path = paper_dir / "extracted_text.json"

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 리스트인지 딕셔너리인지 확인
            pages_list = []
            if isinstance(data, dict) and "pages" in data:
                pages_list = data["pages"]
            elif isinstance(data, list):
                pages_list = data

            # 페이지 찾기 (문자열 변환 비교 필수!)
            for p in pages_list:
                p_num = p.get("page") or p.get("page_number")
                if str(p_num) == str(page_num):
                    return p.get("text", "") or p.get("content", "")

        except Exception:
            pass  # JSON 실패 시 조용히 PDF로 넘어감

    # PDF Fallback
    pdf_files = list(paper_dir.glob("*.pdf"))
    if pdf_files and PdfReader:
        try:
            reader = PdfReader(pdf_files[0])
            idx = int(page_num) - 1
            if 0 <= idx < len(reader.pages):
                return reader.pages[idx].extract_text()
        except Exception:
            pass

    return None


# ==================== Section Analysis Tool ====================

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))


class AnalyzeSectionTool(MCPTool):
    """Analyze a specific section of a paper identified by its heading title."""

    @property
    def name(self) -> str:
        return "analyze_section"

    @property
    def description(self) -> str:
        return (
            "Analyze a specific section of a paper. Extracts text between the given "
            "section heading and the next heading from extracted_text.txt, then provides "
            "a detailed interpretation and explanation in Korean."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="Paper ID (e.g., '1809.04281')",
                required=True,
            ),
            ToolParameter(
                name="section_title",
                type="string",
                description="The section heading title to analyze",
                required=True,
            ),
            ToolParameter(
                name="next_section_title",
                type="string",
                description="The next section heading title (used as end boundary). Empty string means end of document.",
                required=False,
                default="",
            ),
        ]

    @property
    def category(self) -> str:
        return "analysis"

    def _extract_section_from_text(self, full_text: str, section_title: str, next_section_title: str) -> str:
        """Extract text between section_title and next_section_title from extracted text."""
        lines = full_text.split('\n')

        def normalize(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip().lower()

        norm_target = normalize(section_title)
        norm_next = normalize(next_section_title) if next_section_title else ""

        # Find the start line
        start_idx = None
        for i, line in enumerate(lines):
            norm_line = normalize(line)
            if norm_target and (norm_target in norm_line or norm_line in norm_target):
                start_idx = i
                break

        if start_idx is None:
            target_words = set(norm_target.split())
            for i, line in enumerate(lines):
                norm_line = normalize(line)
                line_words = set(norm_line.split())
                if len(target_words) > 0 and len(target_words & line_words) >= len(target_words) * 0.7:
                    start_idx = i
                    break

        if start_idx is None:
            raise ExecutionError(
                f"Section '{section_title}' not found in extracted text.",
                tool_name=self.name,
            )

        # Find the end line
        end_idx = len(lines)
        if norm_next:
            for i in range(start_idx + 1, len(lines)):
                norm_line = normalize(lines[i])
                if norm_next in norm_line or norm_line in norm_next:
                    end_idx = i
                    break
            else:
                next_words = set(norm_next.split())
                for i in range(start_idx + 1, len(lines)):
                    norm_line = normalize(lines[i])
                    line_words = set(norm_line.split())
                    if len(next_words) > 0 and len(next_words & line_words) >= len(next_words) * 0.7:
                        end_idx = i
                        break

        section_lines = lines[start_idx + 1:end_idx]
        section_text = '\n'.join(section_lines).strip()

        if not section_text:
            raise ExecutionError(
                f"No text content found for section '{section_title}'.",
                tool_name=self.name,
            )

        return section_text

    async def execute(
        self,
        paper_id: str,
        section_title: str,
        next_section_title: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        # Read extracted_text.txt
        paper_dir = OUTPUT_DIR / paper_id
        text_file = paper_dir / "extracted_text.txt"

        if not text_file.exists():
            json_file = paper_dir / "extracted_text.json"
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    full_text = data.get("full_text", "")
            else:
                raise ExecutionError(
                    f"텍스트 파일을 찾을 수 없습니다: {text_file}",
                    tool_name=self.name,
                )
        else:
            with open(text_file, "r", encoding="utf-8") as f:
                full_text = f.read()

        if isinstance(full_text, str):
            full_text = full_text.encode("utf-8", "replace").decode("utf-8")

        # Extract section text
        section_text = self._extract_section_from_text(full_text, section_title, next_section_title)

        # Analyze the section using LLM
        prompt = f"""당신은 노련한 AI 연구원으로서, 동료에게 논문 내용을 쉽게 설명해주는 역할을 맡았습니다.
아래 제공된 논문의 '{section_title}' 섹션 내용을 읽고, 한국어로 명확하게 '해설'해 주세요.

[분석할 섹션 내용]:
{section_text[:6000]}

**반드시 지켜야 할 지침:**
1. **단순 번역 금지**: 영어를 한국어로 그대로 옮기지 마십시오. 내용을 완전히 소화한 뒤, 당신의 언어로 다시 서술하세요.
2. **구조화된 출력**: 결과물은 반드시 아래 형식을 따르세요.
   - **3줄 핵심 요약**: 이 섹션에서 가장 중요한 내용을 3문장으로 요약.
   - **상세 해설**: 문단별 번역이 아니라, 논리적 흐름에 따라 이야기를 풀어서 설명.
   - **주요 개념/용어**: 본문에 등장한 어려운 전문 용어나 개념이 있다면, 초보자도 이해할 수 있게 풀이.
3. **톤앤매너**: 전문적이지만 이해하기 쉽게, 친절한 어조로 작성하세요.
"""

        response = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful and expert research assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        analysis_text = response.choices[0].message.content

        return {
            "status": "success",
            "paper_id": paper_id,
            "section_title": section_title,
            "analysis_text": analysis_text,
        }
