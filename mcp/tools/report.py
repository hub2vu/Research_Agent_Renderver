import os
import json
import logging
import re
import time
import arxiv
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI
from ..base import MCPTool, ToolParameter

# LLM Logging
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from logs.llm_logger import get_logger as get_llm_logger, SummaryType

# PDF 라이브러리
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logger = logging.getLogger("mcp.tools.report")
logger.setLevel(logging.INFO)
aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 경로 설정
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/output"))
PDF_DIR = Path(os.getenv("PDF_DIR", "data/pdf"))
NEURIPS_DIR = PDF_DIR / "neurips2025"


class GetReportTool(MCPTool):
    """이미 생성된 요약 리포트(txt)를 읽어옵니다."""

    @property
    def name(self) -> str:
        return "get_report"

    @property
    def description(self) -> str:
        return "Retrieve existing summary report."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id", type="string", description="Paper ID", required=True
            )
        ]

    async def execute(self, paper_id: str, **kwargs) -> Dict[str, Any]:
        # 1. 기본 경로 확인
        report_path = OUTPUT_DIR / paper_id / "summary_report.txt"

        # 2. 없으면 fuzzy folder (예: 115627_Title) 확인
        if not report_path.exists() and OUTPUT_DIR.exists():
            for folder in OUTPUT_DIR.iterdir():
                if folder.is_dir() and folder.name.startswith(f"{paper_id}_"):
                    candidate = folder / "summary_report.txt"
                    if candidate.exists():
                        report_path = candidate
                        break

        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                return {"found": True, "content": f.read()}
        return {"found": False, "message": "Report not found."}


class GenerateReportTool(MCPTool):
    """논문 리포트 생성기 (NeurIPS 경로 자동 탐색 포함)"""

    @property
    def name(self) -> str:
        return "generate_report"

    @property
    def description(self) -> str:
        return "Generate summary report from PDF or text."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id", type="string", description="Paper ID", required=True
            )
        ]

    def _resolve_paper_dir(self, paper_id: str) -> Path:
        """ID로 폴더 찾기 (정확히 일치하거나, ID_Title 형태)"""
        exact = OUTPUT_DIR / paper_id
        if exact.exists():
            return exact

        # Fuzzy search in output dir
        if OUTPUT_DIR.exists():
            for folder in OUTPUT_DIR.iterdir():
                if folder.is_dir() and folder.name.startswith(f"{paper_id}_"):
                    return folder

        # 없으면 새로 생성할 기본 경로
        return exact

    def _get_text_from_file(self, paper_dir: Path) -> str:
        """추출된 텍스트 파일 읽기"""
        for fname in ["extracted_text.json", "extracted_text.txt"]:
            fpath = paper_dir / fname
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        if fname.endswith(".json"):
                            data = json.load(f)
                            return data.get("full_text", "") or ""
                        return f.read()
                except:
                    pass
        return ""

    def _find_pdf_path(self, paper_id: str, paper_dir: Path) -> Optional[Path]:
        """PDF 파일 위치 추적 (NeurIPS 폴더 포함)"""
        # 1. Output 폴더 내
        if paper_dir.exists():
            pdfs = list(paper_dir.glob("*.pdf"))
            if pdfs:
                return pdfs[0]

        # 2. NeurIPS 2025 폴더 (ID_Title.pdf 패턴)
        if NEURIPS_DIR.exists():
            # 정확한 ID 매칭
            exact = NEURIPS_DIR / f"{paper_id}.pdf"
            if exact.exists():
                return exact
            # 접두어 매칭 (115627_Alternating_Gradient...)
            for f in NEURIPS_DIR.glob(f"{paper_id}_*.pdf"):
                return f

        # 3. 기본 PDF 폴더
        if (PDF_DIR / f"{paper_id}.pdf").exists():
            return PDF_DIR / f"{paper_id}.pdf"

        return None

    def _extract_from_pdf_file(self, pdf_path: Path) -> str:
        """PDF 파일에서 텍스트 추출"""
        if not PdfReader:
            return ""
        try:
            reader = PdfReader(str(pdf_path))
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            logger.error(f"PDF read error: {e}")
            return ""

    def _download_arxiv_pdf(self, paper_id: str, save_dir: Path) -> bool:
        if not arxiv:
            return False
        try:
            clean_id = paper_id.split("v")[0]
            paper = next(arxiv.Search(id_list=[clean_id]).results(), None)
            if paper:
                save_dir.mkdir(parents=True, exist_ok=True)
                paper.download_pdf(dirpath=str(save_dir), filename=f"{paper_id}.pdf")
                return True
        except:
            pass
        return False

    async def execute(self, paper_id: str, **kwargs) -> Dict[str, Any]:
        paper_dir = self._resolve_paper_dir(paper_id)

        # 1. 텍스트 확인
        full_text = self._get_text_from_file(paper_dir)

        # 2. 텍스트 없으면 PDF 찾아서 추출
        if not full_text:
            pdf_path = self._find_pdf_path(paper_id, paper_dir)

            if pdf_path:
                logger.info(f"Found PDF at {pdf_path}, extracting...")
                full_text = self._extract_from_pdf_file(pdf_path)
            else:
                # PDF도 없으면 ArXiv 다운로드 시도 (NeurIPS는 이미 다운로드 되었다고 가정)
                if "." in paper_id and self._download_arxiv_pdf(paper_id, paper_dir):
                    full_text = self._extract_from_pdf_file(
                        paper_dir / f"{paper_id}.pdf"
                    )

        if not full_text:
            return {
                "success": False,
                "error": f"논문 파일(PDF)을 찾을 수 없습니다. 경로: {NEURIPS_DIR}/{paper_id}_*.pdf 확인 필요.",
            }

        # 3. 리포트 생성
        try:
            prompt = f"""
            당신은 AI 논문 전문 분석가입니다. 아래 내용을 바탕으로 [종합 요약 보고서]를 작성해 주세요.

            [논문 텍스트 일부]
            {full_text[:50000]}

            [작성 양식]
            1. 📌 논문 제목 및 핵심 기여 (3문장 요약)
            2. 🛠 주요 방법론 (Methodology)
            3. 📊 실험 결과 및 성능 (Experiments)
            4. 💡 결론 및 한계점

            반드시 **한국어**로 작성하세요.
            """

            start_time = time.time()
            response = await aclient.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content

            # LLM Logging - Log report generation
            try:
                llm_logger = get_llm_logger()

                llm_logger.log_summary(
                    paper_id=paper_id,
                    summary_type=SummaryType.COMPREHENSIVE,
                    summary_prompt=prompt[:5000],
                    summary_response=content,
                    source_text_length=len(full_text),
                    model="gpt-4o",
                    temperature=0.3,
                    metadata={
                        "source_truncated": len(full_text) > 50000
                    }
                )

                llm_logger.log_llm_call(
                    model="gpt-4o",
                    prompt=prompt[:5000],
                    response=content[:5000],
                    tool_name="generate_report",
                    temperature=0.3,
                    latency_ms=latency_ms,
                    paper_id=paper_id
                )
            except Exception as log_error:
                logger.warning(f"Failed to log report generation: {log_error}")

            # 저장 (결과 폴더가 없으면 생성)
            paper_dir.mkdir(parents=True, exist_ok=True)
            report_file_path = paper_dir / "summary_report.txt"
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # LLM Logging - Log artifact
            try:
                llm_logger = get_llm_logger()
                llm_logger.log_artifact(
                    artifact_type="summary_report_txt",
                    file_path=str(report_file_path),
                    generation_method="llm",
                    input_sources=[{"type": "full_text", "length": len(full_text)}],
                    paper_id=paper_id
                )
            except Exception as log_error:
                logger.warning(f"Failed to log artifact: {log_error}")

            return {"success": True, "result": {"content": content}}

        except Exception as e:
            logger.error(f"Report gen failed: {e}")
            return {"success": False, "error": str(e)}
