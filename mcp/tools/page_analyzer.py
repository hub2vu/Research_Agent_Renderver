import os
import logging
import json
from typing import Dict, Any
from pathlib import Path
from openai import AsyncOpenAI

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
