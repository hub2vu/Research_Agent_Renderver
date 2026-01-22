import os
import json
from pathlib import Path
from typing import Any, Dict, List

# [1] pdf.py와 똑같은 경로 설정 방식 사용 (이러면 pdf.py랑 같은 곳에 저장됨)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))

# [2] OpenAI 키 가져오기 (환경변수에서)
API_KEY = os.getenv("OPENAI_API_KEY")

# [3] 라이브러리 체크 (없으면 에러 메시지를 명확히 뱉도록 함)
try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from ..base import MCPTool, ToolParameter, ExecutionError


class GenerateReportTool(MCPTool):
    """Generate a structured markdown report from extracted text."""

    @property
    def name(self) -> str:
        return "generate_report"

    @property
    def description(self) -> str:
        return "Generate a txt summary report from extracted text using OpenAI."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="Paper ID (e.g., '1809.04281')",
                required=True,
            )
        ]

    @property
    def category(self) -> str:
        return "report"

    async def execute(self, paper_id: str) -> Dict[str, Any]:
        # --- [체크 1] OpenAI 라이브러리 확인 ---
        if not HAS_OPENAI:
            raise ExecutionError(
                "서버에 'openai' 패키지가 없습니다. (pip install openai 필요)",
                tool_name=self.name,
            )

        # --- [체크 2] API 키 확인 ---
        if not API_KEY:
            raise ExecutionError(
                "환경변수 OPENAI_API_KEY가 설정되지 않았습니다.", tool_name=self.name
            )

        # --- [체크 3] 파일 경로 확인 ---
        paper_dir = OUTPUT_DIR / paper_id
        text_file = paper_dir / "extracted_text.txt"

        if not text_file.exists():
            json_file = paper_dir / "extracted_text.json"
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    full_text = json.load(f).get("full_text", "")
            else:
                raise ExecutionError(
                    f"텍스트 파일을 찾을 수 없습니다. 경로: {text_file}",
                    tool_name=self.name,
                )
        else:
            with open(text_file, "r", encoding="utf-8") as f:
                full_text = f.read()

        # ⭐ [핵심 수정] 여기서도 깨진 문자(Surrogate)를 제거해야 함! ⭐
        # 이 줄이 없으면 report.py가 텍스트 읽다가 죽습니다.
        if isinstance(full_text, str):
            full_text = full_text.encode("utf-8", "replace").decode("utf-8")

        # --- OpenAI 호출 및 리포트 작성 ---
        try:
            client = openai.OpenAI(api_key=API_KEY)

            # 텍스트 길이 제한
            if len(full_text) > 30000:
                full_text = full_text[:30000] + "\n...(truncated)..."

            system_instruction = """
You are a professional IT Tech Journalist who explains complex research to general readers.
Your task is to rewrite the given academic paper into an easy-to-read structured report.

Please follow this format strictly:

# [Title of the Paper] - Easy Review

## 1. Problem Definition (Why did they start this?)
- Explain the limitation of previous technologies in simple terms.
- Avoid complex math symbols (like O(L^2)) or jargon.
- Focus on "what was difficult to do before" and "why it mattered".

## 2. Research Objective (What did they want to solve?)
- Describe the goal clearly in one or two sentences.
- Example: "They wanted to make AI compose music longer than 1 minute without losing the beat."

## 3. Core Claims & Achievements (What is the breakthrough?)
- List 3 key achievements.
- Use analogies if possible to help understanding.

## 4. Summary Report (Narrative)
- Write a 3-paragraph story summarizing the paper.
- Do NOT use bullet points here. Write it like a blog post or news article.
- Start with "This research proposes..." or "The authors introduce..."
- Make it engaging and easy to read for non-experts.

(IMPORTANT: Write the report in English. If the text is truncated, focus on the available parts.)
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {"role": "user", "content": f"Paper text:\n{full_text}"},
                ],
            )
            report_content = response.choices[0].message.content

            # 저장 (txt 파일로 저장)
            report_path = paper_dir / "summary_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            return {
                "status": "success",
                "report_path": str(report_path),
                "preview": report_content[:100],
            }

        except Exception as e:
            raise ExecutionError(f"OpenAI API 호출 실패: {str(e)}", tool_name=self.name)


class GetReportTool(MCPTool):
    """Retrieve an existing report."""

    @property
    def name(self) -> str:
        return "get_report"

    @property
    def description(self) -> str:
        return "Retrieve the generated report."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id", type="string", description="Paper ID", required=True
            )
        ]

    @property
    def category(self) -> str:
        return "report"

    async def execute(self, paper_id: str) -> Dict[str, Any]:
        report_path = OUTPUT_DIR / paper_id / "summary_report.txt"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                return {"found": True, "content": f.read()}
        return {"found": False, "message": "Report not found."}
