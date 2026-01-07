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
        return "Generate a markdown summary report from extracted text using OpenAI."

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
        # --- [체크 1] OpenAI 라이브러리가 깔려있는지 확인 ---
        if not HAS_OPENAI:
            # 이 에러가 뜨면 requirements.txt에 openai를 추가하고 재빌드해야 함
            raise ExecutionError(
                "서버에 'openai' 패키지가 없습니다. (pip install openai 필요)",
                tool_name=self.name,
            )

        # --- [체크 2] API 키가 들어왔는지 확인 ---
        if not API_KEY:
            # 이 에러가 뜨면 docker-compose.yml의 mcp-server 부분에 OPENAI_API_KEY를 추가해야 함
            raise ExecutionError(
                "환경변수 OPENAI_API_KEY가 설정되지 않았습니다.", tool_name=self.name
            )

        # --- [체크 3] 파일 경로 확인 (pdf.py 방식 따름) ---
        paper_dir = OUTPUT_DIR / paper_id
        text_file = paper_dir / "extracted_text.txt"

        if not text_file.exists():
            # 혹시 JSON으로 되어있는지 확인
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

        # --- OpenAI 호출 및 리포트 작성 ---
        try:
            client = openai.OpenAI(api_key=API_KEY)

            # 텍스트 길이 제한
            if len(full_text) > 30000:
                full_text = full_text[:30000] + "\n...(truncated)..."

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the paper into a Markdown report.",
                    },
                    {"role": "user", "content": f"Paper text:\n{full_text}"},
                ],
            )
            report_content = response.choices[0].message.content

            # 저장
            report_path = paper_dir / "summary_report.md"
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
        report_path = OUTPUT_DIR / paper_id / "summary_report.md"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                return {"found": True, "content": f.read()}
        return {"found": False, "message": "Report not found."}
