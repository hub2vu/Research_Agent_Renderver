from typing import Any, Dict, List, Optional
from mcp.base import MCPTool, ToolParameter, ExecutionError
from mcp.registry import execute_tool
import os

class NeuripsPipelineTool(MCPTool):
    @property
    def name(self) -> str:
        return "process_neurips_paper"

    @property
    def description(self) -> str:
        return "NeurIPS 논문 PDF 다운로드, 텍스트 추출, 레퍼런스 추출, 그리고 그래프 생성(Global/Reference)까지 수행하는 통합 파이프라인입니다."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="NeurIPS 논문 ID",
                required=True,
            ),
            ToolParameter(
                name="out_dir",
                type="string",
                description="PDF 저장 경로 (선택)",
                required=False,
                default="/data/pdf/neurips2025"
            ),
            ToolParameter(
                name="similarity_threshold",
                type="number",
                description="Global Graph 생성 시 사용할 유사도 임계값 (기본: 0.75)",
                required=False,
                default=0.75
            )
        ]

    async def execute(self, paper_id: str, out_dir: str = "/data/pdf/neurips2025", similarity_threshold: float = 0.75, **kwargs) -> Dict[str, Any]:
        results_summary = {}

        # ---------------------------------------------------------
        # 1단계: PDF 다운로드
        # ---------------------------------------------------------
        print(f"[Pipeline] 1. Downloading paper {paper_id}...")
        download_res = await execute_tool(
            "neurips2025_download_pdf",
            paper_id=paper_id,
            mode="download",
            out_dir=out_dir
        )

        if not download_res["success"]:
            return {"error": f"Download failed: {download_res.get('error')}"}
        
        dl_data = download_res["result"]
        if not dl_data.get("results"):
             return {"error": "Download tool returned no results."}
             
        saved_path = dl_data["results"][0].get("saved_path")
        if not saved_path:
             return {"error": "PDF was not saved correctly."}

        results_summary["pdf_path"] = saved_path
        
        # ---------------------------------------------------------
        # 2단계: 텍스트 추출
        # ---------------------------------------------------------
        print(f"[Pipeline] 2. Extracting text from {saved_path}...")
        
        extract_res = await execute_tool(
            "extract_all", 
            filename=saved_path
        )

        if not extract_res["success"]:
            return {"error": f"Extraction failed: {extract_res.get('error')}", "partial_result": results_summary}

        # Get the output folder name from extract_all result
        output_directory = extract_res["result"].get("output_directory", "")
        folder_name = os.path.basename(output_directory) if output_directory else ""
        results_summary["text_extracted"] = bool(folder_name)
        results_summary["output_folder"] = folder_name

        # ---------------------------------------------------------
        # 3단계: 레퍼런스 추출 (reference_titles.json 생성)
        # ---------------------------------------------------------
        print(f"[Pipeline] 3. Extracting references from {folder_name}...")

        if folder_name:
            # Use extract_reference_titles (refer.py) to parse references from extracted_text.txt
            # This creates reference_titles.json in the output folder
            ref_res = await execute_tool(
                "extract_reference_titles",
                filename=folder_name,
                save_files=True
            )

            if ref_res["success"]:
                ref_count = ref_res["result"].get("titles_extracted", 0)
                results_summary["ref_count"] = ref_count
                results_summary["references_detected"] = ref_res["result"].get("references_detected", 0)
            else:
                results_summary["ref_warning"] = f"Reference extraction failed: {ref_res.get('error')}"
        else:
            results_summary["ref_warning"] = "No output folder found, skipping reference extraction."

        # ---------------------------------------------------------
        # 4단계: Global Graph 생성 (업데이트)
        # ---------------------------------------------------------
        print(f"[Pipeline] 4. Building Global Graph (threshold={similarity_threshold})...")
        
        skip_emb = os.getenv("SKIP_EMBEDDINGS", "").lower() in ("1", "true", "yes")
        global_graph_res = await execute_tool(
            "build_global_graph",
            similarity_threshold=similarity_threshold,
            use_embeddings=not skip_emb
        )
        
        if global_graph_res["success"]:
            results_summary["global_graph"] = "Updated successfully"
        else:
            results_summary["global_graph_error"] = global_graph_res.get("error")

        # NOTE: Step 5 (build_reference_subgraph) removed
        # PaperGraphPage directly fetches /output/{paperId}/reference_titles.json
        # The subgraph tool was causing UI state to switch to "paper" mode,
        # which forced navigation away from Global Graph page.

        return {
            "message": "Pipeline completed successfully",
            "pipeline_results": results_summary
        }