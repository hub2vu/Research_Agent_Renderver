# mcp/tools/iclr_pipeline.py
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from ..base import MCPTool, ToolParameter, ExecutionError
from ..registry import execute_tool


def _safe_filename(s: str, max_len: int = 180) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len].strip()


@dataclass
class ICLRPaperRow:
    paper_id: str
    title: str
    abstract: str


class ICLRDownloadPdfTool(MCPTool):
    """ICLR 2025 논문 PDF 다운로드 도구"""

    @property
    def name(self) -> str:
        return "iclr2025_download_pdf"

    @property
    def description(self) -> str:
        return (
            "ICLR 2025 논문 paper_id로 OpenReview에서 PDF를 다운로드합니다. "
            "ICLR의 경우 paper_id가 OpenReview note_id와 동일합니다."
        )

    @property
    def category(self) -> str:
        return "iclr"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="ICLR 논문 ID (OpenReview note_id)",
                required=True,
            ),
            ToolParameter(
                name="metadata_path",
                type="string",
                description="metadata CSV 경로",
                required=False,
                default="data/embeddings_ICLR/ICLR2025_accepted_meta.csv",
            ),
            ToolParameter(
                name="out_dir",
                type="string",
                description="PDF 저장 디렉토리",
                required=False,
                default="pdf/iclr2025",
            ),
            ToolParameter(
                name="overwrite",
                type="boolean",
                description="이미 파일이 있으면 덮어쓰기",
                required=False,
                default=False,
            ),
        ]

    async def execute(
        self,
        paper_id: str,
        metadata_path: str = "data/embeddings_ICLR/ICLR2025_accepted_meta.csv",
        out_dir: str = "pdf/iclr2025",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        paper_id = str(paper_id).strip()
        if not paper_id:
            raise ExecutionError("paper_id는 필수입니다.")

        # Load metadata to get title
        mp = Path(metadata_path)
        title = paper_id  # fallback
        abstract = ""

        if mp.exists():
            try:
                df = pd.read_csv(mp)
                row = df[df["paper_id"].astype(str) == paper_id]
                if not row.empty:
                    title = str(row.iloc[0].get("title", paper_id))
                    abstract = str(row.iloc[0].get("abstract", ""))
            except Exception as e:
                print(f"Warning: Could not read metadata: {e}")

        # ICLR uses paper_id directly as OpenReview note_id
        pdf_url = f"https://openreview.net/pdf?id={paper_id}"

        # Prepare output directory
        out_base = Path(out_dir)
        out_base.mkdir(parents=True, exist_ok=True)

        fname = _safe_filename(f"{paper_id}_{title}.pdf")
        fpath = out_base / fname

        if fpath.exists() and not overwrite:
            return {
                "ok": True,
                "count": 1,
                "results": [{
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "pdf_url": pdf_url,
                    "saved_path": str(fpath),
                    "status": "exists",
                }]
            }

        # Download PDF
        try:
            resp = await asyncio.to_thread(
                requests.get, pdf_url, timeout=90
            )
            resp.raise_for_status()
            fpath.write_bytes(resp.content)
            status = "downloaded"
        except Exception as e:
            return {
                "ok": False,
                "error": f"PDF download failed: {e}",
                "results": [{
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "pdf_url": pdf_url,
                    "saved_path": None,
                    "status": "download_failed",
                }]
            }

        return {
            "ok": True,
            "count": 1,
            "results": [{
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "saved_path": str(fpath),
                "status": status,
            }]
        }


class ICLRPipelineTool(MCPTool):
    """ICLR 논문 처리 파이프라인"""

    @property
    def name(self) -> str:
        return "process_iclr_paper"

    @property
    def description(self) -> str:
        return (
            "ICLR 논문 PDF 다운로드, 텍스트 추출, 레퍼런스 추출, 그리고 그래프 생성까지 "
            "수행하는 통합 파이프라인입니다."
        )

    @property
    def category(self) -> str:
        return "iclr"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="ICLR 논문 ID (OpenReview note_id)",
                required=True,
            ),
            ToolParameter(
                name="out_dir",
                type="string",
                description="PDF 저장 경로",
                required=False,
                default="/data/pdf/iclr2025"
            ),
            ToolParameter(
                name="similarity_threshold",
                type="number",
                description="Global Graph 생성 시 사용할 유사도 임계값 (기본: 0.75)",
                required=False,
                default=0.75
            )
        ]

    async def execute(
        self,
        paper_id: str,
        out_dir: str = "/data/pdf/iclr2025",
        similarity_threshold: float = 0.75,
        **kwargs
    ) -> Dict[str, Any]:
        results_summary = {}

        # ---------------------------------------------------------
        # Step 1: Download PDF
        # ---------------------------------------------------------
        print(f"[ICLR Pipeline] 1. Downloading paper {paper_id}...")
        download_res = await execute_tool(
            "iclr2025_download_pdf",
            paper_id=paper_id,
            out_dir=out_dir
        )

        if not download_res["success"]:
            return {"error": f"Download failed: {download_res.get('error')}"}

        dl_data = download_res["result"]
        if not dl_data.get("ok") or not dl_data.get("results"):
            return {"error": "Download tool returned no results."}

        saved_path = dl_data["results"][0].get("saved_path")
        if not saved_path:
            return {"error": "PDF was not saved correctly."}

        results_summary["pdf_path"] = saved_path

        # ---------------------------------------------------------
        # Step 2: Extract text
        # ---------------------------------------------------------
        print(f"[ICLR Pipeline] 2. Extracting text from {saved_path}...")

        extract_res = await execute_tool(
            "extract_all",
            filename=saved_path
        )

        if not extract_res["success"]:
            return {
                "error": f"Extraction failed: {extract_res.get('error')}",
                "partial_result": results_summary
            }

        output_directory = extract_res["result"].get("output_directory", "")
        folder_name = os.path.basename(output_directory) if output_directory else ""
        results_summary["text_extracted"] = bool(folder_name)
        results_summary["output_folder"] = folder_name

        # ---------------------------------------------------------
        # Step 3: Extract references
        # ---------------------------------------------------------
        print(f"[ICLR Pipeline] 3. Extracting references from {folder_name}...")

        if folder_name:
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
        # Step 4: Build Global Graph
        # ---------------------------------------------------------
        print(f"[ICLR Pipeline] 4. Building Global Graph (threshold={similarity_threshold})...")

        global_graph_res = await execute_tool(
            "build_global_graph",
            similarity_threshold=similarity_threshold,
            use_embeddings=True
        )

        if global_graph_res["success"]:
            results_summary["global_graph"] = "Updated successfully"
        else:
            results_summary["global_graph_error"] = global_graph_res.get("error")

        return {
            "message": "ICLR Pipeline completed successfully",
            "pipeline_results": results_summary
        }
