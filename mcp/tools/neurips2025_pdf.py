# mcp/tools/neurips2025_pdf.py
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..base import MCPTool, ToolParameter, ExecutionError


def _norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    # 간단 정규화: LaTeX/기호/중복공백 등으로 OpenReview 제목과 약간 달라도 매칭되게
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_filename(s: str, max_len: int = 180) -> str:
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len].strip()


def _extract_openreview_title(note: Any) -> str:
    """
    OpenReview API v2 노트 content 필드는 경우에 따라:
    - {"title": {"value": "..."}}
    - {"title": "..."}
    형태가 섞여 나올 수 있어 방어적으로 처리.
    """
    content = getattr(note, "content", None) or {}
    title = content.get("title", "")
    if isinstance(title, dict):
        return title.get("value", "") or ""
    if isinstance(title, str):
        return title
    return ""


@dataclass
class PaperRow:
    paper_id: str
    name: str
    abstract: str
    speakers_authors: str
    virtualsite_url: str


class Neurips2025PdfTool(MCPTool):
    @property
    def name(self) -> str:
        return "neurips2025_download_pdf"

    @property
    def description(self) -> str:
        return (
            "NeurIPS 2025 metadata.csv의 paper_id 또는 제목(name)으로 OpenReview에서 논문 PDF를 찾아 "
            "로컬 디렉토리에 저장하거나(pdf), PDF 링크만 반환합니다. "
            "기본 venue_id는 NeurIPS.cc/2025/Conference 입니다."
        )

    @property
    def category(self) -> str:
        return "neurips"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="metadata.csv의 paper_id (단일). 예: '119181'",
                required=False,
            ),
            ToolParameter(
                name="paper_ids",
                type="array",
                items_type="string",
                description="metadata.csv의 paper_id 여러 개",
                required=False,
            ),
            ToolParameter(
                name="title_query",
                type="string",
                description="제목(name) 부분 문자열로 검색(여러 개 매칭될 수 있음)",
                required=False,
            ),
            ToolParameter(
                name="metadata_path",
                type="string",
                description="metadata.csv 경로",
                required=False,
                default="data/embeddings_Neu/metadata.csv",
            ),
            ToolParameter(
                name="venue_id",
                type="string",
                description="OpenReview venue id",
                required=False,
                default="NeurIPS.cc/2025/Conference",
            ),
            ToolParameter(
                name="cache_path",
                type="string",
                description="OpenReview title->note_id 캐시(JSON) 저장 경로",
                required=False,
                default="data/embeddings_Neu/openreview_title_map.json",
            ),
            ToolParameter(
                name="out_dir",
                type="string",
                description="PDF 저장 디렉토리",
                required=False,
                default="pdf/neurips2025",
            ),
            ToolParameter(
                name="mode",
                type="string",
                description="link | download | both (기본 download)",
                required=False,
                default="download",
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="title_query로 여러 개 매칭될 때 상한",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="overwrite",
                type="boolean",
                description="이미 파일이 있으면 덮어쓰기",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="force_rebuild_cache",
                type="boolean",
                description="OpenReview title map 캐시 강제 재생성",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="openreview_username",
                type="string",
                description="OpenReview 로그인(선택). 미지정 시 환경변수 OPENREVIEW_USERNAME 사용",
                required=False,
            ),
            ToolParameter(
                name="openreview_password",
                type="string",
                description="OpenReview 로그인(선택). 미지정 시 환경변수 OPENREVIEW_PASSWORD 사용",
                required=False,
            ),
        ]

    async def execute(
        self,
        paper_id: Optional[str] = None,
        paper_ids: Optional[List[str]] = None,
        title_query: Optional[str] = None,
        metadata_path: str = "data/embeddings_Neu/metadata.csv",
        venue_id: str = "NeurIPS.cc/2025/Conference",
        cache_path: str = "data/embeddings_Neu/openreview_title_map.json",
        out_dir: str = "pdf/neurips2025",
        mode: str = "download",
        max_results: int = 5,
        overwrite: bool = False,
        force_rebuild_cache: bool = False,
        openreview_username: Optional[str] = None,
        openreview_password: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 1) 입력 검증
        mode = (mode or "download").strip().lower()
        if mode not in {"link", "download", "both"}:
            raise ExecutionError("mode는 link | download | both 중 하나여야 합니다.")

        ids: List[str] = []
        if paper_id:
            ids.append(str(paper_id).strip())
        if paper_ids:
            ids.extend([str(x).strip() for x in paper_ids if str(x).strip()])
        ids = list(dict.fromkeys(ids))  # dedup, preserve order

        if not ids and not (title_query and title_query.strip()):
            raise ExecutionError("paper_id/paper_ids 또는 title_query 중 하나는 제공해야 합니다.")

        # 2) metadata 로드
        mp = Path(metadata_path)
        if not mp.exists():
            raise ExecutionError(f"metadata_path를 찾을 수 없습니다: {mp}")

        import pandas as pd  # lazy import – save ~50MB RAM at startup
        df = pd.read_csv(mp)
        required_cols = {"paper_id", "name", "abstract", "speakers/authors", "virtualsite_url"}
        if not required_cols.issubset(set(df.columns)):
            raise ExecutionError(
                f"metadata.csv 컬럼이 예상과 다릅니다. 필요: {sorted(required_cols)} / 현재: {list(df.columns)}"
            )

        # 3) 대상 논문 선택
        rows: List[PaperRow] = []

        if ids:
            sub = df[df["paper_id"].astype(str).isin(ids)]
            for _, r in sub.iterrows():
                rows.append(
                    PaperRow(
                        paper_id=str(r["paper_id"]),
                        name=str(r["name"]),
                        abstract=str(r.get("abstract", "")),
                        speakers_authors=str(r.get("speakers/authors", "")),
                        virtualsite_url=str(r.get("virtualsite_url", "")),
                    )
                )

        if title_query and title_query.strip():
            q = title_query.strip().lower()
            sub = df[df["name"].astype(str).str.lower().str.contains(q, na=False)].head(max_results)
            for _, r in sub.iterrows():
                rows.append(
                    PaperRow(
                        paper_id=str(r["paper_id"]),
                        name=str(r["name"]),
                        abstract=str(r.get("abstract", "")),
                        speakers_authors=str(r.get("speakers/authors", "")),
                        virtualsite_url=str(r.get("virtualsite_url", "")),
                    )
                )

        # dedup by paper_id
        uniq: Dict[str, PaperRow] = {}
        for pr in rows:
            uniq[pr.paper_id] = pr
        rows = list(uniq.values())

        if not rows:
            return {"ok": True, "results": [], "message": "조건에 맞는 논문을 metadata.csv에서 찾지 못했습니다."}

        # 4) OpenReview title->note_id map 준비
        title_map = await self._load_or_build_openreview_title_map(
            venue_id=venue_id,
            cache_path=Path(cache_path),
            force=force_rebuild_cache,
            openreview_username=openreview_username,
            openreview_password=openreview_password,
        )

        # 5) 각 논문 note_id 매칭 + 링크/다운로드
        out_base = Path(out_dir)
        out_base.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for pr in rows:
            nt = _norm_title(pr.name)
            note_id = title_map.get(nt)

            # fuzzy fallback (정규화가 달라 매칭 안 되는 경우)
            if not note_id:
                note_id = self._fuzzy_match_note_id(nt, title_map)

            pdf_url = f"https://openreview.net/pdf?id={note_id}" if note_id else None
            saved_path = None
            status = "no_note_match" if not note_id else "linked"

            if note_id and mode in {"download", "both"}:
                fname = _safe_filename(f"{pr.paper_id}_{pr.name}.pdf")
                fpath = out_base / fname
                if fpath.exists() and not overwrite:
                    saved_path = str(fpath)
                    status = "exists"
                else:
                    pdf_bytes = await self._download_pdf_bytes(
                        note_id=note_id,
                        pdf_url=pdf_url,
                        openreview_username=openreview_username,
                        openreview_password=openreview_password,
                    )
                    fpath.write_bytes(pdf_bytes)
                    saved_path = str(fpath)
                    status = "downloaded"

            results.append(
                {
                    "paper_id": pr.paper_id,
                    "title": pr.name,
                    "abstract": pr.abstract,
                    "speakers_authors": pr.speakers_authors,
                    "virtualsite_url": pr.virtualsite_url,
                    "openreview_note_id": note_id,
                    "pdf_url": pdf_url,
                    "saved_path": saved_path,
                    "status": status,
                }
            )

        return {"ok": True, "count": len(results), "results": results}

    async def _load_or_build_openreview_title_map(
        self,
        venue_id: str,
        cache_path: Path,
        force: bool,
        openreview_username: Optional[str],
        openreview_password: Optional[str],
    ) -> Dict[str, str]:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not force:
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    return {str(k): str(v) for k, v in data.items()}
            except Exception:
                # 캐시 깨졌으면 재생성
                pass

        # openreview-py가 없으면: 캐시 생성 불가 → 링크/다운로드 정확도가 떨어지므로 에러로 처리
        try:
            import openreview  # type: ignore
        except Exception as e:
            raise ExecutionError(
                "openreview-py가 필요합니다. mcp 컨테이너에 `pip install openreview-py`를 추가하세요."
            ) from e

        username = openreview_username or (Path(".") and None)
        password = openreview_password or None

        # env fallback
        import os as _os

        username = username or _os.getenv("OPENREVIEW_USERNAME")
        password = password or _os.getenv("OPENREVIEW_PASSWORD")

        # 로그인 없이도 되는 경우가 있지만, 환경별로 막히는 경우가 있어 우선 시도 후 실패 시 안내
        def _make_client():
            try:
                return openreview.api.OpenReviewClient(
                    baseurl="https://api2.openreview.net",
                    username=username,
                    password=password,
                )
            except TypeError:
                # 어떤 버전은 username/password 없이도 생성됨
                return openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")

        client = await asyncio.to_thread(_make_client)

        def _fetch_all_notes():
            # 공식 가이드: content={'venueid': ...} 로 노트 전체 가져오기 :contentReference[oaicite:1]{index=1}
            return client.get_all_notes(content={"venueid": venue_id})

        try:
            notes = await asyncio.to_thread(_fetch_all_notes)
        except Exception as e:
            raise ExecutionError(
                f"OpenReview에서 venue 노트를 가져오지 못했습니다. "
                f"OPENREVIEW_USERNAME/PASSWORD 설정이 필요할 수 있습니다. 원인: {e}"
            )

        title_map: Dict[str, str] = {}
        for note in notes:
            t = _extract_openreview_title(note)
            if not t:
                continue
            title_map[_norm_title(t)] = getattr(note, "id", None) or getattr(note, "note_id", None)

        # None 값 제거
        title_map = {k: v for k, v in title_map.items() if v}

        cache_path.write_text(json.dumps(title_map, ensure_ascii=False, indent=2), encoding="utf-8")
        return title_map

    def _fuzzy_match_note_id(self, norm_title: str, title_map: Dict[str, str]) -> Optional[str]:
        # 아주 가벼운 fuzzy: 토큰 Jaccard + 길이 가중
        ntoks = set(norm_title.split())
        if not ntoks:
            return None

        best = (0.0, None)
        for k, v in title_map.items():
            ktoks = set(k.split())
            inter = len(ntoks & ktoks)
            union = len(ntoks | ktoks) or 1
            score = inter / union
            if score > best[0]:
                best = (score, v)

        # 너무 약한 매칭은 버림
        return best[1] if best[0] >= 0.72 else None

    async def _download_pdf_bytes(
        self,
        note_id: str,
        pdf_url: Optional[str],
        openreview_username: Optional[str],
        openreview_password: Optional[str],
    ) -> bytes:
        # 1순위: openreview-py로 pdf bytes 받기 (가능하면 이게 안정적)
        try:
            import openreview  # type: ignore
            import os as _os

            username = openreview_username or _os.getenv("OPENREVIEW_USERNAME")
            password = openreview_password or _os.getenv("OPENREVIEW_PASSWORD")

            def _make_client():
                try:
                    return openreview.api.OpenReviewClient(
                        baseurl="https://api2.openreview.net",
                        username=username,
                        password=password,
                    )
                except TypeError:
                    return openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")

            client = await asyncio.to_thread(_make_client)

            # openreview-py는 get_pdf / get_attachment 제공 
            def _get_pdf():
                return client.get_pdf(id=note_id)

            return await asyncio.to_thread(_get_pdf)
        except Exception:
            # 2순위: 그냥 openreview.net/pdf?id= 로 HTTP GET
            if not pdf_url:
                raise ExecutionError("pdf_url이 없어 다운로드를 진행할 수 없습니다.")
            r = await asyncio.to_thread(requests.get, pdf_url, {"timeout": 90})
            # requests.get 인자를 잘못 줬을 수 있으니 방어
            if isinstance(r, dict):
                r = requests.get(pdf_url, timeout=90)
            r.raise_for_status()
            return r.content
