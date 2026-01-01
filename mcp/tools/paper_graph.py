"""
Paper Graph Tools

Provides Graph A (Paper Mode) and Graph B (Global Mode) functionality.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

try:
    from ..base import MCPTool, ToolParameter, ExecutionError
except ImportError:
    class MCPTool: pass
    class ToolParameter:
        def __init__(self, name, type, description, required=False, default=None): pass
    class ExecutionError(Exception):
        def __init__(self, message, tool_name=None): super().__init__(f"Error in {tool_name}: {message}")

# Configuration
PDF_DIR = Path(os.getenv("PDF_DIR", "/data/pdf"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
GRAPH_DIR = OUTPUT_DIR / "graph"
PAPER_GRAPH_DIR = GRAPH_DIR / "paper"
UI_STATE_FILE = GRAPH_DIR / "ui_state.json"
GLOBAL_GRAPH_PATH = GRAPH_DIR / "global_graph.json"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
PAPER_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False



def extract_paper_title_from_text(text_file_path: Path) -> Optional[str]:
    """
    Robust title extraction from extracted_text.txt (first page area).

    Handles:
    - "Title: Subtitle" on 2 lines
    - arXiv metadata lines (ignores the ':' in 'arXiv:')
    - UPPERCASE conference style titles (often split across lines)
    - stops before author/affiliation lines
    """
    if not text_file_path.exists():
        return None

    def clean_line(s: str) -> str:
        # remove page markers even if stuck in same line
        s = re.sub(r"===\s*Page\s*\d+\s*===", "", s, flags=re.IGNORECASE)
        s = s.strip()
        # normalize weird separators like "Abstract ;"
        s = re.sub(r"\s*;\s*$", "", s)
        return s.strip()

    def is_noise_line(s: str) -> bool:
        if not s:
            return True
        if re.match(r"^(abstract)\b", s, flags=re.IGNORECASE):
            return True
        if re.match(r"^(published as|preprint|proceedings)\b", s, flags=re.IGNORECASE):
            return True
        # pure arXiv meta line
        if re.match(r"(?i)^\s*arxiv\s*:\s*\S+", s):
            return True
        return False

    def is_author_or_affil_line(s: str) -> bool:
        if not s:
            return False

        # emails / urls
        if "@" in s or re.search(r"https?://", s):
            return True

        # affiliation keywords (strong boundary)
        if re.search(r"\b(University|Department|Institute|Laboratory|Research|School|College)\b", s):
            return True

        # starts with affiliation index like "1Carnegie Mellon University"
        if re.match(r"^\s*\d{1,2}\s*[A-Za-z]", s):
            return True

        # obvious author list patterns
        if re.search(r"\bet\s+al\.?\b", s, flags=re.IGNORECASE):
            return True

        # comma-separated many names
        if "," in s and len(s) < 220:
            words = re.findall(r"[A-Za-z][A-Za-z\-\*0-9]*", s)
            caps = sum(1 for w in words if w and w[0].isupper())
            if caps >= 4:
                return True

        # ✅ 매우 보수적인 "단일 저자 이름" 판별:
        # - 하이픈이 있으면 제목(예: Test-Time)일 가능성이 커서 저자로 보지 않음
        # - 2~3단어 정도의 짧은 이름만 잡음
        if "-" not in s and ":" not in s and len(s) <= 40:
            if re.match(
                r"^[A-Z][a-z]+(?:[\'][A-Za-z]+)?\*?(?:\s+[A-Z][a-z]+(?:[\'][A-Za-z]+)?\*?){1,2}$",
                s
            ):
                return True

        return False

    try:
        # read more than 5 lines (titles often span lines)
        raw_lines = []
        with open(text_file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= 30:
                    break
                s = clean_line(line.strip())
                if s and not is_noise_line(s):
                    raw_lines.append(s)

        if not raw_lines:
            return None

        # find a good "start line" for title
        start_idx = None
        for i, s in enumerate(raw_lines):
            # ignore ':' in arXiv:
            check = re.sub(r"(?i)\barxiv\s*:", "arXiv", s)
            has_colon = (":" in check)

            letters = re.findall(r"[A-Za-z]", s)
            upper_ratio = (sum(ch.isupper() for ch in letters) / len(letters)) if letters else 0.0

            # candidate if:
            # - has a meaningful colon, OR
            # - looks like title in uppercase / long enough
            if has_colon or (upper_ratio > 0.65 and len(s.split()) >= 3) or (len(s) >= 20 and len(s.split()) >= 4):
                # but skip obvious non-title boundaries
                if re.match(r"^(published as)\b", s, flags=re.IGNORECASE):
                    continue
                # skip pure arXiv meta
                if re.match(r"(?i)^\s*arxiv\s*:\s*\S+", s):
                    continue
                start_idx = i
                break

        if start_idx is None:
            # fallback: first decent line
            for s in raw_lines:
                if len(s) > 8:
                    return s[:200]
            return None

        # build title lines (1~3 lines)
        title_lines = [raw_lines[start_idx]]

        # if the start line contains arXiv metadata + title, strip metadata
        if re.search(r"(?i)\barxiv\s*:\s*\S+", title_lines[0]):
            tmp = re.sub(r"(?i)\barxiv\s*:\s*", "arXiv ", title_lines[0])
            m_year = re.search(r"\b(19|20)\d{2}\b", tmp)
            if m_year:
                tail = tmp[m_year.end():].strip()
                if len(tail) >= 8:
                    title_lines[0] = tail
                else:
                    title_lines[0] = tmp
            else:
                title_lines[0] = tmp

        # append continuation lines until author/affiliation/noise
        for j in range(start_idx + 1, min(start_idx + 3, len(raw_lines))):
            nxt = raw_lines[j]
            if is_noise_line(nxt):
                break
            if is_author_or_affil_line(nxt):
                break

            # allow short ALL-CAPS continuation like "LANGUAGE MODELS"
            title_lines.append(nxt)

        title = " ".join(title_lines).strip()
        title = re.sub(r"\s+", " ", title)
        title = re.sub(r"^(Title|Paper|Article)\s*:\s*", "", title, flags=re.IGNORECASE)

        # last sanity trim
        return title[:200] if len(title) > 5 else None

    except Exception:
        return None


class HasPDFTool(MCPTool):
    @property
    def name(self) -> str: return "has_pdf"
    @property
    def description(self) -> str: return "Check if a PDF file exists"
    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="paper_id", type="string", description="Paper ID", required=True)]
    @property
    def category(self) -> str: return "paper_graph"
    async def execute(self, paper_id: str) -> Dict[str, Any]:
        target_id = paper_id.replace("/", "_").replace(".pdf", "")
        if PDF_DIR.exists():
            for file_path in PDF_DIR.iterdir():
                if file_path.suffix.lower() == ".pdf" and target_id in file_path.name:
                    return {"paper_id": paper_id, "exists": True, "path": str(file_path)}
        return {"paper_id": paper_id, "exists": False, "path": None}

class FetchPaperIfMissingTool(MCPTool):
    @property
    def name(self) -> str: return "fetch_paper_if_missing"
    @property
    def description(self) -> str: return "Download a paper from arXiv if missing"
    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="paper_id", type="string", description="arXiv ID", required=True)]
    @property
    def category(self) -> str: return "paper_graph"
    async def execute(self, paper_id: str) -> Dict[str, Any]:
        has_pdf = HasPDFTool()
        check = await has_pdf.execute(paper_id=paper_id)
        if check["exists"]: return {"paper_id": paper_id, "action": "exists", "path": check["path"]}
        try:
            import arxiv
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results(), None)
            if not paper: raise ExecutionError(f"Not found: {paper_id}", self.name)
            filename = f"{paper_id.replace('/', '_')}.pdf"
            paper.download_pdf(dirpath=str(PDF_DIR), filename=filename)
            return {"paper_id": paper_id, "action": "downloaded", "path": str(PDF_DIR / filename)}
        except Exception as e: raise ExecutionError(f"Fetch failed: {e}", self.name)

class ExtractReferencesTool(MCPTool):
    @property
    def name(self) -> str: return "extract_references"
    @property
    def description(self) -> str: return "Extract references"
    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="paper_id", type="string", description="Paper ID", required=True)]
    @property
    def category(self) -> str: return "paper_graph"
    async def execute(self, paper_id: str) -> Dict[str, Any]:
        safe_id = paper_id.replace("/", "_")
        titles_path = None
        if OUTPUT_DIR.exists():
            if (OUTPUT_DIR / safe_id / "reference_titles.json").exists():
                titles_path = OUTPUT_DIR / safe_id / "reference_titles.json"
            else:
                for folder in OUTPUT_DIR.iterdir():
                    if folder.is_dir() and safe_id in folder.name and (folder / "reference_titles.json").exists():
                        titles_path = folder / "reference_titles.json"
                        break

        if titles_path:
            try:
                with open(titles_path, "r") as f: data = json.load(f)
                refs = [{"title": " ".join(t.split()), "source": "json", "arxiv_id": None} for t in data.get("titles", [])]
                cache = PAPER_GRAPH_DIR / f"{safe_id}_refs.json"
                with open(cache, "w") as f: json.dump({"paper_id": paper_id, "references": refs}, f, indent=2)
                return {"paper_id": paper_id, "references_count": len(refs), "cache_path": str(cache)}
            except: pass

        # PDF Fallback (Simplified)
        return {"paper_id": paper_id, "references_count": 0, "cache_path": None}

class GetReferencesTool(MCPTool):
    @property
    def name(self) -> str: return "get_references"
    @property
    def description(self) -> str: return "Get references"
    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="paper_id", type="string", description="Paper ID", required=True)]
    @property
    def category(self) -> str: return "paper_graph"
    async def execute(self, paper_id: str) -> Dict[str, Any]:
        safe_id = paper_id.replace("/", "_")
        # Try both naming conventions
        paths = [
            PAPER_GRAPH_DIR / f"{safe_id}_refs.json",
            PAPER_GRAPH_DIR / f"{safe_id}.pdf_refs.json"
        ]
        for p in paths:
            if p.exists():
                with open(p, "r") as f: return json.load(f)

        return {"paper_id": paper_id, "references": []}

class FetchPaperMetadataTool(MCPTool):
    @property
    def name(self) -> str: return "fetch_paper_metadata"
    @property
    def description(self) -> str: return "Fetch metadata"
    @property
    def parameters(self) -> List[ToolParameter]:
        return [ToolParameter(name="query", type="string", description="Query", required=True)]
    @property
    def category(self) -> str: return "paper_graph"
    async def execute(self, query: str) -> Dict[str, Any]:
        return {"id": query, "title": query, "found": False} # Simplified for speed

class UpdateGraphViewTool(MCPTool):
    @property
    def name(self) -> str: return "update_graph_view"
    @property
    def description(self) -> str: return "Update UI"
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="mode", type="string", description="mode", required=True),
            ToolParameter(name="focus_id", type="string", description="id", required=False)
        ]
    @property
    def category(self) -> str: return "paper_graph"
    async def execute(self, mode: str, focus_id: str = None, message: str = None) -> Dict[str, Any]:
        with open(UI_STATE_FILE, "w") as f:
            json.dump({"timestamp": time.time(), "mode": mode, "focus_id": focus_id}, f)
        return {"success": True}


class BuildReferenceSubgraphTool(MCPTool):
    """Build a reference subgraph for Graph A (Paper Mode) and update UI."""

    @property
    def name(self) -> str: return "build_reference_subgraph"
    @property
    def description(self) -> str: return "Build a reference subgraph centered on a specific paper."
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="paper_id", type="string", description="Center paper ID", required=True),
            ToolParameter(name="depth", type="integer", description="Depth", required=False, default=1),
            ToolParameter(name="existing_nodes", type="array", description="Existing nodes", required=False, default=None)
        ]
    @property
    def category(self) -> str: return "paper_graph"

    async def execute(self, paper_id: str, depth: int = 1, existing_nodes: List[str] = None) -> Dict[str, Any]:
        safe_id = paper_id.replace("/", "_")

        # 1. _refs.json 파일 찾기 (JSON 파일만 있으면 무조건 그림)
        refs_file_path = None
        if PAPER_GRAPH_DIR.exists():
            # 정확한 매칭 또는 부분 매칭 검색
            for f in PAPER_GRAPH_DIR.iterdir():
                if f.name.endswith("refs.json") and safe_id in f.name:
                    refs_file_path = f
                    break

        nodes = []
        edges = []

        if refs_file_path:
            try:
                with open(refs_file_path, "r") as f: data = json.load(f)

                # Center Node
                nodes.append({
                    "id": paper_id,
                    "title": paper_id,
                    "is_center": True,
                    "cluster": 0,
                    "depth": 0,
                    "has_details": True
                })

                # Reference Nodes
                for ref in data.get("references", []):
                    title = ref.get("title", "Unknown")
                    ref_id = ref.get("arxiv_id")
                    target_id = ref_id if ref_id else title

                    nodes.append({
                        "id": target_id,
                        "title": title,
                        "is_center": False,
                        "cluster": 1,
                        "depth": 1
                    })
                    edges.append({
                        "source": paper_id,
                        "target": target_id,
                        "type": "references"
                    })
            except Exception as e:
                print(f"Error parsing refs: {e}")

        # 파일이 없거나 읽기 실패 시, 최소한 센터 노드는 반환 (에러 방지)
        if not nodes:
            nodes.append({"id": paper_id, "is_center": True})

        # 결과 저장
        graph_data = {
            "center": paper_id,
            "nodes": nodes,          # [중요] Frontend가 찾는 키
            "new_nodes": nodes,      # [안전장치] 혹시 모를 호환성용
            "edges": edges,
            "new_edges": edges,
            "meta": {"depth": depth}
        }

        cache_path = PAPER_GRAPH_DIR / f"{safe_id}.json"
        with open(cache_path, "w") as f: json.dump(graph_data, f, indent=2)

        # UI 업데이트
        try:
            with open(UI_STATE_FILE, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "mode": "paper",
                    "focus_id": paper_id,
                    "message": "Graph loaded"
                }, f)
        except: pass

        return graph_data


class BuildGlobalGraphTool(MCPTool):
    """Build a global graph of all papers (Graph B)."""

    @property
    def name(self) -> str: return "build_global_graph"
    @property
    def description(self) -> str: return "Build global graph with paper titles from extracted_text.txt"
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="similarity_threshold", type="number", description="Threshold", required=False, default=0.7),
            ToolParameter(name="use_embeddings", type="boolean", description="Embeddings", required=False, default=True)
        ]
    @property
    def category(self) -> str: return "paper_graph"

    def _load_paper_metadata(self, output_folder: Path) -> Dict[str, Any]:
        """Load title/author/year metadata from known files inside a paper folder."""

        metadata_paths = [
            output_folder / "metadata.json",
            output_folder / "paper_metadata.json",
            output_folder / "meta.json",
        ]

        info: Dict[str, Any] = {"title": None, "authors": None, "year": None}
        for meta_path in metadata_paths:
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                info["title"] = data.get("title") or data.get("paper_title")
                info["authors"] = data.get("authors") or data.get("author")
                info["year"] = data.get("year") or data.get("published") or data.get("publish_year")
                break
            except Exception:
                continue

        return info

    async def execute(self, similarity_threshold: float = 0.7, use_embeddings: bool = True) -> Dict[str, Any]:
        G = nx.Graph()
        papers = []
        if OUTPUT_DIR.exists():
            for output_folder in OUTPUT_DIR.iterdir():
                if output_folder.is_dir() and output_folder.name != "graph":
                    # Check for extracted_text.txt first (preferred), then .json
                    txt_file_txt = output_folder / "extracted_text.txt"
                    txt_file_json = output_folder / "extracted_text.json"

                    text = ""
                    if txt_file_txt.exists():
                        try:
                            with open(txt_file_txt, "r", encoding="utf-8", errors="ignore") as f:
                                text = f.read()[:2000]  # First 2000 chars for similarity
                        except:
                            pass
                    elif txt_file_json.exists():
                        try:
                            with open(txt_file_json) as f:
                                text = " ".join(
                                    p.get("text", "")[:500]
                                    for p in json.load(f).get("pages", [])[:3]
                                )
                        except Exception:
                            pass

                    if txt_file_txt.exists() or txt_file_json.exists():
                        # Try to extract title from extracted_text.txt first
                        paper_title = extract_paper_title_from_text(txt_file_txt)

                        # Fallback to metadata if title not found
                        if not paper_title:
                            meta = self._load_paper_metadata(output_folder)
                            paper_title = meta.get("title")

                        # Final fallback to folder name
                        if not paper_title:
                            paper_title = output_folder.name

                        meta = self._load_paper_metadata(output_folder)
                        authors = meta.get("authors") or []
                        if isinstance(authors, str):
                            authors = [authors]

                        papers.append({
                            "id": output_folder.name,
                            "text": text,
                            "title": paper_title,
                            "authors": authors,
                            "year": meta.get("year"),
                        })

        nodes = [
            {
                "id": p["id"],
                "title": p["title"],
                "cluster": 0,
                "authors": p.get("authors") or [],
                "year": p.get("year"),
            }
            for p in papers
        ]
        for n in nodes:
            G.add_node(n["id"], **n)

        edges = []
        # Similarity
        embeddings_used = False
        if use_embeddings and HAS_SENTENCE_TRANSFORMERS and len(papers) > 1:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                vecs = model.encode([p["text"] for p in papers])
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(vecs)
                for i in range(len(papers)):
                    for j in range(i + 1, len(papers)):
                        if sims[i][j] >= similarity_threshold and papers[i]["text"] and papers[j]["text"]:
                            edges.append({
                                "source": papers[i]["id"],
                                "target": papers[j]["id"],
                                "weight": float(sims[i][j]),
                                "type": "similarity"
                            })
                embeddings_used = True
            except Exception:
                pass

        # Reference Fallback
        for p in papers:
            # 안전하게 폴더 내의 _refs.json 찾기
            ref_file = None
            safe_id = p['id'].replace("/", "_")
            if PAPER_GRAPH_DIR.exists():
                for f in PAPER_GRAPH_DIR.iterdir():
                    if f.name.endswith("refs.json") and safe_id in f.name:
                        ref_file = f
                        break

            if ref_file:
                try:
                    with open(ref_file) as f:
                        for r in json.load(f).get("references", []):
                            rid = r.get("arxiv_id")
                            if rid:
                                rid = rid.replace("/", "_")
                                if any(x["id"] == rid for x in papers):
                                    edges.append({"source": p["id"], "target": rid, "weight": 1.0, "type": "references"})
                except Exception:
                    pass

        meta = {
            "total_papers": len(nodes),
            "total_edges": len(edges),
            "similarity_threshold": similarity_threshold,
            "used_embeddings": embeddings_used,
            "titles": {n["id"]: n.get("title") for n in nodes},
        }

        graph_data = {"nodes": nodes, "edges": edges, "meta": meta}

        # Save to global_graph.json
        with open(GLOBAL_GRAPH_PATH, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        try:
            with open(UI_STATE_FILE, "w") as f:
                json.dump({"timestamp": time.time(), "mode": "global", "message": "Global graph rebuilt"}, f)
        except Exception:
            pass

        return graph_data


class GetGlobalGraphTool(MCPTool):
    """Load global graph from global_graph.json (Graph B)."""

    @property
    def name(self) -> str:
        return "get_global_graph"

    @property
    def description(self) -> str:
        return "Load the global graph from global_graph.json. Returns cached data, use build_global_graph to refresh."

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    @property
    def category(self) -> str:
        return "paper_graph"

    async def execute(self) -> Dict[str, Any]:
        if not GLOBAL_GRAPH_PATH.exists():
            return {
                "nodes": [],
                "edges": [],
                "meta": {"error": "Global graph not found. Run build_global_graph first."}
            }

        try:
            with open(GLOBAL_GRAPH_PATH, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            return graph_data
        except Exception as e:
            return {
                "nodes": [],
                "edges": [],
                "meta": {"error": f"Failed to load global graph: {str(e)}"}
            }
