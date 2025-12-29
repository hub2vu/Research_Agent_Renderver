"""
Paper Graph Tools

Provides Graph A (Paper Mode) and Graph B (Global Mode) functionality.
- Graph A: Single paper reference exploration (on-demand, incremental)
- Graph B: Global paper relationship overview (batch, embedding-based)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from ..base import MCPTool, ToolParameter, ExecutionError

# Configuration
PDF_DIR = Path(os.getenv("PDF_DIR", "/data/pdf"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))
GRAPH_DIR = OUTPUT_DIR / "graph"
PAPER_GRAPH_DIR = GRAPH_DIR / "paper"

# Ensure directories exist
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
PAPER_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Try to import embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class HasPDFTool(MCPTool):
    """Check if a PDF exists for a given paper ID."""

    @property
    def name(self) -> str:
        return "has_pdf"

    @property
    def description(self) -> str:
        return "Check if a PDF file exists for a given paper ID"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="Paper ID (e.g., arXiv ID '2301.07041' or filename without .pdf)",
                required=True
            )
        ]

    @property
    def category(self) -> str:
        return "paper_graph"

    async def execute(self, paper_id: str) -> Dict[str, Any]:
        # Normalize paper_id
        paper_id = paper_id.replace("/", "_").replace(".pdf", "")

        # Check various possible filenames
        possible_names = [
            f"{paper_id}.pdf",
            f"{paper_id.replace('_', '/')}.pdf",
            paper_id if paper_id.endswith(".pdf") else None
        ]

        for name in possible_names:
            if name and (PDF_DIR / name).exists():
                return {
                    "paper_id": paper_id,
                    "exists": True,
                    "path": str(PDF_DIR / name)
                }

        return {
            "paper_id": paper_id,
            "exists": False,
            "path": None
        }


class FetchPaperIfMissingTool(MCPTool):
    """Fetch a paper from arXiv if not already downloaded."""

    @property
    def name(self) -> str:
        return "fetch_paper_if_missing"

    @property
    def description(self) -> str:
        return "Download a paper from arXiv if it doesn't exist locally"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="arXiv paper ID (e.g., '2301.07041')",
                required=True
            )
        ]

    @property
    def category(self) -> str:
        return "paper_graph"

    async def execute(self, paper_id: str) -> Dict[str, Any]:
        # Check if already exists
        has_pdf_tool = HasPDFTool()
        check_result = await has_pdf_tool.execute(paper_id=paper_id)

        if check_result["exists"]:
            return {
                "paper_id": paper_id,
                "action": "already_exists",
                "path": check_result["path"]
            }

        # Try to download from arXiv
        try:
            import arxiv

            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results(), None)

            if paper is None:
                raise ExecutionError(f"Paper not found on arXiv: {paper_id}", tool_name=self.name)

            # Download
            filename = f"{paper_id.replace('/', '_')}.pdf"
            paper.download_pdf(dirpath=str(PDF_DIR), filename=filename)

            return {
                "paper_id": paper_id,
                "action": "downloaded",
                "path": str(PDF_DIR / filename),
                "title": paper.title
            }

        except ImportError:
            raise ExecutionError("arxiv library not installed", tool_name=self.name)
        except Exception as e:
            raise ExecutionError(f"Failed to fetch paper: {str(e)}", tool_name=self.name)


class ExtractReferencesTool(MCPTool):
    """Extract references from a paper's PDF."""

    @property
    def name(self) -> str:
        return "extract_references"

    @property
    def description(self) -> str:
        return "Extract references from a paper's PDF file"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="Paper ID",
                required=True
            )
        ]

    @property
    def category(self) -> str:
        return "paper_graph"

    async def execute(self, paper_id: str) -> Dict[str, Any]:
        # First check if PDF exists
        has_pdf_tool = HasPDFTool()
        check_result = await has_pdf_tool.execute(paper_id=paper_id)

        if not check_result["exists"]:
            raise ExecutionError(f"PDF not found for paper: {paper_id}", tool_name=self.name)

        # Try to use refer.py tool if available
        try:
            from .refer import ExtractReferencesTool as ReferTool
            refer_tool = ReferTool()
            result = await refer_tool.execute(filename=Path(check_result["path"]).name)
            return result
        except ImportError:
            pass

        # Fallback: basic reference extraction using PyMuPDF
        import fitz

        pdf_path = Path(check_result["path"])
        doc = fitz.open(pdf_path)

        # Extract text from last pages (usually where references are)
        references = []
        ref_text = ""

        for page_num in range(max(0, len(doc) - 5), len(doc)):
            page = doc[page_num]
            ref_text += page.get_text()

        doc.close()

        # Basic parsing - look for arXiv IDs
        import re
        arxiv_pattern = r'(\d{4}\.\d{4,5})'
        found_ids = re.findall(arxiv_pattern, ref_text)

        for arxiv_id in set(found_ids):
            if arxiv_id != paper_id:  # Don't include self-reference
                references.append({
                    "arxiv_id": arxiv_id,
                    "source": "extracted"
                })

        # Save to cache
        cache_path = PAPER_GRAPH_DIR / f"{paper_id.replace('/', '_')}_refs.json"
        with open(cache_path, "w") as f:
            json.dump({"paper_id": paper_id, "references": references}, f, indent=2)

        return {
            "paper_id": paper_id,
            "references_count": len(references),
            "references": references,
            "cache_path": str(cache_path)
        }


class GetReferencesTool(MCPTool):
    """Get references for a paper (from cache or extract)."""

    @property
    def name(self) -> str:
        return "get_references"

    @property
    def description(self) -> str:
        return "Get references for a paper, using cache if available"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="Paper ID",
                required=True
            )
        ]

    @property
    def category(self) -> str:
        return "paper_graph"

    async def execute(self, paper_id: str) -> Dict[str, Any]:
        cache_path = PAPER_GRAPH_DIR / f"{paper_id.replace('/', '_')}_refs.json"

        # Check cache first
        if cache_path.exists():
            with open(cache_path, "r") as f:
                cached = json.load(f)
            cached["from_cache"] = True
            return cached

        # Extract if not cached
        extract_tool = ExtractReferencesTool()
        result = await extract_tool.execute(paper_id=paper_id)
        result["from_cache"] = False
        return result


class BuildReferenceSubgraphTool(MCPTool):
    """Build a reference subgraph for Graph A (Paper Mode)."""

    @property
    def name(self) -> str:
        return "build_reference_subgraph"

    @property
    def description(self) -> str:
        return "Build a reference subgraph centered on a specific paper (Graph A)"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="Center paper ID",
                required=True
            ),
            ToolParameter(
                name="depth",
                type="integer",
                description="How many levels of references to include",
                required=False,
                default=1
            ),
            ToolParameter(
                name="existing_nodes",
                type="array",
                description="List of already loaded node IDs (for incremental updates)",
                required=False,
                default=None
            )
        ]

    @property
    def category(self) -> str:
        return "paper_graph"

    async def execute(
        self,
        paper_id: str,
        depth: int = 1,
        existing_nodes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        existing_nodes = set(existing_nodes or [])
        G = nx.DiGraph()

        # Get paper metadata from arXiv
        async def get_paper_metadata(pid: str) -> Dict:
            try:
                import arxiv
                search = arxiv.Search(id_list=[pid])
                paper = next(search.results(), None)
                if paper:
                    return {
                        "id": pid,
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors][:3],
                        "abstract": paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                        "year": paper.published.year if paper.published else None,
                        "categories": paper.categories
                    }
            except:
                pass
            return {"id": pid, "title": pid, "authors": [], "abstract": "", "year": None}

        # BFS to build subgraph
        to_process = [(paper_id, 0)]
        processed = set()
        new_nodes = []
        new_edges = []

        while to_process:
            current_id, current_depth = to_process.pop(0)

            if current_id in processed:
                continue
            processed.add(current_id)

            # Get metadata
            metadata = await get_paper_metadata(current_id)

            # Add node if new
            if current_id not in existing_nodes:
                new_nodes.append({
                    **metadata,
                    "depth": current_depth,
                    "is_center": current_id == paper_id
                })
                G.add_node(current_id, **metadata)

            # Get references if within depth
            if current_depth < depth:
                try:
                    refs_tool = GetReferencesTool()
                    refs_result = await refs_tool.execute(paper_id=current_id)

                    for ref in refs_result.get("references", []):
                        ref_id = ref.get("arxiv_id")
                        if ref_id:
                            # Add edge
                            if current_id not in existing_nodes or ref_id not in existing_nodes:
                                new_edges.append({
                                    "source": current_id,
                                    "target": ref_id,
                                    "type": "references"
                                })
                            G.add_edge(current_id, ref_id, type="references")

                            # Queue for processing
                            if ref_id not in processed:
                                to_process.append((ref_id, current_depth + 1))
                except:
                    pass

        # Save to cache
        cache_path = PAPER_GRAPH_DIR / f"{paper_id.replace('/', '_')}.json"
        graph_data = {
            "center": paper_id,
            "nodes": new_nodes,
            "edges": new_edges,
            "meta": {
                "depth": depth,
                "total_nodes": len(G.nodes),
                "total_edges": len(G.edges)
            }
        }

        with open(cache_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        # Return diff for incremental update
        return {
            "center": paper_id,
            "new_nodes": new_nodes,
            "new_edges": new_edges,
            "cache_path": str(cache_path),
            "is_incremental": len(existing_nodes) > 0
        }


class BuildGlobalGraphTool(MCPTool):
    """Build a global graph of all papers (Graph B)."""

    @property
    def name(self) -> str:
        return "build_global_graph"

    @property
    def description(self) -> str:
        return "Build a global graph of all papers with embedding-based similarities (Graph B)"

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="similarity_threshold",
                type="number",
                description="Minimum similarity score to create an edge (0-1)",
                required=False,
                default=0.7
            ),
            ToolParameter(
                name="use_embeddings",
                type="boolean",
                description="Whether to use embeddings for similarity",
                required=False,
                default=True
            )
        ]

    @property
    def category(self) -> str:
        return "paper_graph"

    async def execute(
        self,
        similarity_threshold: float = 0.7,
        use_embeddings: bool = True
    ) -> Dict[str, Any]:
        G = nx.Graph()
        papers = []

        # Collect all papers with extracted text
        for output_folder in OUTPUT_DIR.iterdir():
            if output_folder.is_dir() and output_folder.name != "graph":
                text_file = output_folder / "extracted_text.json"
                if text_file.exists():
                    with open(text_file, "r") as f:
                        data = json.load(f)
                    papers.append({
                        "id": output_folder.name,
                        "filename": data.get("filename", output_folder.name),
                        "text": " ".join(p.get("text", "")[:500] for p in data.get("pages", [])[:3])
                    })

        if not papers:
            return {
                "nodes": [],
                "edges": [],
                "meta": {"error": "No processed papers found"}
            }

        # Get metadata for each paper
        nodes = []
        for paper in papers:
            node = {
                "id": paper["id"],
                "filename": paper["filename"],
                "title": paper["id"],  # Will be updated if arXiv metadata available
                "cluster": 0
            }

            # Try to get arXiv metadata
            try:
                import arxiv
                # Check if it looks like an arXiv ID
                paper_id = paper["id"].replace("_", "/")
                if paper_id[0].isdigit():
                    search = arxiv.Search(id_list=[paper_id])
                    arxiv_paper = next(search.results(), None)
                    if arxiv_paper:
                        node["title"] = arxiv_paper.title
                        node["abstract"] = arxiv_paper.summary[:300]
                        node["authors"] = [a.name for a in arxiv_paper.authors][:3]
                        node["year"] = arxiv_paper.published.year if arxiv_paper.published else None
            except:
                pass

            nodes.append(node)
            G.add_node(paper["id"], **node)

        # Build edges based on similarity
        edges = []

        if use_embeddings and HAS_SENTENCE_TRANSFORMERS and len(papers) > 1:
            # Use embeddings for similarity
            model = SentenceTransformer('all-MiniLM-L6-v2')
            texts = [p["text"] for p in papers]
            embeddings = model.encode(texts)

            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)

            for i in range(len(papers)):
                for j in range(i + 1, len(papers)):
                    sim = float(sim_matrix[i][j])
                    if sim >= similarity_threshold:
                        edges.append({
                            "source": papers[i]["id"],
                            "target": papers[j]["id"],
                            "weight": sim,
                            "type": "similarity"
                        })
                        G.add_edge(papers[i]["id"], papers[j]["id"], weight=sim)

            # Cluster detection
            try:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(G)
                for cluster_id, community in enumerate(communities):
                    for node_id in community:
                        for node in nodes:
                            if node["id"] == node_id:
                                node["cluster"] = cluster_id
            except:
                pass

        else:
            # Fallback: reference-based edges only
            for paper in papers:
                refs_cache = PAPER_GRAPH_DIR / f"{paper['id']}_refs.json"
                if refs_cache.exists():
                    with open(refs_cache, "r") as f:
                        refs_data = json.load(f)
                    for ref in refs_data.get("references", []):
                        ref_id = ref.get("arxiv_id", "").replace("/", "_")
                        if ref_id and any(p["id"] == ref_id for p in papers):
                            edges.append({
                                "source": paper["id"],
                                "target": ref_id,
                                "weight": 1.0,
                                "type": "references"
                            })

        # Save global graph
        global_graph_path = GRAPH_DIR / "global_graph.json"
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "total_papers": len(nodes),
                "total_edges": len(edges),
                "similarity_threshold": similarity_threshold,
                "used_embeddings": use_embeddings and HAS_SENTENCE_TRANSFORMERS
            }
        }

        with open(global_graph_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        return graph_data
