# Notion Save 사용법
.env에 다음 2개 추가
    NOTION_PARENT_PAGE_ID=
    NOTION_API_TOKEN=


NOTION_PARENT_PAGE_ID=
<img width="2054" height="784" alt="스크린샷 2026-01-30 111953" src="https://github.com/user-attachments/assets/23f23abd-2b32-443c-a325-cb5a42db8ff2" />

NOTION_API_TOKEN=
https://www.notion.so/profile/integrations 링크에서 API 발급

# Research Agent

A Docker-based research assistant with MCP (Model Context Protocol) architecture and paper graph visualization.

# Reference Graph

<img width="1979" height="1444" alt="스크린샷 2025-12-29 210026" src="https://github.com/user-attachments/assets/65912fb0-2f24-4ea9-a065-4dab52e39179" />

# NeurIPS Graph
<img width="2850" height="1518" alt="스크린샷 2026-01-03 232421" src="https://github.com/user-attachments/assets/a907d3de-baf1-4d77-b3d8-e55d25989146" />



## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Web UI (React)                             │
│  ┌──────────────────────────┐  ┌──────────────────────────────────┐ │
│  │   GlobalGraphPage (B)    │  │      PaperGraphPage (A)          │ │
│  │   - All papers overview  │  │   - Reference exploration        │ │
│  │   - Embedding similarity │  │   - Incremental expansion        │ │
│  └────────────┬─────────────┘  └───────────────┬──────────────────┘ │
│               └────────────────┬───────────────┘                    │
│                                │ HTTP :3000                         │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────┐
│                        Agent Layer                                   │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐          │
│  │ client  │──│ planner  │──│ executor │──│   memory    │          │
│  │ (entry) │  │          │  │          │  │             │          │
│  └────┬────┘  └──────────┘  └─────┬────┘  └─────────────┘          │
│       │         Planning          │         State                   │
│       │         (no side-effects) │         Storage                 │
└───────┼───────────────────────────┼─────────────────────────────────┘
        │                           │
        │      HTTP API :8000       │
        ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          MCP Layer                                   │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────────────────────────┐ │
│  │ server  │──│ registry │──│              tools/                 │ │
│  │ (entry) │  │ (SSOT)   │  │  pdf | arxiv | web_search | graph   │ │
│  └─────────┘  └──────────┘  └─────────────────────────────────────┘ │
│                               Side-effects allowed                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns**
   - Agent Layer: Thinking, planning, decision-making (NO side-effects)
   - MCP Layer: Tool execution, file I/O, API calls (side-effects allowed)
   - Web UI: Visualization and interaction

2. **Single Source of Truth (SSOT)**
   - `registry.py` is the ONLY place where tools are discovered and collected
   - Tools are auto-discovered from `mcp/tools/` directory

3. **Single Entry Points**
   - `mcp/server.py` - MCP server entry point
   - `agent/client.py` - Agent entry point
   - Other modules (planner, executor, memory) are NEVER executed directly

## Project Structure

```
Research_agent/
├─ mcp/                          # MCP Layer (side-effects allowed)
│  ├─ __init__.py
│  ├─ server.py                  # MCP server entrypoint
│  ├─ registry.py                # Tool auto-discovery (SSOT)
│  ├─ base.py                    # Tool base classes
│  └─ tools/
│     ├─ pdf.py                  # PDF extraction tools
│     ├─ refer.py                # Reference extraction tools
│     ├─ arxiv.py                # arXiv API tools
│     ├─ web_search.py           # Web search tools (Tavily)
│     └─ paper_graph.py          # Paper graph tools (Graph A/B)
│
├─ agent/                        # Agent Layer (no side-effects)
│  ├─ __init__.py
│  ├─ client.py                  # Agent entrypoint (orchestration)
│  ├─ planner.py                 # Goal → plan creation
│  ├─ executor.py                # Plan → MCP calls
│  └─ memory.py                  # State storage
│
├─ web/                          # Web UI (React/TypeScript)
│  ├─ api/mcp.ts                 # MCP REST client
│  ├─ components/
│  │  ├─ GraphCanvas.tsx         # D3.js force-directed graph
│  │  ├─ SidePanel.tsx           # Paper details panel
│  │  └─ PaperCard.tsx           # Paper metadata card
│  ├─ pages/
│  │  ├─ GlobalGraphPage.tsx     # Graph B - All papers overview
│  │  └─ PaperGraphPage.tsx      # Graph A - Reference exploration
│  ├─ App.tsx                    # Router setup
│  ├─ main.tsx                   # Entry point
│  └─ package.json
│
├─ requirements/
│  ├─ base.txt                   # Common dependencies
│  ├─ mcp.txt                    # MCP server dependencies
│  └─ agent.txt                  # Agent dependencies
│
├─ docker/
│  ├─ Dockerfile.mcp             # MCP server image
│  ├─ Dockerfile.agent           # Agent image
│  └─ Dockerfile.web             # Web UI image
│
├─ scripts/
│  ├─ run-mcp.sh                 # Start MCP server
│  └─ process-all.sh             # Process all PDFs
│
├─ pdf/                          # Input PDF files
├─ output/                       # Extracted content
│  ├─ images/                    # Extracted images
│  ├─ text/                      # Extracted text
│  └─ graph/                     # Graph cache
│     ├─ global_graph.json       # Global graph (Graph B)
│     └─ paper/                  # Per-paper graphs (Graph A)
│        └─ <paper_id>.json
│
├─ .env.example                  # Environment template
├─ docker-compose.yml            # Service orchestration
└─ README.md
```

## Paper Graph System

### Graph A: Paper Mode (Reference Exploration)
- **Purpose**: Explore references of a specific paper
- **Behavior**: On-demand, incremental expansion
- **Usage**: Double-click nodes to expand their references
- **Center**: Selected paper is fixed at center

### Graph B: Global Mode (All Papers Overview)
- **Purpose**: Visualize relationships across all papers
- **Behavior**: Batch processing with embedding-based similarity
- **Clustering**: Louvain community detection
- **Similarity**: SentenceTransformer embeddings

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repo-url>
cd Research_agent

# Create .env file
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY (required)
# - TAVILY_API_KEY (optional, for web search)
```

### 2. Build Docker Images

```bash
docker compose build
```

### 3. Add PDF Files

```bash
cp /path/to/your/papers/*.pdf ./pdf/
```

### 4. Run

**Start All Services:**
```bash
# Start MCP server and Web UI
docker compose up -d mcp-server web

# Open Web UI
open http://localhost:3000
```

**Interactive Agent:**
```bash
# Run Agent interactively
docker compose run --rm agent
```

**Single Command:**
```bash
docker compose run --rm agent "List all PDFs and extract text from each"
```

**Process All PDFs:**
```bash
./scripts/process-all.sh
```

## Available Tools

### PDF Tools
| Tool | Description |
|------|-------------|
| `list_pdfs` | List all PDF files in the directory |
| `extract_text` | Extract text from a PDF |
| `extract_images` | Extract images from a PDF |
| `extract_all` | Extract text and images |
| `process_all_pdfs` | Process all PDFs |
| `get_pdf_info` | Get PDF metadata |
| `read_extracted_text` | Read previously extracted text |
| `check_github_link` | Find GitHub repository URLs in extracted text |

### arXiv Tools
| Tool | Description |
|------|-------------|
| `arxiv_search` | Search arXiv for papers |
| `arxiv_get_paper` | Get paper details by ID |
| `arxiv_download` | Download paper PDF |

### Web Search Tools
| Tool | Description |
|------|-------------|
| `web_search` | Search the web (Tavily) |
| `web_get_content` | Fetch URL content |
| `web_research` | In-depth topic research |

### Ranking Tools
| Tool | Description |
|------|-------------|
| `update_user_profile` | Update interests/keywords and toggle `exclude_local_papers` (writes to `OUTPUT_DIR/users/profile.json`). |
| `apply_hard_filters` | Apply ALREADY_READ, blacklist keywords, and year filters. |
| `calculate_semantic_scores` | Compute hybrid semantic relevance scores (embeddings + optional LLM for borderline cases). |
| `evaluate_paper_metrics` | Compute dimension scores (keywords/authors/institutions/recency/practicality) and soft penalties. |
| `rank_and_select_top_k` | Combine scores, compute final ranking, and optionally add a contrastive paper. |

### Paper Graph Tools
| Tool | Description |
|------|-------------|
| `has_pdf` | Check if PDF exists for a paper ID |
| `fetch_paper_if_missing` | Download from arXiv if not present |
| `extract_references` | Extract references from a PDF |
| `get_references` | Get cached references for a paper |
| `build_reference_subgraph` | Build Graph A (paper-centered) |
| `build_global_graph` | Build Graph B (all papers) |

## API Endpoints

The MCP server exposes a REST API at `http://localhost:8000`:

```bash
# List all tools
curl http://localhost:8000/tools

# Get tools in OpenAI format
curl http://localhost:8000/tools/schema

# Execute a tool
curl -X POST http://localhost:8000/tools/list_pdfs/execute \
  -H "Content-Type: application/json" \
  -d '{"arguments": {}}'

# Graph endpoints
curl http://localhost:8000/tools/build_global_graph/execute \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"similarity_threshold": 0.7}}'

# Convenience endpoints
curl http://localhost:8000/pdf/list
curl "http://localhost:8000/arxiv/search?query=transformer"
```

## Web UI

Access the web interface at `http://localhost:3000`:

| Page | URL | Description |
|------|-----|-------------|
| Global Graph | `/` | Overview of all papers (Graph B) |
| Paper Graph | `/paper/:id` | Reference exploration for a paper (Graph A) |

### Graph Interactions
- **Click**: Select a node to view details
- **Double-click**: Expand references (Graph A only)
- **Drag**: Reposition nodes
- **Controls**: Adjust similarity threshold, rebuild graph

## Example Usage

```
You: What PDFs do I have?
  [Calling list_pdfs...]
Assistant: You have 3 PDF files: paper1.pdf, paper2.pdf, paper3.pdf

You: Search arXiv for papers about attention mechanisms
  [Calling arxiv_search...]
Assistant: Found 10 papers about attention mechanisms...

You: Download the first one and extract its content
  [Calling arxiv_download...]
  [Calling extract_all...]
Assistant: Downloaded and extracted the paper. Here's a summary...

You: Build a reference graph for paper 2106.09685
  [Calling build_reference_subgraph...]
Assistant: Built reference graph with 15 papers and 23 edges.
```

## Requirements

- Docker & Docker Compose
- OpenAI API key
- (Optional) Tavily API key for web search

## License

MIT
