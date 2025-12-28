# Research Agent

A Docker-based research assistant with MCP (Model Context Protocol) architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent Layer                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ client  │──│ planner  │──│ executor │──│   memory    │  │
│  │ (entry) │  │          │  │          │  │             │  │
│  └────┬────┘  └──────────┘  └─────┬────┘  └─────────────┘  │
│       │         Planning          │         State           │
│       │         (no side-effects) │         Storage         │
└───────┼───────────────────────────┼─────────────────────────┘
        │                           │
        │      HTTP API             │
        ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         MCP Layer                           │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────────────┐   │
│  │ server  │──│ registry │──│         tools/           │   │
│  │ (entry) │  │ (SSOT)   │  │  pdf | arxiv | web_search│   │
│  └─────────┘  └──────────┘  └──────────────────────────┘   │
│                               Side-effects allowed          │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns**
   - Agent Layer: Thinking, planning, decision-making (NO side-effects)
   - MCP Layer: Tool execution, file I/O, API calls (side-effects allowed)

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
│     ├─ refer.py                # reference extraction tools
│     ├─ arxiv.py                # arXiv API tools
│     └─ web_search.py           # Web search tools (Tavily)
│
├─ agent/                        # Agent Layer (no side-effects)
│  ├─ __init__.py
│  ├─ client.py                  # Agent entrypoint (orchestration)
│  ├─ planner.py                 # Goal → plan creation
│  ├─ executor.py                # Plan → MCP calls
│  └─ memory.py                  # State storage
│
├─ requirements/
│  ├─ base.txt                   # Common dependencies
│  ├─ mcp.txt                    # MCP server dependencies
│  └─ agent.txt                  # Agent dependencies
│
├─ docker/
│  ├─ Dockerfile.mcp             # MCP server image
│  └─ Dockerfile.agent           # Agent image
│
├─ scripts/
│  ├─ run-mcp.sh                 # Start MCP server
│  └─ process-all.sh             # Process all PDFs
│
├─ pdf/                          # Input PDF files
├─ output/                       # Extracted content
├─ .env.example                  # Environment template
├─ docker-compose.yml            # Service orchestration
└─ README.md
```

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

**Interactive Mode:**
```bash
# Start MCP server (background)
docker compose up -d mcp-server

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

# Convenience endpoints
curl http://localhost:8000/pdf/list
curl "http://localhost:8000/arxiv/search?query=transformer"
```

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
```

## Requirements

- Docker & Docker Compose
- OpenAI API key
- (Optional) Tavily API key for web search

## License

MIT
