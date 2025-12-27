# PDF Extraction MCP Server

Docker-based MCP (Model Context Protocol) server for extracting text and images from PDF files.
Includes ChatGPT API integration for intelligent document analysis.

## Quick Start

### 1. Build the Docker Image

```bash
./scripts/build.sh
```

Or manually:

```bash
docker build -t pdf-extraction-mcp ./mcp-pdf-server
```

### 2. Add PDF Files

Place your PDF files in the `./pdf` folder:

```bash
cp /path/to/your/papers/*.pdf ./pdf/
```

### 3. Process All PDFs

**Option A: Direct Processing (without MCP)**

```bash
./scripts/process-all.sh
```

**Option B: Using Claude Code with MCP**

The MCP server is configured in `.claude/mcp.json`. When using Claude Code, the PDF extraction tools will be available automatically.

**Option C: Using ChatGPT API**

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run interactive chat
python chatgpt_client.py

# Or run a single command
python chatgpt_client.py "Process all PDFs and summarize the content"
```

## Directory Structure

```
.
├── pdf/                    # Put your PDF files here
├── output/                 # Extracted content will be saved here
│   └── <pdf-name>/
│       ├── extracted_text.txt
│       ├── extracted_text.json
│       ├── image_metadata.json
│       └── images/
│           ├── page1_img1.png
│           └── ...
├── mcp-pdf-server/         # MCP server source code
│   ├── Dockerfile
│   ├── requirements.txt
│   └── server.py
├── scripts/
│   ├── build.sh           # Build Docker image
│   ├── run-mcp.sh         # Run MCP server
│   └── process-all.sh     # Process all PDFs
├── chatgpt_client.py       # ChatGPT API client
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── docker-compose.yml
└── .claude/
    └── mcp.json           # Claude Code MCP configuration
```

## MCP Tools Available

| Tool | Description |
|------|-------------|
| `list_pdfs` | List all PDF files in the /data/pdf directory |
| `extract_text` | Extract text from a specific PDF file |
| `extract_images` | Extract all images from a specific PDF file |
| `extract_all` | Extract both text and images from a specific PDF |
| `process_all_pdfs` | Process all PDFs in the directory |
| `get_pdf_info` | Get metadata and info about a specific PDF |

## Using with Claude Code

1. Build the Docker image first:
   ```bash
   ./scripts/build.sh
   ```

2. The MCP server is auto-configured via `.claude/mcp.json`

3. In Claude Code, you can use commands like:
   - "List all PDFs in the folder"
   - "Extract text from paper.pdf"
   - "Process all PDF papers and extract images"

## Manual MCP Server Usage

Run the MCP server directly:

```bash
./scripts/run-mcp.sh
```

Or with docker-compose:

```bash
docker-compose up pdf-extraction-mcp
```

## Output Format

### Text Extraction

`extracted_text.txt`:
```
=== Page 1 ===
[Text content from page 1]

=== Page 2 ===
[Text content from page 2]
...
```

`extracted_text.json`:
```json
{
  "filename": "paper.pdf",
  "total_pages": 10,
  "pages": [
    {"page_number": 1, "text": "..."},
    {"page_number": 2, "text": "..."}
  ]
}
```

### Image Extraction

Images are saved as `page{N}_img{M}.{ext}` in the `images/` subdirectory.

`image_metadata.json`:
```json
{
  "filename": "paper.pdf",
  "total_pages": 10,
  "total_images_extracted": 15,
  "images": [
    {
      "page_number": 1,
      "image_index": 1,
      "filename": "page1_img1.png",
      "format": "png",
      "size_bytes": 12345
    }
  ]
}
```

## Using ChatGPT API Client

The ChatGPT client uses OpenAI's function calling to interact with PDF tools.

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API key:
   ```bash
   cp .env.example .env
   # Edit .env file:
   # OPENAI_API_KEY=your-key-here
   # OPENAI_MODEL=gpt-4o  (optional, default: gpt-4o)
   ```

### Interactive Mode

```bash
python chatgpt_client.py
```

Commands in interactive mode:
- `/quit` - Exit the program
- `/reset` - Reset conversation history
- `/pdfs` - List PDF files

### Single Command Mode

```bash
# Process and analyze
python chatgpt_client.py "List all PDFs and show their info"
python chatgpt_client.py "Extract text from paper.pdf and summarize it"
python chatgpt_client.py "Process all PDFs and create a summary of each"
```

### Example Session

```
You: What PDFs do I have?
  [Calling list_pdfs...]
Assistant: You have 3 PDF files: paper1.pdf, paper2.pdf, paper3.pdf

You: Extract and summarize paper1.pdf
  [Calling extract_all...]
  [Calling read_extracted_text...]
Assistant: Here's a summary of paper1.pdf...
```

## Requirements

- Docker
- Python 3.8+ (for ChatGPT client)
- OpenAI API key (for ChatGPT client)
- (Optional) Claude Code for MCP integration

## License

MIT
