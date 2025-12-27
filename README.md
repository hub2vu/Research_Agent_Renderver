# PDF Extraction MCP Server

Docker-based MCP (Model Context Protocol) server for extracting text and images from PDF files.

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

## Requirements

- Docker
- (Optional) Claude Code for MCP integration

## License

MIT
