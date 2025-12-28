#!/bin/bash
# Run both MCP server and Agent to process all PDFs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check for PDFs
PDF_COUNT=$(find ./pdf -name "*.pdf" 2>/dev/null | wc -l)

if [ "$PDF_COUNT" -eq 0 ]; then
    echo "No PDF files found in ./pdf folder"
    echo "Please add PDF files first"
    exit 1
fi

echo "Found $PDF_COUNT PDF file(s)"
echo ""

# Start MCP server if not running
if ! docker compose ps mcp-server | grep -q "running"; then
    echo "Starting MCP server..."
    docker compose up -d mcp-server
    echo "Waiting for server to be ready..."
    sleep 5
fi

# Run agent with command
echo "Processing all PDFs..."
docker compose run --rm agent "Process all PDFs in the folder and summarize each one"
