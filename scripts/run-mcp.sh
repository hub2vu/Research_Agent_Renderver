#!/bin/bash
# Run the PDF Extraction MCP server interactively

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting PDF Extraction MCP server..."
cd "$PROJECT_DIR"

docker run -i --rm \
    -v "$PROJECT_DIR/pdf:/data/pdf:ro" \
    -v "$PROJECT_DIR/output:/data/output" \
    pdf-extraction-mcp
