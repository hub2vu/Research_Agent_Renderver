#!/bin/bash
# Build the PDF Extraction MCP Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building PDF Extraction MCP Docker image..."
cd "$PROJECT_DIR"

docker build -t pdf-extraction-mcp ./mcp-pdf-server

echo "Build complete!"
echo "Image: pdf-extraction-mcp"
