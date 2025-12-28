#!/bin/bash
# Run MCP server standalone

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Starting MCP server..."
docker compose up -d mcp-server

echo "MCP server is running at http://localhost:8000"
echo "Use 'docker compose logs -f mcp-server' to view logs"
