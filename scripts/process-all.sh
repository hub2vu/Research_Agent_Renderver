#!/bin/bash
# Process all PDFs in the pdf folder without MCP (direct extraction)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Processing all PDFs in $PROJECT_DIR/pdf..."
cd "$PROJECT_DIR"

# Check if pdf folder has files
PDF_COUNT=$(find ./pdf -name "*.pdf" 2>/dev/null | wc -l)

if [ "$PDF_COUNT" -eq 0 ]; then
    echo "No PDF files found in ./pdf folder"
    echo "Please add PDF files to the ./pdf folder first"
    exit 1
fi

echo "Found $PDF_COUNT PDF file(s)"

# Run extraction using Docker
docker run --rm \
    -v "$PROJECT_DIR/pdf:/data/pdf:ro" \
    -v "$PROJECT_DIR/output:/data/output" \
    pdf-extraction-mcp \
    python -c "
import json
from pathlib import Path
from server import extract_all_from_pdf, list_pdfs, PDF_DIR, OUTPUT_DIR

pdfs = list_pdfs(PDF_DIR)
print(f'Processing {len(pdfs)} PDF(s)...')

for pdf_info in pdfs:
    pdf_path = Path(pdf_info['path'])
    print(f'  Processing: {pdf_path.name}')
    try:
        result = extract_all_from_pdf(pdf_path, OUTPUT_DIR)
        print(f'    ✓ Text: {result[\"text_extraction\"][\"total_pages\"]} pages')
        print(f'    ✓ Images: {result[\"image_extraction\"][\"total_images\"]} images')
    except Exception as e:
        print(f'    ✗ Error: {e}')

print('Done!')
"

echo ""
echo "Output saved to: $PROJECT_DIR/output/"
