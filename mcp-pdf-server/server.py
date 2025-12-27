#!/usr/bin/env python3
"""
MCP PDF Extraction Server
Extracts text and images from PDF files using PyMuPDF.
"""

import os
import json
import base64
import asyncio
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
)

# Configuration from environment
PDF_DIR = Path(os.environ.get("PDF_DIR", "/data/pdf"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/output"))

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create MCP server
server = Server("pdf-extraction-server")


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    result = {
        "filename": pdf_path.name,
        "total_pages": len(doc),
        "pages": []
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        result["pages"].append({
            "page_number": page_num + 1,
            "text": text
        })

    doc.close()
    return result


def extract_images_from_pdf(pdf_path: Path, output_dir: Path) -> dict:
    """Extract all images from a PDF file."""
    doc = fitz.open(pdf_path)
    pdf_name = pdf_path.stem
    image_output_dir = output_dir / pdf_name / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "filename": pdf_path.name,
        "total_pages": len(doc),
        "images": []
    }

    image_count = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                image_path = image_output_dir / image_filename

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                result["images"].append({
                    "page_number": page_num + 1,
                    "image_index": img_index + 1,
                    "filename": image_filename,
                    "path": str(image_path),
                    "format": image_ext,
                    "size_bytes": len(image_bytes)
                })
                image_count += 1
            except Exception as e:
                result["images"].append({
                    "page_number": page_num + 1,
                    "image_index": img_index + 1,
                    "error": str(e)
                })

    doc.close()
    result["total_images_extracted"] = image_count
    return result


def extract_all_from_pdf(pdf_path: Path, output_dir: Path) -> dict:
    """Extract both text and images from a PDF file."""
    pdf_name = pdf_path.stem
    pdf_output_dir = output_dir / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract text
    text_result = extract_text_from_pdf(pdf_path)

    # Save text to file
    text_output_path = pdf_output_dir / "extracted_text.json"
    with open(text_output_path, "w", encoding="utf-8") as f:
        json.dump(text_result, f, ensure_ascii=False, indent=2)

    # Also save as plain text
    plain_text_path = pdf_output_dir / "extracted_text.txt"
    with open(plain_text_path, "w", encoding="utf-8") as f:
        for page in text_result["pages"]:
            f.write(f"=== Page {page['page_number']} ===\n")
            f.write(page["text"])
            f.write("\n\n")

    # Extract images
    image_result = extract_images_from_pdf(pdf_path, output_dir)

    # Save image metadata
    image_meta_path = pdf_output_dir / "image_metadata.json"
    with open(image_meta_path, "w", encoding="utf-8") as f:
        json.dump(image_result, f, ensure_ascii=False, indent=2)

    return {
        "filename": pdf_path.name,
        "output_directory": str(pdf_output_dir),
        "text_extraction": {
            "total_pages": text_result["total_pages"],
            "text_file": str(plain_text_path),
            "json_file": str(text_output_path)
        },
        "image_extraction": {
            "total_images": image_result["total_images_extracted"],
            "metadata_file": str(image_meta_path),
            "images_directory": str(pdf_output_dir / "images")
        }
    }


def list_pdfs(pdf_dir: Path) -> list:
    """List all PDF files in the directory."""
    pdfs = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        stat = pdf_file.stat()
        pdfs.append({
            "filename": pdf_file.name,
            "path": str(pdf_file),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2)
        })
    return sorted(pdfs, key=lambda x: x["filename"])


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available PDF extraction tools."""
    return [
        Tool(
            name="list_pdfs",
            description="List all PDF files in the /data/pdf directory",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="extract_text",
            description="Extract text from a specific PDF file",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the PDF file (e.g., 'paper.pdf')"
                    }
                },
                "required": ["filename"]
            }
        ),
        Tool(
            name="extract_images",
            description="Extract all images from a specific PDF file and save them to output directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the PDF file (e.g., 'paper.pdf')"
                    }
                },
                "required": ["filename"]
            }
        ),
        Tool(
            name="extract_all",
            description="Extract both text and images from a specific PDF file",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the PDF file (e.g., 'paper.pdf')"
                    }
                },
                "required": ["filename"]
            }
        ),
        Tool(
            name="process_all_pdfs",
            description="Process all PDF files in the /data/pdf directory, extracting text and images from each",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_pdf_info",
            description="Get metadata and information about a specific PDF file",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the PDF file (e.g., 'paper.pdf')"
                    }
                },
                "required": ["filename"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Handle tool calls."""

    if name == "list_pdfs":
        pdfs = list_pdfs(PDF_DIR)
        return [TextContent(
            type="text",
            text=json.dumps({
                "pdf_directory": str(PDF_DIR),
                "total_files": len(pdfs),
                "files": pdfs
            }, indent=2)
        )]

    elif name == "extract_text":
        filename = arguments.get("filename")
        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"File not found: {filename}"})
            )]

        result = extract_text_from_pdf(pdf_path)
        return [TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2)
        )]

    elif name == "extract_images":
        filename = arguments.get("filename")
        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"File not found: {filename}"})
            )]

        result = extract_images_from_pdf(pdf_path, OUTPUT_DIR)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    elif name == "extract_all":
        filename = arguments.get("filename")
        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"File not found: {filename}"})
            )]

        result = extract_all_from_pdf(pdf_path, OUTPUT_DIR)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    elif name == "process_all_pdfs":
        pdfs = list_pdfs(PDF_DIR)
        results = []

        for pdf_info in pdfs:
            pdf_path = Path(pdf_info["path"])
            try:
                result = extract_all_from_pdf(pdf_path, OUTPUT_DIR)
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": pdf_info["filename"],
                    "status": "error",
                    "error": str(e)
                })

        return [TextContent(
            type="text",
            text=json.dumps({
                "total_processed": len(results),
                "successful": len([r for r in results if r.get("status") == "success"]),
                "failed": len([r for r in results if r.get("status") == "error"]),
                "results": results
            }, indent=2)
        )]

    elif name == "get_pdf_info":
        filename = arguments.get("filename")
        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"File not found: {filename}"})
            )]

        doc = fitz.open(pdf_path)
        metadata = doc.metadata

        info = {
            "filename": filename,
            "path": str(pdf_path),
            "total_pages": len(doc),
            "metadata": metadata,
            "file_size_bytes": pdf_path.stat().st_size,
            "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2)
        }

        # Count images per page
        image_counts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            image_counts.append({
                "page": page_num + 1,
                "image_count": len(images)
            })

        info["images_per_page"] = image_counts
        info["total_images"] = sum(ic["image_count"] for ic in image_counts)

        doc.close()

        return [TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]

    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"})
        )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
