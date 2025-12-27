#!/usr/bin/env python3
"""
HTTP API Server for PDF Extraction
Wraps MCP tools as REST API endpoints.
"""

import os
import json
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuration
PDF_DIR = Path(os.environ.get("PDF_DIR", "/data/pdf"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/data/output"))

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PDF Extraction API", version="1.0.0")


class FilenameRequest(BaseModel):
    filename: str


# ============== Helper Functions ==============

def list_pdfs() -> dict:
    """List all PDF files in the directory."""
    pdfs = []
    for pdf_file in PDF_DIR.glob("*.pdf"):
        stat = pdf_file.stat()
        pdfs.append({
            "filename": pdf_file.name,
            "path": str(pdf_file),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2)
        })
    return {
        "pdf_directory": str(PDF_DIR),
        "total_files": len(pdfs),
        "files": sorted(pdfs, key=lambda x: x["filename"])
    }


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


def extract_images_from_pdf(pdf_path: Path) -> dict:
    """Extract all images from a PDF file."""
    doc = fitz.open(pdf_path)
    pdf_name = pdf_path.stem
    image_output_dir = OUTPUT_DIR / pdf_name / "images"
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


def extract_all_from_pdf(pdf_path: Path) -> dict:
    """Extract both text and images from a PDF file."""
    pdf_name = pdf_path.stem
    pdf_output_dir = OUTPUT_DIR / pdf_name
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
    image_result = extract_images_from_pdf(pdf_path)

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


def get_pdf_info(pdf_path: Path) -> dict:
    """Get metadata and information about a specific PDF file."""
    doc = fitz.open(pdf_path)
    metadata = doc.metadata

    info = {
        "filename": pdf_path.name,
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
    return info


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "PDF Extraction API",
        "version": "1.0.0",
        "endpoints": [
            "/list_pdfs",
            "/extract_text",
            "/extract_images",
            "/extract_all",
            "/process_all_pdfs",
            "/get_pdf_info",
            "/read_extracted_text"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/list_pdfs")
async def api_list_pdfs():
    """List all PDF files."""
    return list_pdfs()


@app.post("/extract_text")
async def api_extract_text(request: FilenameRequest):
    """Extract text from a PDF file."""
    pdf_path = PDF_DIR / request.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.filename}")
    return extract_text_from_pdf(pdf_path)


@app.post("/extract_images")
async def api_extract_images(request: FilenameRequest):
    """Extract images from a PDF file."""
    pdf_path = PDF_DIR / request.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.filename}")
    return extract_images_from_pdf(pdf_path)


@app.post("/extract_all")
async def api_extract_all(request: FilenameRequest):
    """Extract text and images from a PDF file."""
    pdf_path = PDF_DIR / request.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.filename}")
    return extract_all_from_pdf(pdf_path)


@app.get("/process_all_pdfs")
async def api_process_all_pdfs():
    """Process all PDF files."""
    pdfs_info = list_pdfs()
    results = []

    for pdf_info in pdfs_info["files"]:
        pdf_path = Path(pdf_info["path"])
        try:
            result = extract_all_from_pdf(pdf_path)
            result["status"] = "success"
            results.append(result)
        except Exception as e:
            results.append({
                "filename": pdf_info["filename"],
                "status": "error",
                "error": str(e)
            })

    return {
        "total_processed": len(results),
        "successful": len([r for r in results if r.get("status") == "success"]),
        "failed": len([r for r in results if r.get("status") == "error"]),
        "results": results
    }


@app.post("/get_pdf_info")
async def api_get_pdf_info(request: FilenameRequest):
    """Get PDF metadata."""
    pdf_path = PDF_DIR / request.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.filename}")
    return get_pdf_info(pdf_path)


@app.post("/read_extracted_text")
async def api_read_extracted_text(request: FilenameRequest):
    """Read extracted text from output directory."""
    pdf_name = request.filename.replace(".pdf", "")
    text_file = OUTPUT_DIR / pdf_name / "extracted_text.txt"

    if not text_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No extracted text found for '{pdf_name}'. Run extract_all first."
        )

    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Truncate if too long
    max_chars = 50000
    truncated = False
    if len(content) > max_chars:
        original_len = len(content)
        content = content[:max_chars]
        truncated = True

    result = {
        "filename": pdf_name,
        "text_file": str(text_file),
        "content": content
    }

    if truncated:
        result["truncated"] = True
        result["original_length"] = original_len

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
