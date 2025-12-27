#!/usr/bin/env python3
"""
ChatGPT Client for PDF Extraction

Uses OpenAI's function calling to interact with PDF extraction tools.
Loads API key from .env file.
Supports both Docker and native execution modes.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Support both Docker (/data) and local paths
if os.path.exists("/data/pdf"):
    # Running in Docker
    PDF_DIR = Path("/data/pdf")
    OUTPUT_DIR = Path("/data/output")
    USE_DOCKER = False  # Already in Docker, use native extraction
else:
    # Running locally
    PROJECT_DIR = Path(__file__).parent.absolute()
    PDF_DIR = Path(os.getenv("PDF_DIR", PROJECT_DIR / "pdf"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_DIR / "output"))
    USE_DOCKER = os.getenv("USE_DOCKER", "false").lower() == "true"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try to import PyMuPDF for native extraction
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def get_pdf_tools():
    """Define the tools available for ChatGPT function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_pdfs",
                "description": "List all PDF files in the pdf folder",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_text",
                "description": "Extract text from a specific PDF file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the PDF file (e.g., 'paper.pdf')"
                        }
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_images",
                "description": "Extract all images from a specific PDF file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the PDF file (e.g., 'paper.pdf')"
                        }
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_all",
                "description": "Extract both text and images from a specific PDF file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the PDF file (e.g., 'paper.pdf')"
                        }
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "process_all_pdfs",
                "description": "Process all PDF files in the folder, extracting text and images from each",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_pdf_info",
                "description": "Get metadata and information about a specific PDF file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the PDF file (e.g., 'paper.pdf')"
                        }
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_extracted_text",
                "description": "Read the extracted text from a previously processed PDF",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the PDF file (without .pdf extension)"
                        }
                    },
                    "required": ["filename"]
                }
            }
        }
    ]


# ============== Native PDF Extraction Functions ==============

def native_list_pdfs() -> dict:
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


def native_extract_text(pdf_path: Path) -> dict:
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


def native_extract_images(pdf_path: Path) -> dict:
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


def native_extract_all(pdf_path: Path) -> dict:
    """Extract both text and images from a PDF file."""
    pdf_name = pdf_path.stem
    pdf_output_dir = OUTPUT_DIR / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract text
    text_result = native_extract_text(pdf_path)

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
    image_result = native_extract_images(pdf_path)

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


def native_get_pdf_info(pdf_path: Path) -> dict:
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


def native_process_all_pdfs() -> dict:
    """Process all PDF files in the directory."""
    pdfs_info = native_list_pdfs()
    results = []

    for pdf_info in pdfs_info["files"]:
        pdf_path = Path(pdf_info["path"])
        try:
            result = native_extract_all(pdf_path)
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


# ============== Docker-based Extraction ==============

def call_docker_tool(tool_name: str, arguments: dict) -> str:
    """Call a tool using Docker MCP server."""
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm", "-i",
                "-v", f"{PDF_DIR}:/data/pdf:ro",
                "-v", f"{OUTPUT_DIR}:/data/output",
                "pdf-extraction-mcp",
                "python", "-c", f'''
import json
import asyncio
from server import call_tool

async def main():
    result = await call_tool("{tool_name}", {json.dumps(arguments)})
    for item in result:
        if hasattr(item, "text"):
            print(item.text)

asyncio.run(main())
'''
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return json.dumps({"error": result.stderr or "Unknown error"})

        return result.stdout.strip() or json.dumps({"status": "completed"})

    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Operation timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============== Tool Execution ==============

def read_extracted_text(filename: str) -> str:
    """Read extracted text from output directory."""
    pdf_name = filename.replace(".pdf", "")
    text_file = OUTPUT_DIR / pdf_name / "extracted_text.txt"

    if not text_file.exists():
        return json.dumps({"error": f"No extracted text found for '{pdf_name}'. Run extract_all first."})

    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Truncate if too long
    max_chars = 50000
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n... [Truncated, {len(content) - max_chars} more characters]"

    return json.dumps({
        "filename": pdf_name,
        "text_file": str(text_file),
        "content": content
    })


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""

    # Handle read_extracted_text locally always
    if tool_name == "read_extracted_text":
        return read_extracted_text(arguments.get("filename", ""))

    # Use native extraction if PyMuPDF is available and not forcing Docker
    if HAS_PYMUPDF and not USE_DOCKER:
        try:
            if tool_name == "list_pdfs":
                return json.dumps(native_list_pdfs(), indent=2)

            elif tool_name == "extract_text":
                filename = arguments.get("filename")
                pdf_path = PDF_DIR / filename
                if not pdf_path.exists():
                    return json.dumps({"error": f"File not found: {filename}"})
                return json.dumps(native_extract_text(pdf_path), ensure_ascii=False, indent=2)

            elif tool_name == "extract_images":
                filename = arguments.get("filename")
                pdf_path = PDF_DIR / filename
                if not pdf_path.exists():
                    return json.dumps({"error": f"File not found: {filename}"})
                return json.dumps(native_extract_images(pdf_path), indent=2)

            elif tool_name == "extract_all":
                filename = arguments.get("filename")
                pdf_path = PDF_DIR / filename
                if not pdf_path.exists():
                    return json.dumps({"error": f"File not found: {filename}"})
                return json.dumps(native_extract_all(pdf_path), indent=2)

            elif tool_name == "process_all_pdfs":
                return json.dumps(native_process_all_pdfs(), indent=2)

            elif tool_name == "get_pdf_info":
                filename = arguments.get("filename")
                pdf_path = PDF_DIR / filename
                if not pdf_path.exists():
                    return json.dumps({"error": f"File not found: {filename}"})
                return json.dumps(native_get_pdf_info(pdf_path), indent=2)

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    # Fallback to Docker
    return call_docker_tool(tool_name, arguments)


class ChatGPTClient:
    """ChatGPT client with PDF extraction tool support."""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.tools = get_pdf_tools()
        self.messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant for PDF document analysis.
You can extract text and images from PDF files using the available tools.

Available tools:
- list_pdfs: List all PDF files in the folder
- extract_text: Extract text from a PDF
- extract_images: Extract images from a PDF
- extract_all: Extract both text and images
- process_all_pdfs: Process all PDFs at once
- get_pdf_info: Get PDF metadata
- read_extracted_text: Read previously extracted text

When asked to analyze PDFs:
1. First list available PDFs with list_pdfs
2. Extract content with extract_all or process_all_pdfs
3. Use read_extracted_text to access the content for analysis

Always be helpful and provide clear summaries of the extracted content."""
            }
        ]

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, handling tool calls."""

        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message
            self.messages.append(assistant_message)

            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    print(f"  [Calling {function_name}...]")

                    result = execute_tool(function_name, arguments)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                return assistant_message.content

    def reset(self):
        """Reset conversation history."""
        self.messages = self.messages[:1]


def interactive_mode():
    """Run in interactive chat mode."""
    print("=" * 60)
    print("PDF Extraction ChatGPT Client")
    print("=" * 60)
    print(f"Model: {OPENAI_MODEL}")
    print(f"PDF folder: {PDF_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Extraction mode: {'Native (PyMuPDF)' if HAS_PYMUPDF else 'Docker'}")
    print("-" * 60)
    print("Commands:")
    print("  /quit - Exit the program")
    print("  /reset - Reset conversation")
    print("  /pdfs - List PDF files")
    print("-" * 60)
    print()

    client = ChatGPTClient()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            if user_input.lower() == "/reset":
                client.reset()
                print("Conversation reset.")
                continue

            if user_input.lower() == "/pdfs":
                pdfs = list(PDF_DIR.glob("*.pdf"))
                if pdfs:
                    print(f"PDF files ({len(pdfs)}):")
                    for pdf in pdfs:
                        size_mb = pdf.stat().st_size / (1024 * 1024)
                        print(f"  - {pdf.name} ({size_mb:.2f} MB)")
                else:
                    print("No PDF files found in pdf folder")
                continue

            response = client.chat(user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def single_command(command: str):
    """Run a single command and exit."""
    client = ChatGPTClient()
    response = client.chat(command)
    print(response)


def main():
    """Main entry point."""
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found")
        print("Set OPENAI_API_KEY in .env file or environment variable")
        sys.exit(1)

    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
        single_command(command)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
