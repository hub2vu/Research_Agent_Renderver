#!/usr/bin/env python3
"""
ChatGPT Client for PDF Extraction

Uses OpenAI's function calling to interact with PDF extraction tools.
Loads API key from .env file.
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
PROJECT_DIR = Path(__file__).parent.absolute()
PDF_DIR = PROJECT_DIR / "pdf"
OUTPUT_DIR = PROJECT_DIR / "output"

# Ensure directories exist
PDF_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


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


def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Call a tool using Docker MCP server."""

    # Build MCP request
    mcp_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    try:
        # Run Docker container with MCP request
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


def read_extracted_text(filename: str) -> str:
    """Read extracted text from output directory."""
    # Remove .pdf extension if present
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

    # Handle local tools
    if tool_name == "read_extracted_text":
        return read_extracted_text(arguments.get("filename", ""))

    # Handle MCP tools via Docker
    return call_mcp_tool(tool_name, arguments)


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

            # Check if we need to call tools
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    print(f"  [Calling {function_name}...]")

                    # Execute the tool
                    result = execute_tool(function_name, arguments)

                    # Add tool result to messages
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                # No more tool calls, return the response
                return assistant_message.content

    def reset(self):
        """Reset conversation history."""
        self.messages = self.messages[:1]  # Keep only system message


def check_docker_image():
    """Check if Docker image exists."""
    result = subprocess.run(
        ["docker", "images", "-q", "pdf-extraction-mcp"],
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())


def interactive_mode():
    """Run in interactive chat mode."""
    print("=" * 60)
    print("PDF Extraction ChatGPT Client")
    print("=" * 60)
    print(f"Model: {OPENAI_MODEL}")
    print(f"PDF folder: {PDF_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("-" * 60)
    print("Commands:")
    print("  /quit - Exit the program")
    print("  /reset - Reset conversation")
    print("  /pdfs - List PDF files")
    print("-" * 60)
    print()

    # Check Docker image
    if not check_docker_image():
        print("Warning: Docker image 'pdf-extraction-mcp' not found.")
        print("Run './scripts/build.sh' first to build the image.")
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
                    print("No PDF files found in ./pdf folder")
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
        print("Error: OPENAI_API_KEY not found in .env file")
        print("Create a .env file with your OpenAI API key:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    if len(sys.argv) > 1:
        # Single command mode
        command = " ".join(sys.argv[1:])
        single_command(command)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
