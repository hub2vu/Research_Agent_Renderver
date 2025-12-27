#!/usr/bin/env python3
"""
ChatGPT Client for PDF Extraction

Uses OpenAI's function calling to interact with PDF extraction API.
Loads API key from .env file.
"""

import os
import json
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
PDF_SERVER_URL = os.getenv("PDF_SERVER_URL", "http://pdf-server:8000")

# Local paths for /pdfs command
PDF_DIR = Path(os.getenv("PDF_DIR", "/data/pdf"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))


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


def call_pdf_api(endpoint: str, method: str = "GET", data: dict = None) -> str:
    """Call PDF extraction API."""
    url = f"{PDF_SERVER_URL}/{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=120)
        else:
            response = requests.post(url, json=data, timeout=120)

        if response.status_code == 200:
            return json.dumps(response.json(), indent=2, ensure_ascii=False)
        else:
            return json.dumps({
                "error": f"API error: {response.status_code}",
                "detail": response.text
            })

    except requests.exceptions.ConnectionError:
        return json.dumps({"error": "Cannot connect to PDF server. Is it running?"})
    except requests.exceptions.Timeout:
        return json.dumps({"error": "Request timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool via PDF API."""

    if tool_name == "list_pdfs":
        return call_pdf_api("list_pdfs")

    elif tool_name == "extract_text":
        return call_pdf_api("extract_text", "POST", {"filename": arguments.get("filename")})

    elif tool_name == "extract_images":
        return call_pdf_api("extract_images", "POST", {"filename": arguments.get("filename")})

    elif tool_name == "extract_all":
        return call_pdf_api("extract_all", "POST", {"filename": arguments.get("filename")})

    elif tool_name == "process_all_pdfs":
        return call_pdf_api("process_all_pdfs")

    elif tool_name == "get_pdf_info":
        return call_pdf_api("get_pdf_info", "POST", {"filename": arguments.get("filename")})

    elif tool_name == "read_extracted_text":
        return call_pdf_api("read_extracted_text", "POST", {"filename": arguments.get("filename")})

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


class ChatGPTClient:
    """ChatGPT client with PDF extraction tool support."""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")

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


def check_server():
    """Check if PDF server is available."""
    try:
        response = requests.get(f"{PDF_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def interactive_mode():
    """Run in interactive chat mode."""
    print("=" * 60)
    print("PDF Extraction ChatGPT Client")
    print("=" * 60)
    print(f"Model: {OPENAI_MODEL}")
    print(f"PDF Server: {PDF_SERVER_URL}")
    print("-" * 60)
    print("Commands:")
    print("  /quit  - Exit the program")
    print("  /reset - Reset conversation")
    print("  /pdfs  - List PDF files")
    print("-" * 60)

    # Check server connection
    print("Connecting to PDF server...", end=" ")
    if check_server():
        print("OK")
    else:
        print("FAILED")
        print(f"Warning: Cannot connect to {PDF_SERVER_URL}")
        print("Make sure pdf-server is running.")
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
                if PDF_DIR.exists():
                    pdfs = list(PDF_DIR.glob("*.pdf"))
                    if pdfs:
                        print(f"PDF files ({len(pdfs)}):")
                        for pdf in pdfs:
                            size_mb = pdf.stat().st_size / (1024 * 1024)
                            print(f"  - {pdf.name} ({size_mb:.2f} MB)")
                    else:
                        print("No PDF files found")
                else:
                    print(f"PDF directory not found: {PDF_DIR}")
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
    # Wait for server to be ready
    print("Waiting for PDF server...", end=" ", flush=True)
    import time
    for _ in range(30):
        if check_server():
            print("OK")
            break
        time.sleep(1)
    else:
        print("TIMEOUT")
        print("Warning: PDF server not responding")

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
