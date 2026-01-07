#!/usr/bin/env python3
"""
Agent Client - Main Entrypoint

This is the ONLY executable file in the agent layer.
All other modules (planner, executor, memory) are imported here.

The client orchestrates:
1. User interaction
2. LLM communication
3. Plan creation and execution
4. MCP tool invocation
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

from .memory import Memory, MessageRole
from .planner import Planner
from .executor import Executor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000")

SYSTEM_PROMPT = """You are a research assistant with access to various tools for:
- PDF document analysis and extraction
- arXiv paper search and download
- Web search and research

When the user asks you to do something:
1. Analyze what tools you need
2. Call the appropriate tools
3. Synthesize the results into a helpful response

Be concise and informative. When analyzing documents, provide key insights.
When searching, summarize the most relevant findings.

Available tool categories:
- PDF: list_pdfs, extract_text, extract_images, extract_all, process_all_pdfs, get_pdf_info, read_extracted_text, check_github_link
- arXiv: arxiv_search, arxiv_get_paper, arxiv_download
- Web: web_search, web_get_content, web_research
- Ranking: update_user_profile, apply_hard_filters, calculate_semantic_scores, evaluate_paper_metrics, rank_and_select_top_k
"""


class AgentClient:
    """
    Main Agent Client.

    Orchestrates all agent components:
    - Memory for state management
    - Planner for creating execution plans
    - Executor for running plans via MCP
    - LLM for reasoning and responses
    """

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.mcp_url = MCP_SERVER_URL

        # Initialize components
        self.memory = Memory(system_prompt=SYSTEM_PROMPT)
        self.tools_schema = self._fetch_tools_schema()
        self.planner = Planner(self.tools_schema)
        self.executor = Executor(
            tool_caller=self._call_mcp_tool,
            memory=self.memory
        )

    def _fetch_tools_schema(self) -> List[Dict]:
        """Fetch available tools from MCP server."""
        try:
            response = requests.get(
                f"{self.mcp_url}/tools/schema",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Loaded {len(data.get('tools', []))} tools from MCP server")
                return data.get("tools", [])
            else:
                logger.warning(f"Failed to fetch tools: {response.status_code}")
                return []
        except requests.exceptions.ConnectionError:
            logger.warning(f"Cannot connect to MCP server at {self.mcp_url}")
            return []
        except Exception as e:
            logger.error(f"Error fetching tools: {e}")
            return []

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call an MCP tool via HTTP API."""
        try:
            response = requests.post(
                f"{self.mcp_url}/tools/{tool_name}/execute",
                json={"arguments": arguments},
                timeout=120
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Cannot connect to MCP server at {self.mcp_url}"
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _call_llm(
        self,
        messages: List[Dict],
        tools: List[Dict] = None
    ) -> Any:
        """Call the LLM with messages and optional tools."""
        kwargs = {
            "model": self.model,
            "messages": messages
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.openai.chat.completions.create(**kwargs)
        return response.choices[0].message

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.

        This is the main entry point for conversation.
        """
        self.memory.add_user_message(user_message)

        messages = self.memory.get_context_messages()

        while True:
            # Call LLM with tools
            response = self._call_llm(messages, self.tools_schema)

            # Check for tool calls
            if response.tool_calls:
                # Add assistant's response to context
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response.tool_calls
                    ]
                })

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    print(f"  [Calling {tool_name}...]")

                    result = await self._call_mcp_tool(tool_name, arguments)

                    # Add tool result to context
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

            else:
                # No more tool calls - return the response
                final_response = response.content or ""
                self.memory.add_assistant_message(final_response)
                return final_response

    def reset(self) -> None:
        """Reset the conversation."""
        self.memory.reset(keep_system=True)

    def check_mcp_server(self) -> bool:
        """Check if MCP server is available."""
        try:
            response = requests.get(f"{self.mcp_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


def interactive_mode():
    """Run the agent in interactive mode."""
    print("=" * 60)
    print("Research Agent")
    print("=" * 60)
    print(f"Model: {OPENAI_MODEL}")
    print(f"MCP Server: {MCP_SERVER_URL}")
    print("-" * 60)
    print("Commands:")
    print("  /quit  - Exit")
    print("  /reset - Reset conversation")
    print("  /tools - List available tools")
    print("-" * 60)

    client = AgentClient()

    # Check MCP connection
    print("Connecting to MCP server...", end=" ")
    if client.check_mcp_server():
        print("OK")
        print(f"Loaded {len(client.tools_schema)} tools")
    else:
        print("FAILED")
        print(f"Warning: Cannot connect to {MCP_SERVER_URL}")
    print()

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

            if user_input.lower() == "/tools":
                if client.tools_schema:
                    print("Available tools:")
                    for tool in client.tools_schema:
                        name = tool["function"]["name"]
                        desc = tool["function"]["description"][:50]
                        print(f"  - {name}: {desc}...")
                else:
                    print("No tools loaded. Is MCP server running?")
                continue

            # Process the message
            response = asyncio.run(client.chat(user_input))
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            logger.exception("Error in chat")


def single_command(command: str):
    """Run a single command and exit."""
    import time

    client = AgentClient()

    # Wait for MCP server
    print("Waiting for MCP server...", end=" ", flush=True)
    for _ in range(30):
        if client.check_mcp_server():
            print("OK")
            break
        time.sleep(1)
    else:
        print("TIMEOUT")
        print("Warning: MCP server not responding")

    response = asyncio.run(client.chat(command))
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
