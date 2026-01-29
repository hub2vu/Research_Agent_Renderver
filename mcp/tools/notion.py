"""
Notion API Integration MCP Tool

Saves note content to Notion as a page under the configured parent page.
Environment variables required:
  - NOTION_API_TOKEN: Notion Internal Integration Token
  - NOTION_PARENT_PAGE_ID: Parent page ID where new pages will be created
"""

import os
import logging
from typing import Any, List
import httpx

from mcp.base import MCPTool, ToolParameter, ExecutionError

logger = logging.getLogger(__name__)

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


class SaveToNotionTool(MCPTool):
    """
    Save note content to Notion.
    Creates a new page under the configured parent page with the note content.

    Page title format: [{paper_id}] {paper_title}
    """

    @property
    def name(self) -> str:
        return "save_to_notion"

    @property
    def description(self) -> str:
        return (
            "Save paper notes to Notion. Creates a new page under the configured parent page "
            "with the note content formatted as Notion blocks."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="The paper ID (e.g., arxiv ID)",
                required=True,
            ),
            ToolParameter(
                name="paper_title",
                type="string",
                description="The paper title",
                required=True,
            ),
            ToolParameter(
                name="note_content",
                type="string",
                description="The note content in markdown format",
                required=True,
            ),
            ToolParameter(
                name="tags",
                type="array",
                description="Optional list of tags for the note",
                required=False,
                default=[],
                items_type="string",
            ),
            ToolParameter(
                name="updated_at",
                type="string",
                description="Optional timestamp when the note was last updated",
                required=False,
                default=None,
            ),
        ]

    @property
    def category(self) -> str:
        return "notion"

    def _get_notion_credentials(self) -> tuple[str, str]:
        """Get Notion API credentials from environment variables."""
        token = os.environ.get("NOTION_API_TOKEN", "")
        parent_page_id = os.environ.get("NOTION_PARENT_PAGE_ID", "")

        if not token:
            raise ExecutionError(
                "NOTION_API_TOKEN environment variable is not set",
                tool_name=self.name
            )
        if not parent_page_id:
            raise ExecutionError(
                "NOTION_PARENT_PAGE_ID environment variable is not set",
                tool_name=self.name
            )

        # Normalize parent_page_id: remove hyphens if present
        parent_page_id = parent_page_id.replace("-", "")

        return token, parent_page_id

    def _markdown_to_notion_blocks(self, markdown_content: str) -> List[dict]:
        """
        Convert markdown content to Notion blocks.
        Supports: headings, paragraphs, bullet lists, numbered lists, code blocks, quotes.
        """
        blocks = []
        lines = markdown_content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Code block (```)
            if line.strip().startswith("```"):
                language = line.strip()[3:].strip() or "plain text"
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                blocks.append({
                    "object": "block",
                    "type": "code",
                    "code": {
                        "rich_text": [{"type": "text", "text": {"content": "\n".join(code_lines)[:2000]}}],
                        "language": language if language in self._get_supported_languages() else "plain text"
                    }
                })
                i += 1  # Skip closing ```
                continue

            # Heading 1 (#)
            if line.startswith("# "):
                blocks.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": self._parse_inline_markdown(line[2:].strip())
                    }
                })
                i += 1
                continue

            # Heading 2 (##)
            if line.startswith("## "):
                blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": self._parse_inline_markdown(line[3:].strip())
                    }
                })
                i += 1
                continue

            # Heading 3 (###)
            if line.startswith("### "):
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": self._parse_inline_markdown(line[4:].strip())
                    }
                })
                i += 1
                continue

            # Bullet list (- or *)
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                text = line.strip()[2:]
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": self._parse_inline_markdown(text)
                    }
                })
                i += 1
                continue

            # Numbered list (1. 2. etc)
            if line.strip() and line.strip()[0].isdigit():
                parts = line.strip().split(". ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    blocks.append({
                        "object": "block",
                        "type": "numbered_list_item",
                        "numbered_list_item": {
                            "rich_text": self._parse_inline_markdown(parts[1])
                        }
                    })
                    i += 1
                    continue

            # Quote (>)
            if line.startswith("> "):
                blocks.append({
                    "object": "block",
                    "type": "quote",
                    "quote": {
                        "rich_text": self._parse_inline_markdown(line[2:].strip())
                    }
                })
                i += 1
                continue

            # Horizontal rule (--- or ***)
            if line.strip() in ["---", "***", "___"]:
                blocks.append({
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                })
                i += 1
                continue

            # Default: paragraph
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": self._parse_inline_markdown(line.strip())
                }
            })
            i += 1

        return blocks

    def _parse_inline_markdown(self, text: str) -> List[dict]:
        """
        Parse inline markdown (bold, italic, code, links) and return rich_text array.
        Simplified parser - handles basic cases.
        """
        # Truncate text to Notion's limit (2000 chars per rich_text element)
        if len(text) > 2000:
            text = text[:2000]

        # For simplicity, return as plain text with basic formatting detection
        # A full parser would handle **bold**, *italic*, `code`, [links](url) etc.

        rich_text = []

        # Simple approach: return as single text element
        # In production, you'd want a proper markdown inline parser
        if text:
            rich_text.append({
                "type": "text",
                "text": {"content": text}
            })

        return rich_text

    def _get_supported_languages(self) -> set:
        """Return set of Notion-supported code block languages."""
        return {
            "abap", "arduino", "bash", "basic", "c", "clojure", "coffeescript",
            "cpp", "csharp", "css", "dart", "diff", "docker", "elixir", "elm",
            "erlang", "flow", "fortran", "fsharp", "gherkin", "glsl", "go",
            "graphql", "groovy", "haskell", "html", "java", "javascript", "json",
            "julia", "kotlin", "latex", "less", "lisp", "livescript", "lua",
            "makefile", "markdown", "markup", "matlab", "mermaid", "nix",
            "objective-c", "ocaml", "pascal", "perl", "php", "plain text",
            "powershell", "prolog", "protobuf", "python", "r", "reason", "ruby",
            "rust", "sass", "scala", "scheme", "scss", "shell", "sql", "swift",
            "typescript", "vb.net", "verilog", "vhdl", "visual basic", "webassembly",
            "xml", "yaml", "java/c/c++/c#"
        }

    async def execute(self, **kwargs) -> Any:
        """Execute the save to Notion operation."""
        paper_id = kwargs["paper_id"]
        paper_title = kwargs["paper_title"]
        note_content = kwargs["note_content"]
        tags = kwargs.get("tags", []) or []
        updated_at = kwargs.get("updated_at")

        # Get credentials
        token, parent_page_id = self._get_notion_credentials()

        # Build page title: [{paper_id}] {paper_title}
        page_title = f"[{paper_id}] {paper_title}"

        # Convert markdown to Notion blocks
        content_blocks = self._markdown_to_notion_blocks(note_content)

        # Notion has a limit of 100 blocks per request
        # We'll only send the first 100 blocks
        if len(content_blocks) > 100:
            logger.warning(f"Content has {len(content_blocks)} blocks, truncating to 100")
            content_blocks = content_blocks[:100]

        # Build the page creation payload
        page_payload = {
            "parent": {
                "type": "page_id",
                "page_id": parent_page_id
            },
            "properties": {
                "title": {
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": page_title}
                        }
                    ]
                }
            },
            "children": content_blocks
        }

        # Make API request
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_VERSION
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{NOTION_API_BASE}/pages",
                headers=headers,
                json=page_payload
            )

            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("message", error_detail)
                except Exception:
                    pass

                raise ExecutionError(
                    f"Notion API error ({response.status_code}): {error_detail}",
                    tool_name=self.name
                )

            result = response.json()

        # Extract useful info from response
        page_id = result.get("id", "")
        page_url = result.get("url", "")

        logger.info(f"Created Notion page: {page_title} (ID: {page_id})")

        return {
            "success": True,
            "page_id": page_id,
            "page_url": page_url,
            "page_title": page_title,
            "blocks_created": len(content_blocks),
            "message": f"Successfully saved to Notion: {page_title}"
        }


# Export tools for auto-registration
__all__ = ["SaveToNotionTool"]
