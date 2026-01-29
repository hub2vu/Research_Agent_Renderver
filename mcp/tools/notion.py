"""
Notion API Integration MCP Tool

Saves note content to Notion as a page under the configured parent page.
Environment variables required:
  - NOTION_API_TOKEN: Notion Internal Integration Token
  - NOTION_PARENT_PAGE_ID: Parent page ID where new pages will be created
"""

import os
import json
import logging
from typing import Any, Dict, List

import httpx

from mcp.base import MCPTool, ToolParameter, ExecutionError

logger = logging.getLogger(__name__)

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


class SaveToNotionTool(MCPTool):
    """
    Save note content to Notion.
    Creates a new page under the configured parent page with the note content.

    Page structure:
      [paper_id] paper_title (Page)
        └─ Toggle(Note Title)
            ├─ Toggle(메모)    → content blocks
            ├─ Toggle(번역)    → content blocks
            ├─ Toggle(분석)    → content blocks
            └─ Toggle(Prompt)  → content blocks
    """

    @property
    def name(self) -> str:
        return "save_to_notion"

    @property
    def description(self) -> str:
        return (
            "Save paper notes to Notion. Creates a new page under the configured parent page "
            "with the note content formatted as nested toggle blocks."
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
                description="(Legacy) Single note content in markdown format. Ignored if `notes` is provided.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="notes",
                type="array",
                description=("Optional list of notes to save as individual toggles. "
                             "Each item: {title, memo, translation(or translate), analysis, prompt(or qa)} (title required)."),
                required=False,
                default=[],
                items_type="object",
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

    def _toggle_block(self, title: str) -> dict:
        """Create a toggle block with given title."""
        return {
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": self._parse_inline_markdown(title),
            },
        }

    def _chunk_blocks(self, blocks: List[dict], size: int = 100) -> List[List[dict]]:
        """Split blocks into chunks for Notion API limit (100 blocks per request)."""
        return [blocks[i:i + size] for i in range(0, len(blocks), size)]

    async def _create_paper_page(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        parent_page_id: str,
        page_title: str,
    ) -> Dict[str, str]:
        """Create a new Notion page under the parent page."""
        payload = {
            "parent": {"type": "page_id", "page_id": parent_page_id},
            "properties": {
                "title": {"title": [{"type": "text", "text": {"content": page_title}}]}
            },
        }
        resp = await client.post(f"{NOTION_API_BASE}/pages", headers=headers, json=payload)
        if resp.status_code != 200:
            detail = resp.text
            try:
                detail = resp.json().get("message", detail)
            except Exception:
                pass
            raise ExecutionError(f"Notion API error ({resp.status_code}): {detail}", tool_name=self.name)

        data = resp.json()
        return {"id": data.get("id", ""), "url": data.get("url", "")}

    async def _append_children(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        block_id: str,
        children: List[dict],
    ) -> List[dict]:
        """PATCH /blocks/{block_id}/children. Returns created block objects (contain `id`)."""
        if not children:
            return []
        resp = await client.patch(
            f"{NOTION_API_BASE}/blocks/{block_id}/children",
            headers=headers,
            json={"children": children},
        )
        if resp.status_code != 200:
            detail = resp.text
            try:
                detail = resp.json().get("message", detail)
            except Exception:
                pass
            raise ExecutionError(f"Notion API error ({resp.status_code}): {detail}", tool_name=self.name)

        return resp.json().get("results", []) or []

    async def _append_note_toggle_tree(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        page_id: str,
        note: Dict[str, Any],
    ) -> None:
        """Create per-note toggle tree on Notion page.

        Structure:
          page
            └─ Toggle(<note.title>)        ← Note title toggle
                ├─ Toggle(메모)   → blocks
                ├─ Toggle(번역)   → blocks
                ├─ Toggle(분석)   → blocks
                └─ Toggle(Prompt) → blocks
        """
        title = str(note.get("title", "")).strip()
        if not title:
            return

        # 1) Create note title toggle under the page
        created_note = await self._append_children(client, headers, page_id, [self._toggle_block(title)])
        if not created_note:
            return
        note_toggle_id = created_note[0].get("id", "")
        if not note_toggle_id:
            return

        # Accept both translation/translate, prompt/qa keys
        translation_val = note.get("translation", None)
        if translation_val is None:
            translation_val = note.get("translate", None)

        prompt_val = note.get("prompt", None)
        if prompt_val is None:
            prompt_val = note.get("qa", None)

        # Define sections: (label, content)
        sections = [
            ("메모", note.get("memo", None)),
            ("번역", translation_val),
            ("분석", note.get("analysis", None)),
            ("Prompt", prompt_val),
        ]

        for label, content in sections:
            if content is None:
                continue
            text = str(content).strip()
            if not text:
                continue

            # 2) Create section toggle under note title toggle
            created_sec = await self._append_children(client, headers, note_toggle_id, [self._toggle_block(label)])
            if not created_sec:
                continue
            sec_toggle_id = created_sec[0].get("id", "")
            if not sec_toggle_id:
                continue

            # 3) Append content blocks under section toggle (chunked for API limit)
            blocks = self._markdown_to_notion_blocks(text)
            for chunk in self._chunk_blocks(blocks, 100):
                await self._append_children(client, headers, sec_toggle_id, chunk)

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

        rich_text = []

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

    def _extract_note_title_from_markdown(self, content: str) -> tuple[str, str]:
        """Extract title from first H1 heading if present."""
        lines = content.split("\n")
        title = ""
        body_start = 0

        for i, line in enumerate(lines):
            if line.startswith("# "):
                title = line[2:].strip()
                body_start = i + 1
                break

        body = "\n".join(lines[body_start:])
        return title, body

    def _split_sections_from_markdown(self, content: str) -> Dict[str, str]:
        """Split markdown content into sections based on H2 headers."""
        sections = {"memo": "", "translation": "", "analysis": "", "prompt": ""}
        current_section = "memo"
        current_content = []

        for line in content.split("\n"):
            lower_line = line.lower().strip()
            if lower_line.startswith("## "):
                # Save current section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                    current_content = []

                # Detect section type
                header = lower_line[3:].strip()
                if "번역" in header or "translation" in header:
                    current_section = "translation"
                elif "분석" in header or "analysis" in header:
                    current_section = "analysis"
                elif "prompt" in header or "qa" in header or "질문" in header:
                    current_section = "prompt"
                else:
                    current_section = "memo"
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    async def execute(self, **kwargs) -> Any:
        """Execute the save to Notion operation."""
        paper_id = kwargs["paper_id"]
        paper_title = kwargs["paper_title"]
        note_content = kwargs.get("note_content", "") or ""
        notes_raw = kwargs.get("notes", []) or []
        note_title = kwargs.get("note_title")
        memo = kwargs.get("memo")
        translation = kwargs.get("translation")
        analysis = kwargs.get("analysis")
        prompt = kwargs.get("prompt")
        tags = kwargs.get("tags", []) or []
        updated_at = kwargs.get("updated_at")

        # Get credentials
        token, parent_page_id = self._get_notion_credentials()

        # Build page title: [{paper_id}] {paper_title}
        page_title = f"[{paper_id}] {paper_title}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_VERSION,
        }

        # Normalize notes input:
        # - list[dict] is ideal
        # - some clients may send notes as JSON string
        notes: List[Dict[str, Any]] = []
        if isinstance(notes_raw, str) and notes_raw.strip():
            try:
                parsed = json.loads(notes_raw)
                if isinstance(parsed, list):
                    notes = [x for x in parsed if isinstance(x, dict)]
            except Exception:
                notes = []
        elif isinstance(notes_raw, list):
            notes = [x for x in notes_raw if isinstance(x, dict)]

        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1) Create paper page first (empty)
            created_page = await self._create_paper_page(client, headers, parent_page_id, page_title)
            page_id = created_page.get("id", "")
            page_url = created_page.get("url", "")
            if not page_id:
                raise ExecutionError("Failed to create Notion page (missing id)", tool_name=self.name)

            # 2) Save multiple notes if provided
            if notes:
                for n in notes:
                    await self._append_note_toggle_tree(client, headers, page_id, n)
                notes_saved = len(notes)
            else:
                # Legacy single-note fallback: parse note_content / overrides
                extracted_title, note_body = self._extract_note_title_from_markdown(note_content)
                final_note_title = (note_title or extracted_title or "노트").strip()

                sections = self._split_sections_from_markdown(note_body)
                if memo is not None:
                    sections["memo"] = memo
                if translation is not None:
                    sections["translation"] = translation
                if analysis is not None:
                    sections["analysis"] = analysis
                if prompt is not None:
                    sections["prompt"] = prompt

                legacy_note = {
                    "title": final_note_title,
                    "memo": sections.get("memo", ""),
                    "translation": sections.get("translation", ""),
                    "analysis": sections.get("analysis", ""),
                    "prompt": sections.get("prompt", ""),
                }
                await self._append_note_toggle_tree(client, headers, page_id, legacy_note)
                notes_saved = 1

        logger.info(f"Created Notion page: {page_title} (ID: {page_id})")

        return {
            "success": True,
            "page_id": page_id,
            "page_url": page_url,
            "page_title": page_title,
            "notes_saved": notes_saved,
            "message": f"Successfully saved to Notion: {page_title}",
        }


# Export tools for auto-registration
__all__ = ["SaveToNotionTool"]
