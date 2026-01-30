"""
LLM Logger Module

Comprehensive logging system for LLM usage tracking with support for:
A. Summary generation logs
B. Claim extraction logs
C. Verification logs (Fact Checking Lite)
5. Reference/citation parsing logs
6. Agent decision logs
7. Output/artifact generation logs
"""

import json
import hashlib
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid


# ==============================================================================
# Log Types and Enums
# ==============================================================================

class SummaryType(str, Enum):
    ABSTRACT_LEVEL = "abstract-level"
    SECTION_WISE = "section-wise"
    BULLET = "bullet"
    IMRAD = "IMRAD"
    EXECUTIVE = "executive"
    COMPREHENSIVE = "comprehensive"


class ClaimType(str, Enum):
    METHOD = "method"
    RESULT = "result"
    LIMITATION = "limitation"
    DATASET = "dataset"
    BASELINE = "baseline"
    CONTRIBUTION = "contribution"
    HYPOTHESIS = "hypothesis"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "med"
    LOW = "low"


class LogCategory(str, Enum):
    LLM_CALL = "llm_call"
    SUMMARY = "summary"
    CLAIM_EXTRACTION = "claim_extraction"
    VERIFICATION = "verification"
    REFERENCE = "reference"
    DECISION = "decision"
    ARTIFACT = "artifact"


# ==============================================================================
# Data Classes for Structured Logs
# ==============================================================================

@dataclass
class LLMCallLog:
    """Base LLM call logging."""
    log_id: str
    timestamp: str
    category: str
    model: str
    prompt: str
    response: str
    temperature: float
    token_usage: Dict[str, int]
    latency_ms: float
    tool_name: str
    paper_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceMapping:
    """Maps summary sentences to source chunks."""
    summary_sentence_id: str
    summary_sentence: str
    source_chunks: List[str]  # chunk_id list
    source_pages: List[int]
    confidence: str


@dataclass
class SummaryLog:
    """Summary generation log."""
    log_id: str
    timestamp: str
    paper_id: str
    summary_type: str
    summary_prompt: str
    summary_response: str
    evidence_map: List[Dict]  # List of EvidenceMapping as dicts
    source_text_length: int
    summary_length: int
    compression_ratio: float
    model: str
    temperature: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim:
    """Individual claim extracted from paper."""
    claim_id: str
    claim_text: str
    claim_type: str
    evidence: Dict[str, Any]  # page, chunk_id, snippet
    confidence: str
    source_section: Optional[str] = None


@dataclass
class ClaimExtractionLog:
    """Claim extraction log."""
    log_id: str
    timestamp: str
    paper_id: str
    extraction_prompt: str
    extraction_response: str
    claims: List[Dict]  # List of Claim as dicts
    total_claims: int
    claims_by_type: Dict[str, int]
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationLog:
    """Verification/fact checking log."""
    log_id: str
    timestamp: str
    paper_id: str
    citation_check: Dict[str, Any]  # matched, unmatched, coverage
    number_consistency_check: Dict[str, Any]  # conflicts detected
    hallucination_flags: Dict[str, Any]  # unsupported statements
    verification_prompt: Optional[str] = None
    verification_response: Optional[str] = None
    overall_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceItem:
    """Parsed reference item."""
    ref_id: str
    authors: List[str]
    year: Optional[str]
    title: str
    venue: Optional[str]
    doi: Optional[str]
    parsed_successfully: bool
    raw_text: str


@dataclass
class ReferenceLog:
    """Reference parsing log."""
    log_id: str
    timestamp: str
    paper_id: str
    reference_items: List[Dict]  # List of ReferenceItem as dicts
    total_references: int
    parse_success_rate: float
    dedup_keys: List[str]
    unresolved_refs: List[Dict]
    parsing_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionLog:
    """Agent decision log."""
    log_id: str
    timestamp: str
    decision_id: str
    paper_id: Optional[str]
    context: str
    decision: str
    options: List[str]
    chosen: str
    rationale: str
    evidence_links: List[str]
    agent_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactLog:
    """Output/artifact generation log."""
    log_id: str
    timestamp: str
    artifact_id: str
    artifact_type: str  # summary_txt, notes_json, report_md, etc.
    file_path: str
    file_hash: str
    file_size_bytes: int
    created_at: str
    input_sources: List[Dict]  # chunks, prompts used
    generation_method: str
    paper_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Main Logger Class
# ==============================================================================

class LLMLogger:
    """
    Comprehensive LLM Logger for research agent.

    Logs are saved as JSONL (JSON Lines) format to logs/prompts/*.jsonl
    Each log type has its own file for easy filtering and analysis.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global logger access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: str = "logs/prompts"):
        """Initialize the logger."""
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Log file paths
        self.log_files = {
            LogCategory.LLM_CALL: self.log_dir / "llm_calls.jsonl",
            LogCategory.SUMMARY: self.log_dir / "summaries.jsonl",
            LogCategory.CLAIM_EXTRACTION: self.log_dir / "claims.jsonl",
            LogCategory.VERIFICATION: self.log_dir / "verifications.jsonl",
            LogCategory.REFERENCE: self.log_dir / "references.jsonl",
            LogCategory.DECISION: self.log_dir / "decisions.jsonl",
            LogCategory.ARTIFACT: self.log_dir / "artifacts.jsonl",
        }

        # Thread lock for file writing
        self._write_lock = threading.Lock()
        self._initialized = True

    def _generate_id(self, prefix: str = "log") -> str:
        """Generate unique log ID."""
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _get_timestamp(self) -> str:
        """Get ISO format timestamp."""
        return datetime.now().isoformat()

    def _write_log(self, category: LogCategory, log_data: Dict[str, Any]) -> None:
        """Write log entry to appropriate JSONL file."""
        log_file = self.log_files.get(category)
        if not log_file:
            return

        with self._write_lock:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, default=str)
                    f.write("\n")
            except Exception as e:
                print(f"[LLMLogger] Failed to write log: {e}")

    def _compute_hash(self, content: Union[str, bytes]) -> str:
        """Compute SHA256 hash of content."""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()[:16]

    # ==========================================================================
    # Public Logging Methods
    # ==========================================================================

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tool_name: str,
        temperature: float = 0.0,
        token_usage: Optional[Dict[str, int]] = None,
        latency_ms: float = 0.0,
        paper_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log a basic LLM call.

        Returns:
            log_id for reference
        """
        log = LLMCallLog(
            log_id=self._generate_id("llm"),
            timestamp=self._get_timestamp(),
            category=LogCategory.LLM_CALL.value,
            model=model,
            prompt=prompt,
            response=response,
            temperature=temperature,
            token_usage=token_usage or {},
            latency_ms=latency_ms,
            tool_name=tool_name,
            paper_id=paper_id,
            session_id=self.session_id,
            metadata=metadata or {}
        )

        self._write_log(LogCategory.LLM_CALL, asdict(log))
        return log.log_id

    def log_summary(
        self,
        paper_id: str,
        summary_type: Union[SummaryType, str],
        summary_prompt: str,
        summary_response: str,
        source_text_length: int,
        model: str,
        temperature: float = 0.3,
        evidence_map: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log summary generation.

        Args:
            paper_id: Paper identifier
            summary_type: Type of summary (abstract-level, section-wise, etc.)
            summary_prompt: The prompt used
            summary_response: The generated summary
            source_text_length: Length of source text
            model: Model used
            temperature: Temperature setting
            evidence_map: Mapping of summary sentences to source evidence
            metadata: Additional metadata

        Returns:
            log_id for reference
        """
        summary_length = len(summary_response)
        compression_ratio = source_text_length / summary_length if summary_length > 0 else 0

        log = SummaryLog(
            log_id=self._generate_id("sum"),
            timestamp=self._get_timestamp(),
            paper_id=paper_id,
            summary_type=summary_type.value if isinstance(summary_type, SummaryType) else summary_type,
            summary_prompt=summary_prompt,
            summary_response=summary_response,
            evidence_map=evidence_map or [],
            source_text_length=source_text_length,
            summary_length=summary_length,
            compression_ratio=round(compression_ratio, 2),
            model=model,
            temperature=temperature,
            metadata=metadata or {}
        )

        self._write_log(LogCategory.SUMMARY, asdict(log))
        return log.log_id

    def log_claim_extraction(
        self,
        paper_id: str,
        extraction_prompt: str,
        extraction_response: str,
        claims: List[Dict],
        model: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log claim extraction results.

        Args:
            paper_id: Paper identifier
            extraction_prompt: The prompt used for extraction
            extraction_response: Raw LLM response
            claims: List of extracted claims with structure:
                   {claim_text, claim_type, evidence: {page, chunk_id, snippet}, confidence}
            model: Model used
            metadata: Additional metadata

        Returns:
            log_id for reference
        """
        # Count claims by type
        claims_by_type = {}
        for claim in claims:
            ctype = claim.get("claim_type", "unknown")
            claims_by_type[ctype] = claims_by_type.get(ctype, 0) + 1

        log = ClaimExtractionLog(
            log_id=self._generate_id("claim"),
            timestamp=self._get_timestamp(),
            paper_id=paper_id,
            extraction_prompt=extraction_prompt,
            extraction_response=extraction_response,
            claims=claims,
            total_claims=len(claims),
            claims_by_type=claims_by_type,
            model=model,
            metadata=metadata or {}
        )

        self._write_log(LogCategory.CLAIM_EXTRACTION, asdict(log))
        return log.log_id

    def log_verification(
        self,
        paper_id: str,
        citation_check: Optional[Dict] = None,
        number_consistency_check: Optional[Dict] = None,
        hallucination_flags: Optional[Dict] = None,
        verification_prompt: Optional[str] = None,
        verification_response: Optional[str] = None,
        overall_score: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log verification/fact-checking results.

        Args:
            paper_id: Paper identifier
            citation_check: {matched: int, unmatched: int, coverage: float}
            number_consistency_check: {conflicts: [], consistent: bool}
            hallucination_flags: {unsupported_count: int, unsupported_sentences: []}
            verification_prompt: Optional prompt used
            verification_response: Optional LLM response
            overall_score: Overall verification score (0-1)
            metadata: Additional metadata

        Returns:
            log_id for reference
        """
        log = VerificationLog(
            log_id=self._generate_id("verify"),
            timestamp=self._get_timestamp(),
            paper_id=paper_id,
            citation_check=citation_check or {},
            number_consistency_check=number_consistency_check or {},
            hallucination_flags=hallucination_flags or {},
            verification_prompt=verification_prompt,
            verification_response=verification_response,
            overall_score=overall_score,
            metadata=metadata or {}
        )

        self._write_log(LogCategory.VERIFICATION, asdict(log))
        return log.log_id

    def log_reference_parsing(
        self,
        paper_id: str,
        reference_items: List[Dict],
        parsing_method: str = "regex",
        dedup_keys: Optional[List[str]] = None,
        unresolved_refs: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log reference parsing results.

        Args:
            paper_id: Paper identifier
            reference_items: List of parsed references with structure:
                   {ref_id, authors, year, title, venue, doi, parsed_successfully, raw_text}
            parsing_method: Method used (regex, llm, grobid, etc.)
            dedup_keys: Keys used for deduplication (doi, title+year)
            unresolved_refs: References that couldn't be fully parsed
            metadata: Additional metadata

        Returns:
            log_id for reference
        """
        total = len(reference_items)
        successful = sum(1 for r in reference_items if r.get("parsed_successfully", False))
        parse_rate = successful / total if total > 0 else 0.0

        log = ReferenceLog(
            log_id=self._generate_id("ref"),
            timestamp=self._get_timestamp(),
            paper_id=paper_id,
            reference_items=reference_items,
            total_references=total,
            parse_success_rate=round(parse_rate, 3),
            dedup_keys=dedup_keys or ["doi", "title+year"],
            unresolved_refs=unresolved_refs or [],
            parsing_method=parsing_method,
            metadata=metadata or {}
        )

        self._write_log(LogCategory.REFERENCE, asdict(log))
        return log.log_id

    def log_decision(
        self,
        decision_id: str,
        decision: str,
        options: List[str],
        chosen: str,
        rationale: str,
        agent_name: str,
        context: str = "",
        evidence_links: Optional[List[str]] = None,
        paper_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log agent decision.

        Args:
            decision_id: Unique decision identifier
            decision: Description of the decision made
            options: Available options considered
            chosen: The option that was chosen
            rationale: Why this option was chosen
            agent_name: Name of the agent making the decision
            context: Context in which decision was made
            evidence_links: Links to supporting evidence
            paper_id: Optional paper identifier
            metadata: Additional metadata

        Returns:
            log_id for reference
        """
        log = DecisionLog(
            log_id=self._generate_id("dec"),
            timestamp=self._get_timestamp(),
            decision_id=decision_id,
            paper_id=paper_id,
            context=context,
            decision=decision,
            options=options,
            chosen=chosen,
            rationale=rationale,
            evidence_links=evidence_links or [],
            agent_name=agent_name,
            metadata=metadata or {}
        )

        self._write_log(LogCategory.DECISION, asdict(log))
        return log.log_id

    def log_artifact(
        self,
        artifact_type: str,
        file_path: str,
        generation_method: str,
        input_sources: Optional[List[Dict]] = None,
        paper_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log artifact/output generation.

        Args:
            artifact_type: Type of artifact (summary_txt, notes_json, report_md, etc.)
            file_path: Path where artifact was saved
            generation_method: How it was generated (llm, extraction, merge, etc.)
            input_sources: Sources used to generate (chunks, prompts)
            paper_id: Optional paper identifier
            metadata: Additional metadata

        Returns:
            log_id for reference
        """
        # Get file info
        path = Path(file_path)
        file_size = 0
        file_hash = ""

        if path.exists():
            file_size = path.stat().st_size
            try:
                with open(path, "rb") as f:
                    file_hash = self._compute_hash(f.read())
            except:
                pass

        log = ArtifactLog(
            log_id=self._generate_id("art"),
            timestamp=self._get_timestamp(),
            artifact_id=f"{artifact_type}_{self._compute_hash(file_path)}",
            artifact_type=artifact_type,
            file_path=str(file_path),
            file_hash=file_hash,
            file_size_bytes=file_size,
            created_at=datetime.now().isoformat(),
            input_sources=input_sources or [],
            generation_method=generation_method,
            paper_id=paper_id,
            metadata=metadata or {}
        )

        self._write_log(LogCategory.ARTIFACT, asdict(log))
        return log.log_id

    # ==========================================================================
    # Convenience Methods for Evidence-Based Summary Logging
    # ==========================================================================

    def create_evidence_mapping(
        self,
        summary_sentences: List[str],
        source_chunks: List[Dict],  # {chunk_id, text, page}
        model: str = "semantic"
    ) -> List[Dict]:
        """
        Create evidence mapping between summary sentences and source chunks.

        This is a simplified version - in production, you might use
        semantic similarity or LLM-based attribution.

        Args:
            summary_sentences: List of summary sentences
            source_chunks: List of source chunks with id, text, page
            model: Mapping method (semantic, keyword, llm)

        Returns:
            List of evidence mappings
        """
        evidence_map = []

        for idx, sentence in enumerate(summary_sentences):
            # Simple keyword-based matching (can be enhanced with embeddings)
            sentence_words = set(sentence.lower().split())
            matched_chunks = []
            matched_pages = []

            for chunk in source_chunks:
                chunk_words = set(chunk.get("text", "").lower().split())
                overlap = len(sentence_words & chunk_words)
                if overlap >= 3:  # At least 3 common words
                    matched_chunks.append(chunk.get("chunk_id", ""))
                    if chunk.get("page"):
                        matched_pages.append(chunk["page"])

            confidence = "high" if len(matched_chunks) >= 2 else "med" if matched_chunks else "low"

            evidence_map.append({
                "summary_sentence_id": f"sent_{idx}",
                "summary_sentence": sentence,
                "source_chunks": matched_chunks[:5],  # Top 5
                "source_pages": list(set(matched_pages))[:5],
                "confidence": confidence
            })

        return evidence_map

    def extract_claims_from_text(
        self,
        text: str,
        paper_id: str,
        section: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract claims from text using heuristics.

        This is a simplified extraction - in production, use LLM.

        Args:
            text: Text to extract claims from
            paper_id: Paper identifier
            section: Optional section name

        Returns:
            List of claim dictionaries
        """
        claims = []

        # Simple patterns for different claim types
        patterns = {
            ClaimType.RESULT: [
                r"(?:we|our method|the proposed)\s+(?:achieve|obtain|show|demonstrate)[sd]?\s+(.+?)(?:\.|$)",
                r"(?:improve|outperform)[sd]?\s+(.+?)(?:by|with)\s+(.+?)(?:\.|$)",
            ],
            ClaimType.METHOD: [
                r"(?:we|our)\s+(?:propose|introduce|present)[sd]?\s+(.+?)(?:\.|$)",
                r"(?:novel|new)\s+(.+?)\s+(?:method|approach|technique)",
            ],
            ClaimType.LIMITATION: [
                r"(?:limitation|drawback|weakness)[s]?\s+(?:is|are|include)\s+(.+?)(?:\.|$)",
                r"(?:however|although)\s*,?\s*(.+?)(?:\.|$)",
            ],
        }

        import re
        for claim_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:3]:  # Limit per pattern
                    claim_text = match[0] if isinstance(match, tuple) else match
                    claims.append({
                        "claim_id": self._generate_id("c"),
                        "claim_text": claim_text.strip()[:500],
                        "claim_type": claim_type.value,
                        "evidence": {
                            "page": None,
                            "chunk_id": None,
                            "snippet": text[:200]
                        },
                        "confidence": "med",
                        "source_section": section
                    })

        return claims

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session."""
        stats = {
            "session_id": self.session_id,
            "log_counts": {}
        }

        for category, log_file in self.log_files.items():
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    # Count lines for this session
                    count = sum(1 for line in f if self.session_id in line)
                    stats["log_counts"][category.value] = count

        return stats

    def read_logs(
        self,
        category: LogCategory,
        limit: int = 100,
        paper_id: Optional[str] = None
    ) -> List[Dict]:
        """Read logs from a category."""
        log_file = self.log_files.get(category)
        if not log_file or not log_file.exists():
            return []

        logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    log = json.loads(line)
                    if paper_id and log.get("paper_id") != paper_id:
                        continue
                    logs.append(log)
                    if len(logs) >= limit:
                        break
                except json.JSONDecodeError:
                    continue

        return logs


# ==============================================================================
# Global Logger Access
# ==============================================================================

_logger_instance: Optional[LLMLogger] = None


def get_logger(log_dir: str = "logs/prompts") -> LLMLogger:
    """Get or create the global LLM logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = LLMLogger(log_dir)
    return _logger_instance


# ==============================================================================
# Decorator for Easy LLM Call Logging
# ==============================================================================

def log_llm_call(tool_name: str, model: str = "gpt-4o"):
    """
    Decorator to automatically log LLM calls.

    Usage:
        @log_llm_call("my_tool")
        async def my_llm_function(prompt: str) -> str:
            ...
    """
    def decorator(func):
        import functools
        import time

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger()
            start_time = time.time()

            # Extract prompt from args or kwargs
            prompt = kwargs.get("prompt") or (args[0] if args else "")

            try:
                result = await func(*args, **kwargs)
                latency = (time.time() - start_time) * 1000

                # Log the call
                logger.log_llm_call(
                    model=model,
                    prompt=str(prompt)[:10000],  # Truncate for storage
                    response=str(result)[:10000] if result else "",
                    tool_name=tool_name,
                    latency_ms=latency,
                    paper_id=kwargs.get("paper_id")
                )

                return result
            except Exception as e:
                latency = (time.time() - start_time) * 1000
                logger.log_llm_call(
                    model=model,
                    prompt=str(prompt)[:10000],
                    response=f"ERROR: {str(e)}",
                    tool_name=tool_name,
                    latency_ms=latency,
                    paper_id=kwargs.get("paper_id"),
                    metadata={"error": str(e)}
                )
                raise

        return wrapper
    return decorator
