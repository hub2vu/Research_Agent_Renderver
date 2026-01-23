"""
Translation Tools
Translate academic papers from source language to target language with glossary support.
"""

import os
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# [1] pdf.py와 똑같은 경로 설정 방식 사용
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/output"))

# [2] OpenAI 키 가져오기 (환경변수에서)
API_KEY = os.getenv("OPENAI_API_KEY")

# [3] 라이브러리 체크 (없으면 에러 메시지를 명확히 뱉도록 함)
try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from ..base import MCPTool, ToolParameter, ExecutionError


def _extract_abstract_intro(text: str) -> str:
    """
    Abstract와 Introduction 섹션을 추출
    정규표현식으로 "Abstract", "Introduction" 헤더 찾기
    두 섹션을 결합하여 반환
    """
    # Abstract 섹션 찾기
    abstract_pattern = re.compile(
        r"(?i)^\s*(?:abstract|summary)\s*:?\s*\n(.*?)(?=\n\s*(?:1\.|introduction|keywords|index terms|ccs concepts|acm reference format|introduction:))",
        re.MULTILINE | re.DOTALL,
    )
    
    # Introduction 섹션 찾기
    intro_pattern = re.compile(
        r"(?i)^\s*(?:1\.\s*)?introduction\s*:?\s*\n(.*?)(?=\n\s*(?:2\.|related work|background|methodology|preliminaries|related|method|approach))",
        re.MULTILINE | re.DOTALL,
    )
    
    abstract_match = abstract_pattern.search(text)
    intro_match = intro_pattern.search(text)
    
    sections = []
    
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        if len(abstract_text) > 50:  # 최소 길이 체크
            sections.append(abstract_text)
    
    if intro_match:
        intro_text = intro_match.group(1).strip()
        if len(intro_text) > 100:  # 최소 길이 체크
            sections.append(intro_text)
    
    # 섹션을 찾지 못한 경우, 텍스트 앞부분 사용 (최대 5000자)
    if not sections:
        return text[:5000]
    
    return "\n\n".join(sections)


def _split_into_chunks(text: str, max_chunk_size: int = 5000) -> List[str]:
    """
    문장 단위를 보존하며 청크 분할
    1. 문단 단위로 먼저 분할 (빈 줄 \n\n 기준)
    2. 문단이 max_chunk_size를 초과하면 문장 단위로 분할 (마침표 . 기준)
    3. 문맥이 끊기지 않도록 보장
    """
    chunks = []
    
    # 먼저 문단 단위로 분할 (빈 줄 기준)
    paragraphs = re.split(r"\n\s*\n", text)
    
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # 현재 청크에 이 문단을 추가했을 때 크기 체크
        test_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        if len(test_chunk) <= max_chunk_size:
            # 크기가 적절하면 추가
            current_chunk = test_chunk
        else:
            # 크기를 초과하면
            if current_chunk:
                # 현재 청크를 저장
                chunks.append(current_chunk)
                current_chunk = ""
            
            # 문단 자체가 max_chunk_size를 초과하면 문장 단위로 분할
            if len(para) > max_chunk_size:
                # 문장 단위로 분할 (마침표 뒤 공백 기준)
                sentences = re.split(r"(?<=\.)\s+", para)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    
                    if len(test_chunk) <= max_chunk_size:
                        current_chunk = test_chunk
                    else:
                        # 현재 청크가 있으면 저장
                        if current_chunk:
                            chunks.append(current_chunk)
                        # 새 문장으로 시작
                        current_chunk = sentence
            else:
                # 문단이 크기를 초과하지 않으면 그대로 추가
                current_chunk = para
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]


def _extract_glossary(
    abstract_intro_text: str, source_lang: str, target_lang: str, client
) -> Dict[str, str]:
    """
    Abstract/Introduction에서 전문 용어를 추출하여 용어집 생성
    OpenAI API를 사용하여 용어-번역 쌍 추출
    반환: {"term": "translation", ...}
    """
    if not abstract_intro_text or len(abstract_intro_text) < 100:
        return {}
    
    # 텍스트 길이 제한 (용어 추출용)
    text_for_glossary = abstract_intro_text[:8000]
    
    system_prompt = f"""You are a professional translator specializing in academic papers.
Extract technical terms and their translations from the given text.

Your task:
1. Identify key technical terms, proper nouns, and domain-specific terminology
2. Provide accurate translations from {source_lang} to {target_lang}
3. Return ONLY a JSON object in this format: {{"term1": "translation1", "term2": "translation2", ...}}
4. Include 10-20 most important terms
5. Do NOT include common words or general vocabulary
6. Focus on terms that are specific to this research domain

Return ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Extract technical terms from this text:\n\n{text_for_glossary}",
                },
            ],
            temperature=0.3,
        )
        
        content = response.choices[0].message.content.strip()
        
        # JSON 추출 (코드 블록 제거)
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        content = content.strip()
        
        # JSON 파싱
        glossary = json.loads(content)
        
        # 딕셔너리인지 확인
        if isinstance(glossary, dict):
            return glossary
        else:
            return {}
            
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 빈 딕셔너리 반환
        return {}
    except Exception:
        # 기타 에러도 빈 딕셔너리 반환
        return {}


async def _translate_with_retry(
    client, messages: List[Dict], max_retries: int = 3
) -> str:
    """
    OpenAI API 호출 시 재시도 로직
    - Rate Limit 에러 시 exponential backoff
    - 최대 3회 재시도
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content
            
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff: 1초, 2초, 4초
                time.sleep(wait_time)
                continue
            else:
                raise ExecutionError(
                    f"Rate limit exceeded after {max_retries} retries",
                    tool_name="translate_paper",
                )
        except openai.APITimeoutError:
            if attempt < max_retries - 1:
                time.sleep(1)  # timeout은 짧은 대기
                continue
            else:
                raise ExecutionError(
                    f"API timeout after {max_retries} retries",
                    tool_name="translate_paper",
                )
        except Exception as e:
            # 기타 에러는 재시도하지 않고 즉시 실패
            raise ExecutionError(
                f"OpenAI API 호출 실패: {str(e)}", tool_name="translate_paper"
            )
    
    raise ExecutionError(
        f"Translation failed after {max_retries} retries", tool_name="translate_paper"
    )


class TranslatePaperTool(MCPTool):
    """Translate a paper from source language to target language."""

    @property
    def name(self) -> str:
        return "translate_paper"

    @property
    def description(self) -> str:
        return "Translate a paper from source language to target language with glossary support."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id",
                type="string",
                description="Paper ID (e.g., '1809.04281')",
                required=True,
            ),
            ToolParameter(
                name="target_language",
                type="string",
                description="Target language (default: 'Korean')",
                required=False,
                default="Korean",
            ),
            ToolParameter(
                name="source_language",
                type="string",
                description="Source language (default: 'English')",
                required=False,
                default="English",
            ),
        ]

    @property
    def category(self) -> str:
        return "translation"

    async def execute(
        self, paper_id: str, target_language: str = "Korean", source_language: str = "English"
    ) -> Dict[str, Any]:
        # --- [체크 1] OpenAI 라이브러리 확인 ---
        if not HAS_OPENAI:
            raise ExecutionError(
                "서버에 'openai' 패키지가 없습니다. (pip install openai 필요)",
                tool_name=self.name,
            )

        # --- [체크 2] API 키 확인 ---
        if not API_KEY:
            raise ExecutionError(
                "환경변수 OPENAI_API_KEY가 설정되지 않았습니다.", tool_name=self.name
            )

        # --- [체크 3] 파일 경로 확인 ---
        paper_dir = OUTPUT_DIR / paper_id
        text_file = paper_dir / "extracted_text.txt"

        if not text_file.exists():
            json_file = paper_dir / "extracted_text.json"
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    full_text = data.get("full_text", "")
            else:
                raise ExecutionError(
                    f"텍스트 파일을 찾을 수 없습니다. 경로: {text_file}",
                    tool_name=self.name,
                )
        else:
            with open(text_file, "r", encoding="utf-8") as f:
                full_text = f.read()

        # ⭐ UTF-8 인코딩 정리
        if isinstance(full_text, str):
            full_text = full_text.encode("utf-8", "replace").decode("utf-8")

        # --- 번역 프로세스 시작 ---
        try:
            client = openai.OpenAI(api_key=API_KEY)
            started_at = datetime.now().isoformat()

            # 1. Abstract/Introduction 추출
            abstract_intro = _extract_abstract_intro(full_text)

            # 2. 용어집 생성
            glossary = _extract_glossary(abstract_intro, source_language, target_language, client)

            # 3. 용어집 저장
            glossary_path = paper_dir / "translation_glossary.json"
            with open(glossary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "paper_id": paper_id,
                        "source_language": source_language,
                        "target_language": target_language,
                        "glossary": glossary,
                        "created_at": started_at,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # 4. 청크 분할
            chunks = _split_into_chunks(full_text, max_chunk_size=5000)
            total_chunks = len(chunks)

            # 5. 상태 파일 초기화
            status_path = paper_dir / "translation_status.json"
            status_data = {
                "status": "in_progress",
                "paper_id": paper_id,
                "source_language": source_language,
                "target_language": target_language,
                "total_chunks": total_chunks,
                "completed_chunks": 0,
                "started_at": started_at,
                "completed_at": None,
                "error": None,
            }
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status_data, f, ensure_ascii=False, indent=2)

            # 6. 청크별 번역
            translated_chunks = []
            glossary_str = "\n".join([f"{k}: {v}" for k, v in glossary.items()])

            for i, chunk in enumerate(chunks):
                # 이전 컨텍스트 준비 (최대 2개 청크)
                prev_context = ""
                if i > 0:
                    prev_chunks = translated_chunks[-2:] if len(translated_chunks) >= 2 else translated_chunks
                    prev_context = "\n\n".join(prev_chunks)

                # 구조화된 프롬프트 생성
                system_prompt = f"""You are a professional academic translator.
Translate the given text from {source_language} to {target_language} accurately and naturally.

CRITICAL RULES:
1. Translate accurately without paraphrasing or simplifying
2. Preserve all technical terms consistently using the provided glossary
3. Keep all mathematical expressions, citations, and reference numbers unchanged
4. Do NOT translate or modify any text inside $...$ or $$...$$ (LaTeX formulas)
5. Maintain the original structure and formatting
6. Do NOT add explanations or interpretations"""

                user_prompt = f"""[Context: Previous Paragraphs]
{prev_context if prev_context else "(No previous context)"}

[Glossary]
{glossary_str if glossary_str else "(No glossary available)"}

[Task] 
Translate the following text from {source_language} to {target_language}. 
Maintain the formatting and do not add any comments.
CRITICAL: Do not translate or modify any text inside $...$ or $$...$$ (LaTeX formulas).

[Text to Translate]
{chunk}"""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # 번역 실행 (재시도 로직 포함)
                translated_chunk = await _translate_with_retry(client, messages)
                translated_chunks.append(translated_chunk)

                # 진행 상태 업데이트
                status_data["completed_chunks"] = i + 1
                with open(status_path, "w", encoding="utf-8") as f:
                    json.dump(status_data, f, ensure_ascii=False, indent=2)

                # Rate Limit 방지를 위한 sleep
                if i < total_chunks - 1:  # 마지막 청크가 아니면
                    time.sleep(0.5)

            # 7. 번역 결과 저장
            translated_text = "\n\n".join(translated_chunks)
            translated_path = paper_dir / f"translated_text_{target_language}.txt"
            with open(translated_path, "w", encoding="utf-8") as f:
                f.write(translated_text)

            # 8. 상태 파일 완료 업데이트
            completed_at = datetime.now().isoformat()
            status_data["status"] = "completed"
            status_data["completed_at"] = completed_at
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status_data, f, ensure_ascii=False, indent=2)

            return {
                "status": "success",
                "paper_id": paper_id,
                "translated_path": str(translated_path),
                "glossary_path": str(glossary_path),
                "total_chunks": total_chunks,
                "preview": translated_text[:200],
            }

        except ExecutionError:
            # ExecutionError는 그대로 전파
            raise
        except Exception as e:
            # 상태 파일에 에러 기록
            try:
                status_path = paper_dir / "translation_status.json"
                if status_path.exists():
                    with open(status_path, "r", encoding="utf-8") as f:
                        status_data = json.load(f)
                    status_data["status"] = "failed"
                    status_data["error"] = str(e)
                    status_data["completed_at"] = datetime.now().isoformat()
                    with open(status_path, "w", encoding="utf-8") as f:
                        json.dump(status_data, f, ensure_ascii=False, indent=2)
            except:
                pass  # 상태 파일 업데이트 실패는 무시
            
            raise ExecutionError(
                f"번역 실패: {str(e)}", tool_name=self.name
            )


class GetTranslationTool(MCPTool):
    """Retrieve an existing translation."""

    @property
    def name(self) -> str:
        return "get_translation"

    @property
    def description(self) -> str:
        return "Retrieve the generated translation with status information."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="paper_id", type="string", description="Paper ID", required=True
            ),
            ToolParameter(
                name="target_language",
                type="string",
                description="Target language (default: 'Korean')",
                required=False,
                default="Korean",
            ),
        ]

    @property
    def category(self) -> str:
        return "translation"

    async def execute(
        self, paper_id: str, target_language: str = "Korean"
    ) -> Dict[str, Any]:
        paper_dir = OUTPUT_DIR / paper_id
        translated_path = paper_dir / f"translated_text_{target_language}.txt"
        status_path = paper_dir / "translation_status.json"

        # 상태 파일 읽기
        status = "unknown"
        progress = None
        error = None
        
        if status_path.exists():
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                    status = status_data.get("status", "unknown")
                    total_chunks = status_data.get("total_chunks", 0)
                    completed_chunks = status_data.get("completed_chunks", 0)
                    if total_chunks > 0:
                        progress = f"{completed_chunks}/{total_chunks} chunks"
                    error = status_data.get("error")
            except:
                pass

        # 번역 파일 읽기
        if translated_path.exists():
            with open(translated_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return {
                "found": True,
                "status": status,
                "content": content,
                "progress": progress,
                "error": error,
            }
        
        # 번역 파일이 없지만 상태 파일이 있으면 진행 중 또는 실패
        if status_path.exists():
            return {
                "found": False,
                "status": status,
                "content": None,
                "progress": progress,
                "error": error,
                "message": f"Translation {status}. File not yet created." if status == "in_progress" else f"Translation {status}.",
            }
        
        return {
            "found": False,
            "status": "not_started",
            "content": None,
            "progress": None,
            "error": None,
            "message": "Translation not found. Run translate_paper first.",
        }
