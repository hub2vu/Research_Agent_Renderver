"""
Integration tests for translation tools.

Tests translation functionality:
1. translate_paper - Translate a paper from source to target language
2. get_translation - Retrieve translation status and content
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import pytest
import requests
## Docker Compose로 실행 (같은 네트워크)
#docker compose run --rm -e MCP_SERVER_URL=http://mcp-server:8000 mcp-server pytest tests/test_translate.py::test_translate -v -s
# MCP Server URL
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

# Test paper IDs (should exist in output directory)
TEST_PAPER_IDS = [
    "1809.04281",
    "10.48550_arxiv.2201.07207",
    "10.48550_arxiv.2210.03629",
]


@pytest.fixture
def mcp_server_available() -> bool:
    """Check if MCP server is available."""
    try:
        response = requests.get(f"{MCP_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


@pytest.fixture
def sample_paper_id() -> str:
    """Return a sample paper ID for testing."""
    # Check which paper IDs are available
    output_dir = Path(os.getenv("OUTPUT_DIR", "/data/output"))
    
    for paper_id in TEST_PAPER_IDS:
        paper_dir = output_dir / paper_id
        text_file = paper_dir / "extracted_text.txt"
        json_file = paper_dir / "extracted_text.json"
        
        if text_file.exists() or json_file.exists():
            return paper_id
    
    # Fallback to first test ID
    return TEST_PAPER_IDS[0]


def test_translate_paper_basic(mcp_server_available, sample_paper_id):
    """Test basic translation functionality."""
    if not mcp_server_available:
        pytest.skip("MCP server not available")
    
    print(f"\n📄 논문 번역 시작: {sample_paper_id}")
    
    # 1. 번역 실행
    response = requests.post(
        f"{MCP_URL}/tools/translate_paper/execute",
        json={
            "arguments": {
                "paper_id": sample_paper_id,
                "target_language": "Korean"
            }
        },
        timeout=600  # 10분 타임아웃 (긴 논문 대비)
    )
    
    assert response.status_code == 200, f"HTTP {response.status_code}: {response.text}"
    
    result = response.json()
    print(f"\n✅ 번역 요청 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Assertions
    assert "success" in result, "Response should have 'success' field"
    
    if result.get("success"):
        assert "result" in result, "Successful response should have 'result' field"
        result_data = result["result"]
        
        print(f"\n📊 번역 통계:")
        print(f"  - 총 청크 수: {result_data.get('total_chunks', 'N/A')}")
        print(f"  - 번역 파일: {result_data.get('translated_path', 'N/A')}")
        print(f"  - 용어집 파일: {result_data.get('glossary_path', 'N/A')}")
        print(f"\n📝 미리보기:")
        print(result_data.get('preview', 'N/A'))
        
        # Verify files exist
        if "translated_path" in result_data:
            translated_path = Path(result_data["translated_path"])
            assert translated_path.exists(), f"Translation file should exist: {translated_path}"
        
        if "glossary_path" in result_data:
            glossary_path = Path(result_data["glossary_path"])
            assert glossary_path.exists(), f"Glossary file should exist: {glossary_path}"
    else:
        error = result.get("error", "Unknown error")
        print(f"\n❌ 에러: {error}")
        pytest.fail(f"Translation failed: {error}")


def test_get_translation(mcp_server_available, sample_paper_id):
    """Test getting translation status and content."""
    if not mcp_server_available:
        pytest.skip("MCP server not available")
    
    print(f"\n🔍 번역 상태 확인: {sample_paper_id}")
    
    # Wait a bit for translation to complete if it's in progress
    time.sleep(2)
    
    response = requests.post(
        f"{MCP_URL}/tools/get_translation/execute",
        json={
            "arguments": {
                "paper_id": sample_paper_id,
                "target_language": "Korean"
            }
        },
        timeout=30
    )
    
    assert response.status_code == 200, f"HTTP {response.status_code}: {response.text}"
    
    result = response.json()
    print(f"\n📋 번역 상태:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    assert "success" in result, "Response should have 'success' field"
    
    if result.get("success"):
        result_data = result.get("result", {})
        
        assert "found" in result_data, "Response should have 'found' field"
        assert "status" in result_data, "Response should have 'status' field"
        
        status = result_data.get("status")
        print(f"\n📊 상태: {status}")
        
        if result_data.get("found"):
            content = result_data.get("content", "")
            progress = result_data.get("progress", "N/A")
            
            print(f"  - 진행률: {progress}")
            print(f"  - 번역본 길이: {len(content)} 글자")
            
            if content:
                print(f"\n📝 번역본 앞부분 (500자):")
                print(content[:500])
            
            # Verify status is valid
            assert status in ["completed", "in_progress", "failed", "not_started"], \
                f"Invalid status: {status}"


def test_translate_paper_error_handling(mcp_server_available):
    """Test error handling for invalid paper IDs."""
    if not mcp_server_available:
        pytest.skip("MCP server not available")
    
    # Test with non-existent paper ID
    invalid_paper_id = "nonexistent_paper_12345"
    
    response = requests.post(
        f"{MCP_URL}/tools/translate_paper/execute",
        json={
            "arguments": {
                "paper_id": invalid_paper_id,
                "target_language": "Korean"
            }
        },
        timeout=30
    )
    
    assert response.status_code == 200
    result = response.json()
    
    # Should fail gracefully
    assert "success" in result
    assert result.get("success") == False or "error" in result, \
        "Should return error for invalid paper ID"
    
    print(f"\n✅ 에러 처리 확인: {result.get('error', 'N/A')}")


def test_translation_files_exist(mcp_server_available, sample_paper_id):
    """Verify that translation creates expected files."""
    if not mcp_server_available:
        pytest.skip("MCP server not available")
    
    output_dir = Path(os.getenv("OUTPUT_DIR", "/data/output"))
    paper_dir = output_dir / sample_paper_id
    
    # Check if translation files exist
    translated_file = paper_dir / "translated_text_Korean.txt"
    glossary_file = paper_dir / "translation_glossary.json"
    status_file = paper_dir / "translation_status.json"
    
    files_exist = {
        "translated_text": translated_file.exists(),
        "glossary": glossary_file.exists(),
        "status": status_file.exists()
    }
    
    print(f"\n📁 파일 존재 여부:")
    for name, exists in files_exist.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {name}")
    
    # At least status file should exist if translation was attempted
    # (This is a soft check - files might not exist if translation failed)
    if status_file.exists():
        with open(status_file, "r", encoding="utf-8") as f:
            status_data = json.load(f)
            print(f"\n📋 상태 파일 내용:")
            print(json.dumps(status_data, indent=2, ensure_ascii=False))


def test_translation_glossary_format(mcp_server_available, sample_paper_id):
    """Test that glossary file has correct format."""
    if not mcp_server_available:
        pytest.skip("MCP server not available")
    
    output_dir = Path(os.getenv("OUTPUT_DIR", "/data/output"))
    glossary_file = output_dir / sample_paper_id / "translation_glossary.json"
    
    if not glossary_file.exists():
        pytest.skip("Glossary file does not exist (translation may not have run)")
    
    with open(glossary_file, "r", encoding="utf-8") as f:
        glossary_data = json.load(f)
    
    # Verify structure
    assert "paper_id" in glossary_data
    assert "source_language" in glossary_data
    assert "target_language" in glossary_data
    assert "glossary" in glossary_data
    assert isinstance(glossary_data["glossary"], dict)
    
    print(f"\n📚 용어집 구조 확인:")
    print(f"  - 논문 ID: {glossary_data.get('paper_id')}")
    print(f"  - 원본 언어: {glossary_data.get('source_language')}")
    print(f"  - 목표 언어: {glossary_data.get('target_language')}")
    print(f"  - 용어 개수: {len(glossary_data.get('glossary', {}))}")
    
    # Show sample terms
    if glossary_data.get("glossary"):
        print(f"\n📝 용어 샘플 (최대 5개):")
        for i, (term, translation) in enumerate(list(glossary_data["glossary"].items())[:5]):
            print(f"  {i+1}. {term} → {translation}")


# ========== Main test function (for direct execution) ==========

def test_translate():
    """
    Main test function that can be run directly.
    Usage: python -m pytest tests/test_translate.py::test_translate -v -s
    Or: python tests/test_translate.py
    """
    MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    PAPER_ID = "1809.04281"  # 또는 다른 논문 ID
    
    print(f"📄 논문 번역 시작: {PAPER_ID}")
    
    # 1. 번역 실행
    response = requests.post(
        f"{MCP_URL}/tools/translate_paper/execute",
        json={"arguments": {
            "paper_id": PAPER_ID,
            "target_language": "Korean"
        }},
        timeout=600  # 10분 타임아웃 (긴 논문 대비)
    )
    
    result = response.json()
    print(f"\n✅ 번역 요청 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result.get("success"):
        print(f"\n📊 번역 통계:")
        print(f"  - 총 청크 수: {result.get('result', {}).get('total_chunks', 'N/A')}")
        print(f"  - 번역 파일: {result.get('result', {}).get('translated_path', 'N/A')}")
        print(f"  - 용어집 파일: {result.get('result', {}).get('glossary_path', 'N/A')}")
        print(f"\n📝 미리보기:")
        print(result.get('result', {}).get('preview', 'N/A'))
    else:
        print(f"\n❌ 에러: {result.get('error')}")
        return
    
    # 2. 번역 상태 확인
    print(f"\n\n🔍 번역 상태 확인 중...")
    time.sleep(2)  # 잠시 대기
    
    status_response = requests.post(
        f"{MCP_URL}/tools/get_translation/execute",
        json={"arguments": {
            "paper_id": PAPER_ID,
            "target_language": "Korean"
        }}
    )
    
    status_result = status_response.json()
    print(f"\n📋 번역 상태:")
    print(json.dumps(status_result, indent=2, ensure_ascii=False))
    
    if status_result.get("success") and status_result.get("result", {}).get("found"):
        content = status_result.get("result", {}).get("content", "")
        print(f"\n📄 번역본 길이: {len(content)} 글자")
        print(f"\n📝 번역본 앞부분 (500자):")
        print(content[:500])


if __name__ == "__main__":
    # Direct execution (not via pytest)
    test_translate()
