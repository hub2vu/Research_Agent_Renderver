# 📚 arXiv Tools

이 모듈은 **arXiv Python API**를 기반으로  
논문 검색, 상세 정보 조회, PDF 다운로드 기능을 제공하는 MCP Tool 집합이다.  
연구 보조 에이전트가 **논문 탐색 → 검토 → 수집**을 자동화하는 데 사용된다.

---

## 🧩 Tool 목록

---

## 1. `arxiv_search`

### 설명
arXiv에서 주어진 검색어에 해당하는 **논문 목록을 검색**한다.

### 입력 파라미터
- `query` (string, required)  
  검색 쿼리  
  예: `"transformer attention mechanism"`
- `max_results` (integer, optional, default=10)  
  최대 검색 결과 수
- `sort_by` (string, optional, default="relevance")  
  정렬 기준  
  - `relevance`
  - `lastUpdatedDate`
  - `submittedDate`

### 주요 기능
- arXiv 검색 API 호출
- 정렬 기준 매핑
- 검색 결과 논문 메타데이터 수집

### 출력
- 검색 쿼리
- 전체 결과 수
- 논문 목록:
  - 논문 ID
  - 제목
  - 저자 목록
  - 요약(Abstract)
  - 게시일 / 업데이트일
  - 카테고리
  - PDF URL
  - 주 카테고리

---

## 2. `arxiv_get_paper`

### 설명
특정 arXiv 논문의 **상세 정보**를 조회한다.

### 입력 파라미터
- `paper_id` (string, required)  
  arXiv 논문 ID 또는 arXiv URL  
  예:
  - `2301.07041`
  - `https://arxiv.org/abs/2301.07041`

### 주요 기능
- URL에서 논문 ID 자동 추출
- 단일 논문 메타데이터 조회

### 출력
- 논문 ID
- 제목
- 저자 목록
- 요약(Abstract)
- 게시일 / 업데이트일
- 카테고리
- 주 카테고리
- PDF URL
- DOI (존재 시)
- 관련 링크 목록
- 코멘트
- 저널 레퍼런스

---

## 3. `arxiv_download`

### 설명
arXiv 논문의 **PDF 파일을 로컬 PDF 디렉터리로 다운로드**한다.

### 입력 파라미터
- `paper_id` (string, required)  
  arXiv 논문 ID 또는 URL
- `filename` (string, optional)  
  저장할 파일명 (확장자 제외)  
  미지정 시 논문 ID를 파일명으로 사용

### 주요 기능
- URL 기반 ID 자동 정규화
- PDF 저장 디렉터리 자동 생성
- arXiv API를 통한 PDF 다운로드

### 출력
- 다운로드 성공 여부
- 논문 ID
- 논문 제목
- 저장 경로
- 저장된 파일명

---

## 📊 기능 요약 표

| Tool 이름 | 기능 요약 |
|---------|----------|
| `arxiv_search` | arXiv 논문 검색 |
| `arxiv_get_paper` | 단일 논문 상세 정보 조회 |
| `arxiv_download` | arXiv 논문 PDF 다운로드 |

---

## 🎯 요약

> **arXiv Tools는  
> 논문 탐색 → 정보 조회 → 로컬 수집까지의 연구 흐름을  
> MCP Tool 단위로 자동화한 모듈이다.**

이 모듈은 PDF Processing Tools와 결합하여  
**“논문 검색 → 다운로드 → 분석” 파이프라인**을 완성한다.
