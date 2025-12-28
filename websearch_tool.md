# 🌐 Web Search Tools

이 모듈은 **Tavily API**를 사용하여  
웹 검색, 특정 URL 콘텐츠 추출, 주제 기반 심층 리서치를 수행하는 MCP Tool 집합이다.  
Agent가 **외부 웹 지식 탐색 및 조사(research)**를 수행할 때 사용된다.

> ⚠️ 이 모듈을 사용하려면 환경 변수 `TAVILY_API_KEY`가 필요하다.

---

## 🧩 Tool 목록

---

## 1. `web_search`

### 설명
Tavily API를 사용하여 **웹 전체를 대상으로 검색**을 수행한다.

### 입력 파라미터
- `query` (string, required)  
  검색 쿼리
- `max_results` (integer, optional, default=5)  
  반환할 최대 검색 결과 수
- `search_depth` (string, optional, default="basic")  
  검색 깊이  
  - `"basic"`
  - `"advanced"`
- `include_domains` (array[string], optional)  
  검색 결과에 **포함할 도메인 목록**
- `exclude_domains` (array[string], optional)  
  검색 결과에서 **제외할 도메인 목록**

### 주요 기능
- Tavily 검색 API 호출
- 도메인 기반 검색 필터링
- 검색 결과 요약(answer) 제공

### 출력
- 검색 쿼리
- 전체 결과 수
- 검색 결과 목록:
  - 제목
  - URL
  - 콘텐츠 요약
  - 관련도 점수(score)
- Tavily AI가 생성한 종합 답변(answer)

---

## 2. `web_get_content`

### 설명
특정 URL의 **본문 콘텐츠를 추출**한다.

### 입력 파라미터
- `url` (string, required)  
  콘텐츠를 가져올 웹 페이지 URL

### 주요 기능
- Tavily `extract` API 사용
- 웹 페이지의 원문(raw content) 추출
- 실패 시 오류 상태 반환

### 출력
- URL
- 추출된 콘텐츠
- 성공 여부
- 실패 시 오류 메시지

---

## 3. `web_research`

### 설명
하나의 주제에 대해 **여러 검색을 수행하여 심층 리서치**를 진행한다.

### 입력 파라미터
- `topic` (string, required)  
  리서치할 주제 또는 질문
- `max_results_per_search` (integer, optional, default=5)  
  검색 쿼리당 최대 결과 수

### 주요 기능
- 고급 검색(`search_depth="advanced"`) 수행
- 주제 관련 다수의 웹 소스 수집
- Tavily AI 기반 요약 생성

### 출력
- 리서치 주제
- 수집된 소스 수
- 소스 목록:
  - 제목
  - URL
  - 콘텐츠
  - 점수(score)
- AI 생성 요약(summary)

---

## 📊 기능 요약 표

| Tool 이름 | 기능 요약 |
|----------|-----------|
| `web_search` | 웹 전체 검색 (도메인 필터 지원) |
| `web_get_content` | 특정 URL 콘텐츠 추출 |
| `web_research` | 주제 기반 심층 웹 리서치 |

---

## 🎯 요약

> **Web Search Tools는  
> 외부 웹 지식 탐색 → 콘텐츠 수집 → 주제 요약까지의 과정을  
> MCP Tool 단위로 자동화한 웹 리서치 모듈이다.**

이 모듈은 arXiv Tools, PDF Processing Tools와 결합하여  
**“논문 + 웹 기반 종합 리서치 에이전트”**를 구성한다.
