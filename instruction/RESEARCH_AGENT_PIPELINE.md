# Research Agent Pipeline 문서

## 개요

`research_agent.py`는 LLM-Orchestrated Research Agent의 핵심 구현입니다. 논문을 분석하기 위해 여러 단계를 거쳐 전략적으로 접근합니다.

## 파일 구조

```
research_agent.py
├── Helper Functions
│   ├── _env_or_empty()
│   └── _get_fallback_sections()
│
├── Data Classes
│   ├── ReasoningEntry
│   ├── PaperAnalysisResult
│   └── AgentState
│
├── LLMOrchestrator
│   ├── extract_abstract()
│   ├── determine_analysis_strategy()
│   └── generate_executive_summary()
│
├── FinalReportGenerator
│   └── generate()
│
├── ResearchAgentTool (Main Tool)
│   └── execute()
│       └── _process_single_paper()
│
└── ConferencePipelineTool
    ├── _download_arxiv_paper()
    ├── _download_neurips_paper()
    └── execute()
```

## 주요 파이프라인

### 1. ResearchAgentTool 파이프라인

**엔트리 포인트**: `ResearchAgentTool.execute()`

#### 전체 흐름

```
1. 초기화
   ├── Job ID 생성
   ├── AgentState 초기화
   └── Progress: 0%

2. 각 논문 처리 (0-60%)
   └── _process_single_paper()
       ├── Phase 1: 텍스트 추출 (extract_all)
       ├── Phase 2: 요약 생성 (generate_report)
       ├── Phase 2.5: 초록 추출 (extract_abstract)
       ├── Phase 3: 목차 추출 (extract_paper_sections)
       ├── Phase 4: 전략 결정 (determine_analysis_strategy)
       └── Phase 5: 전략 기반 분석
           ├── analyze_section (각 섹션 분석)
           ├── paper_qa (질문 답변, 선택적)
           └── focus_areas 강조

3. Executive Summary 생성 (70%)
   └── generate_executive_summary()

4. 최종 리포트 생성 (80%)
   └── FinalReportGenerator.generate()

5. 리포트 저장
   └── OUTPUT_DIR/agent_reports/

6. Discord 알림 전송 (90%)
   ├── Full report (discord_webhook_full)
   └── Summary (discord_webhook_summary)

7. 완료 (100%)
   └── state.mark_completed()
```

#### _process_single_paper 상세 흐름

```
입력: paper_id, state (AgentState)

1. 텍스트 추출 확인
   ├── extracted_text.json 존재 확인
   └── 없으면 extract_all 실행

2. 요약 생성
   └── generate_report 툴 호출
       └── 결과: summary_report

3. 초록 추출
   └── LLMOrchestrator.extract_abstract()
       └── 결과: abstract

4. 목차 추출
   └── extract_paper_sections 툴 호출
       └── 결과: sections (List[Dict])

5. 전략 결정 (핵심 단계)
   └── LLMOrchestrator.determine_analysis_strategy()
       입력: summary_report, abstract, sections, goal, mode
       └── LLM이 다음을 결정:
           ├── strategy: "experiment_focused" | "methodology_focused" | ...
           ├── target_sections: 분석할 섹션 목록
           ├── focus_areas: 각 섹션의 집중 영역
           ├── analysis_questions: 답변할 질문들
           └── analysis_depth: 분석 깊이

6. 전략 기반 분석
   각 target_section에 대해:
   ├── analyze_section 툴 호출
   │   └── 기본 섹션 분석
   │
   ├── focus_areas가 있으면 강조 표시 추가
   │
   └── analysis_questions가 있으면
       └── paper_qa 툴로 각 질문 답변
           └── Q&A 섹션 추가

출력: PaperAnalysisResult
```

### 2. ConferencePipelineTool 파이프라인

**엔트리 포인트**: `ConferencePipelineTool.execute()`

#### 전체 흐름

```
1. 검색 (5%)
   ├── arxiv_search 또는 neurips_search
   └── 결과: papers (List[Dict])

2. 랭킹 (15-30%)
   ├── apply_hard_filters (필터링)
   ├── calculate_semantic_scores (관련성 점수)
   ├── evaluate_paper_metrics (메트릭 평가)
   └── rank_and_select_top_k (상위 K개 선택)
       └── 결과: ranked_papers

3. 다운로드 (35-50%)
   각 ranked_paper에 대해:
   ├── arxiv_download 또는 neurips2025_download_pdf
   └── 결과: downloaded_papers

4. 텍스트 추출 (50-60%)
   각 downloaded_paper에 대해:
   └── extract_all
       └── 결과: extracted_papers

5. Research Agent 실행 (60-100%)
   └── ResearchAgentTool.execute()
       └── 위의 ResearchAgentTool 파이프라인 실행
```

## 전략 기반 분석

### 분석 전략 유형

1. **experiment_focused**: 실험 결과 중심
   - 집중: 표, 그래프, 메트릭 비교
   - 질문 예시: "핵심 실험 결과는?", "베이스라인 대비 성능은?"

2. **methodology_focused**: 방법론/알고리즘 중심
   - 집중: 알고리즘 흐름, 핵심 수식
   - 질문 예시: "핵심 알고리즘은?", "수식의 의미는?"

3. **formula_heavy**: 수식/이론 중심
   - 집중: 수식 설명, 증명
   - 질문 예시: "이 수식의 의미는?", "증명의 핵심은?"

4. **implementation_focused**: 구현 세부사항 중심
   - 집중: 코드, 실용적 팁
   - 질문 예시: "구현 시 주의할 점은?", "코드 구조는?"

5. **comparison_focused**: 관련 연구 비교 중심
   - 집중: Related Work 심층 분석
   - 질문 예시: "기존 방법과의 차이는?", "장단점 비교는?"

### 전략 결정 프로세스

```
입력:
├── summary_report: 전체 요약
├── abstract: 초록
├── sections: 목차
├── goal: 사용자 목표
└── mode: 분석 모드 (quick/standard/deep)

LLM 분석:
├── 논문 특성 파악 (실험 논문인지, 이론 논문인지)
├── 사용자 목표 분석
└── 목차 구조 분석

출력:
├── strategy: 분석 전략
├── target_sections: 분석할 섹션
├── focus_areas: 집중 영역
├── analysis_questions: 질문 목록
└── analysis_depth: 분석 깊이
```

## 사용되는 기존 툴

### PDF 처리
- `extract_all`: PDF에서 텍스트와 이미지 추출
- `extract_paper_sections`: 목차 추출

### 리포트 생성
- `generate_report`: 전체 요약 리포트 생성

### 분석
- `analyze_section`: 특정 섹션 분석
- `paper_qa`: 논문에 대한 질문 답변

### 검색 및 다운로드
- `arxiv_search`: arXiv 검색
- `arxiv_download`: arXiv PDF 다운로드
- `neurips_search`: NeurIPS 검색
- `neurips2025_download_pdf`: NeurIPS PDF 다운로드

### 랭킹
- `apply_hard_filters`: 하드 필터 적용
- `calculate_semantic_scores`: 의미적 관련성 점수 계산
- `evaluate_paper_metrics`: 논문 메트릭 평가
- `rank_and_select_top_k`: 상위 K개 선택

### 알림
- `send_discord_notification`: Discord 알림 전송 (스레드 지원)

## 데이터 구조

### AgentState
```python
{
    "job_id": str,
    "goal": str,
    "papers": List[str],
    "analysis_mode": str,  # "quick" | "standard" | "deep"
    "status": str,  # "running" | "completed" | "failed"
    "current_paper_idx": int,
    "current_step": str,
    "progress_percent": float,  # 0-100
    "paper_results": List[PaperAnalysisResult],
    "reasoning_log": List[ReasoningEntry],
    "errors": List[str],
    "created_at": str,
    "updated_at": str
}
```

### PaperAnalysisResult
```python
{
    "paper_id": str,
    "title": str,
    "summary_report": str,
    "selected_sections": List[str],
    "section_analyses": Dict[str, str],  # section_title -> analysis_text
    "selection_reasoning": str  # 전략 정보 포함
}
```

### Strategy (determine_analysis_strategy 반환값)
```python
{
    "strategy": str,  # 전략 유형
    "strategy_reasoning": str,
    "target_sections": List[str],
    "focus_areas": Dict[str, List[str]],  # section -> focus areas
    "analysis_questions": List[str],
    "analysis_depth": str
}
```

## 주요 개선 사항

1. **전략 기반 분석**: 단순 섹션 선택이 아닌 논문 특성에 맞는 전략 결정
2. **초록 활용**: summary + 초록 + 목차를 종합 분석
3. **집중 분석**: 섹션 내 핵심 부분만 선별 분석
4. **질문 기반 심화**: 전략에 맞는 질문 자동 생성 및 답변
5. **코드 정리**: 사용하지 않는 코드 제거, 중복 로직 헬퍼 함수화

## 실행 예시

### ResearchAgentTool
```python
result = await execute_tool(
    "run_research_agent",
    paper_ids=["paper1", "paper2"],
    goal="실험 결과 분석",
    analysis_mode="quick"
)
```

### ConferencePipelineTool
```python
result = await execute_tool(
    "run_conference_pipeline",
    source="arxiv",
    query="transformer attention",
    top_k=3,
    goal="최신 방법론 이해",
    analysis_mode="standard"
)
```
