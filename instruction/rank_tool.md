# `rank_and_filter_papers` Tool 가이드
---

## 1. 개요

### 1.1 Tool 정체성

| 항목 | 내용 |
|------|------|
| **이름** | `rank_and_filter_papers` |
| **역할** | 게이트키퍼 (Gatekeeper) |
| **위치** | `mcp/tools/rank_filter.py` |

### 1.2 한 줄 설명

검색된 논문의 **메타데이터(제목, 저자, 초록)**를 사용자 프로필과 대조하여 필터링하고, 다차원 스코어링으로 **Top-K를 선별**하는 도구

### 1.3 핵심 가치

- **비용 효율성:** 수십 개 논문을 Agent가 직접 분석하면 토큰 비용과 시간이 과다 소모됨
- **선별 자동화:** 메타데이터 기반 경량 평가로 고비용 작업(PDF 전문 분석)의 대상을 사전 선별
- **개인화:** 사용자 프로필 기반 맞춤형 추천

---

## 2. 파일 구조

```
mcp/tools/
├── rank_filter.py                 # 메인 Tool 클래스
│
└── rank_filter_utils/             # 유틸리티 모듈
    ├── __init__.py                # re-export
    ├── types.py                   # 데이터 클래스, 상수
    ├── path_resolver.py           # 경로 해석 유틸리티
    ├── loaders.py                 # 프로필, 히스토리, PDF 로더
    ├── filters.py                 # 필터링 로직
    ├── scorers.py                 # 스코어링 로직
    ├── rankers.py                 # 순위화 및 Contrastive 선택
    └── formatters.py              # 태깅, 결과 포맷, 저장
```

### 2.1 모듈별 책임

| 모듈 | 담당 Phase | 주요 함수 |
|------|-----------|----------|
| `types.py` | - | `PaperInput`, `UserProfile`, `FilteredPaper` 등 |
| `path_resolver.py` | 0 | `resolve_path()`, `ensure_directory()` |
| `loaders.py` | 1 | `load_profile()`, `load_history()`, `scan_local_pdfs()` |
| `filters.py` | 2 | `filter_papers()` |
| `scorers.py` | 3-5 | `calculate_embedding_scores()`, `verify_with_llm_batch()`, `calculate_final_score()` |
| `rankers.py` | 6-7 | `rank_and_select()`, `select_contrastive_paper()` |
| `formatters.py` | 8-9 | `generate_tags()`, `save_and_format_result()` |

---

## 3. Parameters 상세

### 3.1 필수 파라미터

#### `papers`
- **타입:** `Array[Object]`
- **설명:** 검색 도구가 반환한 논문 리스트

**필수 필드:**
```json
{
  "paper_id": "2501.12345",
  "title": "논문 제목",
  "abstract": "초록 내용",
  "authors": ["저자1", "저자2"]
}
```

**선택 필드:** `published`, `categories`, `pdf_url`, `github_url`, `affiliations`

### 3.2 선택 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `top_k` | integer | 5 | 반환할 최대 논문 수 |
| `profile_path` | string | `"config/profile.json"` | 사용자 프로필 경로 |
| `purpose` | enum | `"general"` | 사용 목적 |
| `ranking_mode` | enum | `"balanced"` | 순위 결정 기준 |
| `include_contrastive` | boolean | `false` | 대조적 논문 포함 여부 |
| `contrastive_type` | enum | `"method"` | 대조 기준 |
| `history_path` | string | `"history/read_papers.json"` | 읽은 논문 기록 |
| `local_pdf_dir` | string | `"pdf/"` | 로컬 PDF 디렉토리 |
| `enable_llm_verification` | boolean | `true` | LLM 검증 활성화 |

### 3.3 Purpose 선택 가이드

| Purpose | 사용 시점 | 특성 |
|---------|----------|------|
| `general` | 일반적인 추천 | 균형 잡힌 기준 |
| `literature_review` | 서베이, 관련 연구 조사 | 넓은 범위, 연도 제한 완화 |
| `implementation` | 직접 구현, 코드 실행 | 코드 필수, 없으면 탈락 |
| `idea_generation` | 트렌드 파악, 아이디어 탐색 | 최신성 극대화 |

### 3.4 Ranking Mode 선택 가이드

| Mode | 사용 시점 | 강조점 |
|------|----------|--------|
| `balanced` | 특별한 요구 없음 | 모든 요소 균형 |
| `novelty` | "새로운 거", "신선한 거" | 최신성, 독창성 |
| `practicality` | "바로 쓸 수 있는", "실용적인" | 코드 공개, 재현성 |
| `diversity` | "다양한 관점", "여러 접근법" | 중복 감점, 다양성 |

### 3.5 Contrastive Type 설명

| Type | 의미 | 예시 |
|------|------|------|
| `method` | 같은 문제, 다른 방법론 | Transformer 관심 → RNN/SSM 추천 |
| `assumption` | 같은 방법, 다른 가정 | Supervised 관심 → Self-supervised 추천 |
| `domain` | 같은 기술, 다른 분야 | NLP 관심 → Vision 적용 추천 |

---

## 4. 내부 처리 흐름

### 4.1 Phase 개요

```
Phase 0: 경로 해석 (Path Resolution)
    ↓
Phase 1: 사전 준비 (프로필, 히스토리, 로컬 PDF 로드)
    ↓
Phase 2: 필터링 (부적절한 논문 제거)
    ↓
Phase 3: 스코어링
    ├─ Stage 1: 임베딩 기반 빠른 평가 (전체)
    └─ Stage 2: LLM Batch 검증 (중간 그룹만)
    ↓
Phase 4: Soft 감점 적용
    ↓
Phase 5: 가중치 적용 및 최종 점수 계산
    ↓
Phase 6: 순위화 및 Top-K 선택
    ↓
Phase 7: Contrastive 논문 선택 (조건부)
    ↓
Phase 8: 태깅
    ↓
Phase 9: 결과 저장 및 반환
```

### 4.2 Phase 2: 필터링 상세

필터링 순서 (각 단계에서 제거 사유 기록):

1. **ALREADY_READ:** 히스토리에 있는 논문
2. **BLACKLIST_KEYWORD:** hard 제외 키워드 포함
3. **TOO_OLD:** min_year 이전 출판 (literature_review는 5년 완화)
4. **NO_CODE_REQUIRED:** implementation 모드에서 코드 없음

### 4.3 Phase 3: 스코어링 상세

`RankAndFilterPapersTool`의 점수 산정 방식은 단순한 선형 계산을 넘어 **시맨틱 분석, LLM 검증, 다차원 지표, 그리고 사용자의 연구 목적**을 결합한 하이브리드 시스템입니다.

점수 매기기 과정을 3가지 주요 단계로 나누어 상세히 설명해 드립니다.

---

#### 1단계: 시맨틱 스코어링 (기초 적합도 산출)
- **모델:** `sentence-transformers/all-MiniLM-L6-v2`
- **방식:** 사용자 interests와 논문 "제목+초록"의 코사인 유사도.

1.  **관심사 텍스트 가중치화**:
    *   사용자의 `UserProfile`에 정의된 관심 분야를 중요도에 따라 반복하여 하나의 긴 쿼리 텍스트로 만듭니다.
    *   **Primary (1.0)**: 텍스트 내 3회 반복 (가장 강력한 영향력)
    *   **Secondary (0.7)**: 텍스트 내 2회 반복
    *   **Exploratory (0.4)**: 텍스트 내 1회 반복
2.  **임베딩 유사도 계산 (`all-MiniLM-L6-v2`)**:
    *   생성된 쿼리 텍스트와 각 논문의 "제목 + 초록"을 벡터로 변환하여 **코사인 유사도**를 구합니다. (0.0 ~ 1.0 범위)
3.  **LLM 배치 검증 (Hybrid 방식)**:
    *   유사도 점수가 **0.4 ~ 0.7 사이인 '애매한' 그룹(Mid-group)**은 LLM(GPT-4o 등)에게 보내 한 번 더 검증합니다.
    *   LLM은 사용자의 관심사와 논문의 초록을 직접 읽고 0.0 ~ 1.0 사이의 점수를 부여합니다.
    *   최종 시맨틱 점수는 **High/Low 그룹은 임베딩 점수**를, **Mid 그룹은 LLM 점수**를 우선적으로 채택하여 정확도를 높입니다.
    - ** 효과 ** : API 호출 최소화, 정확도 향상

#### 2단계: 6대 차원별 지표 계산 (Dimension Scoring)
시맨틱 점수 외에 논문의 가치를 결정하는 6가지 요소를 각각 0.0 ~ 1.0 점수로 수치화합니다.

1.  **Semantic Relevance**: 1단계에서 산출된 최종 시맨틱 점수.
2.  **Must Keywords**: `must_include` 키워드가 제목/초록에 포함되었는지 확인. (없으면 1.0점, 요구사항이 있는데 안 맞으면 비율에 따라 감점)
3.  **Author Trust**: 사용자가 선호하는 저자(`preferred_authors`)가 포함되었는지 확인. (포함 시 1.0, 미포함 시 0.0. 단, 선호 저자 리스트가 없으면 모두 1.0)
4.  **Institution Trust**: 선호 기관(`preferred_institutions`) 소속 저자인지 확인. (기관명 부분 일치 시 1.0)
5.  **Recency (최신성)**: 발표일 기준으로 점수 차등 부여.
    *   2주 이내: 1.0 | 1개월 이내: 0.85 | 3개월 이내: 0.7 | 1년 초과: 0.1
6.  **Practicality (실용성)**: 현재는 주석 처리리
    *   GitHub URL 존재 시 **+0.5**
    *   이미 로컬에 PDF를 가지고 있을 시 **+0.3**
    *   합산하여 최대 1.0으로 클램핑.

#### 3단계: 가중치 적용 및 최종 점수 확정

이 단계에서 사용자의 **연구 목적(`purpose`)**과 **순위화 모드(`ranking_mode`)**가 개입합니다.

##### 1. 목적별 가중치(Weights) 적용
각 차원 점수에 가중치를 곱해 합산합니다.
*   **General**: 모든 지표를 균형 있게 합산.
*   **Literature Review**: 최신성(`Recency`) 비중을 낮추고, 실용성(`Practicality`)과 시맨틱 점수 비중을 높임.
*   **Implementation**: 실용성(`Practicality`) 가중치를 극대화.
*   **Idea Generation**: 최신성(`Recency`)과 시맨틱 점수 비중을 높임.

##### 2. 모드별 미세 조정 (Ranking Mode)
*   **Novelty 모드**: `Recency` 점수에 +0.1 보너스 부여.
*   **Practicality 모드**: `Practicality` 점수에 +0.1 보너스 부여.

##### 3. 소프트 패널티 (Soft Penalty) 적용
*   `exclude.soft` 키워드(피하고 싶은 주제)가 발견될 때마다 **최종 점수에서 -0.15점씩 감점**합니다. (최대 -0.3점)


#### 4단계: 다양성 필터링 (Diversity Adjustment)
상위권 논문들이 너무 비슷한 내용만 있을 경우를 방지합니다. (`ranking_mode="diversity"`인 경우)

1.  1위 논문을 먼저 확정합니다.
2.  다음 순위 논문을 뽑을 때, 이미 선택된 논문들과의 **임베딩 유사도**를 검사합니다.
3.  유사도가 **0.8 이상(매우 비슷함)**이면 해당 논문의 최종 점수를 **-0.2점 감점**한 후 다시 정렬합니다.
4.  이 과정을 반복하여 내용이 중복되지 않는 최적의 논문 세트를 구성합니다.

---

### 요약: 최종 점수 공식
> **최종 점수** = `Σ(차원 점수[1~6] × 목적별 가중치)` + `모드별 보너스` - `소프트 패널티`

이 결과는 **-0.5에서 1.5 사이**의 값으로 산출되며, 에이전트는 이 점수를 기준으로 `top_k`개의 논문을 최종 추천하게 됩니다. 
---
## 5. 결과 강화

### 5.1 대조 논문 선택 (_select_contrastive_paper) (조건부)
- 이미 선택된 상위 논문들과 성격이 다른 논문을 하나 포함시킵니다.(filter bubble 방지)
- 방법론(Method), 가정(Assumption), 도메인(Domain) 측면에서 차이점이 가장 뚜렷한 논문을 후보군 중 최종 점수가 높은 순으로 선정.

Step 1: 선택된 논문들의 공통 특성 추출
    ↓
Step 2: contrastive_type에 따라 대조 후보 필터링 (rankers.py 265~)
    ↓
Step 3: 후보 중 점수가 가장 높은 논문 선택
    ↓
Step 4: contrastive_info 생성 (대조 차원 정보)

### 5.2 태깅 및 비교 노트
- 태그 생성: CODE_AVAILABLE, MUST_KEYWORD_MATCH, OLDER_PAPER, CONTRASTIVE_PICK 등 시각적 태그 부여.
- 비교 노트: 유사한 접근 방식을 가진 논문 쌍을 찾고, 공통 키워드와 차별점(Differentiator)을 텍스트로 생성.

### 5.3 저장 (_save_and_format_result)
- 타임스탬프 기반 파일명(예: 2024-05-20_143005_ranked.json)으로 output/rankings/ 디렉토리에 저장.

## 6. 사용자 프로필

### 6.1 프로필 구조

```json
{
  "interests": {
    "primary": ["핵심 관심 주제"],
    "secondary": ["관련 관심 주제"],
    "exploratory": ["탐색적 관심 주제"]
  },
  "keywords": {
    "must_include": ["필수 키워드"],
    "exclude": {
      "hard": ["무조건 제외"],
      "soft": ["감점만 적용"]
    }
  },
  "preferred_authors": ["선호 저자"],
  "preferred_institutions": ["선호 기관"],
  "constraints": {
    "min_year": 2024,
    "require_code": false
  }
}
```

### 5.2 프로필 예시

```json
{
  "interests": {
    "primary": ["LLM inference optimization", "efficient transformers"],
    "secondary": ["model compression", "quantization"],
    "exploratory": ["neural architecture search"]
  },
  "keywords": {
    "must_include": [],
    "exclude": {
      "hard": ["medical", "clinical", "drug discovery"],
      "soft": ["survey", "benchmark", "dataset"]
    }
  },
  "preferred_authors": ["Tri Dao", "Song Han"],
  "preferred_institutions": ["Stanford", "MIT", "OpenAI", "DeepMind"],
  "constraints": {
    "min_year": 2024,
    "require_code": false
  }
}
```

### 5.3 프로필이 없을 때

- **필터링:** 최소한의 기본 필터만 적용
- **스코어링:** 최신성, 코드 공개 여부 등 프로필 무관 요소만 반영

---

## 6. 출력 구조

### 6.1 반환 객체

```json
{
  "success": true,
  "error": null,
  "summary": {
    "input_count": 15,
    "filtered_count": 3,
    "scored_count": 12,
    "output_count": 5,
    "purpose": "general",
    "ranking_mode": "balanced",
    "profile_used": "config/profile.json",
    "llm_verification_used": true,
    "llm_calls_made": 1
  },
  "ranked_papers": [...],
  "filtered_papers": [...],
  "contrastive_paper": {...},
  "comparison_notes": [...],
  "output_path": "output/rankings/2025-01-03_143000_ranked.json",
  "generated_at": "2025-01-03T14:30:00"
}
```

### 6.2 ranked_papers 항목 예시

```json
{
  "rank": 1,
  "paper_id": "2501.12345",
  "title": "논문 제목",
  "authors": ["저자1", "저자2"],
  "published": "2025-01-15",
  "score": {
    "final": 0.87,
    "breakdown": {
      "semantic_relevance": 0.92,
      "must_keywords": 1.0,
      "author_trust": 1.0,
      "institution_trust": 0.8,
      "recency": 0.95,
      "practicality": 0.8
    },
    "soft_penalty": -0.15,
    "penalty_keywords": ["survey"],
    "evaluation_method": "embedding+llm"
  },
  "tags": ["SEMANTIC_HIGH_MATCH", "PREFERRED_AUTHOR", "CODE_AVAILABLE"],
  "local_status": {
    "already_downloaded": false,
    "local_path": null
  },
  "original_data": {...}
}
```

### 6.3 Selection Tags

#### 긍정 태그

| 태그                | 조건                       | UI 배지        |
|---------------------|---------------------------|---------------|
| SEMANTIC_HIGH_MATCH | semantic ≥ 0.7            | "높은 관련성"  |
| PREFERRED_AUTHOR    | 선호 저자 포함            | "주목 저자"    |
| PREFERRED_INSTITUTION | 선호 기관 소속          | "주요 기관"    |
| CODE_AVAILABLE      | github_url 존재           | "코드 공개"    |
| VERY_RECENT         | 2주 이내                  | "최신"         |
| ALREADY_DOWNLOADED  | 로컬 보유                 | "보유 중"      |

#### 주의 태그

| 태그          | 조건         |
|---------------|--------------|
| NO_CODE       | github_url 없음 |
| OLDER_PAPER   | 3개월 이상 경과 |
| SOFT_PENALTY:{keyword} | soft 키워드 포함 |

#### 특수 태그

| 태그                 | 조건          |
|----------------------|--------------|
| CONTRASTIVE_PICK     | 대조적 선택      |
| CONTRASTIVE_METHOD   | 방법론 대조      |
| CONTRASTIVE_ASSUMPTION | 가정 대조    |
| CONTRASTIVE_DOMAIN   | 도메인 대조     |

---

## 7. 테스트
- 필터링 테스트: 블랙리스트 키워드 및 중복 논문 제거 확인.
- 점수 테스트: 다차원 점수 합산 및 소프트 감점 정확도 검증.
- 모드 테스트: 다양성(Diversity) 모드 작동 및 대조 논문 선택 로직 검증.
- 예외 처리: 빈 입력값, 프로필 부재, 잘못된 경로에 대한 안정성 확인.
- 통합 파이프라인: 10개 이상의 Mock 데이터를 이용한 전체 프로세스(execute) 성공 여부 확인.

-> 전부 통과 확인인
---

## 8. 에러 처리

| 상황               | 처리                 | 결과                    |
|--------------------|----------------------|-------------------------|
| 빈 papers 배열     | 정상 처리            | success: true, 빈 결과  |
| papers 형식 오류   | 즉시 중단            | success: false, 에러 메시지 |
| 프로필 파일 없음   | 기본값 진행          | profile_used: null      |
| 임베딩 모델 로드 실패 | 키워드 매칭 폴백     | 정상 진행              |
| LLM 호출 실패      | 임베딩 점수만 사용   | 정상 진행              |
| 모든 논문 필터링됨 | 정상 처리            | 빈 ranked_papers        |

