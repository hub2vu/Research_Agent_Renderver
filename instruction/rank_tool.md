## 1. 개요

### 1.1 Tool 정체성

| 항목 | 내용 |
| --- | --- |
| **이름** | `rank_and_filter_papers` |
| **역할** | 게이트키퍼 (Gatekeeper) - 메타데이터 기반 저비용 선별 |
| **한 줄 설명** | 검색된 논문의 메타데이터(제목, 저자, 초록)를 사용자 프로필과 대조하여 필터링하고, 다차원 스코어링으로 Top-K를 선별한다 |

### 1.2 핵심 가치

**비용 효율성:** arxiv_search가 반환한 수십 개 논문을 Agent가 직접 분석하면 토큰 비용과 시간이 과다 소모됩니다. 이 Tool은 메타데이터만 활용한 경량 평가로 고비용 작업(PDF 전문 분석)의 대상을 사전 선별합니다.

### 1.3 이 Tool이 하는 것 / 하지 않는 것

**한다:**

- 메타데이터(제목, 저자, 소속, 초록) 기반 필터링 및 스코어링
- 임베딩 기반 의미 유사도 계산 (로컬 모델)
- 애매한 케이스에 대한 LLM Batch 검증 (선택적)
- 로컬 보유 현황(pdf/ 디렉토리) 확인 및 태깅
- 대조적 논문 선택 및 비교 데이터 생성
- 선정 사유 태그 및 비교 정보를 데이터 형태로 제공

**하지 않는다:**

- 논문 검색 (arxiv_search의 역할)
- PDF 본문(Full-text) 분석 (paper_analyzer의 역할)
- 코드 재현성 검증 (repro_checker의 역할)
- 외부 API를 통한 인용 수 조회
- 자연어 리포트/문장 생성 (Agent의 역할)

**paper_analyzer와의 핵심 차이:**

| 구분 | 이 Tool | paper_analyzer |
| --- | --- | --- |
| 분석 대상 | 초록 (Abstract) | PDF 본문 전체 (Full-text) |
| 처리 비용 | 낮음 | 높음 |
| 목적 | 분석 대상 선별 | 심층 내용 분석 |

---

## 2. 입력 (Parameters)

### 2.1 필수 파라미터

### `papers`

- **타입:** 배열 (Array of Objects)
- **설명:** 검색 도구가 반환한 논문 리스트

**각 논문 객체의 필수 필드:**

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `paper_id` | string | arXiv ID (예: "2501.12345") |
| `title` | string | 논문 제목 |
| `abstract` | string | 초록 |
| `authors` | array | 저자 리스트 |

**각 논문 객체의 선택 필드:**

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `published` | string | 출판일 (YYYY-MM-DD) |
| `categories` | array | arXiv 카테고리 |
| `pdf_url` | string | PDF 다운로드 링크 |
| `github_url` | string | GitHub 저장소 링크 |
| `affiliations` | array | 저자 소속 기관 리스트 |
** 현재는 github_url 관련 코드 주석 처리 **

### 2.2 선택 파라미터

### `top_k`

- **타입:** 정수
- **기본값:** 5
- **범위:** 1-20 권장
- **설명:** 반환할 최대 논문 수

### `profile_path`

- **타입:** 문자열
- **기본값:** `"config/profile.json"` (상대 경로)
- **설명:** 사용자 프로필 파일 경로. 상대 경로 입력 시 환경 변수 `OUTPUT_DIR` 기준으로 해석. 파일 없으면 기본 스코어링만 수행.

### `purpose`

- **타입:** 문자열 (열거형)
- **선택지:** `"general"`, `"literature_review"`, `"implementation"`, `"idea_generation"`
- **기본값:** `"general"`
- **설명:** 사용자의 현재 목적

| 목적 | 특성 |
| --- | --- |
| `general` | 범용 추천, 균형 잡힌 기준 |
| `literature_review` | 넓은 범위, 다양성 중시, 연도 제한 완화 |
| `implementation` | 코드 필수, 코드 없으면 탈락 |
| `idea_generation` | 최신성 극대화, 독창적 접근 우선 |

### `ranking_mode`

- **타입:** 문자열 (열거형)
- **선택지:** `"balanced"`, `"novelty"`, `"practicality"`, `"diversity"`
- **기본값:** `"balanced"`
- **설명:** 스코어링 시 강조할 세부 기준

| 모드 | 강조점 |
| --- | --- |
| `balanced` | 모든 요소 균형 |
| `novelty` | 최신성, 독창적 방법론 |
| `practicality` | 코드 공개, 재현 용이성 |
| `diversity` | 다양한 관점, 중복 내용 감점 |

### `include_contrastive`

- **타입:** 불리언
- **기본값:** `false`
- **설명:** `true`일 경우 Top-K 중 마지막 1개를 대조적 관점의 논문으로 대체

### `contrastive_type`

- **타입:** 문자열 (열거형)
- **선택지:** `"method"`, `"assumption"`, `"domain"`
- **기본값:** `"method"`
- **조건:** `include_contrastive`가 `true`일 때만 유효

| 타입 | 의미 |
| --- | --- |
| `method` | 같은 문제, 다른 방법론 |
| `assumption` | 같은 방법, 다른 가정 |
| `domain` | 같은 기술, 다른 적용 분야 |

### `history_path`

- **타입:** 문자열
- **기본값:** `"history/read_papers.json"` (상대 경로)
- **설명:** 이미 읽은 논문 ID 목록

### `local_pdf_dir`

- **타입:** 문자열
- **기본값:** `"pdf/"` (상대 경로)
- **설명:** 로컬 PDF 저장 디렉토리

### `enable_llm_verification`

- **타입:** 불리언
- **기본값:** `true`
- **설명:** 임베딩 점수가 애매한 논문에 대해 LLM 검증 수행 여부. `false`면 임베딩 점수만 사용 (더 빠름).

---

## 3. 사용자 프로필 구조

### 3.1 profile.json 스키마

```
{
  "interests": {
    "primary": [핵심 연구 주제],
    "secondary": [관련 관심 주제],
    "exploratory": [탐색적 관심 주제]
  },
  "keywords": {
    "must_include": [반드시 포함 키워드],
    "exclude": {
      "hard": [무조건 제외],
      "soft": [감점만 적용]
    }
  },
  "preferred_authors": [선호 저자 이름],
  "preferred_institutions": [선호 기관/연구소],
  "constraints": {
    "min_year": 최소 출판 연도,
    "require_code": 코드 필수 여부
  }
}

```

### 3.2 프로필 예시

```
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
  "preferred_authors": ["Tri Dao", "Song Han", "William Dally"],
  "preferred_institutions": ["Stanford", "MIT", "Berkeley", "OpenAI", "DeepMind"],
  "constraints": {
    "min_year": 2024,
    "require_code": false
  }
}

```

### 3.3 프로필 없을 때 기본 동작

- 필터링: 최소한의 기본 필터만 적용
- 스코어링: 최신성, 코드 공개 여부 등 프로필 무관 요소만 반영

---

## 4. 내부 처리 흐름

### Phase 0: 경로 해석 (Path Resolution)

모든 파일 경로에 대해 다음 순서로 해석:

1. 절대 경로인 경우: 그대로 사용
2. 상대 경로인 경우:
    - 환경 변수 `OUTPUT_DIR`이 설정되어 있으면: `{OUTPUT_DIR}/{relative_path}`
    - 환경 변수 `PDF_DIR`이 설정되어 있으면 (pdf 경로에 한해): `{PDF_DIR}/{relative_path}`
    - 환경 변수 없으면: 현재 작업 디렉토리 기준

**Docker 환경 예시:**

```
호스트: ./output → 컨테이너: /data/output
환경 변수: OUTPUT_DIR=/data/output

입력: profile_path="config/profile.json"
해석: /data/output/config/profile.json

```

### Phase 1: 사전 준비

**1-1. 프로필 로드**

- 경로 해석 후 프로필 로드
- 파일 없으면 기본값 사용, 로그에 기록

**1-2. 히스토리 로드**

- 읽은 논문 ID 목록 로드
- 파일 없으면 빈 리스트

**1-3. 로컬 보유 현황 스캔**

- `local_pdf_dir` 스캔
- 파일명에서 paper_id 추출
- 보유 중인 논문 ID 목록 생성

### Phase 2: 필터링 (제거 단계)

순서대로 적용, 각 제거 건은 사유와 함께 기록:

**2-1. 중복 제거**

- `history`에 있는 paper_id → 제거
- 사유: `ALREADY_READ`

**2-2. Hard 키워드 필터**

- `keywords.exclude.hard` 단어가 제목/초록에 존재 → 제거
- 사유: `BLACKLIST_KEYWORD:{keyword}`

**2-3. 연도 필터**

- `constraints.min_year`보다 이전 → 제거
- 사유: `TOO_OLD:{year}`
- 예외: `purpose`가 `literature_review`면 min_year를 5년 전으로 완화

**2-4. 코드 필수 필터 (조건부)**

- `purpose`가 `implementation`이거나 `constraints.require_code`가 true
- `github_url` 없음 → 제거
- 사유: `NO_CODE_REQUIRED`

### Phase 3: 스코어링 (2단계 평가)

### Stage 1: 임베딩 기반 빠른 평가 (전체 논문)

로컬 임베딩 모델(sentence-transformers)을 사용한 벡터 유사도 계산:

**3-1-1. 임베딩 생성**

- 사용자 interests (primary/secondary/exploratory)를 하나의 텍스트로 결합 → 임베딩
- 각 논문의 "제목 + 초록"을 결합 → 임베딩

**3-1-2. 코사인 유사도 계산**

- 각 논문과 사용자 interests 간 유사도 계산
- 0~1 범위의 `embedding_score` 산출

**3-1-3. 논문 분류**

- 상위 그룹 (embedding_score ≥ 0.7): 높은 관련성
- 중간 그룹 (0.4 ≤ embedding_score < 0.7): 애매한 케이스
- 하위 그룹 (embedding_score < 0.4): 낮은 관련성

### Stage 2: LLM Batch 검증 (중간 그룹만, 조건부)

`enable_llm_verification`이 `true`이고 중간 그룹이 존재할 때만 실행:

**3-2-1. Batch 구성**

- 중간 그룹 논문들을 10~15개 단위로 묶음
- 각 논문: {paper_id, title, abstract 앞 500자}

**3-2-2. Batch LLM 호출**

프롬프트 구조:

```
사용자 연구 관심사:
- Primary: {interests.primary}
- Secondary: {interests.secondary}

다음 논문들이 위 관심사와 얼마나 관련있는지 평가하세요.
각 논문에 대해 0.0~1.0 점수와 한 줄 판단 근거를 제공하세요.

논문 목록:
1. [ID: 2501.001] 제목: ... / 초록: ...
2. [ID: 2501.002] 제목: ... / 초록: ...
...

응답 형식 (JSON):
[
  {"paper_id": "2501.001", "relevance": 0.75, "reason": "효율적 추론 관련"},
  ...
]

```

**3-2-3. 점수 병합**

- LLM 검증 결과로 중간 그룹의 `semantic_score` 갱신
- 상위/하위 그룹은 embedding_score를 그대로 사용

### Stage 3: 다차원 점수 계산 (전체 논문)

6가지 요소 점수 계산 (각 0~1):

| 요소 | 계산 방법 |
| --- | --- |
| **의미적 관련성** | Stage 1-2에서 산출된 점수 |
| **필수 키워드** | must_include 키워드 포함 비율 |
| **저자 신뢰도** | preferred_authors 포함 시 1.0 |
| **기관 신뢰도** | preferred_institutions 부분 문자열 매칭 |
| **최신성** | 출판일 기준 (2주 이내 1.0 → 1년 초과 0.1) |
| **실용성** | 코드 공개 +0.5, 로컬 보유 +0.3 |
현재난 단순한 키워드 매칭칭

### Phase 4: Soft 감점 적용

- `keywords.exclude.soft` 키워드 포함 시 -0.15씩
- 최대 감점: -0.3

### Phase 5: 가중치 적용 및 최종 점수

**5-1. Purpose별 기본 가중치**

| 요소 | general | literature_review | implementation | idea_generation |
| --- | --- | --- | --- | --- |
| 의미적 관련성 | 0.30 | 0.25 | 0.20 | 0.25 |
| 필수 키워드 | 0.10 | 0.10 | 0.10 | 0.15 |
| 저자 신뢰도 | 0.15 | 0.15 | 0.10 | 0.10 |
| 기관 신뢰도 | 0.10 | 0.10 | 0.10 | 0.05 |
| 최신성 | 0.20 | 0.15 | 0.10 | 0.35 |
| 실용성 | 0.15 | 0.10 | 0.40 | 0.10 |

**5-2. Ranking Mode 미세 조정**

- `novelty`: 최신성 +0.1, 저자/기관 -0.05씩
- `practicality`: 실용성 +0.1, 최신성 -0.1
- `diversity`: Phase 6에서 별도 처리

**5-3. 최종 점수**

```
final_score = Σ(요소별 점수 × 가중치) + soft_penalty

```

### Phase 6: 순위화 및 선별

**6-1. 기본 정렬**

- final_score 내림차순

**6-2. Diversity 모드 처리**

- Top-1 선택 후, 이후 논문은 이미 선택된 논문과의 임베딩 유사도 계산
- 유사도 높으면 감점 (-0.2)

**6-3. Top-K 추출**

- include_contrastive가 true면 K-1개 선택

### Phase 7: Contrastive 논문 선택 (조건부)

`include_contrastive`가 `true`일 때:

**7-1. 기존 선택 논문의 공통 특성 추출**

- 선택된 K-1개 논문의 임베딩 평균 계산
- 주요 카테고리, 방법론 키워드 추출

**7-2. 대조 후보 탐색**

- 필터링 통과했지만 Top-K 미선정 논문 중에서
- contrastive_type에 따라:
    - `method`: 다른 카테고리 또는 방법론 키워드
    - `assumption`: 다른 학습 패러다임 키워드 (supervised vs unsupervised 등)
    - `domain`: 다른 arXiv 카테고리

**7-3. 대조 정보 데이터 생성**

자연어 문장이 아닌 구조화된 데이터로 제공:

```
contrastive_info: {
  "type": "method",
  "selected_papers_common_traits": ["transformer", "attention", "cs.CL"],
  "this_paper_traits": ["state-space-model", "linear-complexity", "cs.LG"],
  "contrast_dimensions": [
    {"dimension": "architecture", "others": "Transformer", "this": "SSM"},
    {"dimension": "complexity", "others": "quadratic", "this": "linear"}
  ]
}

```

### Phase 8: 태깅

**8-1. Selection Reason Tags**

긍정 태그:

| 태그 | 조건 | UI 배지 |
| --- | --- | --- |
| `SEMANTIC_HIGH_MATCH` | 의미적 관련성 ≥ 0.7 | "높은 관련성" |
| `PREFERRED_AUTHOR` | 선호 저자 포함 | "주목 저자" |
| `PREFERRED_INSTITUTION` | 선호 기관 소속 | "주요 기관" |
| `CODE_AVAILABLE` | github_url 존재 | "코드 공개" |
| `VERY_RECENT` | 2주 이내 | "최신" |
| `ALREADY_DOWNLOADED` | 로컬 보유 | "보유 중" |
| `MUST_KEYWORD_MATCH` | 필수 키워드 포함 | "키워드 일치" |
| `LLM_VERIFIED` | LLM 검증 통과 | - |

주의 태그:

| 태그 | 조건 |
| --- | --- |
| `NO_CODE` | github_url 없음 |
| `OLDER_PAPER` | 3개월 이상 경과 |
| `SOFT_PENALTY:{keyword}` | soft 제외 키워드 포함 |

특수 태그:

| 태그 | 조건 |
| --- | --- |
| `CONTRASTIVE_PICK` | 대조적 선택 |
| `CONTRASTIVE_METHOD` | 방법론 대조 |
| `CONTRASTIVE_ASSUMPTION` | 가정 대조 |
| `CONTRASTIVE_DOMAIN` | 도메인 대조 |

**8-2. Comparison Notes (데이터 형태)**

Agent가 리포트 작성 시 활용할 비교 정보:

```
comparison_notes: [
  {
    "paper_ids": ["2501.001", "2501.002"],
    "relation": "similar_approach",
    "shared_traits": ["transformer", "efficient-attention"],
    "differentiator": "2501.001 focuses on training, 2501.002 on inference"
  },
  {
    "paper_ids": ["2501.001", "2501.005"],
    "relation": "contrastive",
    "contrast_point": "architecture_type"
  }
]

```

### Phase 9: 결과 저장

`{OUTPUT_DIR}/rankings/{date}_{timestamp}_ranked.json`에 저장

---

## 5. 출력 (Return Value)

### 5.1 반환 객체 전체 구조

```
{
  "success": true/false,
  "error": null 또는 에러 메시지,

  "summary": {
    "input_count": 입력 논문 수,
    "filtered_count": 필터링 제거 수,
    "scored_count": 스코어링 대상 수,
    "output_count": 최종 반환 수,
    "purpose": 사용된 목적,
    "ranking_mode": 사용된 모드,
    "profile_used": 프로필 경로 또는 null,
    "llm_verification_used": true/false,
    "llm_calls_made": LLM 호출 횟수 (Batch 단위)
  },

  "ranked_papers": [...],
  "filtered_papers": [...],
  "contrastive_paper": {...} 또는 null,
  "comparison_notes": [...],

  "output_path": 결과 저장 경로,
  "generated_at": ISO 타임스탬프
}

```

### 5.2 ranked_papers 각 항목

```
{
  "rank": 순위,
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
    "evaluation_method": "embedding+llm" 또는 "embedding_only"
  },

  "tags": ["SEMANTIC_HIGH_MATCH", "PREFERRED_AUTHOR", "CODE_AVAILABLE"],

  "local_status": {
    "already_downloaded": true/false,
    "local_path": "pdf/2501.12345.pdf" 또는 null
  },

  "original_data": { 원본 논문 객체 }
}

```

### 5.3 filtered_papers 각 항목

```
{
  "paper_id": "2501.00456",
  "title": "제거된 논문 제목",
  "filter_reason": "BLACKLIST_KEYWORD:medical",
  "filter_phase": 2
}

```

### 5.4 contrastive_paper (해당 시)

```
{
  "paper_id": "2501.00789",
  "title": "대조 논문 제목",
  "score": { ... },
  "tags": ["CONTRASTIVE_PICK", "CONTRASTIVE_METHOD"],

  "contrastive_info": {
    "type": "method",
    "selected_papers_common_traits": ["transformer", "attention"],
    "this_paper_traits": ["state-space-model", "linear-complexity"],
    "contrast_dimensions": [
      {"dimension": "architecture", "others": "Transformer", "this": "SSM"}
    ]
  },

  "original_data": { ... }
}

```

### 5.5 comparison_notes

```
[
  {
    "paper_ids": ["2501.001", "2501.002"],
    "relation": "similar_approach",
    "shared_traits": ["transformer", "efficient-attention"],
    "differentiator": "training_vs_inference"
  }
]

```

---

## 6. 파일 I/O 명세

### 6.1 환경 변수

| 변수명 | 용도 | 기본값 |
| --- | --- | --- |
| `OUTPUT_DIR` | output 관련 파일 기준 경로 | 현재 작업 디렉토리 |
| `PDF_DIR` | PDF 파일 기준 경로 | `OUTPUT_DIR` |

### 6.2 읽기 (Read)

| 파일 | 필수 | 없을 때 |
| --- | --- | --- |
| `{profile_path}` | 선택 | 기본값 사용 |
| `{history_path}` | 선택 | 중복 체크 스킵 |
| `{local_pdf_dir}` | 선택 | 보유 현황 스킵 |

### 6.3 쓰기 (Write)

| 파일 | 용도 |
| --- | --- |
| `{OUTPUT_DIR}/rankings/{date}_{timestamp}_ranked.json` | 전체 결과 |

---

## 7. 에러 처리

| 상황 | 처리 | 반환 |
| --- | --- | --- |
| `papers` 빈 배열 | 정상 처리 | success: true, 빈 결과 |
| `papers` 형식 오류 | 즉시 중단 | success: false, 에러 메시지 |
| 프로필 파일 없음 | 기본값 진행 | profile_used: null |
| 프로필 형식 오류 | 즉시 중단 | success: false, 에러 메시지 |
| 임베딩 모델 로드 실패 | 키워드 매칭으로 폴백 | 경고 로그, 정상 진행 |
| LLM 호출 실패 | 임베딩 점수만 사용 | 경고 로그, 정상 진행 |
| 모든 논문 필터링됨 | 정상 처리 | 빈 ranked_papers |
| Contrastive 후보 없음 | 스킵 | contrastive_paper: null |

---

## 8. Agent를 위한 Description

```
검색 결과의 메타데이터(제목, 저자, 초록)를 분석하여
사용자 맞춤으로 평가하고 선별하는 도구입니다.

[분석 범위]
- 이 도구: 초록(Abstract)까지만 분석
- paper_analyzer: PDF 본문 전체 분석
→ 이 도구로 먼저 선별 후, 선별된 논문만 paper_analyzer로 심층 분석

[언제 사용하나요?]
- arxiv_search로 논문 리스트를 받은 직후
- "내 관심사에 맞게 골라줘", "중요한 것만 추려줘"
- 분석 대상을 좁힐 때

[언제 사용하면 안 되나요?]
- 논문 검색 → arxiv_search
- 논문 내용 상세 분석 → paper_analyzer
- 코드 재현성 검증 → repro_checker

[purpose 선택]
- "트렌드", "요즘 핫한" → idea_generation
- "직접 구현", "코드 돌려볼" → implementation
- "서베이", "관련 연구 조사" → literature_review
- 그 외 → general

[ranking_mode 선택]
- "새로운", "신선한" → novelty
- "실용적", "바로 쓸 수 있는" → practicality
- "다양한 관점" → diversity
- 그 외 → balanced

[include_contrastive 선택]
- "다른 시각", "반대 의견" → true
- "관련된 것만" → false

[결과 활용]
- tags: UI 배지 표시용
- comparison_notes: 논문 간 비교 설명 작성 시 참고
- contrastive_info: 대조 논문 설명 작성 시 참고

```

---

## 9. 성능 특성

### 9.1 예상 처리 시간

| 입력 규모 | 임베딩만 | 임베딩+LLM |
| --- | --- | --- |
| 10개 | ~1초 | ~3초 |
| 30개 | ~2초 | ~5초 (1 batch) |
| 50개 | ~3초 | ~8초 (2 batches) |

### 9.2 LLM 호출 최적화

- Batch 크기: 10~15개
- 중간 그룹(0.4~0.7)만 LLM 검증
- 예상 LLM 호출: 입력 30개 기준 0~2회