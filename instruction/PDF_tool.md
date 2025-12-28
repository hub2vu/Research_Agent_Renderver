# 📄 PDF Processing Tools

이 모듈은 **PyMuPDF(fitz)** 기반으로 PDF 파일을 분석·처리하기 위한 MCP Tool 집합이다.  
PDF 목록 조회부터 텍스트/이미지 추출, 메타데이터 분석, 일괄 처리까지  
**PDF 처리의 전체 라이프사이클을 포괄**한다.

---

## 📂 기본 동작 디렉터리

- **PDF 입력 디렉터리**: `PDF_DIR` (기본값: `/data/pdf`)
- **출력 디렉터리**: `OUTPUT_DIR` (기본값: `/data/output`)

환경 변수로 경로를 변경할 수 있다.

---

## 🧩 Tool 목록

### 1. `list_pdfs`

**설명**  
PDF 디렉터리 내의 모든 PDF 파일을 나열한다.

**주요 기능**
- `*.pdf` 파일 탐색
- 파일명 기준 정렬
- 파일 크기(bytes / MB) 계산

**출력**
- PDF 디렉터리 경로
- 전체 PDF 개수
- 파일 목록 (파일명, 경로, 크기)

---

### 2. `extract_text`

**설명**  
단일 PDF 파일에서 **모든 페이지의 텍스트를 추출**한다.

**입력 파라미터**
- `filename` (string, required): PDF 파일명

**주요 기능**
- 페이지 단위 텍스트 추출
- 페이지 번호 포함

**출력**
- 파일명
- 전체 페이지 수
- 페이지별 텍스트 리스트

---

### 3. `extract_images`

**설명**  
단일 PDF 파일에서 **모든 이미지를 추출하여 파일로 저장**한다.

**입력 파라미터**
- `filename` (string, required)

**주요 기능**
- 페이지별 이미지 탐색
- 이미지 파일 저장 (`output/{pdf_name}/images/`)
- 이미지 포맷, 크기 기록
- 이미지 추출 실패 시 오류 기록

**출력**
- 파일명
- 페이지 수
- 이미지 메타데이터 목록
- 전체 추출 이미지 개수

---

### 4. `extract_all`

**설명**  
단일 PDF 파일에 대해 **텍스트 + 이미지 추출을 한 번에 수행**한다.

**입력 파라미터**
- `filename` (string, required)

**주요 기능**
1. 텍스트 추출
2. 텍스트 결과 저장
   - `extracted_text.json`
   - `extracted_text.txt`
3. 이미지 추출
4. 이미지 메타데이터 저장 (`image_metadata.json`)

**출력**
- 출력 디렉터리 경로
- 텍스트 추출 결과 파일 경로
- 이미지 추출 결과 요약

---

### 5. `process_all_pdfs`

**설명**  
PDF 디렉터리 내 **모든 PDF 파일을 일괄 처리**한다.

**주요 기능**
- 모든 PDF에 대해 `extract_all` 실행
- 파일별 성공/실패 상태 기록

**출력**
- 전체 처리 개수
- 성공 개수
- 실패 개수
- PDF별 처리 결과 목록

---

### 6. `get_pdf_info`

**설명**  
단일 PDF 파일의 **메타데이터 및 구조 정보**를 조회한다.

**입력 파라미터**
- `filename` (string, required)

**주요 기능**
- PDF 메타데이터 조회
- 페이지 수 계산
- 파일 크기 계산
- 페이지별 이미지 개수 계산

**출력**
- 파일 경로
- 페이지 수
- 메타데이터
- 파일 크기
- 페이지별 이미지 수
- 전체 이미지 수

---

### 7. `read_extracted_text`

**설명**  
이전에 추출된 **텍스트 결과를 다시 읽어온다**.

**입력 파라미터**
- `filename` (string, required, 확장자 생략 가능)

**주요 기능**
- `extracted_text.txt` 로드
- 최대 50,000자 제한 자동 적용 (초과 시 truncation)

**출력**
- 파일명
- 텍스트 파일 경로
- 텍스트 내용
- truncation 여부

---

## 📊 기능 요약 표

| Tool 이름 | 기능 요약 |
|---------|----------|
| `list_pdfs` | PDF 파일 목록 조회 |
| `extract_text` | PDF 텍스트 추출 |
| `extract_images` | PDF 이미지 추출 |
| `extract_all` | 텍스트 + 이미지 통합 추출 |
| `process_all_pdfs` | 전체 PDF 일괄 처리 |
| `get_pdf_info` | PDF 메타데이터/구조 조회 |
| `read_extracted_text` | 기존 텍스트 결과 재조회 |

---

## 🎯 요약

> **PDF Processing Tools는  
> PDF 수집 → 분석 → 저장 → 재활용까지의 전 과정을  
> MCP Tool 단위로 완결한 처리 파이프라인이다.**

