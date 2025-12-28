##1️⃣ list_pdfs

#기능

설정된 PDF 디렉터리(PDF_DIR) 내의 모든 PDF 파일을 나열

#세부 동작

*.pdf 파일 스캔

각 파일에 대해:

파일명

전체 경로

파일 크기 (bytes / MB)

파일명을 기준으로 정렬

출력 정보

PDF 디렉터리 경로

전체 PDF 개수

PDF 파일 목록 + 크기 정보

##2️⃣ extract_text

기능

단일 PDF 파일에서 모든 페이지의 텍스트를 추출

입력

filename (PDF 파일명)

세부 동작

PDF 파일 열기

각 페이지별로:

전체 텍스트 추출 (page.get_text())

페이지 번호 포함

출력 정보

파일명

전체 페이지 수

페이지별 텍스트 리스트

##3️⃣ extract_images

기능

단일 PDF 파일에서 모든 이미지를 추출하여 파일로 저장

입력

filename

세부 동작

PDF 파일 열기

페이지별 이미지 탐색

각 이미지에 대해:

실제 이미지 바이너리 추출

output/{pdf_name}/images/에 저장

이미지 추출 실패 시 오류 기록

출력 정보

파일명

페이지 수

이미지 메타데이터 목록

전체 추출된 이미지 개수

##4️⃣ extract_all

기능

단일 PDF 파일에 대해 텍스트 + 이미지 전체 추출을 한 번에 수행

입력

filename

세부 동작

extract_text 실행

결과를:

extracted_text.json

extracted_text.txt
로 저장

extract_images 실행

이미지 메타데이터를 image_metadata.json으로 저장

출력 정보

출력 디렉터리 경로

텍스트 추출 결과 파일 경로

이미지 추출 결과 및 이미지 저장 경로

##5️⃣ process_all_pdfs

기능

PDF 디렉터리 내 모든 PDF 파일을 일괄 처리

세부 동작

list_pdfs로 전체 PDF 목록 획득

각 PDF에 대해 extract_all 실행

파일별 성공 / 실패 상태 기록

출력 정보

전체 처리 개수

성공 개수

실패 개수

PDF별 처리 결과 목록

##6️⃣ get_pdf_info

기능

단일 PDF 파일의 메타데이터 및 구조 정보 조회

입력

filename

세부 동작

PDF 메타데이터 읽기

페이지 수 계산

파일 크기 계산

페이지별 이미지 개수 계산

출력 정보

파일 경로

페이지 수

메타데이터

파일 크기

페이지별 이미지 개수

전체 이미지 수

##7️⃣ read_extracted_text

기능

이전에 추출된 텍스트 결과를 다시 읽어오기

입력

filename (확장자 유무 무관)

세부 동작

output/{pdf_name}/extracted_text.txt 로드

텍스트가 너무 길 경우 자동 truncation (50,000자)

출력 정보

파일명

텍스트 파일 경로

텍스트 내용

truncation 여부

📌 기능 범위 요약 표
Tool 이름	기능 요약
list_pdfs	PDF 목록 조회
extract_text	단일 PDF 텍스트 추출
extract_images	단일 PDF 이미지 추출
extract_all	텍스트 + 이미지 통합 추출
process_all_pdfs	전체 PDF 일괄 처리
get_pdf_info	PDF 메타데이터/구조 조회
read_extracted_text	기존 텍스트 결과 재조회
