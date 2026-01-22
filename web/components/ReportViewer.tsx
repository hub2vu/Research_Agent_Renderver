/**
 * ReportViewer Component
 *
 * Fetches and displays the plain text report for a specific paper.
 */

import React, { useState } from 'react';

interface ReportViewerProps {
    paperId: string;
}

export default function ReportViewer({ paperId }: ReportViewerProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [reportContent, setReportContent] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleToggle = async () => {
        // 1. 닫혀있으면 -> 연다 (데이터 없으면 가져오기)
        if (!isOpen) {
            if (!reportContent) {
                await fetchReport();
            }
            setIsOpen(true);
        }
        // 2. 열려있으면 -> 닫는다
        else {
            setIsOpen(false);
        }
    };

    const fetchReport = async () => {
        setIsLoading(true);
        setError(null);
        try {
            // ✅ 백엔드 API 호출 (서버 주소에 맞게 수정 필요, 예: /api/reports/...)
            // setupProxy.js를 쓰고 있다면 상대 경로 '/api/reports' 사용
            // 직접 호출이라면 'http://localhost:8000/reports' 등 사용
            const res = await fetch(`http://localhost:8000/reports/${paperId}`);

            if (res.status === 404) {
                throw new Error("리포트가 아직 생성되지 않았습니다. 에이전트에게 요청해주세요.");
            }
            if (!res.ok) {
                throw new Error("리포트를 불러오는데 실패했습니다.");
            }

            const data = await res.json();
            // 백엔드가 { "markdown": "내용..." } 또는 { "content": "내용..." } 줄 경우 대비
            setReportContent(data.markdown || data.content || data);
        } catch (err: any) {
            console.error(err);
            setError(err.message || "Error loading report");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{ marginTop: '16px', borderTop: '1px solid #e2e8f0', paddingTop: '16px' }}>
            <button
                onClick={handleToggle}
                disabled={isLoading}
                style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: isOpen ? '#ebf8ff' : '#f7fafc',
                    color: isOpen ? '#2b6cb0' : '#4a5568',
                    border: '1px solid #cbd5e0',
                    borderRadius: '6px',
                    cursor: isLoading ? 'not-allowed' : 'pointer',
                    fontWeight: 600,
                    fontSize: '13px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    transition: 'all 0.2s'
                }}
            >
                <span>📝 요약 리포트 보기</span>
                <span>{isOpen ? '▲' : '▼'}</span>
            </button>

            {/* 로딩 표시 */}
            {isLoading && (
                <div style={{ padding: '12px', textAlign: 'center', color: '#718096', fontSize: '13px' }}>
                    불러오는 중...
                </div>
            )}

            {/* 에러 메시지 */}
            {error && !isLoading && isOpen && (
                <div style={{
                    marginTop: '10px',
                    padding: '10px',
                    backgroundColor: '#fff5f5',
                    color: '#c53030',
                    borderRadius: '6px',
                    fontSize: '12px'
                }}>
                    ⚠️ {error}
                </div>
            )}

            {/* 리포트 내용 (텍스트 뷰어) */}
            {isOpen && reportContent && !isLoading && (
                <div style={{
                    marginTop: '10px',
                    padding: '12px',
                    backgroundColor: '#ffffff',
                    border: '1px solid #e2e8f0',
                    borderRadius: '6px',
                    fontSize: '13px',
                    lineHeight: '1.6',
                    color: '#2d3748',
                    whiteSpace: 'pre-wrap',       // ⭐ 줄바꿈 유지 (txt 파일 핵심)
                    fontFamily: 'monospace',      // 텍스트 파일 느낌 (선택사항)
                    maxHeight: '400px',           // 너무 길면 스크롤
                    overflowY: 'auto'
                }}>
                    {reportContent}
                </div>
            )}
        </div>
    );
}