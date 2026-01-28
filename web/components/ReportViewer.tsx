/**
 * ReportViewer Component
 *
 * Fetches and displays the plain text report for a specific paper.
 */

import React, { useEffect, useState } from 'react';
import { getOrCreateReport } from '../lib/mcp';
import { LatexDiv } from './LatexText';
interface ReportViewerProps {
    paperId: string;
}

export default function ReportViewer({ paperId }: ReportViewerProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [reportContent, setReportContent] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // ✅ paperId가 바뀌면 이전 논문 리포트가 남아있는 문제를 방지
    // (새 논문 클릭 시 "이전 reportContent 재사용"을 막기 위해 reset)
    useEffect(() => {
        setIsOpen(false);
        setReportContent(null);
        setError(null);
        setIsLoading(false);
    }, [paperId]);

    const handleToggle = async () => {
        // 1. 닫혀있으면 -> 연다 (데이터 없으면 가져오기)
        if (!isOpen) {
            if (!reportContent) {
                await fetchOrCreateReport();
            }
            setIsOpen(true);
        }
        // 2. 열려있으면 -> 닫는다
        else {
            setIsOpen(false);
        }
    };

    const fetchOrCreateReport = async () => {
        setIsLoading(true);
        setError(null);
        try {
            // ✅ MCP tools/report.py 기반:
            // - summary_report.txt 있으면: get_report로 바로 로드
            // - 없으면: generate_report(오직 이 경우에만) → get_report 재시도
            const result = await getOrCreateReport(paperId);
            setReportContent(result.content);
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

            {/* 리포트 내용 (텍스트 뷰어 - LaTeX 수식 지원) */}
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
                    fontFamily: 'monospace',      // 텍스트 파일 느낌 (선택사항)
                    maxHeight: '400px',           // 너무 길면 스크롤
                    overflowY: 'auto'
                }}>
                    <LatexDiv>{reportContent}</LatexDiv>
                </div>
            )}
        </div>
    );
}