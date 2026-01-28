import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { getReport, generateReport, executeTool } from '../lib/mcp';

// --- [타입 정의] ---
type AnyEdge = { source: string; target: string; weight?: number; type?: string; };
type AnyNode = { id: string; title?: string; label?: string; cluster?: number | string; abstract?: string;[key: string]: any; };
type ReportState = | { status: 'idle' } | { status: 'loading' } | { status: 'missing' } | { status: 'found'; content: string } | { status: 'error'; message: string };

const INITIAL_VISIBLE_COUNT = 20;
const LOAD_MORE_STEP = 50;

// --- [헬퍼 함수들] ---
function truncateText(s: string, n: number) {
  if (!s) return '';
  const t = s.replace(/\s+/g, ' ').trim();
  return t.length > n ? t.slice(0, n) + '…' : t;
}

function buildAdjacency(nodes: AnyNode[], edges: AnyEdge[]) {
  const nodeSet = new Set(nodes.map(n => n.id));
  const adj = new Map<string, Set<string>>();
  const degree = new Map<string, number>();
  for (const n of nodes) { adj.set(n.id, new Set()); degree.set(n.id, 0); }
  for (const e of edges || []) {
    const a = e.source; const b = e.target;
    if (!nodeSet.has(a) || !nodeSet.has(b)) continue;
    const isSimilarity = (e.type === 'similarity') || (e.type == null && typeof e.weight === 'number');
    if (!isSimilarity) continue;
    adj.get(a)!.add(b); adj.get(b)!.add(a);
    degree.set(a, (degree.get(a) || 0) + 1); degree.set(b, (degree.get(b) || 0) + 1);
  }
  return { adj, degree };
}

// 연결성 높은(중요한) 순서로 정렬하는 함수
function orderByConnectivity(groupNodes: AnyNode[], adj: Map<string, Set<string>>, degree: Map<string, number>) {
  return [...groupNodes].sort((a, b) => (degree.get(b.id) || 0) - (degree.get(a.id) || 0));
}

// ✅ [서베이 생성 함수] (백엔드 툴 호출)
async function generateSurvey(topicName: string, papers: AnyNode[]) {
  const topPapers = papers.slice(0, 20);

  const context = topPapers.map((p, i) => `
[Paper ${i + 1}]
Title: ${p.title || p.id}
Authors: ${p.authors ? p.authors.slice(0, 2).join(', ') : 'N/A'}
Abstract: ${p.abstract ? truncateText(p.abstract, 500) : 'No abstract available'}
`).join('\n');

  try {
    const result = await executeTool('generate_cluster_survey', {
      topic: topicName,
      papers_context: context,
      language: "Korean"
    });

    if (result.success && result.result && result.result.survey_content) {
      return result.result.survey_content;
    } else {
      console.error("Tool execution error:", result);
      return "⚠️ Error: 서베이 생성에 실패했습니다. (백엔드 로그를 확인하세요)";
    }
  } catch (error) {
    console.error("System error:", error);
    return "⚠️ Error: 서버 통신 중 오류가 발생했습니다.";
  }
}


export default function PaperListView(props: {
  nodes: AnyNode[];
  edges: AnyEdge[];
  groupBy?: (n: AnyNode) => string | number;
  groupTitle?: (groupKey: string) => string;
  onOpenPaper?: (paperId: string) => void;
  initialPrefetchCount?: number;
}) {
  const { nodes, edges, groupBy, groupTitle, onOpenPaper, initialPrefetchCount = 60 } = props;
  const { adj, degree } = useMemo(() => buildAdjacency(nodes, edges), [nodes, edges]);

  const groups = useMemo(() => {
    const g = new Map<string, AnyNode[]>();
    for (const n of nodes) {
      const keyRaw = groupBy ? groupBy(n) : (n.cluster ?? '0');
      const key = String(keyRaw ?? '0');
      if (!g.has(key)) g.set(key, []);
      g.get(key)!.push(n);
    }
    return g;
  }, [nodes, groupBy]);

  const sortedGroupKeys = useMemo(() => {
    const keys = Array.from(groups.keys());
    const allNumeric = keys.every(k => /^-?\d+(\.\d+)?$/.test(k));
    return keys.sort((a, b) => {
      if (allNumeric) return Number(a) - Number(b);
      return a.localeCompare(b);
    });
  }, [groups]);

  const [reportMap, setReportMap] = useState<Record<string, ReportState>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [pipelineState, setPipelineState] = useState<Record<string, 'idle' | 'loading' | 'done' | 'error'>>({});
  const inflightRef = useRef<Set<string>>(new Set());
  const generatingRef = useRef<Set<string>>(new Set());
  const [visibleCounts, setVisibleCounts] = useState<Record<string, number>>({});

  // ✅ [핵심 변경 1] 상태 분리: 데이터(Data)와 가시성(Visibility) 분리
  const [surveyData, setSurveyData] = useState<Record<string, string>>({});   // 내용 저장 (캐시)
  const [surveyVisible, setSurveyVisible] = useState<Record<string, boolean>>({}); // 보이기/숨기기 상태
  const [surveyLoading, setSurveyLoading] = useState<Record<string, boolean>>({}); // 로딩 상태

  const setReportState = useCallback((paperId: string, st: ReportState) => { setReportMap(prev => ({ ...prev, [paperId]: st })); }, []);
  const ensureReport = useCallback(async (paperId: string) => {
    const cur = reportMap[paperId]; if (cur && cur.status !== 'idle') return; if (inflightRef.current.has(paperId)) return;
    inflightRef.current.add(paperId); setReportState(paperId, { status: 'loading' });
    try { const r = await getReport(paperId); if (r.found) setReportState(paperId, { status: 'found', content: r.content || '' }); else setReportState(paperId, { status: 'missing' }); } catch (e) { setReportState(paperId, { status: 'error', message: e instanceof Error ? e.message : 'Err' }); } finally { inflightRef.current.delete(paperId); }
  }, [reportMap, setReportState]);
  const handleGenerate = useCallback(async (paperId: string) => {
    if (generatingRef.current.has(paperId)) return; generatingRef.current.add(paperId); setReportState(paperId, { status: 'loading' });
    try { const gen = await generateReport(paperId); if (!gen?.status && gen?.success === false) throw new Error('fail'); const r = await getReport(paperId); if (r.found) setReportState(paperId, { status: 'found', content: r.content || '' }); else setReportState(paperId, { status: 'missing' }); } catch (e) { setReportState(paperId, { status: 'error', message: String(e) }); } finally { generatingRef.current.delete(paperId); }
  }, [setReportState]);
  const handleDownloadAndProcess = async (paperId: string) => {
    if (pipelineState[paperId] === 'loading') return; setPipelineState(prev => ({ ...prev, [paperId]: 'loading' }));
    try { const result = await executeTool('process_neurips_paper', { paper_id: paperId, out_dir: '/data/pdf/neurips2025' }); setPipelineState(prev => ({ ...prev, [paperId]: 'done' })); alert(result.result?.pipeline_results ? "PDF Saved!" : "Process Complete!"); ensureReport(paperId); } catch (err) { setPipelineState(prev => ({ ...prev, [paperId]: 'error' })); alert(`Error: ${String(err)}`); }
  };

  const handleLoadMore = (groupKey: string) => {
    setVisibleCounts(prev => ({ ...prev, [groupKey]: (prev[groupKey] || INITIAL_VISIBLE_COUNT) + LOAD_MORE_STEP }));
  };

  // ✅ [핵심 변경 2] 버튼 핸들러: 데이터가 있으면 토글, 없으면 생성
  const handleWriteSurveyButton = async (groupKey: string, papers: AnyNode[]) => {
    if (surveyLoading[groupKey]) return;

    // 1. 이미 데이터가 있으면 -> 단순히 보이기/숨기기 토글
    if (surveyData[groupKey]) {
      setSurveyVisible(prev => ({ ...prev, [groupKey]: !prev[groupKey] }));
      return;
    }

    // 2. 데이터가 없으면 -> 생성 요청
    setSurveyLoading(prev => ({ ...prev, [groupKey]: true }));
    try {
      const title = groupTitle ? groupTitle(groupKey) : `Cluster ${groupKey}`;
      const ordered = orderByConnectivity(papers, adj, degree);
      const resultText = await generateSurvey(title, ordered);

      // 데이터 저장 및 보여주기
      setSurveyData(prev => ({ ...prev, [groupKey]: resultText }));
      setSurveyVisible(prev => ({ ...prev, [groupKey]: true }));
    } catch (e) {
      alert("Failed to write survey.");
    } finally {
      setSurveyLoading(prev => ({ ...prev, [groupKey]: false }));
    }
  };

  // ✅ [핵심 변경 3] 재생성(Regenerate) 버튼 기능 추가
  const handleRegenerate = async (groupKey: string, papers: AnyNode[]) => {
    if (!confirm("서베이를 다시 작성하시겠습니까? (기존 내용은 사라집니다)")) return;
    // 데이터 삭제 후 다시 호출
    setSurveyData(prev => { const n = { ...prev }; delete n[groupKey]; return n; });
    handleWriteSurveyButton(groupKey, papers);
  };

  useEffect(() => { const flat = nodes.slice(0, Math.min(nodes.length, initialPrefetchCount)); for (const n of flat) if (!reportMap[n.id]) setReportState(n.id, { status: 'idle' }); }, [nodes, initialPrefetchCount]);
  useEffect(() => { const flat = nodes.slice(0, Math.min(nodes.length, initialPrefetchCount)); (async () => { for (const n of flat) { if (!reportMap[n.id]) continue; if (reportMap[n.id].status === 'idle') await ensureReport(n.id); } })(); }, [nodes, initialPrefetchCount, ensureReport]);

  const rowStyle: React.CSSProperties = { display: 'grid', gridTemplateColumns: '420px 1fr', gap: '16px', padding: '10px 12px', borderBottom: '1px solid #e2e8f0', alignItems: 'start' };

  return (
    <div style={{ height: '100%', overflow: 'auto', background: '#fff' }}>
      {sortedGroupKeys.map(gk => {
        const rawNodes = groups.get(gk) || [];
        const ordered = orderByConnectivity(rawNodes, adj, degree);
        const currentLimit = visibleCounts[gk] || INITIAL_VISIBLE_COUNT;
        const visibleNodes = ordered.slice(0, currentLimit);
        const remainingCount = ordered.length - visibleNodes.length;

        const isLoading = surveyLoading[gk];
        const hasData = !!surveyData[gk];
        const isVisible = surveyVisible[gk];

        return (
          <div key={gk} style={{ borderBottom: '1px solid #cbd5e0' }}>

            {/* --- [그룹 헤더] --- */}
            <div style={{ position: 'sticky', top: 0, zIndex: 2, background: '#f7fafc', padding: '10px 12px', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '13px', fontWeight: 700, color: '#2d3748' }}>
                  {groupTitle ? groupTitle(gk) : `Group ${gk}`}
                </span>

                {/* ✅ [버튼 상태 분기] */}
                <button
                  onClick={() => handleWriteSurveyButton(gk, rawNodes)}
                  disabled={isLoading}
                  style={{
                    padding: '4px 12px',
                    fontSize: '11px',
                    borderRadius: '20px',
                    border: '1px solid #805ad5',
                    backgroundColor: isVisible ? '#805ad5' : '#fff',
                    color: isVisible ? '#fff' : '#805ad5',
                    cursor: 'pointer',
                    fontWeight: 600,
                    display: 'flex', alignItems: 'center', gap: '4px',
                    transition: 'all 0.2s'
                  }}
                >
                  {isLoading ? 'Writing...' : (
                    hasData
                      ? (isVisible ? '📝 Hide Survey' : '📝 Show Survey (Cached)')
                      : '📝 Write Survey'
                  )}
                </button>
              </div>
              <span style={{ fontSize: '12px', fontWeight: 500, color: '#718096' }}>{rawNodes.length} papers</span>
            </div>

            {/* ✅ [결과 화면] 가시성(isVisible) 체크 */}
            {hasData && isVisible && (
              <div style={{ padding: '20px', backgroundColor: '#faf5ff', borderBottom: '1px solid #e9d8fd' }}>
                <div style={{ fontWeight: 700, color: '#553c9a', marginBottom: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span>🤖 Agent Generated Survey</span>
                    {/* 재생성 버튼 (작게) */}
                    <button onClick={() => handleRegenerate(gk, rawNodes)} style={{ fontSize: '10px', padding: '2px 6px', border: '1px solid #d6bcfa', background: '#fff', color: '#805ad5', borderRadius: '4px', cursor: 'pointer' }}>↻ Regenerate</button>
                  </div>
                  {/* 닫기 버튼: 이제 삭제하지 않고 숨기기만 함 */}
                  <button onClick={() => setSurveyVisible(prev => ({ ...prev, [gk]: false }))} style={{ border: 'none', background: 'transparent', cursor: 'pointer', fontSize: '16px', color: '#553c9a' }}>×</button>
                </div>

                <div style={{ whiteSpace: 'pre-wrap', fontSize: '13.5px', lineHeight: 1.6, color: '#44337a', fontFamily: 'sans-serif' }}>
                  {surveyData[gk]}
                </div>
              </div>
            )}

            {visibleNodes.map(n => {
              const title = n.title || n.label || n.id;
              const deg = degree.get(n.id) || 0;
              const hasLink = deg > 0;
              const st = reportMap[n.id] ?? { status: 'idle' as const };
              const isExpanded = !!expanded[n.id];
              const pStatus = pipelineState[n.id] || 'idle';
              const abstractText = (n as any).abstract || (n as any).summary || '';

              const rightContent = (() => {
                if (st.status === 'loading') return <span style={{ color: '#718096' }}>Loading report...</span>;
                if (st.status === 'found') {
                  const text = isExpanded ? st.content : truncateText(st.content, 420);
                  return (
                    <div>
                      <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, fontSize: '12.5px', color: '#1a202c' }}>{text || '(empty)'}</div>
                      {!!st.content && st.content.length > 450 && (
                        <button onClick={() => setExpanded(prev => ({ ...prev, [n.id]: !prev[n.id] }))} style={{ marginTop: '6px', border: 'none', background: 'transparent', color: '#3182ce', cursor: 'pointer', fontSize: '12px', padding: 0 }}>{isExpanded ? 'Show less' : 'Show more'}</button>
                      )}
                    </div>
                  );
                }
                const isIdle = st.status === 'idle';
                const isError = st.status === 'error';
                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      {isError && <span style={{ color: '#e53e3e', fontSize: '11px' }}>Error</span>}
                      {isIdle ? <button onClick={() => ensureReport(n.id)} style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #a0aec0', background: '#fff', cursor: 'pointer', fontSize: '12px', fontWeight: 600, color: '#4a5568' }}>Load report</button> : <button onClick={() => handleGenerate(n.id)} style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #3182ce', background: '#ebf8ff', color: '#2b6cb0', cursor: 'pointer', fontSize: '12px', fontWeight: 600 }}>{isError ? 'Retry' : 'Generate'}</button>}
                      <button onClick={(e) => { e.stopPropagation(); handleDownloadAndProcess(n.id); }} disabled={pStatus === 'loading' || pStatus === 'done'} style={{ padding: '6px 12px', borderRadius: '4px', border: 'none', backgroundColor: pStatus === 'loading' ? '#cbd5e0' : (pStatus === 'done' ? '#2f855a' : (pStatus === 'error' ? '#f56565' : '#48bb78')), color: '#fff', fontSize: '12px', fontWeight: 600, display: 'flex', alignItems: 'center', cursor: (pStatus === 'loading' || pStatus === 'done') ? 'default' : 'pointer' }}>{pStatus === 'loading' ? 'DL...' : (pStatus === 'done' ? '✓ PDF' : 'Download PDF')}</button>
                    </div>
                    {abstractText ? (
                      <div style={{ backgroundColor: '#f7fafc', padding: '12px', borderRadius: '6px', border: '1px solid #edf2f7' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: '#718096', marginBottom: '6px', textTransform: 'uppercase' }}>Abstract Preview</div>
                        <div style={{ fontSize: '12.5px', color: '#4a5568', lineHeight: '1.5' }}>{isExpanded ? abstractText : truncateText(abstractText, 350)}</div>
                        {abstractText.length > 350 && <button onClick={(e) => { e.stopPropagation(); setExpanded(prev => ({ ...prev, [n.id]: !prev[n.id] })); }} style={{ marginTop: '6px', border: 'none', background: 'transparent', color: '#3182ce', cursor: 'pointer', fontSize: '12px', padding: 0, fontWeight: 500 }}>{isExpanded ? 'Show less' : 'Show more'}</button>}
                      </div>
                    ) : <div style={{ fontSize: '12px', color: '#a0aec0', fontStyle: 'italic' }}>(No abstract)</div>}
                  </div>
                );
              })();

              return (
                <div key={n.id} style={rowStyle}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ width: '18px', textAlign: 'center', marginTop: '2px', color: hasLink ? '#2b6cb0' : '#cbd5e0' }}>{hasLink ? '⟷' : '·'}</div>
                    <div style={{ flex: 1 }}>
                      <div onClick={(e) => { e.stopPropagation(); onOpenPaper?.(n.id); }} style={{ fontSize: '13px', fontWeight: 700, color: '#1a202c', cursor: onOpenPaper ? 'pointer' : 'default', lineHeight: 1.35 }} title={title}>{title}</div>
                      <div style={{ marginTop: '4px', fontSize: '11px', color: '#718096' }}>{n.id} {hasLink ? `• links: ${deg}` : ''}</div>
                    </div>
                  </div>
                  <div>{rightContent}</div>
                </div>
              );
            })}

            {remainingCount > 0 && (
              <div style={{ padding: '12px', textAlign: 'center' }}>
                <button onClick={() => handleLoadMore(gk)} style={{ padding: '8px 24px', borderRadius: '20px', border: '1px solid #cbd5e0', background: '#fff', color: '#4a5568', cursor: 'pointer', fontSize: '13px', fontWeight: 600, boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>👇 Show {Math.min(remainingCount, LOAD_MORE_STEP)} more ({remainingCount} remaining)</button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}