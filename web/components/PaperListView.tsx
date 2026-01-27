import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { getReport, generateReport, executeTool } from '../lib/mcp';

type AnyEdge = { source: string; target: string; weight?: number; type?: string; };
type AnyNode = { id: string; title?: string; label?: string; cluster?: number | string;[key: string]: any; };
type ReportState = | { status: 'idle' } | { status: 'loading' } | { status: 'missing' } | { status: 'found'; content: string } | { status: 'error'; message: string };

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

function orderByConnectivity(groupNodes: AnyNode[], adj: Map<string, Set<string>>, degree: Map<string, number>) {
  const inGroup = new Set(groupNodes.map(n => n.id));
  const visited = new Set<string>();
  const nodesByDeg = [...groupNodes].sort((a, b) => (degree.get(b.id) || 0) - (degree.get(a.id) || 0));
  const ordered: AnyNode[] = [];
  for (const start of nodesByDeg) {
    if (visited.has(start.id)) continue;
    const stack = [start.id]; visited.add(start.id);
    while (stack.length) {
      const cur = stack.pop()!;
      const node = groupNodes.find(n => n.id === cur);
      if (node) ordered.push(node);
      const neighbors = Array.from(adj.get(cur) || []).filter(x => inGroup.has(x) && !visited.has(x)).sort((x, y) => (degree.get(y) || 0) - (degree.get(x) || 0));
      for (const nb of neighbors) { visited.add(nb); stack.push(nb); }
    }
  }
  return ordered;
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

  const setReportState = useCallback((paperId: string, st: ReportState) => {
    setReportMap(prev => ({ ...prev, [paperId]: st }));
  }, []);

  const ensureReport = useCallback(async (paperId: string) => {
    const cur = reportMap[paperId];
    if (cur && cur.status !== 'idle') return;
    if (inflightRef.current.has(paperId)) return;
    inflightRef.current.add(paperId);
    setReportState(paperId, { status: 'loading' });
    try {
      const r = await getReport(paperId);
      if (r.found) setReportState(paperId, { status: 'found', content: r.content || '' });
      else setReportState(paperId, { status: 'missing' });
    } catch (e) {
      setReportState(paperId, { status: 'error', message: e instanceof Error ? e.message : 'Failed to get report' });
    } finally {
      inflightRef.current.delete(paperId);
    }
  }, [reportMap, setReportState]);

  const handleGenerate = useCallback(async (paperId: string) => {
    if (generatingRef.current.has(paperId)) return;
    generatingRef.current.add(paperId);
    setReportState(paperId, { status: 'loading' });
    try {
      const gen = await generateReport(paperId);
      if (!gen?.status && gen?.success === false) throw new Error(gen?.error || 'generate_report failed');
      const r = await getReport(paperId);
      if (r.found) setReportState(paperId, { status: 'found', content: r.content || '' });
      else setReportState(paperId, { status: 'missing' });
    } catch (e) {
      setReportState(paperId, { status: 'error', message: e instanceof Error ? e.message : 'Failed to generate report' });
    } finally {
      generatingRef.current.delete(paperId);
    }
  }, [setReportState]);

  const handleDownloadAndProcess = async (paperId: string) => {
    if (pipelineState[paperId] === 'loading') return;
    setPipelineState(prev => ({ ...prev, [paperId]: 'loading' }));
    try {
      const result = await executeTool('process_neurips_paper', {
        paper_id: paperId,
        out_dir: '/data/pdf/neurips2025'
      });
      setPipelineState(prev => ({ ...prev, [paperId]: 'done' }));

      if (result.result && result.result.pipeline_results) {
        const res = result.result.pipeline_results;
        const msg = `Process Complete!\n\n- PDF Saved: ${res.pdf_path || 'unknown'}\n- References Found: ${res.ref_count || 0}`;
        alert(msg);
      } else {
        alert("Process Complete!");
      }
      ensureReport(paperId);
    } catch (err) {
      console.error(err);
      setPipelineState(prev => ({ ...prev, [paperId]: 'error' }));
      alert(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  useEffect(() => {
    const flat = nodes.slice(0, Math.min(nodes.length, initialPrefetchCount));
    for (const n of flat) if (!reportMap[n.id]) setReportState(n.id, { status: 'idle' });
  }, [nodes, initialPrefetchCount]);

  useEffect(() => {
    const flat = nodes.slice(0, Math.min(nodes.length, initialPrefetchCount));
    (async () => {
      for (const n of flat) {
        if (!reportMap[n.id]) continue;
        if (reportMap[n.id].status === 'idle') await ensureReport(n.id);
      }
    })();
  }, [nodes, initialPrefetchCount, ensureReport]);

  const rowStyle: React.CSSProperties = {
    display: 'grid', gridTemplateColumns: '420px 1fr', gap: '16px', padding: '10px 12px', borderBottom: '1px solid #e2e8f0', alignItems: 'start',
  };

  return (
    <div style={{ height: '100%', overflow: 'auto', background: '#fff' }}>
      {sortedGroupKeys.map(gk => {
        const rawNodes = groups.get(gk) || [];
        const ordered = orderByConnectivity(rawNodes, adj, degree);

        return (
          <div key={gk} style={{ borderBottom: '1px solid #cbd5e0' }}>
            <div style={{ position: 'sticky', top: 0, zIndex: 2, background: '#f7fafc', padding: '10px 12px', borderBottom: '1px solid #e2e8f0', fontSize: '13px', fontWeight: 700, color: '#2d3748', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>{groupTitle ? groupTitle(gk) : `Group ${gk}`}</span>
              <span style={{ fontWeight: 500, color: '#718096' }}>{rawNodes.length} papers</span>
            </div>

            {ordered.map(n => {
              const title = n.title || n.label || n.id;
              const deg = degree.get(n.id) || 0;
              const hasLink = deg > 0;
              const st = reportMap[n.id] ?? { status: 'idle' as const };

              // ✅ Abstract용 토글 버튼
              const isExpanded = !!expanded[n.id];

              const pStatus = pipelineState[n.id] || 'idle';
              const abstractText = (n as any).abstract || (n as any).summary || '';

              const rightContent = (() => {
                if (st.status === 'loading') return <span style={{ color: '#718096' }}>Loading report...</span>;

                // 1. 리포트 생성 완료
                if (st.status === 'found') {
                  const text = isExpanded ? st.content : truncateText(st.content, 420);
                  return (
                    <div>
                      <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, fontSize: '12.5px', color: '#1a202c' }}>{text || '(empty)'}</div>
                      {!!st.content && st.content.length > 450 && (
                        <button onClick={() => setExpanded(prev => ({ ...prev, [n.id]: !prev[n.id] }))} style={{ marginTop: '6px', border: 'none', background: 'transparent', color: '#3182ce', cursor: 'pointer', fontSize: '12px', padding: 0 }}>
                          {isExpanded ? 'Show less' : 'Show more'}
                        </button>
                      )}
                    </div>
                  );
                }

                // 2. 대기/에러 상태 -> Abstract 미리보기
                const isIdle = st.status === 'idle';
                const isError = st.status === 'error';

                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      {isError && <span style={{ color: '#e53e3e', fontSize: '11px' }}>오류 발생</span>}

                      {isIdle ? (
                        <button
                          onClick={() => ensureReport(n.id)}
                          style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #a0aec0', background: '#fff', cursor: 'pointer', fontSize: '12px', fontWeight: 600, color: '#4a5568' }}
                        >
                          Load report
                        </button>
                      ) : (
                        <button
                          onClick={() => handleGenerate(n.id)}
                          style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #3182ce', background: '#ebf8ff', color: '#2b6cb0', cursor: 'pointer', fontSize: '12px', fontWeight: 600 }}
                        >
                          {isError ? 'Retry Gen' : 'Generate'}
                        </button>
                      )}

                      <button
                        onClick={(e) => { e.stopPropagation(); handleDownloadAndProcess(n.id); }}
                        disabled={pStatus === 'loading' || pStatus === 'done'}
                        style={{
                          padding: '6px 12px', borderRadius: '4px', border: 'none',
                          backgroundColor: pStatus === 'loading' ? '#cbd5e0' : (pStatus === 'done' ? '#2f855a' : (pStatus === 'error' ? '#f56565' : '#48bb78')),
                          color: '#fff', fontSize: '12px', fontWeight: 600, display: 'flex', alignItems: 'center', cursor: (pStatus === 'loading' || pStatus === 'done') ? 'default' : 'pointer'
                        }}
                      >
                        {pStatus === 'loading' ? 'Downloading...' : (pStatus === 'done' ? '✓ Complete' : 'Download PDF')}
                      </button>
                    </div>

                    {/* ✅ Abstract 미리보기 (더 보기 버튼 추가) */}
                    {abstractText ? (
                      <div style={{ backgroundColor: '#f7fafc', padding: '12px', borderRadius: '6px', border: '1px solid #edf2f7' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: '#718096', marginBottom: '6px', textTransform: 'uppercase' }}>Abstract Preview</div>
                        <div style={{ fontSize: '12.5px', color: '#4a5568', lineHeight: '1.5' }}>
                          {/* 내용이 길면 잘라 보여주고, 버튼 클릭 시 전체 표시 */}
                          {isExpanded ? abstractText : truncateText(abstractText, 350)}
                        </div>

                        {/* 350자 넘으면 더보기 버튼 표시 */}
                        {abstractText.length > 350 && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setExpanded(prev => ({ ...prev, [n.id]: !prev[n.id] }));
                            }}
                            style={{
                              marginTop: '6px',
                              border: 'none',
                              background: 'transparent',
                              color: '#3182ce',
                              cursor: 'pointer',
                              fontSize: '12px',
                              padding: 0,
                              fontWeight: 500
                            }}
                          >
                            {isExpanded ? 'Show less' : 'Show more'}
                          </button>
                        )}
                      </div>
                    ) : (
                      <div style={{ fontSize: '12px', color: '#a0aec0', fontStyle: 'italic' }}>(No abstract available)</div>
                    )}
                  </div>
                );
              })();

              return (
                <div key={n.id} style={rowStyle}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ width: '18px', textAlign: 'center', marginTop: '2px', color: hasLink ? '#2b6cb0' : '#cbd5e0' }}>{hasLink ? '⟷' : '·'}</div>
                    <div style={{ flex: 1 }}>
                      <div onClick={() => onOpenPaper?.(n.id)} style={{ fontSize: '13px', fontWeight: 700, color: '#1a202c', cursor: onOpenPaper ? 'pointer' : 'default', lineHeight: 1.35 }} title={title}>{title}</div>
                      <div style={{ marginTop: '4px', fontSize: '11px', color: '#718096' }}>{n.id} {hasLink ? `• links: ${deg}` : ''}</div>
                    </div>
                  </div>
                  <div>{rightContent}</div>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}