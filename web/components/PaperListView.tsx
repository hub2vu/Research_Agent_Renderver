// web/components/PaperListView.tsx
import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { getReport, generateReport } from '../lib/mcp';

type AnyEdge = {
  source: string;
  target: string;
  weight?: number;
  type?: string;
};

type AnyNode = {
  id: string;
  title?: string;
  label?: string;
  cluster?: number | string;
};

type ReportState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'missing' }
  | { status: 'found'; content: string }
  | { status: 'error'; message: string };

function truncateText(s: string, n: number) {
  if (!s) return '';
  const t = s.replace(/\s+/g, ' ').trim();
  return t.length > n ? t.slice(0, n) + '…' : t;
}

function buildAdjacency(nodes: AnyNode[], edges: AnyEdge[]) {
  const nodeSet = new Set(nodes.map(n => n.id));
  const adj = new Map<string, Set<string>>();
  const degree = new Map<string, number>();

  for (const n of nodes) {
    adj.set(n.id, new Set());
    degree.set(n.id, 0);
  }

  for (const e of edges || []) {
    const a = e.source;
    const b = e.target;
    if (!nodeSet.has(a) || !nodeSet.has(b)) continue;

    // "기존 유사도"를 재사용: type이 similarity이거나, (neurips처럼) type이 없어도 weight만 있으면 유사도 엣지로 간주
    const isSimilarity = (e.type === 'similarity') || (e.type == null && typeof e.weight === 'number');
    if (!isSimilarity) continue;

    adj.get(a)!.add(b);
    adj.get(b)!.add(a);
    degree.set(a, (degree.get(a) || 0) + 1);
    degree.set(b, (degree.get(b) || 0) + 1);
  }

  return { adj, degree };
}

// "연결된 논문은 인접하게 정렬"을 위해, 같은 그룹 내에서 연결 컴포넌트 단위로 순서를 만든다.
function orderByConnectivity(groupNodes: AnyNode[], adj: Map<string, Set<string>>, degree: Map<string, number>) {
  const inGroup = new Set(groupNodes.map(n => n.id));
  const visited = new Set<string>();
  const nodesByDeg = [...groupNodes].sort((a, b) => (degree.get(b.id) || 0) - (degree.get(a.id) || 0));
  const ordered: AnyNode[] = [];

  for (const start of nodesByDeg) {
    if (visited.has(start.id)) continue;

    // BFS/DFS
    const stack = [start.id];
    visited.add(start.id);

    while (stack.length) {
      const cur = stack.pop()!;
      const node = groupNodes.find(n => n.id === cur);
      if (node) ordered.push(node);

      const neighbors = Array.from(adj.get(cur) || [])
        .filter(x => inGroup.has(x) && !visited.has(x))
        .sort((x, y) => (degree.get(y) || 0) - (degree.get(x) || 0));

      for (const nb of neighbors) {
        visited.add(nb);
        stack.push(nb);
      }
    }
  }

  // 고립 노드도 포함되지만, degree 낮은 것들은 뒤로 밀리는 효과가 있음
  return ordered;
}

export default function PaperListView(props: {
  nodes: AnyNode[];
  edges: AnyEdge[];
  groupBy?: (n: AnyNode) => string | number;          // 그룹 키
  groupTitle?: (groupKey: string) => string;          // 헤더 제목
  onOpenPaper?: (paperId: string) => void;            // 제목 클릭 액션
  initialPrefetchCount?: number;                      // 처음 자동 get_report 호출 개수
}) {
  const {
    nodes,
    edges,
    groupBy,
    groupTitle,
    onOpenPaper,
    initialPrefetchCount = 60,
  } = props;

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
    // 숫자형 그룹이면 숫자 정렬, 아니면 문자열
    const allNumeric = keys.every(k => /^-?\d+(\.\d+)?$/.test(k));
    return keys.sort((a, b) => {
      if (allNumeric) return Number(a) - Number(b);
      return a.localeCompare(b);
    });
  }, [groups]);

  // report cache
  const [reportMap, setReportMap] = useState<Record<string, ReportState>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
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
      if (r.found) {
        setReportState(paperId, { status: 'found', content: r.content || '' });
      } else {
        setReportState(paperId, { status: 'missing' });
      }
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
      if (!gen?.status && gen?.success === false) {
        throw new Error(gen?.error || 'generate_report failed');
      }
      // 생성 후 재조회
      const r = await getReport(paperId);
      if (r.found) {
        setReportState(paperId, { status: 'found', content: r.content || '' });
      } else {
        setReportState(paperId, { status: 'missing' });
      }
    } catch (e) {
      setReportState(paperId, { status: 'error', message: e instanceof Error ? e.message : 'Failed to generate report' });
    } finally {
      generatingRef.current.delete(paperId);
    }
  }, [setReportState]);

  // 초기 일부는 자동으로 get_report를 호출해서 "버튼 조건"을 빠르게 확정
  useEffect(() => {
    const flat = nodes.slice(0, Math.min(nodes.length, initialPrefetchCount));
    for (const n of flat) {
      if (!reportMap[n.id]) setReportState(n.id, { status: 'idle' });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, initialPrefetchCount]);

  useEffect(() => {
    const flat = nodes.slice(0, Math.min(nodes.length, initialPrefetchCount));
    (async () => {
      for (const n of flat) {
        if (!reportMap[n.id]) continue;
        if (reportMap[n.id].status === 'idle') {
          await ensureReport(n.id);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, initialPrefetchCount, ensureReport]);

  const rowStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: '420px 1fr',
    gap: '16px',
    padding: '10px 12px',
    borderBottom: '1px solid #e2e8f0',
    alignItems: 'start',
  };

  return (
    <div style={{ height: '100%', overflow: 'auto', background: '#fff' }}>
      {sortedGroupKeys.map(gk => {
        const rawNodes = groups.get(gk) || [];
        const ordered = orderByConnectivity(rawNodes, adj, degree);

        return (
          <div key={gk} style={{ borderBottom: '1px solid #cbd5e0' }}>
            <div style={{
              position: 'sticky',
              top: 0,
              zIndex: 2,
              background: '#f7fafc',
              padding: '10px 12px',
              borderBottom: '1px solid #e2e8f0',
              fontSize: '13px',
              fontWeight: 700,
              color: '#2d3748',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span>{groupTitle ? groupTitle(gk) : `Group ${gk}`}</span>
              <span style={{ fontWeight: 500, color: '#718096' }}>{rawNodes.length} papers</span>
            </div>

            {ordered.map(n => {
              const title = n.title || n.label || n.id;
              const deg = degree.get(n.id) || 0;
              const hasLink = deg > 0;

              const st = reportMap[n.id] ?? { status: 'idle' as const };
              const isExpanded = !!expanded[n.id];

              const rightContent = (() => {
                if (st.status === 'loading') return <span style={{ color: '#718096' }}>Loading report…</span>;
                if (st.status === 'found') {
                  const text = isExpanded ? st.content : truncateText(st.content, 420);
                  return (
                    <div>
                      <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, fontSize: '12.5px', color: '#1a202c' }}>
                        {text || '(empty)'}
                      </div>
                      {!!st.content && st.content.length > 450 && (
                        <button
                          onClick={() => setExpanded(prev => ({ ...prev, [n.id]: !prev[n.id] }))}
                          style={{
                            marginTop: '6px',
                            border: 'none',
                            background: 'transparent',
                            color: '#3182ce',
                            cursor: 'pointer',
                            fontSize: '12px',
                            padding: 0
                          }}
                        >
                          {isExpanded ? 'Show less' : 'Show more'}
                        </button>
                      )}
                    </div>
                  );
                }
                if (st.status === 'missing') {
                  return (
                    <button
                      onClick={() => handleGenerate(n.id)}
                      style={{
                        padding: '8px 10px',
                        borderRadius: '6px',
                        border: '1px solid #3182ce',
                        background: '#ebf8ff',
                        color: '#2b6cb0',
                        cursor: 'pointer',
                        fontSize: '12px',
                        fontWeight: 600,
                      }}
                      title="summary_report.txt가 없어서 생성합니다."
                    >
                      Generate report
                    </button>
                  );
                }
                if (st.status === 'error') {
                  return (
                    <div>
                      <div style={{ color: '#e53e3e', fontSize: '12px' }}>{st.message}</div>
                      <div style={{ display: 'flex', gap: '8px', marginTop: '6px' }}>
                        <button
                          onClick={() => ensureReport(n.id)}
                          style={{
                            padding: '6px 10px',
                            borderRadius: '6px',
                            border: '1px solid #a0aec0',
                            background: '#fff',
                            cursor: 'pointer',
                            fontSize: '12px',
                          }}
                        >
                          Retry get
                        </button>
                        <button
                          onClick={() => handleGenerate(n.id)}
                          style={{
                            padding: '6px 10px',
                            borderRadius: '6px',
                            border: '1px solid #3182ce',
                            background: '#ebf8ff',
                            cursor: 'pointer',
                            fontSize: '12px',
                          }}
                        >
                          Generate
                        </button>
                      </div>
                    </div>
                  );
                }
                // idle
                return (
                  <button
                    onClick={() => ensureReport(n.id)}
                    style={{
                      padding: '6px 10px',
                      borderRadius: '6px',
                      border: '1px solid #a0aec0',
                      background: '#fff',
                      cursor: 'pointer',
                      fontSize: '12px',
                      color: '#4a5568'
                    }}
                    title="클릭 시 get_report를 호출합니다."
                  >
                    Load report
                  </button>
                );
              })();

              return (
                <div key={n.id} style={rowStyle}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                    <div style={{ width: '18px', textAlign: 'center', marginTop: '2px', color: hasLink ? '#2b6cb0' : '#cbd5e0' }}>
                      {hasLink ? '⟷' : '·'}
                    </div>

                    <div style={{ flex: 1 }}>
                      <div
                        onClick={() => onOpenPaper?.(n.id)}
                        style={{
                          fontSize: '13px',
                          fontWeight: 700,
                          color: '#1a202c',
                          cursor: onOpenPaper ? 'pointer' : 'default',
                          lineHeight: 1.35
                        }}
                        title={title}
                      >
                        {title}
                      </div>

                      <div style={{ marginTop: '4px', fontSize: '11px', color: '#718096' }}>
                        {n.id} {hasLink ? `• links: ${deg}` : ''}
                      </div>
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
