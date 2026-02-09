/**
 * PaperGraphPage (Graph A)
 *
 * Center paper → reference titles graph (local JSON)
 * Loads directly from /output/{paper_id}/reference_titles.json
 *
 * [UPDATED]
 * - Color persistence: use stableKey-based mapping (stableKey 우선)
 * - Reference nodes get stableKey derived from normalized title hash
 * - Reference node id also set to stable stableKey (ref:<hash>)
 */

import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import GraphCanvas from '../components/GraphCanvas';
import SidePanel from '../components/SidePanel';
import { GraphNode, GraphEdge } from '../lib/mcp';
import { useNodeColors } from '../hooks/useNodeColors';

/* ----------------------------- Types ----------------------------- */

interface PaperGraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  loading: boolean;
  error: string | null;
  centerId: string;
}

interface ReferenceTitlesJson {
  filename: string;
  original_filename?: string;
  references_detected: number;
  titles_extracted: number;
  titles: string[];
}

type FetchRefResult = {
  usedId: string;
  json: ReferenceTitlesJson;
};

/* ----------------------- Helpers: ID Normalization ---------------------- */

function normalizeArxivToDoiLike(id: string): string {
  if (id.startsWith('10.48550_arxiv.')) return id;
  const m = id.match(/^(\d{4}\.\d{4,5})(v\d+)?$/);
  if (m) return `10.48550_arxiv.${m[1]}`;
  return id;
}

async function fetchReferenceTitlesResolved(paperId: string): Promise<FetchRefResult | null> {
  const candidates = [paperId, normalizeArxivToDoiLike(paperId)].filter(
    (v, i, a) => a.indexOf(v) === i
  );

  for (const id of candidates) {
    try {
      const res = await fetch(`/output/${id}/reference_titles.json`, { cache: 'no-store' });
      if (res.ok) {
        const json = (await res.json()) as ReferenceTitlesJson;
        return { usedId: id, json };
      }
    } catch {
      // continue
    }
  }
  return null;
}

/* ----------------------- Helpers: Stable Key ---------------------- */

// title 정규화(같은 제목이면 웬만하면 같은 키가 나오도록)
function normalizeTitleKey(title: string): string {
  return String(title ?? '')
    .replace(/\u00A0/g, ' ')        // nbsp
    .replace(/\s+/g, ' ')          // collapse whitespace
    .trim()
    .replace(/[.]+$/g, '')         // trailing dots
    .toLowerCase();
}

// FNV-1a 32-bit hash (deterministic)
function fnv1a32(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// stable reference key from title
function refStableKeyFromTitle(title: string): string {
  const norm = normalizeTitleKey(title);
  const hash = fnv1a32(norm).toString(36);
  return `ref:${hash}`;
}

function nodeKeyOf(node: any): string {
  return String(node?.stableKey ?? node?.id ?? '');
}

/* ----------------------------- Page ------------------------------- */

export default function PaperGraphPage() {
  const { paperId: rawId } = useParams();
  const paperId = rawId ? decodeURIComponent(rawId) : '';

  const navigate = useNavigate();

  const [state, setState] = useState<PaperGraphState>({
    nodes: [],
    edges: [],
    loading: true,
    error: null,
    centerId: paperId
  });

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  /* ------------------- Node color map (file-based persistence) ------------------- */

  const {
    nodeColorMap,
    setNodeColor: handleNodeColorChange,
    resetNodeColor: handleNodeColorReset
  } = useNodeColors();

  /* ----------------------- Load from reference_titles.json ----------------------- */

  const loadGraph = useCallback(async () => {
    if (!paperId) return;

    setState(prev => ({ ...prev, loading: true, error: null, centerId: paperId }));

    try {
      console.log(`Loading reference graph for: ${paperId}`);

      const resolved = await fetchReferenceTitlesResolved(paperId);
      if (!resolved) throw new Error(`reference_titles.json not found for ${paperId}`);

      const { usedId, json } = resolved;
      const titles = json.titles || [];

      console.log(`Found ${titles.length} reference titles (folder used: ${usedId})`);

      // canonical center folder id
      const centerCanonicalId = usedId;

      // ✅ Center node: stableKey는 paper prefix로 고정
      const centerNode: GraphNode = {
        id: centerCanonicalId,
        stableKey: `paper:${centerCanonicalId}` as any,
        title: centerCanonicalId,
        authors: [],
        abstract: '',
        year: undefined,
        isCenter: true,
        hasDetails: true,
        cluster: 0,
        depth: 0
      } as any;

      // ✅ Reference nodes: stableKey = ref:<hash(title)>
      const refNodes: GraphNode[] = titles.map((title) => {
        const stableKey = refStableKeyFromTitle(title);

        return {
          id: stableKey,                 // id도 고정(순서 idx 변화에 영향 없음)
          stableKey: stableKey as any,   // GraphCanvas가 stableKey 우선으로 색/좌표 캐시 사용
          title: title,
          authors: [],
          abstract: '',
          year: undefined,
          isCenter: false,
          hasDetails: false,
          cluster: 1,
          depth: 1
        } as any;
      });

      const edges: GraphEdge[] = refNodes.map(refNode => ({
        source: centerCanonicalId,
        target: refNode.id,
        type: 'references' as const
      }));

      const allNodes = [centerNode, ...refNodes];

      setState({
        nodes: allNodes,
        edges,
        loading: false,
        error: null,
        centerId: centerCanonicalId
      });

      setSelectedNode(centerNode);
    } catch (err) {
      console.error('Graph load failed:', err);
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load reference graph'
      }));
    }
  }, [paperId]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  /* ------------------------ Click Handlers ------------------------ */

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
  }, []);

  const handleNodeDoubleClick = useCallback(
    (node: GraphNode) => {
      // Reference node: navigate only if its TITLE contains arXiv id
      if (!(node as any).isCenter) {
        const title = String((node as any).title || '');
        const arxivMatch = title.match(/(\d{4}\.\d{4,5})/);
        if (arxivMatch) {
          navigate(`/paper/${encodeURIComponent(`10.48550_arxiv.${arxivMatch[1]}`)}`);
          return;
        }
      }

      // Center node: refresh
      if ((node as any).isCenter) {
        loadGraph();
      }
    },
    [navigate, loadGraph]
  );

  /* ------------------------ Navigation ------------------------ */

  const handleBackToGlobal = useCallback(() => {
    navigate('/');
  }, [navigate]);
  const handleOpenNote = () => navigate(`/note/${paperId}`);

  /* --------- Extra content: arXiv / OpenReview search buttons for reference nodes --------- */

  const referenceSearchContent = useMemo(() => {
    if (!selectedNode) return undefined;
    // 센터 노드(원래 논문)에는 표시하지 않음
    if ((selectedNode as any).isCenter) return undefined;

    const title = String((selectedNode as any).title || selectedNode.label || '').trim();
    if (!title) return undefined;

    const encoded = encodeURIComponent(title);
    const arxivUrl = `https://arxiv.org/search/?query=${encoded}&searchtype=all`;
    const openReviewUrl = `https://openreview.net/search?term=${encoded}&content=all`;
    const scholarUrl = `https://scholar.google.com/scholar?q=${encoded}`;

    const btnStyle: React.CSSProperties = {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      width: '100%',
      padding: '10px 14px',
      border: '1px solid #e2e8f0',
      borderRadius: '6px',
      cursor: 'pointer',
      fontSize: '13px',
      fontWeight: 500,
      textDecoration: 'none',
      transition: 'background 0.15s',
    };

    return (
      <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ fontSize: '11px', color: '#6b7280', textTransform: 'uppercase', fontWeight: 600, marginBottom: '2px' }}>
          논문 검색 / 다운로드
        </div>

        <a
          href={arxivUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{ ...btnStyle, backgroundColor: '#fef3c7', color: '#92400e', borderColor: '#fbbf24' }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#fde68a')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#fef3c7')}
        >
          <span style={{ fontSize: '16px' }}>📄</span>
          arXiv에서 검색
        </a>

        <a
          href={openReviewUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{ ...btnStyle, backgroundColor: '#ede9fe', color: '#5b21b6', borderColor: '#c4b5fd' }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#ddd6fe')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#ede9fe')}
        >
          <span style={{ fontSize: '16px' }}>🔍</span>
          OpenReview에서 검색
        </a>

        <a
          href={scholarUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{ ...btnStyle, backgroundColor: '#ecfdf5', color: '#065f46', borderColor: '#6ee7b7' }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#d1fae5')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#ecfdf5')}
        >
          <span style={{ fontSize: '16px' }}>🎓</span>
          Google Scholar에서 검색
        </a>
      </div>
    );
  }, [selectedNode]);

  /* ----------------------------- UI ----------------------------- */

  const selectedKey = selectedNode ? nodeKeyOf(selectedNode as any) : '';

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#f5f5f5' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <header
          style={{
            padding: '16px 24px',
            backgroundColor: '#fff',
            borderBottom: '1px solid #e2e8f0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <div>
            <button
              onClick={handleBackToGlobal}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                backgroundColor: '#edf2f7',
                border: 'none',
                cursor: 'pointer',
                marginRight: '12px'
              }}
            >
              ← Back
            </button>

            <button
              onClick={handleOpenNote}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                backgroundColor: '#e6fffa',
                border: 'none',
                cursor: 'pointer',
                marginRight: '12px'
              }}
            >
              ✎ Note
            </button>

            <span style={{ fontWeight: 600 }}>Paper Reference Graph</span>
          </div>
          <div style={{ fontSize: '12px', color: '#718096' }}>
            ID: {paperId} | Refs: {Math.max(0, state.nodes.length - 1)}
          </div>
        </header>

        <div style={{ flex: 1, position: 'relative' }}>
          {state.loading && <CenteredText text="Loading references..." />}
          {state.error && <CenteredText text={state.error} error />}

          {!state.loading && !state.error && state.nodes.length === 0 && (
            <CenteredText text="No reference data available for this paper." />
          )}

          {!state.loading && !state.error && state.nodes.length > 0 && (
            <GraphCanvas
              nodes={state.nodes}
              edges={state.edges}
              mode="paper"
              centerId={state.centerId}
              onNodeClick={handleNodeClick}
              onNodeDoubleClick={handleNodeDoubleClick}
              selectedNodeId={selectedNode?.id}
              nodeColorMap={nodeColorMap} // ✅ GraphCanvas가 stableKey 우선 조회
            />
          )}
        </div>

        {/* Stats Bar */}
        {!state.loading && !state.error && state.nodes.length > 0 && (
          <div
            style={{
              padding: '8px 24px',
              backgroundColor: '#fff',
              borderTop: '1px solid #e2e8f0',
              display: 'flex',
              gap: '24px',
              fontSize: '12px',
              color: '#718096'
            }}
          >
            <span>Center: {state.centerId}</span>
            <span>References: {Math.max(0, state.nodes.length - 1)}</span>
          </div>
        )}
      </div>

      <SidePanel
        selectedNode={selectedNode}
        mode="paper"
        onAction={() => {}}
        onClose={() => setSelectedNode(null)}
        // ✅ Node color controls (stableKey 우선)
        nodeColorMap={nodeColorMap}
        nodeColor={
          selectedNode
            ? (nodeColorMap[selectedKey] ?? nodeColorMap[(selectedNode as any).id])
            : undefined
        }
        onNodeColorChange={(keyOrId, color) => {
          const key = selectedNode ? nodeKeyOf(selectedNode as any) : keyOrId;
          handleNodeColorChange(key, color);
        }}
        onNodeColorReset={(keyOrId) => {
          const key = selectedNode ? nodeKeyOf(selectedNode as any) : keyOrId;
          handleNodeColorReset(key);
        }}
        extraContent={referenceSearchContent}
      />
    </div>
  );
}

function CenteredText({ text, error }: { text: string; error?: boolean }) {
  return (
    <div
      style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        color: error ? '#e53e3e' : '#718096',
        fontSize: '14px',
        fontWeight: 500
      }}
    >
      {text}
    </div>
  );
}
