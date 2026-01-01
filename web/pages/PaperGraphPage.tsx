/**
 * PaperGraphPage (Graph A)
 *
 * Center paper → reference titles expansion (local JSON)
 * Loads directly from /output/{paper_id}/reference_titles.json
 *
 */

import React, { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import GraphCanvas from '../components/GraphCanvas';
import SidePanel from '../components/SidePanel';
import { GraphNode, GraphEdge } from '../lib/mcp';

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
  // Try multiple ID formats
  const candidates = [paperId, normalizeArxivToDoiLike(paperId)].filter(
    (v, i, a) => a.indexOf(v) === i
  ); // unique

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

  /* ------------------- Node color map (custom colors) ------------------- */

  const [nodeColorMap, setNodeColorMap] = useState<Record<string, string>>(() => {
    try {
      return JSON.parse(localStorage.getItem('nodeColorMap') || '{}');
    } catch {
      return {};
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('nodeColorMap', JSON.stringify(nodeColorMap));
    } catch {
      // ignore
    }
  }, [nodeColorMap]);

  const handleNodeColorChange = useCallback((nodeId: string, color: string) => {
    setNodeColorMap(prev => ({ ...prev, [nodeId]: color }));
  }, []);

  const handleNodeColorReset = useCallback((nodeId: string) => {
    setNodeColorMap(prev => {
      const next = { ...prev };
      delete next[nodeId];
      return next;
    });
  }, []);

  /* ----------------------- Load from reference_titles.json ----------------------- */

  const loadGraph = useCallback(async () => {
    if (!paperId) return;

    setState(prev => ({ ...prev, loading: true, error: null, centerId: paperId }));

    try {
      console.log(`Loading reference graph for: ${paperId}`);

      // Load reference_titles.json directly (with ID fallback)
      const resolved = await fetchReferenceTitlesResolved(paperId);

      if (!resolved) {
        throw new Error(`reference_titles.json not found for ${paperId}`);
      }

      const { usedId, json } = resolved;
      const titles = json.titles || [];

      console.log(`Found ${titles.length} reference titles (folder used: ${usedId})`);

      // ✅ Use the resolved folder id as the canonical center id
      const centerCanonicalId = usedId;

      // Create center node (add safe defaults for PaperCard)
      const centerNode: GraphNode = {
        id: centerCanonicalId,
        title: centerCanonicalId,
        authors: [],
        abstract: '',
        year: undefined,
        isCenter: true,
        hasDetails: true,
        cluster: 0,
        depth: 0
      } as any;

      // Create reference nodes from titles
      // - Stable-ish id based on index + sanitized title (good enough for color mapping)
      const refNodes: GraphNode[] = titles.map((title, idx) => {
        const safe = String(title)
          .slice(0, 60)
          .replace(/\s+/g, ' ')
          .trim()
          .replace(/[^a-zA-Z0-9]/g, '_');

        return {
          id: `ref_${idx}_${safe}`,
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

      // Create edges from center to each reference
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
      // If it's a reference node, try to navigate if title looks like an arXiv ID
      if (!(node as any).isCenter) {
        const title = String((node as any).title || '');
        const arxivMatch = title.match(/(\d{4}\.\d{4,5})/);
        if (arxivMatch) {
          navigate(`/paper/${encodeURIComponent(`10.48550_arxiv.${arxivMatch[1]}`)}`);
          return;
        }
      }

      // For center node, just refresh
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

  /* ----------------------------- UI ----------------------------- */

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
              nodeColorMap={nodeColorMap} // ✅ custom colors applied here
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
        // ✅ Node color controls
        nodeColorMap={nodeColorMap}
        nodeColor={selectedNode ? nodeColorMap[selectedNode.id] : undefined}
        onNodeColorChange={handleNodeColorChange}
        onNodeColorReset={handleNodeColorReset}
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
