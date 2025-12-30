/**
 * PaperGraphPage (Graph A)
 *
 * Center paper → reference titles expansion (local JSON)
 * Reference node → arXiv fetch on demand
 *
 * Fixes:
 * - Robustly load /output/<id>/reference_titles.json with ID fallbacks
 */

import React, { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import GraphCanvas from '../components/GraphCanvas';
import SidePanel from '../components/SidePanel';
import {
  getPaperGraph,
  hasPdf,
  fetchPaperIfMissing,
  GraphNode,
  GraphEdge
} from '../lib/mcp';

/* ----------------------------- Types ----------------------------- */

interface PaperGraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  loading: boolean;
  expanding: boolean;
  error: string | null;
  centerId: string;
}

/* ----------------------- Helpers: ID + Fetch ---------------------- */

function normalizeArxivToDoiLike(id: string): string {
  if (id.startsWith('10.48550_arxiv.')) return id;
  const m = id.match(/^(\d{4}\.\d{4,5})(v\d+)?$/);
  if (m) return `10.48550_arxiv.${m[1]}`;
  return id;
}

async function fetchJsonTitles(url: string): Promise<string[] | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const json = await res.json();
    const titlesRaw: unknown = json?.titles;
    if (!Array.isArray(titlesRaw)) return [];
    const seen = new Set<string>();
    const out: string[] = [];
    for (const t of titlesRaw) {
      const s = String(t ?? '').trim();
      if (!s) continue;
      if (seen.has(s)) continue;
      seen.add(s);
      out.push(s);
    }
    return out;
  } catch {
    return null;
  }
}

async function loadReferenceTitles(centerId: string, urlPaperId: string): Promise<string[]> {
  const candidates = [
    centerId,
    urlPaperId,
    normalizeArxivToDoiLike(centerId),
    normalizeArxivToDoiLike(urlPaperId)
  ]
    .map(s => s.trim())
    .filter(Boolean);

  const seen = new Set<string>();
  const uniq: string[] = [];
  for (const c of candidates) {
    if (seen.has(c)) continue;
    seen.add(c);
    uniq.push(c);
  }

  for (const id of uniq) {
    const titles = await fetchJsonTitles(`/output/${id}/reference_titles.json`);
    if (titles !== null) return titles;
  }

  return [];
}

/* ----------------------------- Page ------------------------------- */

export default function PaperGraphPage() {
  const { paperId: rawId } = useParams();
  const urlPaperId = rawId ? decodeURIComponent(rawId) : '';
  const navigate = useNavigate();

  const [state, setState] = useState<PaperGraphState>({
    nodes: [],
    edges: [],
    loading: true,
    expanding: false,
    error: null,
    centerId: urlPaperId
  });

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  /* ----------------------- Initial Load ----------------------- */

  const loadGraph = useCallback(async () => {
    if (!urlPaperId) return;

    setState(prev => ({ ...prev, loading: true, error: null, centerId: urlPaperId }));

    try {
      const data = await getPaperGraph(urlPaperId, 0);

      if (!data || !data.nodes || data.nodes.length === 0) {
        throw new Error('No graph data returned from agent.');
      }

      // robust center match
      let effectiveCenterId = urlPaperId;
      const exactCenter = data.nodes.find((n: GraphNode) => n.id === urlPaperId);
      if (!exactCenter) {
        const fuzzyCenter = data.nodes.find(
          (n: GraphNode) => n.id.includes(urlPaperId) || urlPaperId.includes(n.id)
        );
        if (fuzzyCenter) effectiveCenterId = fuzzyCenter.id;
      }

      const nodes = data.nodes.map((n: GraphNode) => ({
        ...n,
        isCenter: n.id === effectiveCenterId,
        hasDetails: n.id === effectiveCenterId
      }));

      setState({
        nodes,
        edges: data.edges,
        loading: false,
        expanding: false,
        error: null,
        centerId: effectiveCenterId
      });

      setSelectedNode(nodes.find((n: GraphNode) => n.id === effectiveCenterId) || null);

      // background pdf check
      hasPdf(urlPaperId).then(check => {
        if (!check.exists) {
          fetchPaperIfMissing(urlPaperId).catch(() => {});
        }
      });
    } catch (err) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load graph'
      }));
    }
  }, [urlPaperId]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  /* ------------------------ Click Handlers ------------------------ */

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
  }, []);

  const handleNodeDoubleClick = useCallback(
    async (node: GraphNode) => {
      if (state.expanding) return;

      setState(prev => ({ ...prev, expanding: true }));

      try {
        // Center Node Expansion
        if (node.id === state.centerId) {
          const titles = await loadReferenceTitles(state.centerId, urlPaperId);

          const existingIds = new Set(state.nodes.map(n => n.id));
          const newNodes: GraphNode[] = [];
          const newEdges: GraphEdge[] = [];

          for (const refTitle of titles) {
            if (existingIds.has(refTitle)) continue;

            newNodes.push({
              id: refTitle,
              title: refTitle,
              authors: [],
              abstract: '',
              year: undefined,
              isCenter: false,
              hasDetails: false
            });

            newEdges.push({
              source: state.centerId,
              target: refTitle,
              type: 'references'
            });
          }

          setState(prev => ({
            ...prev,
            nodes: [...prev.nodes, ...newNodes],
            edges: [...prev.edges, ...newEdges],
            expanding: false
          }));
          return;
        }

        // Navigation
        navigate(`/paper/${encodeURIComponent(node.id)}`);
      } finally {
        setState(prev => ({ ...prev, expanding: false }));
      }
    },
    [state.nodes, state.centerId, state.expanding, navigate, urlPaperId]
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
            <span style={{ fontWeight: 600 }}>Paper Graph</span>
          </div>
          <div style={{ fontSize: '12px', color: '#718096' }}>ID: {urlPaperId}</div>
        </header>

        <div style={{ flex: 1, position: 'relative' }}>
          {state.loading && <CenteredText text="Loading graph..." />}
          {state.expanding && <CenteredText text="Expanding references..." />}
          {state.error && <CenteredText text={state.error} error />}

          {!state.loading && !state.error && state.nodes.length === 0 && (
            <CenteredText text="No graph data available for this paper." />
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
            />
          )}
        </div>
      </div>

      <SidePanel
        selectedNode={selectedNode}
        mode="paper"
        onAction={() => {}}
        onClose={() => setSelectedNode(null)}
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
