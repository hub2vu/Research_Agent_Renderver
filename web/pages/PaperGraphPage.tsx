/**
 * PaperGraphPage (Graph A)
 *
 * Center paper → reference titles expansion (local JSON)
 * Reference node → arXiv fetch on demand
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

/* ----------------------- Helper: Load titles ---------------------- */

async function loadReferenceTitles(paperId: string): Promise<string[]> {
  try {
    const res = await fetch(`/output/${paperId}/reference_titles.json`);
    if (!res.ok) return [];
    const json = await res.json();
    return json.titles || [];
  } catch (e) {
    return [];
  }
}

/* ----------------------------- Page ------------------------------- */

export default function PaperGraphPage() {
  // [FIX 1] Decode the ID from URL to handle special characters correctly
  const { paperId: rawId } = useParams();
  const paperId = rawId ? decodeURIComponent(rawId) : "";
  
  const navigate = useNavigate();

  const [state, setState] = useState<PaperGraphState>({
    nodes: [],
    edges: [],
    loading: true, // Start in loading state
    expanding: false,
    error: null,
    centerId: paperId
  });

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  /* ----------------------- Initial Load ----------------------- */

  const loadGraph = useCallback(async () => {
    if (!paperId) return;

    setState(prev => ({ ...prev, loading: true, error: null, centerId: paperId }));

    try {
      console.log(`Loading graph for ID: ${paperId}`);
      
      // [FIX 2] Call API immediately without waiting for PDF check
      const data = await getPaperGraph(paperId, 0);

      // Handle empty data case
      if (!data || !data.nodes || data.nodes.length === 0) {
        throw new Error("No graph data returned from agent.");
      }

      console.log(`Graph loaded. Nodes: ${data.nodes.length}, Edges: ${data.edges.length}`);

      // [FIX 3] Robust Center Logic: Try to find center by exact ID, then by inclusion
      let effectiveCenterId = paperId;
      const exactCenter = data.nodes.find(n => n.id === paperId);
      
      if (!exactCenter) {
        // If exact ID not found, find a node that *contains* the ID (handling prefix differences)
        const fuzzyCenter = data.nodes.find(n => n.id.includes(paperId) || paperId.includes(n.id));
        if (fuzzyCenter) {
          console.log(`Center ID adjusted: ${paperId} -> ${fuzzyCenter.id}`);
          effectiveCenterId = fuzzyCenter.id;
        }
      }

      const nodes = data.nodes.map(n => ({
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

      setSelectedNode(nodes.find(n => n.id === effectiveCenterId) || null);

      // [OPTIONAL] Background PDF check (Non-blocking)
      hasPdf(paperId).then(check => {
        if (!check.exists) {
          fetchPaperIfMissing(paperId).catch(e => console.warn("Background fetch warning:", e));
        }
      });

    } catch (err) {
      console.error("Graph load failed:", err);
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load graph'
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
    async (node: GraphNode) => {
      if (state.expanding) return;

      setState(prev => ({ ...prev, expanding: true }));

      try {
        // Center Node Expansion
        if (node.id === state.centerId) {
          const titles = await loadReferenceTitles(state.centerId);
          const existingIds = new Set(state.nodes.map(n => n.id));
          const newNodes: GraphNode[] = [];
          const newEdges: GraphEdge[] = [];

          titles.forEach(title => {
            if (existingIds.has(title)) return;
            newNodes.push({
              id: title, title, authors: [], abstract: '', year: undefined,
              isCenter: false, hasDetails: false
            });
            newEdges.push({
              source: state.centerId, target: title, type: 'references'
            });
          });

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
      } catch (err) {
        console.error(err);
      } finally {
        setState(prev => ({ ...prev, expanding: false }));
      }
    },
    [state.nodes, state.centerId, state.expanding, navigate]
  );

  /* ------------------------ Navigation ------------------------ */

  const handleBackToGlobal = useCallback(() => {
    navigate('/');
  }, [navigate]);

  /* ----------------------------- UI ----------------------------- */

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#f5f5f5' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <header style={{
          padding: '16px 24px', backgroundColor: '#fff', borderBottom: '1px solid #e2e8f0',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center'
        }}>
          <div>
            <button onClick={handleBackToGlobal} style={{
              padding: '6px 12px', borderRadius: '6px', backgroundColor: '#edf2f7',
              border: 'none', cursor: 'pointer', marginRight: '12px'
            }}>← Back</button>
            <span style={{ fontWeight: 600 }}>Paper Graph</span>
          </div>
          <div style={{ fontSize: '12px', color: '#718096' }}>ID: {paperId}</div>
        </header>

        <div style={{ flex: 1, position: 'relative' }}>
          {state.loading && <CenteredText text="Loading graph..." />}
          {state.expanding && <CenteredText text="Expanding references..." />}
          {state.error && <CenteredText text={state.error} error />}
          
          {/* Show "No Data" if loading done but no nodes */}
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
    <div style={{
      position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
      color: error ? '#e53e3e' : '#718096', fontSize: '14px', fontWeight: 500
    }}>
      {text}
    </div>
  );
}