/**
 * PaperGraphPage (Graph A)
 *
 * Displays a paper-centered reference graph that can be incrementally
 * expanded by double-clicking on nodes.
 */

import React, { useEffect, useState, useCallback } from 'react';
import GraphCanvas from '../components/GraphCanvas';
import SidePanel from '../components/SidePanel';
import {
  getPaperGraph,
  expandPaperGraph,
  hasPdf,
  fetchPaperIfMissing,
  GraphNode,
  GraphEdge
} from '../api/mcp';

interface PaperGraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  loading: boolean;
  expanding: boolean;
  error: string | null;
  centerId: string;
}

interface PaperGraphPageProps {
  paperId: string;
}

export default function PaperGraphPage({ paperId }: PaperGraphPageProps) {
  const [state, setState] = useState<PaperGraphState>({
    nodes: [],
    edges: [],
    loading: true,
    expanding: false,
    error: null,
    centerId: paperId
  });

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [depth, setDepth] = useState(1);

  // Load initial graph
  const loadGraph = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      // First check if PDF exists, if not try to fetch it
      const pdfCheck = await hasPdf(paperId);
      if (!pdfCheck.exists) {
        // Try to fetch from arXiv
        await fetchPaperIfMissing(paperId);
      }

      // Build the reference subgraph
      const data = await getPaperGraph(paperId, depth);

      // Mark center node
      const nodes = data.nodes.map(node => ({
        ...node,
        isCenter: node.id === paperId
      }));

      setState({
        nodes,
        edges: data.edges,
        loading: false,
        expanding: false,
        error: null,
        centerId: paperId
      });

      // Auto-select center node
      const centerNode = nodes.find(n => n.isCenter);
      if (centerNode) {
        setSelectedNode(centerNode);
      }
    } catch (err) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load graph'
      }));
    }
  }, [paperId, depth]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  // Handle node click (selection)
  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
  }, []);

  // Handle node double-click (expansion)
  const handleNodeDoubleClick = useCallback(async (node: GraphNode) => {
    if (state.expanding) return;

    setState(prev => ({ ...prev, expanding: true }));

    try {
      const existingNodeIds = state.nodes.map(n => n.id);
      const diff = await expandPaperGraph(node.id, existingNodeIds);

      if (diff.new_nodes.length === 0) {
        // No new nodes to add
        setState(prev => ({ ...prev, expanding: false }));
        return;
      }

      // Merge new nodes and edges
      setState(prev => ({
        ...prev,
        nodes: [...prev.nodes, ...diff.new_nodes],
        edges: [...prev.edges, ...diff.new_edges],
        expanding: false
      }));
    } catch (err) {
      console.error('Failed to expand node:', err);
      setState(prev => ({ ...prev, expanding: false }));
    }
  }, [state.nodes, state.expanding]);

  // Handle expand action from side panel
  const handleExpandFromPanel = useCallback((nodeId: string) => {
    const node = state.nodes.find(n => n.id === nodeId);
    if (node) {
      handleNodeDoubleClick(node);
    }
  }, [state.nodes, handleNodeDoubleClick]);

  // Reset graph to initial state
  const handleReset = useCallback(() => {
    loadGraph();
  }, [loadGraph]);

  // Navigate back to global graph
  const handleBackToGlobal = useCallback(() => {
    window.location.href = '/';
  }, []);

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#f5f5f5' }}>
      {/* Main Graph Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <header
          style={{
            padding: '16px 24px',
            backgroundColor: '#fff',
            borderBottom: '1px solid #e2e8f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <button
              onClick={handleBackToGlobal}
              style={{
                padding: '8px 12px',
                backgroundColor: '#edf2f7',
                color: '#4a5568',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              ← Back to Global
            </button>
            <div>
              <h1 style={{ margin: 0, fontSize: '20px', color: '#1a202c' }}>
                Paper Reference Graph
              </h1>
              <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#718096' }}>
                {paperId} • Double-click nodes to expand
              </p>
            </div>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '13px', color: '#4a5568' }}>
                Initial Depth:
              </label>
              <select
                value={depth}
                onChange={(e) => setDepth(parseInt(e.target.value))}
                style={{
                  padding: '6px 10px',
                  borderRadius: '4px',
                  border: '1px solid #e2e8f0',
                  fontSize: '13px'
                }}
              >
                <option value={1}>1 level</option>
                <option value={2}>2 levels</option>
                <option value={3}>3 levels</option>
              </select>
            </div>

            <button
              onClick={handleReset}
              disabled={state.loading}
              style={{
                padding: '8px 16px',
                backgroundColor: state.loading ? '#cbd5e0' : '#4299e1',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                cursor: state.loading ? 'not-allowed' : 'pointer'
              }}
            >
              {state.loading ? 'Loading...' : 'Reset Graph'}
            </button>
          </div>
        </header>

        {/* Graph Canvas */}
        <div style={{ flex: 1, position: 'relative' }}>
          {/* Loading overlay */}
          {state.loading && (
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center',
                zIndex: 10
              }}
            >
              <div
                style={{
                  width: '40px',
                  height: '40px',
                  border: '3px solid #e2e8f0',
                  borderTopColor: '#4299e1',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite',
                  margin: '0 auto 12px'
                }}
              />
              <p style={{ color: '#718096', fontSize: '14px' }}>
                Building reference graph...
              </p>
            </div>
          )}

          {/* Expanding indicator */}
          {state.expanding && (
            <div
              style={{
                position: 'absolute',
                top: '16px',
                left: '50%',
                transform: 'translateX(-50%)',
                padding: '8px 16px',
                backgroundColor: 'rgba(66, 153, 225, 0.9)',
                color: '#fff',
                borderRadius: '6px',
                fontSize: '13px',
                zIndex: 10,
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <div
                style={{
                  width: '14px',
                  height: '14px',
                  border: '2px solid rgba(255,255,255,0.3)',
                  borderTopColor: '#fff',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}
              />
              Expanding references...
            </div>
          )}

          {/* Error display */}
          {state.error && (
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center',
                color: '#e53e3e'
              }}
            >
              <p style={{ fontSize: '16px', marginBottom: '8px' }}>Error</p>
              <p style={{ fontSize: '14px' }}>{state.error}</p>
              <button
                onClick={handleReset}
                style={{
                  marginTop: '12px',
                  padding: '8px 16px',
                  backgroundColor: '#e53e3e',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer'
                }}
              >
                Retry
              </button>
            </div>
          )}

          {/* Graph */}
          {!state.loading && !state.error && (
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

        {/* Stats Bar */}
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
          <span>Papers: {state.nodes.length}</span>
          <span>References: {state.edges.length}</span>
          <span>Center: {state.centerId}</span>
        </div>
      </div>

      {/* Side Panel */}
      <SidePanel
        selectedNode={selectedNode}
        mode="paper"
        onAction={handleExpandFromPanel}
        onClose={() => setSelectedNode(null)}
      />

      {/* CSS for spinner animation */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
