/**
 * GlobalGraphPage (Graph B)
 *
 * Displays the global paper relationship overview with embedding-based
 * similarity clustering. All papers are shown at once.
 */

import React, { useEffect, useState, useCallback } from 'react';
import GraphCanvas from '../components/GraphCanvas';
import SidePanel from '../components/SidePanel';
import { getGlobalGraph, GraphNode, GraphEdge } from '../lib/mcp';

interface GlobalGraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  loading: boolean;
  error: string | null;
  meta: {
    total_papers: number;
    total_edges: number;
    similarity_threshold: number;
    used_embeddings: boolean;
  } | null;
}

export default function GlobalGraphPage() {
  const [state, setState] = useState<GlobalGraphState>({
    nodes: [],
    edges: [],
    loading: true,
    error: null,
    meta: null
  });

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [useEmbeddings, setUseEmbeddings] = useState(true);

  // Load global graph on mount or when parameters change
  const loadGraph = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await getGlobalGraph(similarityThreshold, useEmbeddings);
      setState({
        nodes: data.nodes,
        edges: data.edges,
        loading: false,
        error: null,
        meta: data.meta
      });
    } catch (err) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load graph'
      }));
    }
  }, [similarityThreshold, useEmbeddings]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  // Handle node selection
  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
  }, []);

  // Handle navigation to paper graph (Graph A)
  const handleViewPaperGraph = useCallback((paperId: string) => {
    // Navigate to paper graph page
    window.location.href = `/paper/${encodeURIComponent(paperId)}`;
  }, []);

  // Rebuild graph with new parameters
  const handleRebuild = useCallback(() => {
    loadGraph();
  }, [loadGraph]);

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
          <div>
            <h1 style={{ margin: 0, fontSize: '20px', color: '#1a202c' }}>
              Global Paper Graph
            </h1>
            <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#718096' }}>
              Overview of all papers with similarity-based connections
            </p>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '13px', color: '#4a5568' }}>
                Similarity Threshold:
              </label>
              <input
                type="range"
                min="0.3"
                max="0.95"
                step="0.05"
                value={similarityThreshold}
                onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
                style={{ width: '100px' }}
              />
              <span style={{ fontSize: '13px', color: '#718096', minWidth: '40px' }}>
                {similarityThreshold.toFixed(2)}
              </span>
            </div>

            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: '#4a5568' }}>
              <input
                type="checkbox"
                checked={useEmbeddings}
                onChange={(e) => setUseEmbeddings(e.target.checked)}
              />
              Use Embeddings
            </label>

            <button
              onClick={handleRebuild}
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
              {state.loading ? 'Loading...' : 'Rebuild Graph'}
            </button>
          </div>
        </header>

        {/* Graph Canvas */}
        <div style={{ flex: 1, position: 'relative' }}>
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
                Building global graph...
              </p>
            </div>
          )}

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
                onClick={handleRebuild}
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

          {!state.loading && !state.error && (
            <GraphCanvas
              nodes={state.nodes}
              edges={state.edges}
              mode="global"
              onNodeClick={handleNodeClick}
              selectedNodeId={selectedNode?.id}
            />
          )}
        </div>

        {/* Stats Bar */}
        {state.meta && (
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
            <span>Papers: {state.meta.total_papers}</span>
            <span>Connections: {state.meta.total_edges}</span>
            <span>
              Method: {state.meta.used_embeddings ? 'Embedding Similarity' : 'Reference-based'}
            </span>
          </div>
        )}
      </div>

      {/* Side Panel */}
      <SidePanel
        selectedNode={selectedNode}
        mode="global"
        onAction={handleViewPaperGraph}
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
