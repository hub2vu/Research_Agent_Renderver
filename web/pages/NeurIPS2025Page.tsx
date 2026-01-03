/**
 * NeurIPS2025Page
 *
 * Displays NeurIPS 2025 papers as a graph with embedding-based clustering.
 * - Loads paper metadata from /api/neurips/papers
 * - Loads similarity edges from /api/neurips/similarities
 * - Nodes attract based on embedding similarity
 * - SidePanel shows paper details + PDF download button
 */

import React, { useEffect, useState, useCallback, useRef } from 'react';
import GraphCanvas from '../components/GraphCanvas';
import SidePanel from '../components/SidePanel';
import { GraphNode, GraphEdge } from '../lib/mcp';
import { useNodeColors } from '../hooks/useNodeColors';

interface NeurIPSPaper {
  paper_id: string;
  name: string;
  abstract: string;
  'speakers/authors': string;
  virtualsite_url: string;
}

interface SimilarityEdge {
  source: string;
  target: string;
  similarity: number;
}

interface NeurIPSGraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// Generate stable key for NeurIPS paper
function neuripsStableKey(paperId: string): string {
  return `neurips:${paperId}`;
}

export default function NeurIPS2025Page() {
  const [graphState, setGraphState] = useState<NeurIPSGraphState>({
    nodes: [],
    edges: [],
  });
  const [papers, setPapers] = useState<Map<string, NeurIPSPaper>>(new Map());
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadingPdf, setDownloadingPdf] = useState(false);

  const {
    nodeColorMap,
    setNodeColor: handleNodeColorChange,
    resetNodeColor: handleNodeColorReset
  } = useNodeColors();

  // Load papers and similarities
  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Load papers
      const papersRes = await fetch('/api/neurips/papers');
      if (!papersRes.ok) throw new Error(`Failed to load papers: ${papersRes.status}`);
      const papersData = await papersRes.json();

      const paperList: NeurIPSPaper[] = papersData.papers || [];
      const paperMap = new Map<string, NeurIPSPaper>();
      paperList.forEach(p => paperMap.set(p.paper_id, p));
      setPapers(paperMap);

      // Create nodes
      const nodes: GraphNode[] = paperList.map((p, idx) => ({
        id: p.paper_id,
        label: p.name.length > 60 ? p.name.substring(0, 57) + '...' : p.name,
        stableKey: neuripsStableKey(p.paper_id),
        type: 'neurips_paper',
        // Initial positions in a grid to help clustering
        x: (idx % 50) * 30,
        y: Math.floor(idx / 50) * 30,
      }));

      // Load similarities
      let edges: GraphEdge[] = [];
      try {
        const simRes = await fetch('/api/neurips/similarities');
        if (simRes.ok) {
          const simData = await simRes.json();
          const simEdges: SimilarityEdge[] = simData.edges || [];

          edges = simEdges
            .filter(e => paperMap.has(e.source) && paperMap.has(e.target))
            .map(e => ({
              source: e.source,
              target: e.target,
              weight: e.similarity,
            }));
        }
      } catch (e) {
        console.warn('Could not load similarities:', e);
      }

      setGraphState({ nodes, edges });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle node click
  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
  }, []);

  // Handle PDF download
  const handleDownloadPdf = useCallback(async () => {
    if (!selectedNode) return;

    const paper = papers.get(selectedNode.id);
    if (!paper) return;

    setDownloadingPdf(true);
    try {
      const res = await fetch('/api/tools/call', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool_name: 'neurips2025_download_pdf',
          arguments: {
            paper_id: paper.paper_id,
            action: 'pdf'
          }
        }),
      });

      const result = await res.json();
      if (result.error) {
        alert(`Download failed: ${result.error}`);
      } else {
        alert(`PDF downloaded successfully!\n${result.result || ''}`);
      }
    } catch (err) {
      alert(`Download error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setDownloadingPdf(false);
    }
  }, [selectedNode, papers]);

  // Get selected paper details
  const selectedPaper = selectedNode ? papers.get(selectedNode.id) : null;

  // Custom SidePanel content for NeurIPS papers
  const renderNeurIPSDetails = () => {
    if (!selectedPaper) return null;

    return (
      <div style={{ marginTop: '16px' }}>
        <div style={{ marginBottom: '12px' }}>
          <label style={{ fontSize: '11px', color: '#a0aec0', textTransform: 'uppercase' }}>
            Authors
          </label>
          <div style={{ fontSize: '13px', color: '#e2e8f0', marginTop: '4px' }}>
            {selectedPaper['speakers/authors'] || 'N/A'}
          </div>
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label style={{ fontSize: '11px', color: '#a0aec0', textTransform: 'uppercase' }}>
            Abstract
          </label>
          <div style={{
            fontSize: '12px',
            color: '#cbd5e0',
            marginTop: '4px',
            maxHeight: '200px',
            overflowY: 'auto',
            lineHeight: 1.5,
          }}>
            {selectedPaper.abstract || 'N/A'}
          </div>
        </div>

        {selectedPaper.virtualsite_url && (
          <div style={{ marginBottom: '12px' }}>
            <a
              href={selectedPaper.virtualsite_url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: '#63b3ed',
                fontSize: '13px',
                textDecoration: 'none',
              }}
            >
              View on NeurIPS Virtual Site
            </a>
          </div>
        )}

        <button
          onClick={handleDownloadPdf}
          disabled={downloadingPdf}
          style={{
            width: '100%',
            padding: '10px',
            marginTop: '8px',
            border: 'none',
            borderRadius: '6px',
            backgroundColor: downloadingPdf ? '#4a5568' : '#48bb78',
            color: '#fff',
            fontSize: '13px',
            fontWeight: 500,
            cursor: downloadingPdf ? 'not-allowed' : 'pointer',
          }}
        >
          {downloadingPdf ? 'Downloading...' : 'Download PDF'}
        </button>
      </div>
    );
  };

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: 'calc(100vh - 60px)',
        color: '#a0aec0',
        fontSize: '16px',
      }}>
        Loading NeurIPS 2025 papers...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: 'calc(100vh - 60px)',
        color: '#fc8181',
        fontSize: '16px',
      }}>
        Error: {error}
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 60px)' }}>
      <div style={{ flex: 1, position: 'relative' }}>
        <GraphCanvas
          nodes={graphState.nodes}
          edges={graphState.edges}
          onNodeClick={handleNodeClick}
          selectedNodeId={selectedNode?.id}
          nodeColorMap={nodeColorMap}
          mode="global"
        />

        {/* Stats overlay */}
        <div style={{
          position: 'absolute',
          bottom: '16px',
          left: '16px',
          backgroundColor: 'rgba(26, 32, 44, 0.9)',
          padding: '8px 12px',
          borderRadius: '6px',
          fontSize: '12px',
          color: '#a0aec0',
        }}>
          {graphState.nodes.length} papers | {graphState.edges.length} similarity edges
        </div>
      </div>

      <SidePanel
        selectedNode={selectedNode ? {
          ...selectedNode,
          label: selectedPaper?.name || selectedNode.label,
        } : null}
        onClose={() => setSelectedNode(null)}
        onNodeColorChange={(key, color) => {
          const k = selectedNode?.stableKey || key;
          handleNodeColorChange(k, color);
        }}
        onNodeColorReset={(key) => {
          const k = selectedNode?.stableKey || key;
          handleNodeColorReset(k);
        }}
        nodeColor={selectedNode?.stableKey ? nodeColorMap[selectedNode.stableKey] : undefined}
        extraContent={renderNeurIPSDetails()}
      />
    </div>
  );
}
