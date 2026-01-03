/**
 * NeurIPS2025Page
 *
 * Displays NeurIPS 2025 papers as a graph with embedding-based clustering.
 * - Loads paper metadata from /api/neurips/papers
 * - Loads similarity edges from /api/neurips/similarities
 * - Nodes attract based on embedding similarity
 * - SidePanel shows paper details + PDF download button
 */

import React, { useEffect, useState, useCallback, useMemo } from 'react';
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
}

interface ClusterData {
  paper_id_to_cluster: Record<string, number>;
  cluster_sizes: Record<string, number>;
  k: number;
}

interface ClusterCenters {
  [clusterId: string]: { x: number; y: number };
}

// Generate stable key for NeurIPS paper
function neuripsStableKey(paperId: string): string {
  return `neurips:${paperId}`;
}

// Generate cluster centers in a grid layout
function generateClusterCenters(k: number): ClusterCenters {
  const centers: ClusterCenters = {};
  const cols = Math.ceil(Math.sqrt(k));
  const spacing = 600;

  for (let i = 0; i < k; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    centers[String(i)] = {
      x: col * spacing + 400,
      y: row * spacing + 400
    };
  }
  return centers;
}

export default function NeurIPS2025Page() {
  const [graphState, setGraphState] = useState<NeurIPSGraphState>({
    nodes: [],
  });
  const [papers, setPapers] = useState<Map<string, NeurIPSPaper>>(new Map());
  const [rawSimEdges, setRawSimEdges] = useState<SimilarityEdge[]>([]);
  const [clusterCenters, setClusterCenters] = useState<ClusterCenters>({});
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadingPdf, setDownloadingPdf] = useState(false);

  // ✅ 유사도 기준(슬라이더)
  const [minSim, setMinSim] = useState<number>(0.75);

  // ✅ 클러스터 수 (k) 선택
  const [numClusters, setNumClusters] = useState<number>(15);

  // ✅ 클러스터 인력 강도
  const [clusterStrength, setClusterStrength] = useState<number>(0.15);

  const {
    nodeColorMap,
    setNodeColor: handleNodeColorChange,
    resetNodeColor: handleNodeColorReset
  } = useNodeColors();

  // Load papers, similarities, and clusters
  const loadData = useCallback(async (k: number = 15) => {
    setIsLoading(true);
    setError(null);

    try {
      // Load papers and clusters in parallel
      const [papersRes, clustersRes] = await Promise.all([
        fetch('/api/neurips/papers'),
        fetch(`/api/neurips/clusters?k=${k}`)
      ]);

      if (!papersRes.ok) throw new Error(`Failed to load papers: ${papersRes.status}`);
      const papersData = await papersRes.json();

      // Parse clusters
      let clusterMap: Record<string, number> = {};
      let actualK = k;
      if (clustersRes.ok) {
        const clusterData: ClusterData = await clustersRes.json();
        clusterMap = clusterData.paper_id_to_cluster || {};
        actualK = clusterData.k || k;
      }

      // Generate cluster centers
      const centers = generateClusterCenters(actualK);
      setClusterCenters(centers);

      const paperList: NeurIPSPaper[] = papersData.papers || [];
      const paperMap = new Map<string, NeurIPSPaper>();
      paperList.forEach(p => paperMap.set(p.paper_id, p));
      setPapers(paperMap);

      // Create nodes with cluster info and paper name as title
      const nodes: GraphNode[] = paperList.map((p, idx) => {
        const clusterId = clusterMap[p.paper_id] ?? 0;
        const center = centers[String(clusterId)];

        return {
          id: p.paper_id,
          label: p.name.length > 40 ? p.name.substring(0, 37) + '...' : p.name,
          title: p.name,  // Full paper name for display
          stableKey: neuripsStableKey(p.paper_id),
          type: 'neurips_paper',
          cluster: clusterId,
          // Initial position near cluster center
          x: center ? center.x + (Math.random() - 0.5) * 200 : (idx % 50) * 30,
          y: center ? center.y + (Math.random() - 0.5) * 200 : Math.floor(idx / 50) * 30,
        };
      });

      // Load similarities (raw, no filtering here)
      let simEdges: SimilarityEdge[] = [];
      try {
        const simRes = await fetch('/api/neurips/similarities');
        if (simRes.ok) {
          const simData = await simRes.json();
          simEdges = simData.edges || [];
        }
      } catch (e) {
        console.warn('Could not load similarities:', e);
      }

      // ✅ raw edges 저장
      setRawSimEdges(simEdges);

      // ✅ nodes만 graphState에 저장
      setGraphState({ nodes });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load data when numClusters changes
  useEffect(() => {
    loadData(numClusters);
  }, [loadData, numClusters]);

  // ✅ minSim에 따라 edges를 “실시간”으로 재구성
  const filteredEdges: GraphEdge[] = useMemo(() => {
    if (rawSimEdges.length === 0 || papers.size === 0) return [];

    const edges = rawSimEdges
      .filter(e => e.similarity >= minSim)
      .filter(e => papers.has(e.source) && papers.has(e.target))
      .map(e => ({
        source: e.source,
        target: e.target,
        weight: e.similarity,
      }));

    return edges;
  }, [rawSimEdges, papers, minSim]);

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

        {/* ✅ Control Panel: Clusters + Similarity */}
        <div style={{
          position: 'absolute',
          top: '16px',
          left: '16px',
          zIndex: 5,
          backgroundColor: 'rgba(26, 32, 44, 0.95)',
          padding: '12px 14px',
          borderRadius: '8px',
          fontSize: '12px',
          color: '#a0aec0',
          width: '260px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
        }}>
          {/* Cluster Count (k) Slider */}
          <div style={{ marginBottom: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span>Clusters (k)</span>
              <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{numClusters}</span>
            </div>
            <input
              type="range"
              min={5}
              max={30}
              step={1}
              value={numClusters}
              onChange={(e) => setNumClusters(parseInt(e.target.value, 10))}
              style={{ width: '100%' }}
            />
            <div style={{ marginTop: '4px', color: '#718096', fontSize: '11px' }}>
              Adjust number of topic clusters.
            </div>
          </div>

          {/* Cluster Strength Slider */}
          <div style={{ marginBottom: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span>Cluster strength</span>
              <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{clusterStrength.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={clusterStrength}
              onChange={(e) => setClusterStrength(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ marginTop: '4px', color: '#718096', fontSize: '11px' }}>
              How strongly nodes attract to cluster centers.
            </div>
          </div>

          {/* Similarity Threshold Slider */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span>Min similarity</span>
              <span style={{ color: '#e2e8f0' }}>{minSim.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={minSim}
              onChange={(e) => setMinSim(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ marginTop: '4px', color: '#718096', fontSize: '11px' }}>
              Increase to reduce links (stricter).
            </div>
          </div>
        </div>

        <GraphCanvas
          nodes={graphState.nodes}
          edges={filteredEdges}
          onNodeClick={handleNodeClick}
          selectedNodeId={selectedNode?.id}
          nodeColorMap={nodeColorMap}
          clusterCenters={clusterCenters}
          clusterStrength={clusterStrength}
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
          {graphState.nodes.length} papers | {numClusters} clusters | {filteredEdges.length} edges
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
