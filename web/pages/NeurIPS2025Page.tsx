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
import NeurIPSSearchSidebar from '../components/NeurIPSSearchSidebar';
import NeurIPSRankedList from '../components/NeurIPSRankedList';
import PaperListView from '../components/PaperListView';
import { GraphNode, GraphEdge, executeNeurIPSSearchAndRank, executeTool } from '../lib/mcp';
import { ScoredPaper } from '../components/PaperResultCard';
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

const PAGE_SIZE = 20;

export default function NeurIPS2025Page() {
  const [viewMode, setViewMode] = useState<'graph' | 'list'>('graph');
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

  //  토글
  const [showControls, setShowControls] = useState(true);
  const [showSearchUI, setShowSearchUI] = useState(true);

  // ✅ 유사도 기준(슬라이더)
  const [minSim, setMinSim] = useState<number>(0.75);

  // ✅ 클러스터 수 (k) 선택
  const [numClusters, setNumClusters] = useState<number>(15);

  // ✅ 클러스터 인력 강도
  const [clusterStrength, setClusterStrength] = useState<number>(0.15);

  // Search & Ranking state
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<ScoredPaper[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [highlightedPaperIds, setHighlightedPaperIds] = useState<Set<string>>(new Set());
  const [focusNodeId, setFocusNodeId] = useState<string | undefined>(undefined);
  const [clusterMap, setClusterMap] = useState<Record<string, number>>({});

  // ✅ Graph pagination state (Load more – 20개씩)
  const [graphOffset, setGraphOffset] = useState(0);
  const [graphHasMore, setGraphHasMore] = useState(true);
  const [graphLoadingMore, setGraphLoadingMore] = useState(false);
  const [totalPapers, setTotalPapers] = useState(0);

  // ✅ List pagination state (Load more – 20개씩)
  const [listNodes, setListNodes] = useState<GraphNode[]>([]);
  const [listOffset, setListOffset] = useState(0);
  const [listHasMore, setListHasMore] = useState(true);
  const [listLoadingMore, setListLoadingMore] = useState(false);

  const {
    nodeColorMap,
    setNodeColor: handleNodeColorChange,
    resetNodeColor: handleNodeColorReset
  } = useNodeColors();

  // Helper: create GraphNode from NeurIPSPaper
  const makePaperNode = useCallback((p: NeurIPSPaper, localClusterMap: Record<string, number>, centers: ClusterCenters, idx: number): GraphNode => {
    const clusterId = localClusterMap[p.paper_id] ?? 0;
    const center = centers[String(clusterId)];
    return {
      id: p.paper_id,
      label: p.name.length > 40 ? p.name.substring(0, 37) + '...' : p.name,
      title: p.name,
      stableKey: neuripsStableKey(p.paper_id),
      type: 'neurips_paper',
      cluster: clusterId,
      abstract: p.abstract,
      authors: p['speakers/authors'] ? p['speakers/authors'].split(',').map(s => s.trim()) : [],
      x: center ? center.x + (Math.random() - 0.5) * 200 : (idx % 50) * 30,
      y: center ? center.y + (Math.random() - 0.5) * 200 : Math.floor(idx / 50) * 30,
    };
  }, []);

  // Load first PAGE_SIZE papers + clusters + similarities (Graph mode)
  const loadData = useCallback(async (k: number = 15) => {
    setIsLoading(true);
    setError(null);
    // Reset graph pagination
    setGraphOffset(0);
    setGraphHasMore(true);
    setPapers(new Map());
    setGraphState({ nodes: [] });

    try {
      // Load first page of papers, clusters, and similarities in parallel
      const [papersRes, clustersRes, simRes] = await Promise.all([
        fetch(`/api/neurips/papers?offset=0&limit=${PAGE_SIZE}`),
        fetch(`/api/neurips/clusters?k=${k}`),
        fetch('/api/neurips/similarities').catch(() => null),
      ]);

      if (!papersRes.ok) throw new Error(`Failed to load papers: ${papersRes.status}`);
      const papersData = await papersRes.json();

      // Parse clusters
      let localClusterMap: Record<string, number> = {};
      let actualK = k;
      if (clustersRes.ok) {
        const clusterData: ClusterData = await clustersRes.json();
        localClusterMap = clusterData.paper_id_to_cluster || {};
        actualK = clusterData.k || k;
        setClusterMap(localClusterMap);
      }

      // Generate cluster centers
      const centers = generateClusterCenters(actualK);
      setClusterCenters(centers);

      const paperList: NeurIPSPaper[] = papersData.papers || [];
      const paperMap = new Map<string, NeurIPSPaper>();
      paperList.forEach(p => paperMap.set(p.paper_id, p));
      setPapers(paperMap);

      // Create nodes (only first PAGE_SIZE)
      const nodes: GraphNode[] = paperList.map((p, idx) => makePaperNode(p, localClusterMap, centers, idx));

      // Load similarities (raw, no filtering here)
      let simEdges: SimilarityEdge[] = [];
      if (simRes && simRes.ok) {
        try {
          const simData = await simRes.json();
          simEdges = simData.edges || [];
        } catch (e) {
          console.warn('Could not parse similarities:', e);
        }
      }

      setRawSimEdges(simEdges);
      setGraphState({ nodes });
      setGraphOffset(papersData.nextOffset ?? PAGE_SIZE);
      setGraphHasMore(!!papersData.hasMore);
      setTotalPapers(papersData.total ?? paperList.length);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  }, [makePaperNode]);

  // Load more papers for Graph mode (20개씩 추가)
  const loadMoreGraph = useCallback(async () => {
    if (graphLoadingMore || !graphHasMore) return;

    setGraphLoadingMore(true);
    try {
      const res = await fetch(`/api/neurips/papers?offset=${graphOffset}&limit=${PAGE_SIZE}`);
      if (!res.ok) throw new Error(`Failed to load papers: ${res.status}`);
      const data = await res.json();
      const paperList: NeurIPSPaper[] = data.papers || [];

      // Merge new papers into existing Map
      setPapers(prev => {
        const next = new Map(prev);
        paperList.forEach(p => next.set(p.paper_id, p));
        return next;
      });

      // Add new nodes to graph
      const newNodes = paperList.map((p, idx) => makePaperNode(p, clusterMap, clusterCenters, graphState.nodes.length + idx));
      setGraphState(prev => ({ nodes: [...prev.nodes, ...newNodes] }));

      setGraphOffset(data.nextOffset ?? (graphOffset + PAGE_SIZE));
      setGraphHasMore(!!data.hasMore);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Load more failed');
    } finally {
      setGraphLoadingMore(false);
    }
  }, [graphOffset, graphHasMore, graphLoadingMore, clusterMap, clusterCenters, graphState.nodes.length, makePaperNode]);

  // Load more for List mode
  const loadMoreList = useCallback(async () => {
    if (listLoadingMore || !listHasMore) return;

    setListLoadingMore(true);
    try {
      // Ensure clusters are loaded (for grouping by cluster)
      if (Object.keys(clusterMap).length === 0) {
        const clustersRes = await fetch(`/api/neurips/clusters?k=${numClusters}`);
        if (clustersRes.ok) {
          const clusterData: ClusterData = await clustersRes.json();
          setClusterMap(clusterData.paper_id_to_cluster || {});
        }
      }

      const res = await fetch(`/api/neurips/papers?offset=${listOffset}&limit=${PAGE_SIZE}`);
      if (!res.ok) throw new Error(`Failed to load papers: ${res.status}`);

      const data = await res.json();
      const paperList: NeurIPSPaper[] = data.papers || [];

      const newNodes: GraphNode[] = paperList.map((p, idx) => {
        const clusterId = (clusterMap[p.paper_id] ?? 0);
        return {
          id: p.paper_id,
          label: p.name.length > 40 ? p.name.substring(0, 37) + '...' : p.name,
          title: p.name,
          stableKey: neuripsStableKey(p.paper_id),
          type: 'neurips_paper',
          cluster: clusterId,
          abstract: p.abstract,
          authors: p['speakers/authors'] ? p['speakers/authors'].split(',').map(s => s.trim()) : [],
          x: 0 + (idx % 20),
          y: 0 + Math.floor(idx / 20),
        };
      });

      setListNodes(prev => [...prev, ...newNodes]);
      setListOffset(data.nextOffset ?? (listOffset + PAGE_SIZE));
      setListHasMore(!!data.hasMore);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setListLoadingMore(false);
    }
  }, [listOffset, listHasMore, listLoadingMore, numClusters, clusterMap]);

  // Load data when view mode or cluster count changes
  useEffect(() => {
    if (viewMode === 'graph') {
      loadData(numClusters);
      return;
    }

    // list mode: reset and load first page
    setIsLoading(false);
    setError(null);
    setSelectedNode(null);
    setListNodes([]);
    setListOffset(0);
    setListHasMore(true);

    // kick off first page
    (async () => {
      await Promise.resolve();
      // ensure the latest offset is used
      setListOffset(0);
    })();
  }, [viewMode, numClusters, loadData]);

  // When listOffset resets to 0 in list mode, load first page
  useEffect(() => {
    if (viewMode !== 'list') return;
    if (listNodes.length === 0 && listOffset === 0 && listHasMore && !listLoadingMore) {
      loadMoreList();
    }
  }, [viewMode, listOffset, listNodes.length, listHasMore, listLoadingMore, loadMoreList]);

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

  // Handle search
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      return;
    }

    setIsSearching(true);
    setError(null);
    setSearchResults([]);
    setHighlightedPaperIds(new Set());

    try {
      const result = await executeNeurIPSSearchAndRank(
        searchQuery.trim(),
        'users/profile.json',
        10, // topK
        numClusters // clusterK
      );

      if (result.success && result.ranked_papers) {
        setSearchResults(result.ranked_papers);

        // Update highlighted paper IDs
        const highlightedSet = new Set(result.ranked_papers.map(p => p.paper_id));
        setHighlightedPaperIds(highlightedSet);
      } else {
        throw new Error(result.error || 'Search failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to search papers');
    } finally {
      setIsSearching(false);
    }
  }, [searchQuery, numClusters]);

  // Handle paper click from list
  const handlePaperClick = useCallback((paperId: string) => {
    // Find the corresponding node
    const node = graphState.nodes.find(n => n.id === paperId);
    if (node) {
      setSelectedNode(node);
      setFocusNodeId(paperId);
      // Clear focus after animation
      setTimeout(() => setFocusNodeId(undefined), 700);
    }
  }, [graphState.nodes]);

  // Handle PDF download (for SidePanel)
  const handleDownloadPdf = useCallback(async () => {
    if (!selectedNode) return;
    const paper = papers.get(selectedNode.id);
    if (!paper) return;

    setDownloadingPdf(true);
    try {
      const result = await executeTool('process_neurips_paper', {
        paper_id: paper.paper_id,
        out_dir: '/data/pdf/neurips2025'
      });

      if (!result.success) {
        alert(`Pipeline failed: ${result.error}`);
      } else {
        const info = result.result?.pipeline_results;
        const refCount = info?.ref_count || 0;
        alert(`Process Complete!\n\n- PDF Saved: ${info?.pdf_path}\n- References Found: ${refCount}`);
      }
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`);
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
          <label style={{ fontSize: '11px', color: '#000000', textTransform: 'uppercase' }}>
            Authors
          </label>
          <div style={{ fontSize: '13px', color: '#000000', marginTop: '4px' }}>
            {selectedPaper['speakers/authors'] || 'N/A'}
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

  if (isLoading && viewMode === 'graph') {
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

  const loadedCount = viewMode === 'list' ? listNodes.length : graphState.nodes.length;
  const statsEdgesCount = viewMode === 'list' ? 0 : filteredEdges.length;

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 60px)' }}>
      <div style={{ flex: 1, position: 'relative' }}>

        {/* ✅ Control Panel: toggle */}
        {showControls ? (
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
            {/* 헤더 + Hide 버튼 */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '10px'
            }}>
              <div style={{ fontWeight: 700, color: '#e2e8f0' }}>Controls</div>
              <button
                onClick={() => setShowControls(false)}
                style={{
                  border: '1px solid #718096',
                  background: 'transparent',
                  color: '#e2e8f0',
                  borderRadius: '6px',
                  padding: '4px 8px',
                  cursor: 'pointer',
                  fontSize: '11px'
                }}
              >
                Hide
              </button>
            </div>

            {/* Node/List 토글 */}
            <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
              <button
                onClick={() => setViewMode('graph')}
                style={{
                  padding: '6px 10px',
                  borderRadius: '6px',
                  border: '1px solid #718096',
                  background: viewMode === 'graph' ? '#edf2f7' : 'transparent',
                  color: viewMode === 'graph' ? '#1a202c' : '#a0aec0',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                Node
              </button>
              <button
                onClick={() => setViewMode('list')}
                style={{
                  padding: '6px 10px',
                  borderRadius: '6px',
                  border: '1px solid #718096',
                  background: viewMode === 'list' ? '#edf2f7' : 'transparent',
                  color: viewMode === 'list' ? '#1a202c' : '#a0aec0',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                List
              </button>
            </div>

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

            {/* Similarity Threshold Slider (graph only) */}
            {viewMode === 'graph' && (
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
            )}
          </div>
        ) : (
          <button
            onClick={() => setShowControls(true)}
            style={{
              position: 'absolute',
              top: '16px',
              left: '16px',
              zIndex: 6,
              border: '1px solid #718096',
              backgroundColor: 'rgba(26, 32, 44, 0.95)',
              color: '#e2e8f0',
              borderRadius: '8px',
              padding: '8px 10px',
              cursor: 'pointer',
              fontSize: '12px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
            }}
          >
            Show Controls
          </button>
        )}

        {/* Search Sidebar: toggle */}
        {showSearchUI ? (
          <>
            <button
              onClick={() => setShowSearchUI(false)}
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                zIndex: 6,
                border: '1px solid #718096',
                backgroundColor: 'rgba(26, 32, 44, 0.95)',
                color: '#e2e8f0',
                borderRadius: '8px',
                padding: '8px 10px',
                cursor: 'pointer',
                fontSize: '12px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
              }}
            >
              Hide Search
            </button>

            <NeurIPSSearchSidebar
              searchQuery={searchQuery}
              onSearchQueryChange={setSearchQuery}
              onSearch={handleSearch}
              isSearching={isSearching}
            />
          </>
        ) : (
          <button
            onClick={() => setShowSearchUI(true)}
            style={{
              position: 'absolute',
              top: '16px',
              right: '16px',
              zIndex: 6,
              border: '1px solid #718096',
              backgroundColor: 'rgba(26, 32, 44, 0.95)',
              color: '#e2e8f0',
              borderRadius: '8px',
              padding: '8px 10px',
              cursor: 'pointer',
              fontSize: '12px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
            }}
          >
            Show Search
          </button>
        )}

        {/* Search Results List */}
        {showSearchUI && (isSearching ? (
          <div style={{
            position: 'absolute',
            bottom: '16px',
            right: '16px',
            zIndex: 5,
            backgroundColor: 'rgba(26, 32, 44, 0.95)',
            padding: '16px',
            borderRadius: '8px',
            width: '400px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            color: '#a0aec0',
            fontSize: '14px',
            textAlign: 'center',
          }}>
            Searching and ranking papers...
          </div>
        ) : searchResults.length > 0 ? (
          <NeurIPSRankedList
            papers={searchResults}
            onPaperClick={handlePaperClick}
            clusterMap={clusterMap}
            onClose={() => {
              setSearchResults([]);
              setSearchQuery('');
              setHighlightedPaperIds(new Set());
            }}
          />
        ) : searchQuery.trim() && !isSearching ? (
          <div style={{
            position: 'absolute',
            bottom: '16px',
            right: '16px',
            zIndex: 5,
            backgroundColor: 'rgba(26, 32, 44, 0.95)',
            padding: '16px',
            borderRadius: '8px',
            width: '400px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            color: '#a0aec0',
            fontSize: '14px',
            textAlign: 'center',
          }}>
            No papers found. Try a different search query.
          </div>
        ) : null)}

        {viewMode === 'graph' && (
          <>
            <GraphCanvas
              nodes={graphState.nodes}
              edges={filteredEdges as any}
              onNodeClick={handleNodeClick}
              selectedNodeId={selectedNode?.id}
              nodeColorMap={nodeColorMap}
              clusterCenters={clusterCenters}
              clusterStrength={clusterStrength}
              highlightedNodeIds={Array.from(highlightedPaperIds)}
              focusNodeId={focusNodeId}
              mode="global"
            />

            {/* Load More button for graph mode */}
            {graphHasMore && (
              <div style={{
                position: 'absolute',
                bottom: '50px',
                left: '50%',
                transform: 'translateX(-50%)',
                zIndex: 5,
              }}>
                <button
                  onClick={loadMoreGraph}
                  disabled={graphLoadingMore}
                  style={{
                    padding: '10px 24px',
                    borderRadius: '999px',
                    border: '1px solid rgba(255,255,255,0.3)',
                    background: 'rgba(26, 32, 44, 0.95)',
                    color: '#e2e8f0',
                    cursor: graphLoadingMore ? 'not-allowed' : 'pointer',
                    fontWeight: 700,
                    fontSize: '13px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                  }}
                >
                  {graphLoadingMore ? 'Loading...' : `Load more (${PAGE_SIZE})`}
                </button>
              </div>
            )}
          </>
        )}

        {viewMode === 'list' && (
          <div style={{ position: 'absolute', inset: 0, paddingTop: '60px' }}>
            <PaperListView
              nodes={listNodes as any}
              edges={[] as any}
              groupBy={(n) => (n.cluster ?? 0)}
              groupTitle={(k) => `Cluster ${k}`}
              onOpenPaper={(paperId) => window.location.href = `/paper/${encodeURIComponent(paperId)}`}
              initialPrefetchCount={30}
            />

            <div style={{
              position: 'sticky',
              bottom: 0,
              padding: '12px',
              background: 'rgba(255,255,255,0.95)',
              borderTop: '1px solid #e2e8f0',
              textAlign: 'center'
            }}>
              {listHasMore ? (
                <button
                  onClick={loadMoreList}
                  disabled={listLoadingMore}
                  style={{
                    padding: '10px 18px',
                    borderRadius: '999px',
                    border: '1px solid #cbd5e0',
                    background: '#fff',
                    cursor: listLoadingMore ? 'not-allowed' : 'pointer',
                    fontWeight: 700
                  }}
                >
                  {listLoadingMore ? 'Loading...' : `Load more (${PAGE_SIZE})`}
                </button>
              ) : (
                <div style={{ color: '#718096', fontSize: '13px' }}>No more papers.</div>
              )}
            </div>
          </div>
        )}

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
          {loadedCount}{totalPapers > 0 ? ` / ${totalPapers}` : ''} papers | {numClusters} clusters | {statsEdgesCount} edges
        </div>
      </div>

      {viewMode === 'graph' && (
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
      )}
    </div>
  );
}
