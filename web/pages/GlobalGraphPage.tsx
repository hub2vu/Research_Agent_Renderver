/**
 * GlobalGraphPage (Graph B)
 *
 * - ui_state.json 폴링(에이전트 네비게이션/리프레시)
 * - global_graph.json 변경(Last-Modified/ETag) 폴링 → 즉시 UI 반영
 *
 * [UPDATED]
 * - nodeColorMap 저장 키를 stableKey 우선으로 사용
 * - loadGraph 시 nodes에 stableKey를 주입(없으면 생성)
 * - SidePanel에 nodeColor(현재색)도 stableKey 기반으로 전달
 */

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

import GraphCanvas from '../components/GraphCanvas';
import SidePanel from '../components/SidePanel';
import ArxivSearchSidebar from '../components/ArxivSearchSidebar';
import ArxivRankedList from '../components/ArxivRankedList';
import { getGlobalGraph, rebuildGlobalGraph, GraphNode, executeRankFilterPipeline } from '../lib/mcp';
import { ScoredPaper } from '../components/PaperResultCard';
import { useNodeColors } from '../hooks/useNodeColors';

interface GlobalGraphState {
  nodes: GraphNode[];
  edges: any[];
  loading: boolean;
  error: string | null;
  meta: {
    total_papers: number;
    total_edges: number;
    similarity_threshold: number;
    used_embeddings: boolean;
  } | null;
}

/* ----------------------- Helpers: stableKey ---------------------- */

function normalizeArxivToDoiLike(id: string): string {
  if (!id) return id;
  if (id.startsWith('10.48550_arxiv.')) return id;
  const m = id.match(/^(\d{4}\.\d{4,5})(v\d+)?$/);
  if (m) return `10.48550_arxiv.${m[1]}`;
  return id;
}

function getStableKey(node: any): string {
  // backend가 stableKey를 주면 최우선
  if (node?.stableKey) return String(node.stableKey);

  // global graph는 대부분 paper 노드이므로 paper prefix로 충돌 방지
  const id = String(node?.id ?? '');
  const normalized = normalizeArxivToDoiLike(id);
  return `paper:${normalized || id}`;
}

export default function GlobalGraphPage() {
  const navigate = useNavigate();

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

  // Search & Ranking state
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<ScoredPaper[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [highlightedPaperIds, setHighlightedPaperIds] = useState<Set<string>>(new Set());
  const [focusNodeId, setFocusNodeId] = useState<string | undefined>(undefined);
  const [searchError, setSearchError] = useState<string | null>(null);

  // ui_state.json 폴링용
  const lastTimestampRef = useRef<number>(0);

  // global_graph.json 변경 감지용
  const lastGraphSigRef = useRef<string | null>(null);

  /* ------------------- Node color map (file-based persistence) ------------------- */

  const {
    nodeColorMap,
    setNodeColor: handleNodeColorChange,
    resetNodeColor: handleNodeColorReset
  } = useNodeColors();

  /* ----------------------- Load global graph ----------------------- */

  const loadGraph = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await getGlobalGraph();

      // ✅ stableKey 주입 (GraphCanvas가 stableKey 우선으로 색/좌표 캐시를 씀)
      const nodesWithKey = (data.nodes || []).map((n: any) => ({
        ...n,
        stableKey: getStableKey(n)
      })) as any as GraphNode[];

      setState({
        nodes: nodesWithKey,
        edges: data.edges || [],
        loading: false,
        error: data.meta?.error || null,
        meta: data.meta
      });

      // 선택 노드가 있고, 새 nodes에도 있다면 stableKey 기준으로 다시 매칭
      setSelectedNode(prev => {
        if (!prev) return prev;
        const prevKey = getStableKey(prev as any);
        const found = (nodesWithKey as any[]).find(nn => getStableKey(nn) === prevKey);
        return (found as any) || prev;
      });
    } catch (err) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load graph'
      }));
    }
  }, []);

  // Rebuild global graph with new parameters
  const handleRebuild = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const data = await rebuildGlobalGraph(similarityThreshold, useEmbeddings);

      const nodesWithKey = (data.nodes || []).map((n: any) => ({
        ...n,
        stableKey: getStableKey(n)
      })) as any as GraphNode[];

      setState({
        nodes: nodesWithKey,
        edges: data.edges || [],
        loading: false,
        error: null,
        meta: data.meta
      });
    } catch (err) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to rebuild graph'
      }));
    }
  }, [similarityThreshold, useEmbeddings]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  /* ----------------------- Agent Polling (ui_state.json) ----------------------- */
  useEffect(() => {
    const checkUiState = async () => {
      try {
        const res = await fetch('/output/graph/ui_state.json', { cache: 'no-store' });
        if (!res.ok) return;

        const data = await res.json();

        if (data.timestamp > lastTimestampRef.current) {
          lastTimestampRef.current = data.timestamp;

          if (data.mode === 'paper' && data.focus_id) {
            navigate(`/paper/${encodeURIComponent(data.focus_id)}`);
          } else if (data.mode === 'global') {
            loadGraph();
          }
        }
      } catch {
        // ignore
      }
    };

    const interval = window.setInterval(checkUiState, 1000);
    return () => window.clearInterval(interval);
  }, [navigate, loadGraph]);

  /* ----------------------- global_graph.json Polling ----------------------- */
  useEffect(() => {
    const url = '/output/graph/global_graph.json';

    const checkGlobalGraph = async () => {
      try {
        const head = await fetch(url, { method: 'HEAD', cache: 'no-store' });
        const sig = head.headers.get('etag') ?? head.headers.get('last-modified');

        if (sig) {
          if (lastGraphSigRef.current && lastGraphSigRef.current !== sig) {
            await loadGraph();
          }
          lastGraphSigRef.current = sig;
          return;
        }

        const res = await fetch(`${url}?t=${Date.now()}`, { cache: 'no-store' });
        if (!res.ok) return;

        const text = await res.text();
        const first = text.length ? text.charCodeAt(0) : 0;
        const last = text.length ? text.charCodeAt(text.length - 1) : 0;
        const sig2 = `${text.length}:${first}:${last}`;

        if (lastGraphSigRef.current && lastGraphSigRef.current !== sig2) {
          await loadGraph();
        }
        lastGraphSigRef.current = sig2;
      } catch {
        // ignore
      }
    };

    checkGlobalGraph();
    const interval = window.setInterval(checkGlobalGraph, 1000);
    return () => window.clearInterval(interval);
  }, [loadGraph]);

  /* ------------------------ Click Handlers ------------------------ */

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
  }, []);

  const handleNodeDoubleClick = useCallback(
    (node: GraphNode) => {
      if (!node?.id) return;
      // ✅ 라우팅은 stableKey 말고 실제 id로
      navigate(`/paper/${encodeURIComponent(node.id)}`);
    },
    [navigate]
  );

  const handleViewPaperGraph = useCallback(
    (paperId: string) => navigate(`/paper/${encodeURIComponent(paperId)}`),
    [navigate]
  );

  // Handle search
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      return;
    }

    setIsSearching(true);
    setSearchError(null);
    setSearchResults([]);
    setHighlightedPaperIds(new Set());

    try {
      const result = await executeRankFilterPipeline({
        query: searchQuery.trim(),
        max_results: 50,
        top_k: 10,
      });

      if (result.success && result.ranked_papers) {
        setSearchResults(result.ranked_papers);
        
        // Update highlighted paper IDs (convert arxiv IDs to match graph node IDs)
        const highlightedSet = new Set<string>();
        result.ranked_papers.forEach(paper => {
          // Try to match with existing nodes
          const matchingNode = state.nodes.find(n => {
            const nodeId = String(n.id || '');
            const paperId = String(paper.paper_id || '');
            return nodeId === paperId || nodeId.includes(paperId) || paperId.includes(nodeId);
          });
          if (matchingNode) {
            highlightedSet.add(matchingNode.id);
          } else {
            // Also add original paper_id in case format differs
            highlightedSet.add(paper.paper_id);
          }
        });
        setHighlightedPaperIds(highlightedSet);
      } else {
        throw new Error(result.error || 'Search failed');
      }
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Failed to search papers');
    } finally {
      setIsSearching(false);
    }
  }, [searchQuery, state.nodes]);

  // Handle paper click from list
  const handlePaperClick = useCallback((paperId: string) => {
    // Find the corresponding node
    const node = state.nodes.find(n => {
      const nodeId = String(n.id || '');
      const pId = String(paperId || '');
      return nodeId === pId || nodeId.includes(pId) || pId.includes(nodeId);
    });
    if (node) {
      setSelectedNode(node);
      setFocusNodeId(node.id);
      // Clear focus after animation
      setTimeout(() => setFocusNodeId(undefined), 700);
    }
  }, [state.nodes]);

  /* ----------------------------- UI ----------------------------- */

  const selectedKey = selectedNode ? getStableKey(selectedNode as any) : '';

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#f5f5f5' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
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
            <h1 style={{ margin: 0, fontSize: '20px', color: '#1a202c' }}>Global Paper Graph</h1>
            <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#718096' }}>
              Overview of all papers
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '13px', color: '#4a5568' }}>Similarity:</label>
              <input
                type="range"
                min="0.3"
                max="0.95"
                step="0.05"
                value={similarityThreshold}
                onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
                style={{ width: '80px' }}
              />
              <span style={{ fontSize: '13px', width: '30px' }}>{similarityThreshold.toFixed(2)}</span>
            </div>

            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}>
              <input
                type="checkbox"
                checked={useEmbeddings}
                onChange={(e) => setUseEmbeddings(e.target.checked)}
              />
              Embeddings
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
                cursor: state.loading ? 'not-allowed' : 'pointer'
              }}
            >
              {state.loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </header>

        <div style={{ flex: 1, position: 'relative' }}>
          {/* Search Sidebar */}
          {!state.loading && !state.error && (
            <ArxivSearchSidebar
              searchQuery={searchQuery}
              onSearchQueryChange={setSearchQuery}
              onSearch={handleSearch}
              isSearching={isSearching}
            />
          )}

          {/* Search Results List */}
          {!state.loading && !state.error && isSearching ? (
            <div style={{
              position: 'absolute',
              bottom: '16px',
              right: '16px',
              zIndex: 5,
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              padding: '16px',
              borderRadius: '8px',
              width: '400px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              color: '#718096',
              fontSize: '14px',
              textAlign: 'center',
            }}>
              Searching arXiv and ranking papers...
            </div>
          ) : !state.loading && !state.error && searchResults.length > 0 ? (
            <ArxivRankedList
              papers={searchResults}
              onPaperClick={handlePaperClick}
            />
          ) : !state.loading && !state.error && searchQuery.trim() && !isSearching ? (
            <div style={{
              position: 'absolute',
              bottom: '16px',
              right: '16px',
              zIndex: 5,
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              padding: '16px',
              borderRadius: '8px',
              width: '400px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              color: '#718096',
              fontSize: '14px',
              textAlign: 'center',
            }}>
              No papers found. Try a different search query.
            </div>
          ) : null}

          {/* Error Message */}
          {searchError && (
            <div style={{
              position: 'absolute',
              bottom: '80px',
              right: '16px',
              zIndex: 6,
              backgroundColor: '#f56565',
              padding: '12px 16px',
              borderRadius: '8px',
              width: '400px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              color: '#fff',
              fontSize: '13px',
            }}>
              <div style={{ fontWeight: 600, marginBottom: '4px' }}>Error</div>
              <div>{searchError}</div>
            </div>
          )}

          {!state.loading && !state.error && (
            <GraphCanvas
              nodes={state.nodes}
              edges={state.edges as any}
              mode="global"
              selectedNodeId={selectedNode?.id}
              onNodeClick={handleNodeClick}
              onNodeDoubleClick={handleNodeDoubleClick}
              nodeColorMap={nodeColorMap} // ✅ GraphCanvas는 stableKey 우선 조회
              highlightedNodeIds={Array.from(highlightedPaperIds)}
              focusNodeId={focusNodeId}
            />
          )}
        </div>

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
            <span>Edges: {state.meta.total_edges}</span>
          </div>
        )}
      </div>

      <SidePanel
        selectedNode={selectedNode}
        mode="global"
        onAction={() => {
          if (selectedNode?.id) handleViewPaperGraph(selectedNode.id);
        }}
        onNavigate={(node) => handleViewPaperGraph(node.id)}
        onClose={() => setSelectedNode(null)}
        // ✅ color controls (stableKey 기반)
        nodeColorMap={nodeColorMap}
        nodeColor={
          selectedNode
            ? (nodeColorMap[selectedKey] ?? nodeColorMap[selectedNode.id])
            : undefined
        }
        onNodeColorChange={(keyOrId, color) => {
          // SidePanel이 id를 줄 수도 있어서 안정적으로 처리
          // - stableKey가 있는 경우 stableKey로 저장되도록 강제
          const k = selectedNode ? getStableKey(selectedNode as any) : keyOrId;
          handleNodeColorChange(k, color);
        }}
        onNodeColorReset={(keyOrId) => {
          const k = selectedNode ? getStableKey(selectedNode as any) : keyOrId;
          handleNodeColorReset(k);
        }}
      />
    </div>
  );
}
