/**
 * MCP REST Client
 *
 * Handles all communication with the MCP backend server.
 * Web UI should NEVER perform side-effects directly.
 */

// Use /api prefix for Vite proxy to forward requests to MCP server
const MCP_BASE_URL = '/api';

// Types
export interface GraphNode {
  id: string;
  title: string;
  paper_id?: string;
  label?: string;
  stableKey?: string;
  type?: string;
  x?: number;
  y?: number;
  authors?: string[];
  abstract?: string;
  year?: number;
  cluster?: number;
  depth?: number;
  is_center?: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight?: number;
  type?: 'references' | 'similarity' | string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  meta?: Record<string, any>;
}

export interface GraphDiff {
  new_nodes: GraphNode[];
  new_edges: GraphEdge[];
  is_incremental: boolean;
}

// API Functions

/**
 * Health check
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${MCP_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get list of available tools
 */
export async function getTools(): Promise<any[]> {
  const response = await fetch(`${MCP_BASE_URL}/tools`);
  if (!response.ok) throw new Error('Failed to fetch tools');
  const data = await response.json();
  return data.tools;
}

/**
 * Execute a tool
 */
export async function executeTool(
  toolName: string,
  args: Record<string, any> = {}
): Promise<any> {
  const response = await fetch(`${MCP_BASE_URL}/tools/${toolName}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arguments: args })
  });

  if (!response.ok) {
    throw new Error(`Tool execution failed: ${response.statusText}`);
  }

  return response.json();
}
// ==================== Report API (tools/report.py) ====================
export async function getReport(paperId: string): Promise<{ found: boolean; content?: string; message?: string }> {
  const result = await executeTool('get_report', { paper_id: paperId });
  if (!result.success) throw new Error(result.error || 'get_report failed');
  return result.result as any;
}

export async function generateReport(paperId: string): Promise<any> {
  const result = await executeTool('generate_report', { paper_id: paperId });
  if (!result.success) throw new Error(result.error || 'generate_report failed');
  return result.result;
}
// ==================== Graph B (Global) API ====================

/**
 * Get global graph data (Graph B) - loads from global_graph.json
 */
export async function getGlobalGraph(): Promise<GraphData> {
  const result = await executeTool('get_global_graph', {});

  if (!result.success) {
    throw new Error(result.error || 'Failed to get global graph');
  }

  return result.result;
}

/**
 * Rebuild global graph with new parameters
 */
export async function rebuildGlobalGraph(
  similarityThreshold: number = 0.7,
  useEmbeddings: boolean = true
): Promise<GraphData> {
  const result = await executeTool('build_global_graph', {
    similarity_threshold: similarityThreshold,
    use_embeddings: useEmbeddings
  });

  if (!result.success) {
    throw new Error(result.error || 'Failed to rebuild global graph');
  }

  return result.result;
}

// ==================== Graph A (Paper) API ====================

/**
 * Get paper reference subgraph (Graph A) - Initial load
 */
export async function getPaperGraph(paperId: string, depth: number = 1): Promise<GraphData> {
  const result = await executeTool('build_reference_subgraph', {
    paper_id: paperId,
    depth: depth,
    existing_nodes: []
  });

  if (!result.success) {
    throw new Error(result.error || 'Failed to build paper graph');
  }

  return {
    nodes: result.result.new_nodes,
    edges: result.result.new_edges,
    meta: { center: result.result.center }
  };
}

/**
 * Expand paper graph (Graph A) - Incremental update
 */
export async function expandPaperGraph(
  paperId: string,
  existingNodeIds: string[]
): Promise<GraphDiff> {
  const result = await executeTool('build_reference_subgraph', {
    paper_id: paperId,
    depth: 1,
    existing_nodes: existingNodeIds
  });

  if (!result.success) {
    throw new Error(result.error || 'Failed to expand paper graph');
  }

  return {
    new_nodes: result.result.new_nodes,
    new_edges: result.result.new_edges,
    is_incremental: result.result.is_incremental
  };
}

/**
 * Check if paper PDF exists
 */
export async function hasPdf(paperId: string): Promise<{ exists: boolean }> {
  const result = await executeTool('has_pdf', { paper_id: paperId });
  return { exists: result.success && result.result?.exists };
}

/**
 * Fetch paper if missing
 */
export async function fetchPaperIfMissing(paperId: string): Promise<{
  action: 'already_exists' | 'downloaded';
  path: string;
}> {
  const result = await executeTool('fetch_paper_if_missing', {
    paper_id: paperId
  });

  if (!result.success) {
    throw new Error(result.error || 'Failed to fetch paper');
  }

  return result.result;
}

/**
 * Get paper references
 */
export async function getReferences(paperId: string): Promise<{
  references: Array<{ arxiv_id: string }>;
  from_cache: boolean;
}> {
  const result = await executeTool('get_references', { paper_id: paperId });

  if (!result.success) {
    throw new Error(result.error || 'Failed to get references');
  }

  return result.result;
}

// ==================== PDF API ====================

/**
 * List all PDFs
 */
export async function listPdfs(): Promise<Array<{
  filename: string;
  size_mb: number;
}>> {
  const result = await executeTool('list_pdfs', {});

  if (!result.success) {
    throw new Error(result.error || 'Failed to list PDFs');
  }

  return result.result.files;
}

/**
 * Process all PDFs
 */
export async function processAllPdfs(): Promise<any> {
  const result = await executeTool('process_all_pdfs', {});

  if (!result.success) {
    throw new Error(result.error || 'Failed to process PDFs');
  }

  return result.result;
}

// ==================== Rank Filter API ====================

export interface PaperInput {
  paper_id: string;
  title: string;
  abstract: string;
  authors: string[];
  published?: string;
  categories?: string[];
  pdf_url?: string;
  github_url?: string | null;
}

export interface RankFilterPipelineParams {
  query: string;
  max_results?: number;
  purpose?: string;
  ranking_mode?: string;
  top_k?: number;
  include_contrastive?: boolean;
  contrastive_type?: string;
}

export interface UserProfile {
  interests: {
    primary: string[];
    secondary: string[];
    exploratory: string[];
  };
  keywords: {
    must_include: string[];
    exclude: {
      hard: string[];
      soft: string[];
    };
  };
  preferred_authors: string[];
  preferred_institutions: string[];
  constraints: {
    min_year: number;
    require_code: boolean;
    exclude_local_papers: boolean;
  };
  purpose?: string;
  ranking_mode?: string;
  top_k?: number;
  include_contrastive?: boolean;
  contrastive_type?: string;
}

/**
 * Search arXiv and convert results to PaperInput format
 */
export async function searchArxivForRanking(
  query: string,
  maxResults: number = 50
): Promise<{
  query: string;
  total_results: number;
  papers: PaperInput[];
}> {
  const response = await fetch(`${MCP_BASE_URL}/arxiv/search-for-ranking`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, max_results: maxResults })
  });

  if (!response.ok) {
    throw new Error(`Failed to search arXiv: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Execute the full rank and filter pipeline
 */
export async function executeRankFilterPipeline(
  params: RankFilterPipelineParams
): Promise<any> {
  const response = await fetch(`${MCP_BASE_URL}/rank-filter/execute-pipeline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || 'Failed to execute pipeline');
  }

  return response.json();
}

/**
 * Get user profile
 */
export async function getUserProfile(
  profilePath: string = 'users/profile.json'
): Promise<UserProfile> {
  const response = await fetch(`${MCP_BASE_URL}/rank-filter/profile?profile_path=${encodeURIComponent(profilePath)}`);

  if (!response.ok) {
    throw new Error(`Failed to get profile: ${response.statusText}`);
  }

  const data = await response.json();
  return data.profile;
}

/**
 * Update user profile
 */
export async function updateUserProfile(
  profile: Partial<UserProfile>,
  profilePath: string = 'users/profile.json'
): Promise<any> {
  const response = await fetch(`${MCP_BASE_URL}/rank-filter/profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      profile_path: profilePath,
      ...profile
    })
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || 'Failed to update profile');
  }

  return response.json();
}

// ==================== NeurIPS Search & Rank API ====================

/**
 * Execute NeurIPS search and rank pipeline
 */
export async function executeNeurIPSSearchAndRank(
  query: string,
  profilePath: string = 'users/profile.json',
  topK: number = 10,
  clusterK?: number
): Promise<{
  ranked_papers: Array<{
    rank: number;
    paper_id: string;
    title: string;
    authors: string[];
    published?: string;
    score: {
      final: number;
      breakdown: any;
      soft_penalty: number;
      penalty_keywords: string[];
      evaluation_method: string;
    };
    tags: string[];
    local_status: {
      already_downloaded: boolean;
      local_path: string | null;
    };
    original_data: any;
    reasoning?: string;
  }>;
  success: boolean;
  error?: string;
}> {
  try {
    // Step 1: Search NeurIPS papers
    const searchResult = await executeTool('neurips_search', {
      query,
      max_results: 100,
      profile_path: profilePath,
    });

    if (!searchResult.success) {
      return {
        ranked_papers: [],
        success: false,
        error: searchResult.error || 'NeurIPS search failed',
      };
    }

    if (!searchResult.result?.papers || searchResult.result.papers.length === 0) {
      return {
        ranked_papers: [],
        success: true,
      };
    }

    // Step 2: Apply hard filters
    const filterResult = await executeTool('apply_hard_filters', {
      papers: searchResult.result.papers,
      profile_path: profilePath,
    });

    if (!filterResult.success) {
      throw new Error(filterResult.error || 'Hard filters failed');
    }

    const passedPapers = filterResult.result?.passed_papers || [];

    if (passedPapers.length === 0) {
      return {
        ranked_papers: [],
        success: true,
      };
    }

    // Step 3: Calculate semantic scores
    const semanticResult = await executeTool('calculate_semantic_scores', {
      papers: passedPapers,
      query,
      profile_path: profilePath,
    });

    if (!semanticResult.success) {
      throw new Error(semanticResult.error || 'Semantic scoring failed');
    }

    const semanticScores = semanticResult.result?.scores || {};

    // Step 4: Evaluate metrics (with NeurIPS cluster map)
    // Load cluster map with cluster_k parameter (default: 15)
    const clusterKToUse = clusterK ?? 15;
    let neuripsClusterMap: Record<string, number> = {};
    try {
      const clusterRes = await fetch(`/api/neurips/clusters?k=${clusterKToUse}`);
      if (clusterRes.ok) {
        const clusterData = await clusterRes.json();
        neuripsClusterMap = clusterData.paper_id_to_cluster || {};
      }
    } catch (e) {
      console.warn('Failed to load cluster map:', e);
    }

    const metricsResult = await executeTool('evaluate_paper_metrics', {
      papers: passedPapers,
      semantic_scores: semanticScores,
      neurips_cluster_map: neuripsClusterMap,
      profile_path: profilePath,
    });

    if (!metricsResult.success) {
      throw new Error(metricsResult.error || 'Metrics evaluation failed');
    }

    const metricsScores = metricsResult.result?.scores || {};

    // Step 5: Rank and select top K
    const rankResult = await executeTool('rank_and_select_top_k', {
      papers: passedPapers,
      semantic_scores: semanticScores,
      metrics_scores: metricsScores,
      neurips_cluster_map: neuripsClusterMap,
      top_k: topK,
      profile_path: profilePath,
      cluster_k: clusterKToUse,
    });

    if (!rankResult.success) {
      throw new Error(rankResult.error || 'Ranking failed');
    }

    return {
      ranked_papers: rankResult.result?.ranked_papers || [],
      success: true,
    };
  } catch (error) {
    return {
      ranked_papers: [],
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}