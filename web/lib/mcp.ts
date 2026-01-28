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

/**
 * get_report -> 없으면 generate_report(오직 이때만) -> get_report 재시도
 * ReportViewer 토글에서 "자동 생성" 동작을 위한 헬퍼
 */
export async function getOrCreateReport(
  paperId: string
): Promise<{ content: string }> {
  const first = await getReport(paperId);
  if (first.found) {
    return { content: first.content || '' };
  }

  // ✅ report txt가 없는 경우에만 생성 메카니즘 작동
  await generateReport(paperId);

  const second = await getReport(paperId);
  if (second.found) {
    return { content: second.content || '' };
  }

  throw new Error(second.message || 'Report still not found after generation');
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

// ==================== ICLR Search & Rank API ====================

/**
 * Execute ICLR search and rank pipeline
 */
export async function executeICLRSearchAndRank(
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
    // Step 1: Search ICLR papers
    const searchResult = await executeTool('iclr_search', {
      query,
      max_results: 100,
      profile_path: profilePath,
    });

    if (!searchResult.success) {
      return {
        ranked_papers: [],
        success: false,
        error: searchResult.error || 'ICLR search failed',
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

    // Step 4: Evaluate metrics (with ICLR cluster map)
    const clusterKToUse = clusterK ?? 15;
    let iclrClusterMap: Record<string, number> = {};
    try {
      const clusterRes = await fetch(`/api/iclr/clusters?k=${clusterKToUse}`);
      if (clusterRes.ok) {
        const clusterData = await clusterRes.json();
        iclrClusterMap = clusterData.paper_id_to_cluster || {};
      }
    } catch (e) {
      console.warn('Failed to load ICLR cluster map:', e);
    }

    const metricsResult = await executeTool('evaluate_paper_metrics', {
      papers: passedPapers,
      semantic_scores: semanticScores,
      neurips_cluster_map: iclrClusterMap,
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
      neurips_cluster_map: iclrClusterMap,
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

// ==================== Research Agent Pipeline API ====================

export interface LocalPdfInfo {
  filename: string;
  path: string;
  size_bytes: number;
  size_mb: number;
}

export interface PipelineConfig {
  paper_ids: string[];
  goal?: string;
  analysis_mode?: 'quick' | 'standard' | 'deep';
  slack_webhook_full?: string;
  slack_webhook_summary?: string;
  source?: 'arxiv' | 'neurips' | 'iclr' | 'local';
}

export interface PipelineResult {
  success: boolean;
  papers_analyzed?: number;
  report_path?: string;
  executive_summary?: string;
  reasoning_log_count?: number;
  notifications?: {
    slack_full?: { success: boolean; error?: string };
    slack_summary?: { success: boolean; error?: string };
  };
  errors?: string[];
}

/**
 * List all local PDFs available for analysis
 */
export async function listLocalPdfs(): Promise<LocalPdfInfo[]> {
  const result = await executeTool('list_pdfs', {});
  
  if (!result.success) {
    throw new Error(result.error || 'Failed to list PDFs');
  }
  
  return result.result.files || [];
}

/**
 * Run the LLM-orchestrated research agent pipeline in background
 */
export async function runResearchAgent(config: PipelineConfig): Promise<{ success: boolean; job_id?: string; error?: string }> {
  const response = await fetch(`${MCP_BASE_URL}/agent/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arguments: {
      paper_ids: config.paper_ids,
      goal: config.goal || 'general understanding',
      analysis_mode: config.analysis_mode || 'quick',
      slack_webhook_full: config.slack_webhook_full || '',
      slack_webhook_summary: config.slack_webhook_summary || '',
      source: config.source || 'local',
    }})
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    return {
      success: false,
      error: error.detail || 'Failed to start pipeline',
    };
  }
  
  const data = await response.json();
  return {
    success: true,
    job_id: data.job_id,
  };
}

/**
 * Get the status of a running pipeline job
 */
export async function getAgentStatus(jobId: string): Promise<{
  success: boolean;
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  current_step: string;
  progress_percent: number;
  papers: string[];
  current_paper_idx: number;
  paper_results_count: number;
  reasoning_log_count: number;
  errors: string[];
  created_at: string;
  updated_at: string;
  result?: {
    report_path: string;
    report_exists: boolean;
  };
}> {
  const response = await fetch(`${MCP_BASE_URL}/agent/status/${jobId}`);
  
  if (!response.ok) {
    throw new Error(`Failed to get status: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * List all agent jobs
 */
export async function listAgentJobs(): Promise<Array<{
  job_id: string;
  status: string;
  goal: string;
  papers_count: number;
  progress_percent: number;
  created_at: string;
  updated_at: string;
}>> {
  const response = await fetch(`${MCP_BASE_URL}/agent/jobs`);
  
  if (!response.ok) {
    throw new Error(`Failed to list jobs: ${response.statusText}`);
  }
  
  const data = await response.json();
  return data.jobs || [];
}

// ==================== Slack Config (.env-backed) ====================

export async function getSlackConfig(): Promise<{
  slack_webhook_full: string;
  slack_webhook_summary: string;
  dotenv_path?: string;
}> {
  const response = await fetch(`${MCP_BASE_URL}/config/slack`);
  if (!response.ok) {
    throw new Error(`Failed to load Slack config: ${response.statusText}`);
  }
  const data = await response.json();
  return {
    slack_webhook_full: data.slack_webhook_full || '',
    slack_webhook_summary: data.slack_webhook_summary || '',
    dotenv_path: data.dotenv_path,
  };
}

export async function updateSlackConfig(payload: {
  slack_webhook_full: string;
  slack_webhook_summary: string;
}): Promise<{ success: boolean; dotenv_path?: string }> {
  const response = await fetch(`${MCP_BASE_URL}/config/slack`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || 'Failed to update Slack config');
  }
  const data = await response.json();
  return { success: !!data.success, dotenv_path: data.dotenv_path };
}

/**
 * Test notification configuration
 */
export async function testNotifications(
  slackWebhookFull?: string,
  slackWebhookSummary?: string
): Promise<{
  slack_full?: { success: boolean; error?: string };
  slack_summary?: { success: boolean; error?: string };
  environment_status: Record<string, string>;
}> {
  const result = await executeTool('test_notifications', {
    slack_webhook_full: slackWebhookFull || '',
    slack_webhook_summary: slackWebhookSummary || '',
  });
  
  if (!result.success) {
    throw new Error(result.error || 'Failed to test notifications');
  }
  
  return {
    slack_full: result.result.test_results?.slack_full,
    slack_summary: result.result.test_results?.slack_summary,
    environment_status: result.result.environment_status || {},
  };
}