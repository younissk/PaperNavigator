/**
 * TypeScript types for Paper Navigator API.
 */

export interface Paper {
  paper_id: string;
  title: string;
  abstract?: string | null;
  year?: number | null;
  citation_count: number;
  discovered_from?: string | null;
  edge_type: string;
  depth: number;
  judge_reason: string;
  judge_confidence: number;
  rank?: number;
  elo?: number;
  elo_change?: number;
  wins?: number;
  losses?: number;
  draws?: number;
}

export interface LeaderboardEntry {
  paper_id: string;
  title: string;
  elo: number;
  wins?: number;
  losses?: number;
}

export interface ResearchItem {
  title: string;
  summary: string;
  paper_ids: string[];
}

export interface OpenProblem {
  title: string;
  text: string;
  paper_ids: string[];
}

export interface PaperCard {
  id: string;
  title: string;
  claim: string;
  paradigm_tags: string[];
  year?: number;
  citation_count: number;
  elo_rating?: number;
}

export interface ReportData {
  query: string;
  generated_at: string;
  total_papers_used: number;
  introduction: string;
  current_research: ResearchItem[];
  open_problems: OpenProblem[];
  conclusion: string;
  paper_cards: PaperCard[];
}

export interface SnowballData {
  query: string;
  total_accepted: number;
  papers: Paper[];
}

export interface QueryMetadata {
  query: string;
  created_at?: string;
  last_updated?: string;
  snowball_file?: string;
  snowball_count?: number;
  elo_file?: string;
  report_file?: string;
  report_generated_at?: string;
  report_papers_used?: number;
  report_sections?: number;
}

export interface AllResultsResponse {
  report: ReportData | null;
  graph: Record<string, unknown> | null;
  timeline: Record<string, unknown> | null;
  clusters: Record<string, unknown> | null;
  snowball: SnowballData | null;
}

export interface PipelineRequest {
  query: string;
  num_results?: number;
  max_iterations?: number;
  max_accepted?: number;
  top_n?: number;
  k_factor?: number;
  pairing?: "swiss" | "random";
  early_stop?: boolean;
  elo_concurrency?: number;
  report_top_k?: number;
  notification_email?: string;
}

export interface PipelineEvent {
  ts?: string;
  type: string;
  level?: "info" | "warning" | "error";
  phase?: string;
  message: string;
}

export interface PipelineResponse {
  job_id: string;
  status:
    | "queued"
    | "searching"
    | "ranking"
    | "reporting"
    | "completed"
    | "failed";
  query: string;
  created_at?: string;
  updated_at?: string;
  phase?: string;
  phase_step?: number;
  phase_step_name?: string;
  phase_progress?: number;
  phase_total?: number;
  progress_message?: string;
  papers?: LeaderboardEntry[];
  report_data?: ReportData | null;
  error?: string;
  error_code?: string | null;
  events?: PipelineEvent[];
  alerts?: PipelineEvent[];
}

export interface HealthResponse {
  status: string;
  version: string;
  storage?: string;
  database?: string;
}

export interface ReadinessResponse {
  status: "ok" | "degraded";
  ready: boolean;
  version: string;
  checks: {
    storage: string;
    database: string;
    service_bus: string;
    openai?: string;
  };
  signals?: {
    openai?: {
      ok: boolean;
      latency_ms?: number | null;
      error?: string | null;
    };
  };
}

export interface MonitoringDailyCount {
  date: string; // YYYY-MM-DD
  count: number;
}

export interface MonitoringReportsResponse {
  window_days: number;
  reports_generated: number;
  daily: MonitoringDailyCount[];
  sample_limit: number;
  sampled_jobs: number;
}

export interface MonitoringPipelinesResponse {
  window_days: number;
  sample_limit: number;
  sampled_jobs: number;
  duration_sec: {
    avg: number | null;
    p50: number | null;
    p95: number | null;
    count: number;
  };
  per_phase_avg_duration_sec: Record<string, number | null>;
}

export interface MonitoringCostsResponse {
  window_days: number;
  sample_limit: number;
  sampled_jobs: number;
  cost_proxies: {
    bytes_uploaded_total: number;
    avg_bytes_uploaded_per_pipeline: number | null;
    artifact_count_total: number;
    avg_artifacts_per_pipeline: number | null;
    avg_duration_sec: number | null;
    duration_p95_sec: number | null;
    coverage: {
      bytes_samples: number;
      artifact_samples: number;
      duration_samples: number;
    };
  };
  openai: {
    estimated_cost_usd_total: number | null;
    avg_estimated_cost_usd_per_pipeline: number | null;
    total_tokens_total: number;
    avg_total_tokens_per_pipeline: number | null;
    pipelines_with_usage: number;
    pipelines_with_priced_cost: number;
    by_model: Record<
      string,
      {
        requests: number;
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
        estimated_cost_usd: number | null;
      }
    >;
  };
  notes: string[];
}

export interface RecentReportSummary {
  query_slug: string;
  query: string;
  generated_at: string;
  total_papers_used: number;
  research_themes: string[];
}

export interface RecentReportsResponse {
  reports: RecentReportSummary[];
}

export interface QueryWithMetadataResponse {
  query: string;
  slug: string;
  metadata: QueryMetadata | null;
}

export interface AllQueriesMetadataResponse {
  queries: QueryWithMetadataResponse[];
}
