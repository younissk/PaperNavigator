import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import type { PipelineResponse, PipelineEvent, LeaderboardEntry } from "@/lib/types";
import { readinessCheck } from "@/lib/api";
import { GameLauncher } from "./GameLauncher";

// Brutalist coral shadow styles
const brutalShadow = { boxShadow: "3px 3px 0 #F3787A" };
const brutalShadowSmall = { boxShadow: "1px 1px 0 #F3787A" };

interface ProgressIndicatorProps {
  status: PipelineResponse;
  queryTitle?: string;
}

const PHASE_NAMES: Record<string, string> = {
  search: "searching for papers",
  ranking: "ranking papers",
  report: "generating report",
};

const PIPELINE_PHASES = ["search", "ranking", "report"] as const;
const STALE_WARNING_MINUTES = 20;
const QUEUE_WARNING_MINUTES = 2;

/**
 * Format elapsed time as MM:SS or HH:MM:SS
 */
function formatElapsedTime(seconds: number): string {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Live elapsed timer component - brutalist box style
 */
function ElapsedTimer({ createdAt }: { createdAt?: string }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!createdAt) return;

    const startTime = new Date(createdAt).getTime();

    const updateElapsed = () => {
      const now = Date.now();
      setElapsed(Math.floor((now - startTime) / 1000));
    };

    updateElapsed();
    const interval = setInterval(updateElapsed, 1000);

    return () => clearInterval(interval);
  }, [createdAt]);

  if (!createdAt) return null;

  return (
    <div
      className="inline-flex items-center px-4 py-2 border-2 border-black bg-white text-xl sm:text-2xl font-bold"
      style={brutalShadow}
    >
      <span>{formatElapsedTime(elapsed)}</span>
    </div>
  );
}

/**
 * Horizontal pipeline timeline showing phase progress - brutalist style
 */
function PipelineTimeline({ currentPhase }: { currentPhase: string }) {
  const currentIndex = PIPELINE_PHASES.indexOf(
    currentPhase as (typeof PIPELINE_PHASES)[number]
  );

  return (
    <div className="flex items-center justify-center gap-0 my-6">
      {PIPELINE_PHASES.map((phase, idx) => {
        const isCompleted = idx < currentIndex;
        const isCurrent = idx === currentIndex;
        const isPending = idx > currentIndex;

        return (
          <div key={phase} className="flex items-center">
            {/* Step square */}
            <div className="flex flex-col items-center">
              <div
                className={`
                  w-10 h-10 flex items-center justify-center text-sm font-bold
                  transition-all duration-300 border-2 border-black
                  ${isCompleted ? "bg-black text-white" : ""}
                  ${isCurrent ? "bg-black text-white" : ""}
                  ${isPending ? "bg-white text-gray-400" : ""}
                `}
                style={isCurrent ? brutalShadowSmall : undefined}
              >
                {isCompleted ? (
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  idx + 1
                )}
              </div>
              <span
                className={`
                  text-xs mt-2 font-medium lowercase
                  ${isCompleted || isCurrent ? "text-black" : "text-gray-400"}
                `}
              >
                {phase === "search" ? "search" : phase === "ranking" ? "rank" : "report"}
              </span>
            </div>

            {/* Connector line (except after last) */}
            {idx < PIPELINE_PHASES.length - 1 && (
              <div
                className={`
                  w-8 sm:w-12 h-0.5 mx-2
                  ${idx < currentIndex ? "bg-black" : "bg-gray-300"}
                `}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/**
 * Rank badge for top positions - brutalist style
 */
function RankBadge({ rank }: { rank: number }) {
  if (rank === 1) {
    return (
      <span
        className="inline-flex items-center justify-center w-6 h-6 bg-black text-white font-bold text-xs border-2 border-black"
        style={brutalShadowSmall}
      >
        1
      </span>
    );
  }
  if (rank === 2) {
    return (
      <span className="inline-flex items-center justify-center w-6 h-6 bg-white text-black font-bold text-xs border-2 border-black">
        2
      </span>
    );
  }
  if (rank === 3) {
    return (
      <span className="inline-flex items-center justify-center w-6 h-6 bg-white text-black font-bold text-xs border-2 border-black">
        3
      </span>
    );
  }
  return <span className="text-gray-500 font-medium w-6 text-center">{rank}</span>;
}

/**
 * Leaderboard table showing top papers - brutalist style
 */
function Leaderboard({ papers }: { papers: LeaderboardEntry[] }) {
  if (papers.length === 0) return null;

  return (
    <div className="mt-6 text-left">
      <h3 className="text-xs font-bold text-black uppercase tracking-wide mb-2 lowercase">
        live leaderboard
      </h3>
      <div className="overflow-hidden border-2 border-black" style={brutalShadow}>
        <table className="w-full text-sm">
          <thead className="bg-black text-white text-xs uppercase tracking-wider">
            <tr>
              <th className="px-3 py-2 text-center w-10">#</th>
              <th className="px-3 py-2 text-left">paper</th>
              <th className="px-2 py-2 text-center w-10">w</th>
              <th className="px-2 py-2 text-center w-10">l</th>
              <th className="px-3 py-2 text-right w-16">elo</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-black bg-white">
            {papers.map((paper, idx) => {
              const rank = idx + 1;

              return (
                <tr
                  key={paper.paper_id}
                  className="transition-colors hover:bg-gray-50"
                >
                  <td className="px-3 py-2 text-center">
                    <RankBadge rank={rank} />
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className="block truncate max-w-[280px] sm:max-w-[320px] text-black cursor-default"
                      title={paper.title}
                    >
                      {paper.title}
                    </span>
                  </td>
                  <td className="px-2 py-2 text-center">
                    <span className="text-black font-bold">
                      {paper.wins ?? "-"}
                    </span>
                  </td>
                  <td className="px-2 py-2 text-center">
                    <span className="text-gray-500">
                      {paper.losses ?? "-"}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <span className="font-bold text-black">
                      {Number.isFinite(paper.elo) ? Math.round(paper.elo) : "-"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatEventTime(ts?: string): string {
  if (!ts) return "";
  const parsed = Date.parse(ts);
  if (Number.isNaN(parsed)) return "";
  return new Date(parsed).toLocaleTimeString();
}

function getStaleInfo(updatedAt?: string) {
  if (!updatedAt) return { isStale: false, minutesSince: null, label: "" };
  const parsed = Date.parse(updatedAt);
  if (Number.isNaN(parsed)) return { isStale: false, minutesSince: null, label: "" };
  const minutesSince = Math.floor((Date.now() - parsed) / 60000);
  const isStale = minutesSince >= STALE_WARNING_MINUTES;
  const label = new Date(parsed).toLocaleString();
  return { isStale, minutesSince, label };
}

/**
 * Progress indicator for pipeline jobs.
 */
export function ProgressIndicator({
  status,
  queryTitle,
}: ProgressIndicatorProps) {
  const [logsExpanded, setLogsExpanded] = useState(false);
  const [showGame, setShowGame] = useState(false);
  const phase = status.phase || "";
  const normalizedPhase = PIPELINE_PHASES.includes(phase as (typeof PIPELINE_PHASES)[number])
    ? phase
    : "search";
  const title = PHASE_NAMES[phase] || status.status || "processing...";
  const events = status.events ?? [];
  const alerts = status.alerts ?? [];
  const recentEvents: PipelineEvent[] = events.slice(-10);
  const { isStale, minutesSince, label: updatedLabel } = getStaleInfo(status.updated_at);
  const queuedHint = (status.progress_message || "").toLowerCase().startsWith("queued");
  const isQueuedUi = status.status === "queued" || queuedHint;
  const isQueueSlow = minutesSince !== null && minutesSince >= QUEUE_WARNING_MINUTES && isQueuedUi;

  const showDiagnostics = isStale || isQueueSlow;
  const readiness = useQuery({
    queryKey: ["ready", "pipeline_diagnostics"],
    queryFn: readinessCheck,
    enabled: showDiagnostics,
    refetchInterval: showDiagnostics ? 15000 : false,
    retry: false,
    staleTime: 0,
  });

  // Leaderboard data during ranking phase
  const leaderboard =
    phase === "ranking" && status.papers
      ? [...status.papers].sort((a, b) => (b.elo ?? 0) - (a.elo ?? 0)).slice(0, 5)
      : [];

  // Calculate progress percentage
  let percent = 0;
  if (status.phase_total && status.phase_progress !== undefined) {
    percent = (status.phase_progress / status.phase_total) * 100;
  }

  // Phase details
  let details = "";
  if (status.status === "queued" || queuedHint) {
    const queuedPhase = normalizedPhase || "search";
    details = `queued for ${queuedPhase}`;
  } else if (phase === "search") {
    details = `step: ${status.phase_step_name || ""}`;
  } else if (phase === "ranking") {
    const progress = status.phase_progress ?? 0;
    const total = status.phase_total ?? "?";
    details = `matches: ${progress} / ${total}`;
  } else if (phase === "report") {
    details = `step: ${status.phase_step_name || ""}`;
  }

  return (
    <div className="min-h-[calc(100vh-10rem)] flex flex-col justify-center px-4 py-8 sm:py-12">
      <div className="max-w-xl w-full mx-auto text-center">
        {/* Header with title and timer - side by side on desktop, stacked on mobile */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mb-6 sm:mb-8">
          <h2 className="text-xl sm:text-2xl md:text-3xl font-bold text-black lowercase text-shadow-brutal">
            {title}
          </h2>
          <ElapsedTimer createdAt={status.created_at} />
        </div>

        {queryTitle && (
          <p className="text-gray-600 text-sm sm:text-base mb-6 sm:mb-8 lowercase">
            generating report for "<span className="font-medium text-black">{queryTitle}</span>"
          </p>
        )}

        {/* Pipeline timeline */}
        <PipelineTimeline currentPhase={normalizedPhase} />

        {/* Progress bar */}
        <div className="h-3 bg-white border-2 border-black overflow-hidden mb-4">
          <div
            className="h-full bg-black transition-[width] duration-300 ease-out"
            style={{ width: `${Math.min(percent, 100)}%`, ...brutalShadow }}
          />
        </div>

        <p className="text-xs sm:text-sm text-gray-600 lowercase">
          {status.progress_message || "starting..."}
        </p>

        {details && (
          <p className="text-xs sm:text-sm text-black mt-1 font-medium lowercase">{details}</p>
        )}

        {updatedLabel && (
          <p className={`text-xs mt-2 lowercase ${isStale ? "text-red-600" : "text-gray-400"}`}>
            last update: {updatedLabel}
            {minutesSince !== null ? ` (${minutesSince}m ago)` : ""}
          </p>
        )}

        {isStale && (
          <div className="mt-4 text-left p-4 border-2 border-black bg-white">
            <strong className="text-black lowercase">pipeline appears stalled.</strong>{" "}
            No progress updates in {minutesSince} minutes. You can keep waiting, or
            start a new run if it doesn’t recover.
          </div>
        )}

        {showDiagnostics && (readiness.data || readiness.error) && (
          <div className="mt-4 text-left p-4 border-2 border-black bg-white">
            <div className="text-sm font-bold lowercase text-black">system diagnostics</div>
            {readiness.error ? (
              <div className="text-xs mt-2 text-gray-700">
                failed to load readiness:{" "}
                {readiness.error instanceof Error ? readiness.error.message : "unknown error"}
              </div>
            ) : (
              <>
                <div className="mt-2 grid gap-2">
                  {Object.entries(readiness.data?.checks ?? {}).map(([name, st]) => (
                    <div key={name} className="flex justify-between text-xs border-b border-black pb-2 last:border-b-0 last:pb-0">
                      <span className="text-gray-600 lowercase">{name}</span>
                      <span className="flex items-center gap-2 font-medium text-black">
                        <span
                          className={`w-4 h-4 flex items-center justify-center text-xs ${
                            st === "connected" ? "bg-black text-white" : "border border-black"
                          }`}
                        >
                          {st === "connected" ? "✓" : "×"}
                        </span>
                        <span className="lowercase">{st}</span>
                      </span>
                    </div>
                  ))}
                </div>

                {readiness.data?.signals?.openai && (
                  <p className="text-xs text-gray-600 mt-3 lowercase">
                    openai:{" "}
                    <span className="text-black font-medium">
                      {readiness.data.signals.openai.ok ? "ok" : "unavailable"}
                    </span>
                    {Number.isFinite(readiness.data.signals.openai.latency_ms) ? (
                      <>
                        {" · "}latency{" "}
                        <span className="text-black font-medium">
                          {Math.round(readiness.data.signals.openai.latency_ms!)}ms
                        </span>
                      </>
                    ) : null}
                    {readiness.data.signals.openai.error ? (
                      <>
                        {" · "}error{" "}
                        <span className="text-black font-medium">
                          {readiness.data.signals.openai.error}
                        </span>
                      </>
                    ) : null}
                  </p>
                )}

                {isQueueSlow && readiness.data?.checks?.service_bus === "connected" && (
                  <p className="text-xs text-gray-700 mt-3 lowercase">
                    queue is reachable, but this job is still queued. this usually means the worker trigger isn't consuming messages (trigger sync/cold start/restarts).
                  </p>
                )}
              </>
            )}
          </div>
        )}

        {status.error && status.status !== "failed" && (
          <div className="mt-4 text-left p-3 border-2 border-black bg-white">
            <div className="text-sm font-bold lowercase text-red-600">last error</div>
            <div className="text-xs mt-1 text-gray-700">{status.error}</div>
          </div>
        )}

        {alerts.length > 0 && (
          <div className="mt-4 text-left space-y-2">
            {alerts.slice(-3).map((alert, idx) => (
              <div
                key={`${alert.ts ?? "alert"}-${idx}`}
                className="p-3 border-2 border-black"
                style={alert.level === "error" ? { borderColor: "#e53e3e" } : undefined}
              >
                <div className="text-sm font-bold lowercase">
                  {alert.level === "error" ? "error" : "warning"}
                  {alert.phase ? ` · ${alert.phase}` : ""}
                </div>
                <div className="text-xs mt-1 text-gray-600">{alert.message}</div>
              </div>
            ))}
          </div>
        )}

        {/* Leaderboard */}
        <Leaderboard papers={leaderboard} />

        {/* Logs Accordion */}
        {recentEvents.length > 0 && (
          <div className="mt-6">
            <button
              type="button"
              onClick={() => setLogsExpanded(!logsExpanded)}
              className="flex items-center justify-between w-full px-3 py-2 text-xs font-bold text-black border-2 border-black bg-white hover:bg-gray-50 transition-colors lowercase"
            >
              <span>recent logs ({recentEvents.length})</span>
              <svg
                className={`w-4 h-4 transition-transform ${logsExpanded ? "rotate-180" : ""}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            {logsExpanded && (
              <div className="border-2 border-t-0 border-black p-3 text-left">
                <div className="space-y-1 text-xs text-gray-600 font-mono">
                  {recentEvents.map((ev, idx) => (
                    <div key={`${ev.ts ?? "event"}-${idx}`} className="flex gap-2 flex-wrap sm:flex-nowrap">
                      <span className="text-gray-400 w-14 shrink-0">
                        {formatEventTime(ev.ts)}
                      </span>
                      <span
                        className={`font-bold ${
                          ev.level === "error"
                            ? "text-red-600"
                            : ev.level === "warning"
                              ? "text-black"
                              : "text-gray-500"
                        }`}
                      >
                        {ev.level ?? "info"}
                      </span>
                      <span className="text-gray-700 break-all">{ev.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tired of waiting? Play a game */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <span className="text-sm text-gray-500 lowercase">tired of waiting?</span>
            <button
              type="button"
              onClick={() => setShowGame(true)}
              className="px-4 py-2 text-sm font-bold lowercase border-2 border-black bg-white hover:bg-black hover:text-white transition-colors"
              style={brutalShadow}
            >
              play a game
            </button>
          </div>
        </div>

      </div>

      {/* Game Launcher Overlay */}
      {showGame && <GameLauncher onClose={() => setShowGame(false)} />}
    </div>
  );
}
