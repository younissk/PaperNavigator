import { SEO, DocLoader } from "@/components";
import {
  useHealthCheck,
  useMonitoringCosts,
  useMonitoringPipelines,
  useMonitoringReports,
  useReadinessCheck,
} from "@/hooks";

// Brutalist shadow styles
const brutalShadow = { boxShadow: "3px 3px 0 #F3787A" };
const brutalShadowSmall = { boxShadow: "1px 1px 0 #F3787A" };

function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  if (value < 60) {
    return `${value.toFixed(1)}s`;
  }
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}m ${seconds.toFixed(0)}s`;
}

function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  const units = ["B", "KB", "MB", "GB", "TB"] as const;
  let n = value;
  let u = 0;
  while (n >= 1024 && u < units.length - 1) {
    n /= 1024;
    u += 1;
  }
  const decimals = u === 0 ? 0 : 1;
  // Use non-breaking space to prevent unit from wrapping to next line
  return `${n.toFixed(decimals)}\u00A0${units[u]}`;
}

function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(decimals);
}

type MonitorState = "ok" | "degraded" | "down";

function EcgPulse({ state }: { state: MonitorState }) {
  // Brutalist: use black for ok/degraded, coral for pulse
  const strokeColor = state === "down" ? "#000" : "#000";

  // ECG waveform: flat → small P wave → flat → sharp QRS → flat → small T wave → flat
  const ecgPath =
    "M0 20 L15 20 L18 18 L21 20 L30 20 L33 20 L36 22 L38 5 L42 35 L45 18 L48 20 L60 20 L63 17 L69 20 L80 20";

  if (state === "down") {
    return (
      <svg
        viewBox="0 0 80 40"
        className="w-24 h-8"
        role="img"
        aria-label="ECG offline"
      >
        <line
          x1="0"
          y1="20"
          x2="80"
          y2="20"
          stroke={strokeColor}
          strokeWidth="2"
          opacity="0.3"
        />
      </svg>
    );
  }

  return (
    <svg
      viewBox="0 0 80 40"
      className="w-24 h-8"
      role="img"
      aria-label="ECG pulse"
    >
      <path
        d={ecgPath}
        fill="none"
        stroke={strokeColor}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="ecg-trace"
      />
    </svg>
  );
}

function StatusBadge({ state }: { state: MonitorState }) {
  const label = state === "ok" ? "ready" : state === "degraded" ? "degraded" : "offline";
  const icon = state === "ok" ? "✓" : state === "degraded" ? "!" : "×";
  
  return (
    <span
      className="inline-flex items-center gap-2 px-3 py-1 text-sm border-2 border-black bg-white text-black lowercase"
      style={brutalShadowSmall}
    >
      <span
        className={`w-5 h-5 flex items-center justify-center text-xs font-bold ${
          state === "ok"
            ? "bg-black text-white"
            : state === "degraded"
              ? "bg-white text-black border border-black"
              : "bg-white text-black border border-black"
        }`}
      >
        {icon}
      </span>
      {label}
    </span>
  );
}

/**
 * Monitoring page - brutalist design.
 */
export default function MonitoringPage() {
  const windowDays = 30;

  const health = useHealthCheck();
  const readiness = useReadinessCheck();
  const reports = useMonitoringReports(windowDays);
  const pipelines = useMonitoringPipelines(windowDays);
  const costs = useMonitoringCosts(windowDays);

  const monitorState: MonitorState = (() => {
    if (health.isLoading) {
      return "degraded";
    }
    if (health.error || !health.data) {
      return "down";
    }
    if (readiness.data?.ready) {
      return "ok";
    }
    return "degraded";
  })();

  const daily = reports.data?.daily ?? [];
  const maxDaily = Math.max(1, ...daily.map((d) => d.count));

  return (
    <>
      <SEO
        title="Monitoring"
        description="API and system monitoring dashboard"
      />
      <div className="container container-lg py-12 px-4">
        <div className="stack stack-xl">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-4">
              <h1 className="mb-0 text-3xl md:text-4xl font-bold text-black text-shadow-brutal lowercase">
                monitoring
              </h1>
              <EcgPulse state={monitorState} />
            </div>
            <StatusBadge state={monitorState} />
          </div>

          <div className="grid gap-6 md:grid-cols-3">
            {/* Reports Card */}
            <div className="bg-white border-2 border-black p-6" style={brutalShadow}>
              <div className="stack stack-sm">
                <h2 className="text-base font-bold text-black lowercase">
                  reports (last {windowDays} days)
                </h2>
                {reports.isLoading ? (
                  <div className="flex items-center gap-2 text-gray-600 lowercase">
                    <DocLoader size="sm" />
                    <span>loading…</span>
                  </div>
                ) : reports.error ? (
                  <div className="text-sm text-black">
                    {reports.error instanceof Error
                      ? reports.error.message
                      : "failed to load report metrics"}
                  </div>
                ) : (
                  <>
                    <div className="text-4xl font-bold text-black text-shadow-brutal">
                      {reports.data?.reports_generated ?? 0}
                    </div>
                    {daily.length > 0 && (
                      <div className="mt-3">
                        <div className="flex items-end gap-0.5 h-10">
                          {daily.slice(-windowDays).map((d) => (
                            <div
                              key={d.date}
                              className="w-1 bg-black"
                              style={{
                                height: `${Math.max(
                                  8,
                                  Math.round((d.count / maxDaily) * 100),
                                )}%`,
                              }}
                              title={`${d.date}: ${d.count}`}
                            />
                          ))}
                        </div>
                        <p className="text-xs text-gray-600 mt-2 lowercase">
                          daily counts (sparkline)
                        </p>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Pipeline Time Card */}
            <div className="bg-white border-2 border-black p-6" style={brutalShadow}>
              <div className="stack stack-sm">
                <h2 className="text-base font-bold text-black lowercase">
                  pipeline time (last {windowDays} days)
                </h2>
                {pipelines.isLoading ? (
                  <div className="flex items-center gap-2 text-gray-600 lowercase">
                    <DocLoader size="sm" />
                    <span>loading…</span>
                  </div>
                ) : pipelines.error ? (
                  <div className="text-sm text-black">
                    {pipelines.error instanceof Error
                      ? pipelines.error.message
                      : "failed to load pipeline metrics"}
                  </div>
                ) : (
                  <>
                    <div className="text-sm text-gray-600 lowercase">
                      avg:{" "}
                      <span className="font-bold text-black">
                        {formatSeconds(pipelines.data?.duration_sec.avg)}
                      </span>
                      {" · "}p95:{" "}
                      <span className="font-bold text-black">
                        {formatSeconds(pipelines.data?.duration_sec.p95)}
                      </span>
                    </div>

                    <div className="mt-3">
                      <h3 className="text-xs font-bold text-black lowercase tracking-wide mb-2">
                        avg per phase
                      </h3>
                      <div className="stack stack-sm">
                        {Object.entries(
                          pipelines.data?.per_phase_avg_duration_sec ?? {},
                        ).map(([phase, sec]) => (
                          <div key={phase} className="flex justify-between text-sm">
                            <span className="text-gray-600 lowercase">{phase}</span>
                            <span className="font-medium text-black">
                              {formatSeconds(sec)}
                            </span>
                          </div>
                        ))}
                        {Object.keys(
                          pipelines.data?.per_phase_avg_duration_sec ?? {},
                        ).length === 0 && (
                          <p className="text-sm text-gray-600 lowercase">
                            no phase timing data yet.
                          </p>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Cost Signals Card */}
            <div className="bg-white border-2 border-black p-6" style={brutalShadow}>
              <div className="stack stack-sm">
                <h2 className="text-base font-bold text-black lowercase">
                  cost signals
                </h2>
                {costs.isLoading ? (
                  <div className="flex items-center gap-2 text-gray-600 lowercase">
                    <DocLoader size="sm" />
                    <span>loading…</span>
                  </div>
                ) : costs.error ? (
                  <div className="text-sm text-black">
                    {costs.error instanceof Error
                      ? costs.error.message
                      : "failed to load cost signals"}
                  </div>
                ) : (
                  <>
                    <div className="stack stack-sm">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 lowercase">bytes uploaded (total)</span>
                        <span className="font-medium text-black">
                          {formatBytes(costs.data?.cost_proxies.bytes_uploaded_total)}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 lowercase">avg bytes / pipeline</span>
                        <span className="font-medium text-black">
                          {formatBytes(
                            costs.data?.cost_proxies.avg_bytes_uploaded_per_pipeline,
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 lowercase">artifacts (total)</span>
                        <span className="font-medium text-black">
                          {costs.data?.cost_proxies.artifact_count_total ?? 0}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600 lowercase">avg artifacts / pipeline</span>
                        <span className="font-medium text-black">
                          {formatNumber(costs.data?.cost_proxies.avg_artifacts_per_pipeline)}
                        </span>
                      </div>
                    </div>
                    <p className="text-xs text-gray-600 mt-2 lowercase">
                      openai/llm costs intentionally excluded for now.
                    </p>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Readiness Checks Card */}
          {(readiness.error || readiness.data) && (
            <div className="bg-white border-2 border-black p-6" style={brutalShadow}>
              <h2 className="text-base font-bold text-black mb-3 lowercase">readiness checks</h2>
              {readiness.error ? (
                <p className="text-sm text-black">
                  {readiness.error instanceof Error
                    ? readiness.error.message
                    : "failed to load readiness"}
                </p>
              ) : (
                <div className="grid gap-2">
                  {Object.entries(readiness.data?.checks ?? {}).map(
                    ([name, status]) => (
                      <div
                        key={name}
                        className="flex justify-between text-sm border-b border-black pb-2 last:border-b-0 last:pb-0"
                      >
                        <span className="text-gray-600 lowercase">{name}</span>
                        <span className="flex items-center gap-2 font-medium text-black">
                          <span
                            className={`w-4 h-4 flex items-center justify-center text-xs ${
                              status === "connected"
                                ? "bg-black text-white"
                                : "border border-black"
                            }`}
                          >
                            {status === "connected" ? "✓" : "×"}
                          </span>
                          <span className="lowercase">{status}</span>
                        </span>
                      </div>
                    ),
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
