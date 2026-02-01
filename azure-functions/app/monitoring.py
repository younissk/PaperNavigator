"""Monitoring metrics aggregation for PaperPilot.

All functions here are pure(ish) helpers used by HTTP routes.
They aggregate primarily from Cosmos job documents to avoid expensive Blob scans.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

import azure.functions as func

from .clients import get_jobs_container


def _clamp_int(value: str | None, *, default: int, min_value: int, max_value: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(min_value, min(max_value, parsed))


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # now_iso() uses datetime.now(UTC).isoformat(), which is parseable by fromisoformat.
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _percentile(sorted_values: list[float], p: float) -> float | None:
    """Compute percentile using linear interpolation on a sorted list."""
    if not sorted_values:
        return None
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def _completed_pipeline_jobs_since(cutoff_epoch: int, *, limit: int) -> list[dict[str, Any]]:
    container = get_jobs_container()
    # Use c._ts (server-side last modified time) for windowing; completion updates the doc.
    query = (
        "SELECT c.id, c.job_id, c.job_type, c.status, c.created_at, c.updated_at, c.events, c.result "
        "FROM c WHERE c.job_type = 'pipeline' AND c.status = 'completed' AND c._ts >= @cutoff"
    )
    items_iter: Iterable[dict[str, Any]] = container.query_items(
        query=query,
        parameters=[{"name": "@cutoff", "value": cutoff_epoch}],
        enable_cross_partition_query=True,
    )

    jobs: list[dict[str, Any]] = []
    for item in items_iter:
        jobs.append(item)
        if len(jobs) >= limit:
            break
    return jobs


def _phase_durations_from_events(events: list[dict[str, Any]] | None) -> dict[str, float]:
    if not events:
        return {}

    starts: dict[str, datetime] = {}
    completes: dict[str, datetime] = {}

    for ev in events:
        ev_type = ev.get("type")
        phase = ev.get("phase")
        ts = _parse_iso(ev.get("ts"))
        if not isinstance(phase, str) or not ts:
            continue
        if ev_type == "phase_start":
            starts[phase] = min(starts.get(phase, ts), ts)
        elif ev_type == "phase_complete":
            completes[phase] = max(completes.get(phase, ts), ts)

    durations: dict[str, float] = {}
    for phase, start_ts in starts.items():
        end_ts = completes.get(phase)
        if not end_ts:
            continue
        sec = (end_ts - start_ts).total_seconds()
        if sec >= 0:
            durations[phase] = sec
    return durations


def get_report_metrics(req: func.HttpRequest) -> dict[str, Any]:
    window_days = _clamp_int(req.params.get("window_days"), default=30, min_value=1, max_value=365)
    limit = _clamp_int(req.params.get("limit"), default=5000, min_value=1, max_value=20000)

    cutoff_epoch = int((_now_utc() - timedelta(days=window_days)).timestamp())
    jobs = _completed_pipeline_jobs_since(cutoff_epoch, limit=limit)

    per_day_counts: Counter[str] = Counter()
    for job in jobs:
        updated_at = _parse_iso(job.get("updated_at"))
        if not updated_at:
            continue
        per_day_counts[updated_at.date().isoformat()] += 1

    daily = [{"date": d, "count": per_day_counts[d]} for d in sorted(per_day_counts.keys())]

    return {
        "window_days": window_days,
        "reports_generated": len(jobs),
        "daily": daily,
        "sample_limit": limit,
        "sampled_jobs": len(jobs),
    }


def get_pipeline_metrics(req: func.HttpRequest) -> dict[str, Any]:
    window_days = _clamp_int(req.params.get("window_days"), default=30, min_value=1, max_value=365)
    limit = _clamp_int(req.params.get("limit"), default=2000, min_value=1, max_value=20000)

    cutoff_epoch = int((_now_utc() - timedelta(days=window_days)).timestamp())
    jobs = _completed_pipeline_jobs_since(cutoff_epoch, limit=limit)

    durations: list[float] = []
    per_phase_values: dict[str, list[float]] = defaultdict(list)

    for job in jobs:
        created_at = _parse_iso(job.get("created_at"))
        updated_at = _parse_iso(job.get("updated_at"))
        if created_at and updated_at:
            total_sec = (updated_at - created_at).total_seconds()
            if total_sec >= 0:
                durations.append(total_sec)

        phase_durations = _phase_durations_from_events(job.get("events"))
        for phase, sec in phase_durations.items():
            per_phase_values[phase].append(sec)

    durations_sorted = sorted(durations)
    avg = (sum(durations_sorted) / len(durations_sorted)) if durations_sorted else None

    per_phase_avg = {
        phase: (sum(vals) / len(vals)) if vals else None
        for phase, vals in sorted(per_phase_values.items())
    }

    return {
        "window_days": window_days,
        "sample_limit": limit,
        "sampled_jobs": len(jobs),
        "duration_sec": {
            "avg": avg,
            "p50": _percentile(durations_sorted, 50),
            "p95": _percentile(durations_sorted, 95),
            "count": len(durations_sorted),
        },
        "per_phase_avg_duration_sec": per_phase_avg,
    }


def get_costs_metrics(req: func.HttpRequest) -> dict[str, Any]:
    window_days = _clamp_int(req.params.get("window_days"), default=30, min_value=1, max_value=365)
    limit = _clamp_int(req.params.get("limit"), default=2000, min_value=1, max_value=20000)

    cutoff_epoch = int((_now_utc() - timedelta(days=window_days)).timestamp())
    jobs = _completed_pipeline_jobs_since(cutoff_epoch, limit=limit)

    def _as_int(v: Any) -> int:
        try:
            n = int(v)
        except Exception:
            return 0
        return max(0, n)

    def _as_float(v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except Exception:
            return None

    total_bytes = 0
    total_artifacts = 0
    bytes_samples = 0
    artifacts_samples = 0

    durations: list[float] = []

    openai_pipelines_with_usage = 0
    openai_pipelines_with_priced_cost = 0
    openai_total_tokens = 0
    openai_total_cost_usd: float = 0.0
    openai_by_model: dict[str, dict[str, Any]] = {}

    for job in jobs:
        created_at = _parse_iso(job.get("created_at"))
        updated_at = _parse_iso(job.get("updated_at"))
        if created_at and updated_at:
            sec = (updated_at - created_at).total_seconds()
            if sec >= 0:
                durations.append(sec)

        result = job.get("result") or {}
        if isinstance(result, dict):
            artifact_count = result.get("artifact_count")
            if isinstance(artifact_count, int) and artifact_count >= 0:
                total_artifacts += artifact_count
                artifacts_samples += 1
            else:
                artifacts = result.get("artifacts")
                if isinstance(artifacts, list):
                    total_artifacts += len(artifacts)
                    artifacts_samples += 1

            artifact_bytes_total = result.get("artifact_bytes_total")
            if isinstance(artifact_bytes_total, int) and artifact_bytes_total >= 0:
                total_bytes += artifact_bytes_total
                bytes_samples += 1

            usage = result.get("openai_usage")
            if isinstance(usage, dict):
                totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
                total_tokens = _as_int(totals.get("total_tokens"))
                if total_tokens > 0:
                    openai_pipelines_with_usage += 1
                    openai_total_tokens += total_tokens

                cost = _as_float(totals.get("estimated_cost_usd"))
                if cost is not None and total_tokens > 0:
                    openai_pipelines_with_priced_cost += 1
                    openai_total_cost_usd += float(cost)

                by_model = usage.get("by_model") if isinstance(usage.get("by_model"), dict) else {}
                for model, mvals in by_model.items():
                    if not isinstance(model, str) or not isinstance(mvals, dict):
                        continue
                    bucket = openai_by_model.setdefault(
                        model,
                        {
                            "requests": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "estimated_cost_usd": 0.0,
                            "has_unpriced": False,
                        },
                    )
                    bucket["requests"] += _as_int(mvals.get("requests"))
                    bucket["prompt_tokens"] += _as_int(mvals.get("prompt_tokens"))
                    bucket["completion_tokens"] += _as_int(mvals.get("completion_tokens"))
                    bucket["total_tokens"] += _as_int(mvals.get("total_tokens"))
                    mc = _as_float(mvals.get("estimated_cost_usd"))
                    if mc is None:
                        bucket["has_unpriced"] = True
                    else:
                        bucket["estimated_cost_usd"] = float(bucket["estimated_cost_usd"]) + float(mc)

    durations_sorted = sorted(durations)
    avg_duration = (sum(durations_sorted) / len(durations_sorted)) if durations_sorted else None

    pipelines = len(jobs)
    avg_bytes = (total_bytes / bytes_samples) if bytes_samples else None
    avg_artifacts = (total_artifacts / artifacts_samples) if artifacts_samples else None
    avg_openai_tokens = (openai_total_tokens / openai_pipelines_with_usage) if openai_pipelines_with_usage else None
    avg_openai_cost = (openai_total_cost_usd / openai_pipelines_with_priced_cost) if openai_pipelines_with_priced_cost else None

    openai_by_model_out: dict[str, Any] = {}
    for model, vals in sorted(openai_by_model.items()):
        if vals.get("has_unpriced"):
            cost_val: float | None = None
        else:
            cost_val = float(vals.get("estimated_cost_usd", 0.0))
        openai_by_model_out[model] = {
            "requests": vals.get("requests", 0),
            "prompt_tokens": vals.get("prompt_tokens", 0),
            "completion_tokens": vals.get("completion_tokens", 0),
            "total_tokens": vals.get("total_tokens", 0),
            "estimated_cost_usd": cost_val,
        }

    return {
        "window_days": window_days,
        "sample_limit": limit,
        "sampled_jobs": pipelines,
        "cost_proxies": {
            "bytes_uploaded_total": total_bytes,
            "avg_bytes_uploaded_per_pipeline": avg_bytes,
            "artifact_count_total": total_artifacts,
            "avg_artifacts_per_pipeline": avg_artifacts,
            "avg_duration_sec": avg_duration,
            "duration_p95_sec": _percentile(durations_sorted, 95),
            "coverage": {
                "bytes_samples": bytes_samples,
                "artifact_samples": artifacts_samples,
                "duration_samples": len(durations_sorted),
            },
        },
        "openai": {
            "estimated_cost_usd_total": openai_total_cost_usd if openai_pipelines_with_priced_cost else None,
            "avg_estimated_cost_usd_per_pipeline": avg_openai_cost,
            "total_tokens_total": openai_total_tokens,
            "avg_total_tokens_per_pipeline": avg_openai_tokens,
            "pipelines_with_usage": openai_pipelines_with_usage,
            "pipelines_with_priced_cost": openai_pipelines_with_priced_cost,
            "by_model": openai_by_model_out,
        },
        "notes": [
            "OpenAI costs are estimates derived from token usage, not billing statements.",
            "Default pricing assumptions can be overridden with OPENAI_PRICING_JSON.",
            "Bytes/artifact metrics depend on job result fields; older jobs may not include them.",
        ],
    }
