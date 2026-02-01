"""HTTP API routes for PaperPilot Azure Functions."""

from __future__ import annotations

import azure.functions as func

from .http_utils import cors_preflight, json_response, safe
from .jobs import create_job, enqueue_job, get_job, test_openai_connection
from .parsing import normalize_pipeline_payload, normalize_search_payload, parse_json
from .jobs import test_cosmos_connection, test_service_bus_connection
from .results import get_all_query_metadata, get_query_metadata, get_query_results, list_recent_reports, list_result_slugs, test_storage_connection
from .monitoring import (
    get_costs_metrics,
    get_pipeline_metrics,
    get_report_metrics,
)

bp = func.Blueprint()


@bp.route(route="{*path}", methods=["OPTIONS"])
def preflight_any(req: func.HttpRequest) -> func.HttpResponse:
    return cors_preflight()


@bp.route(route="health", methods=["GET", "OPTIONS"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        storage_ok = test_storage_connection()
        database_ok = test_cosmos_connection()
        all_ok = storage_ok and database_ok

        return json_response({
            "status": "ok" if all_ok else "degraded",
            "version": "0.1.0",
            "storage": "connected" if storage_ok else "unavailable",
            "database": "connected" if database_ok else "unavailable",
        })

    return safe(handler)


@bp.route(route="ready", methods=["GET", "OPTIONS"])
def readiness_check(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        storage_ok = test_storage_connection()
        database_ok = test_cosmos_connection()
        service_bus_ok = test_service_bus_connection()
        openai_ok, openai_latency_ms, openai_error = test_openai_connection()

        ready = storage_ok and database_ok and service_bus_ok and openai_ok

        return json_response({
            "status": "ok" if ready else "degraded",
            "ready": bool(ready),
            "version": "0.1.0",
            "checks": {
                "storage": "connected" if storage_ok else "unavailable",
                "database": "connected" if database_ok else "unavailable",
                "service_bus": "connected" if service_bus_ok else "unavailable",
                "openai": "connected" if openai_ok else "unavailable",
            },
            "signals": {
                "openai": {
                    "ok": bool(openai_ok),
                    "latency_ms": openai_latency_ms,
                    "error": openai_error,
                },
            },
        })

    return safe(handler)


@bp.route(route="", methods=["GET", "OPTIONS"])
def root(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()
    return safe(lambda: json_response({
        "message": "PaperPilot API (Azure Functions)",
        "version": "0.1.0",
        "endpoints": {
            "health": "GET /api/health",
            "ready": "GET /api/ready",
            "create_pipeline": "POST /api/pipeline",
            "create_search": "POST /api/search",
            "create_job": "POST /api/jobs",
            "get_job": "GET /api/jobs/{job_id}",
            "get_job_events": "GET /api/jobs/{job_id}/events",
            "pipeline_status": "GET /api/pipeline/{job_id}",
            "results": "GET /api/results",
            "results_recent": "GET /api/results/recent",
            "monitoring_reports": "GET /api/monitoring/reports",
            "monitoring_pipelines": "GET /api/monitoring/pipelines",
            "monitoring_costs": "GET /api/monitoring/costs",
        },
    }))


@bp.route(route="jobs", methods=["POST", "OPTIONS"])
def create_job_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        data = parse_json(req)
        if data is None:
            return json_response({"error": "Invalid JSON body"}, status=400)

        job_type = str(data.get("job_type", "pipeline"))
        try:
            if job_type == "pipeline":
                payload = normalize_pipeline_payload(data)
            elif job_type == "search":
                payload = normalize_search_payload(data)
            else:
                payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
                query = data.get("query")
                if query:
                    payload["query"] = query
                if "query" not in payload:
                    raise ValueError("Missing required field: query")
        except ValueError as exc:
            return json_response({"error": str(exc)}, status=400)

        job_id = create_job(job_type=job_type, query=payload["query"], payload=payload)
        if not job_id:
            return json_response({"error": "Failed to create job. Database may be unavailable."}, status=503)

        if not enqueue_job(job_id, job_type, payload):
            return json_response({"error": "Failed to enqueue job. Service Bus may be unavailable."}, status=503)

        return json_response(
            {
                "job_id": job_id,
                "status": "queued",
                "message": "Job created and queued for processing",
            },
            status=202,
        )

    return safe(handler)


@bp.route(route="pipeline", methods=["POST", "OPTIONS"])
def start_pipeline(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        data = parse_json(req)
        if data is None:
            return json_response({"error": "Invalid JSON body"}, status=400)

        try:
            payload = normalize_pipeline_payload(data)
        except ValueError as exc:
            return json_response({"error": str(exc)}, status=400)

        job_id = create_job("pipeline", payload["query"], payload)
        if not job_id:
            return json_response({"error": "Failed to create job. Database may be unavailable."}, status=503)

        if not enqueue_job(job_id, "pipeline", payload):
            return json_response({"error": "Failed to enqueue job. Service Bus may be unavailable."}, status=503)

        return json_response(
            {
                "job_id": job_id,
                "status": "queued",
                "message": "Pipeline job queued for processing",
            },
            status=202,
        )

    return safe(handler)


@bp.route(route="search", methods=["POST", "OPTIONS"])
def start_search(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        data = parse_json(req)
        if data is None:
            return json_response({"error": "Invalid JSON body"}, status=400)

        try:
            payload = normalize_search_payload(data)
        except ValueError as exc:
            return json_response({"error": str(exc)}, status=400)

        job_id = create_job("search", payload["query"], payload)
        if not job_id:
            return json_response({"error": "Failed to create job. Database may be unavailable."}, status=503)

        if not enqueue_job(job_id, "search", payload):
            return json_response({"error": "Failed to enqueue job. Service Bus may be unavailable."}, status=503)

        return json_response(
            {
                "job_id": job_id,
                "status": "queued",
                "message": "Search job queued for processing",
            },
            status=202,
        )

    return safe(handler)


@bp.route(route="jobs/{job_id}", methods=["GET", "OPTIONS"])
def get_job_status(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        job_id = req.route_params.get("job_id")
        if not job_id:
            return json_response({"error": "Missing job_id"}, status=400)

        job = get_job(job_id)
        if not job:
            return json_response({"error": "Job not found"}, status=404)

        include_events = str(req.params.get("include_events", "")).lower() in {"1", "true", "yes"}
        response = {
            "job_id": job.get("job_id"),
            "job_type": job.get("job_type", "unknown"),
            "status": job.get("status", "unknown"),
            "query": job.get("query", ""),
            "created_at": job.get("created_at", ""),
            "updated_at": job.get("updated_at", ""),
            "progress": job.get("progress"),
            "result": job.get("result"),
            "error_message": job.get("error_message"),
            "error_code": job.get("error_code"),
        }

        if include_events:
            response["events"] = job.get("events", [])

        return json_response(response)

    return safe(handler)


@bp.route(route="jobs/{job_id}/events", methods=["GET", "OPTIONS"])
def get_job_events(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        job_id = req.route_params.get("job_id")
        if not job_id:
            return json_response({"error": "Missing job_id"}, status=400)

        job = get_job(job_id)
        if not job:
            return json_response({"error": "Job not found"}, status=404)

        events = job.get("events", []) or []
        limit_raw = req.params.get("limit")
        if limit_raw:
            try:
                limit = max(1, int(limit_raw))
                events = events[-limit:]
            except ValueError:
                return json_response({"error": "Invalid limit parameter"}, status=400)

        return json_response({
            "job_id": job_id,
            "events": events,
        })

    return safe(handler)


@bp.route(route="pipeline/{job_id}", methods=["GET", "OPTIONS"])
def get_pipeline_status(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        job_id = req.route_params.get("job_id")
        if not job_id:
            return json_response({"error": "Missing job_id"}, status=400)

        job = get_job(job_id)
        if not job:
            return json_response({"error": "Job not found"}, status=404)

        internal_status = job.get("status", "unknown")
        progress = job.get("progress", {}) or {}
        phase = progress.get("phase", "")
        events = job.get("events", []) or []

        # Limit returned events to keep payload lightweight
        limit_raw = req.params.get("events_limit")
        try:
            event_limit = max(1, int(limit_raw)) if limit_raw else 20
        except ValueError:
            event_limit = 20

        recent_events = events[-event_limit:] if events else []
        alert_events = [e for e in events if e.get("level") in {"warning", "error"}]
        alerts = alert_events[-10:] if alert_events else []

        if internal_status == "running":
            if phase == "search":
                frontend_status = "searching"
            elif phase == "ranking":
                frontend_status = "ranking"
            elif phase in {"report", "upload"}:
                frontend_status = "reporting"
            else:
                frontend_status = "searching"
        else:
            frontend_status = internal_status

        return json_response({
            "job_id": job.get("job_id"),
            "status": frontend_status,
            "query": job.get("query", ""),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "phase": phase or None,
            "phase_step": progress.get("step"),
            "phase_step_name": progress.get("step_name") or progress.get("phase"),
            "phase_progress": progress.get("current"),
            "phase_total": progress.get("total"),
            "progress_message": progress.get("message"),
            "papers": (job.get("result") or {}).get("top_papers"),
            "report_data": None,
            "error": job.get("error_message"),
            "error_code": job.get("error_code"),
            "events": recent_events,
            "alerts": alerts,
        })

    return safe(handler)


@bp.route(route="results", methods=["GET", "OPTIONS"])
def list_results(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()
    return safe(lambda: json_response({
        "queries": [slug.replace("_", " ").title() for slug in list_result_slugs()]
    }))


@bp.route(route="results/metadata", methods=["GET", "OPTIONS"])
def get_all_results_metadata(req: func.HttpRequest) -> func.HttpResponse:
    """Return all queries with their metadata in a single batch request."""
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        slugs = list_result_slugs()
        metadata_map = get_all_query_metadata()

        # Build response with query display names and metadata
        queries = []
        for slug in slugs:
            query_name = slug.replace("_", " ").title()
            queries.append({
                "query": query_name,
                "slug": slug,
                "metadata": metadata_map.get(slug),
            })

        return json_response({"queries": queries})

    return safe(handler)


@bp.route(route="results/recent", methods=["GET", "OPTIONS"])
def get_recent_results(req: func.HttpRequest) -> func.HttpResponse:
    """Return the most recently generated reports with metadata."""
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        limit_raw = req.params.get("limit")
        try:
            limit = max(1, min(20, int(limit_raw))) if limit_raw else 5
        except ValueError:
            limit = 5

        reports = list_recent_reports(limit=limit)
        return json_response({"reports": reports})

    return safe(handler)


@bp.route(route="results/{query_slug}", methods=["GET", "OPTIONS"])
def get_results_metadata(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        query_slug = req.route_params.get("query_slug")
        if not query_slug:
            return json_response({"error": "Missing query"}, status=400)

        metadata = get_query_metadata(query_slug)
        if not metadata:
            return json_response({"error": f"No results found for query: {query_slug}"}, status=404)

        return json_response({
            "query": query_slug,
            "metadata": metadata,
        })

    return safe(handler)


@bp.route(route="results/{query_slug}/all", methods=["GET", "OPTIONS"])
def get_all_results(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        query_slug = req.route_params.get("query_slug")
        if not query_slug:
            return json_response({"error": "Missing query"}, status=400)

        results = get_query_results(query_slug)

        if not results["report"] and not results["snowball"]:
            return json_response({"error": f"No results found for query: {query_slug}"}, status=404)

        return json_response(results)

    return safe(handler)


@bp.route(route="results/{query_slug}/report", methods=["GET", "OPTIONS"])
def get_report_results(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()

    def handler():
        query_slug = req.route_params.get("query_slug")
        if not query_slug:
            return json_response({"error": "Missing query"}, status=400)

        results = get_query_results(query_slug)
        if not results["report"]:
            return json_response({"error": f"No report found for query: {query_slug}"}, status=404)

        return json_response(results["report"])

    return safe(handler)


@bp.route(route="monitoring/reports", methods=["GET", "OPTIONS"])
def monitoring_reports(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()
    return safe(lambda: json_response(get_report_metrics(req)))


@bp.route(route="monitoring/pipelines", methods=["GET", "OPTIONS"])
def monitoring_pipelines(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()
    return safe(lambda: json_response(get_pipeline_metrics(req)))


@bp.route(route="monitoring/costs", methods=["GET", "OPTIONS"])
def monitoring_costs(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return cors_preflight()
    return safe(lambda: json_response(get_costs_metrics(req)))
