"""Cosmos DB job storage + Service Bus enqueue helpers."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from .clients import get_jobs_container, get_service_bus_client
from .config import COSMOS_ENDPOINT, COSMOS_KEY, MAX_EVENTS, QUEUE_NAME, SERVICE_BUS_CONNECTION, logger
from .utils import expires_at, now_iso
from .telemetry import log_event

_jobs_container_pk_path: str | None = None
_jobs_container_pk_field: str | None = None


def _get_jobs_partition_key_field(container) -> str | None:
    """Best-effort discovery of the Cosmos container partition key field.

    Returns the top-level field name (e.g. "job_id", "jobId", "id") or None if unknown.
    """
    global _jobs_container_pk_path, _jobs_container_pk_field
    if _jobs_container_pk_field is not None:
        return _jobs_container_pk_field

    try:
        props = container.read()
        paths = (props.get("partitionKey") or {}).get("paths") or []
        pk_path = paths[0] if paths else None
    except Exception:
        pk_path = None

    if not isinstance(pk_path, str) or not pk_path:
        _jobs_container_pk_path = None
        _jobs_container_pk_field = None
        return None

    _jobs_container_pk_path = pk_path
    field = pk_path.lstrip("/")
    # Only support top-level partition keys for fast point reads/patches.
    if "/" in field or not field:
        _jobs_container_pk_field = None
        return None

    _jobs_container_pk_field = field
    return field


def _get_jobs_partition_key_value(job_id: str, *, job: dict[str, Any] | None = None) -> Any | None:
    """Resolve the Cosmos partition key value for a given job_id (if possible)."""
    try:
        container = get_jobs_container()
    except Exception:
        return None

    field = _get_jobs_partition_key_field(container)
    if not field:
        return None

    if field in {"id", "job_id", "jobId"}:
        return job_id

    if job and field in job:
        return job[field]

    return None


def test_cosmos_connection() -> bool:
    """Test if Cosmos DB is accessible. Returns True if connected, False otherwise."""
    if not COSMOS_ENDPOINT or not COSMOS_KEY:
        return False

    try:
        container = get_jobs_container()
        # Just check if we can query (lightweight operation)
        list(container.query_items(query="SELECT TOP 1 c.id FROM c", enable_cross_partition_query=True))
        return True
    except Exception as exc:
        logger.warning("Cosmos DB connection test failed: %s", exc)
        return False


def test_service_bus_connection() -> bool:
    """Test if Service Bus is accessible. Returns True if connected, False otherwise."""
    if not SERVICE_BUS_CONNECTION:
        return False

    try:
        # Creating a sender exercises DNS/auth/connection without sending messages.
        with get_service_bus_client() as sb_client:
            with sb_client.get_queue_sender(QUEUE_NAME):
                pass
        return True
    except Exception as exc:
        logger.warning("Service Bus connection test failed: %s", exc)
        return False


def test_openai_connection(*, timeout_sec: float = 5.0) -> tuple[bool, float | None, str | None]:
    """Best-effort OpenAI connectivity check.

    Returns (ok, latency_ms, error_message).
    """
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key or api_key.startswith("@Microsoft.KeyVault"):
        return False, None, "OPENAI_API_KEY not available at runtime"

    # Use a very lightweight request; this does not generate tokens.
    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = getattr(resp, "status", None)
            if status and 200 <= status < 300:
                return True, (time.time() - start) * 1000.0, None
            return False, (time.time() - start) * 1000.0, f"OpenAI returned status {status}"
    except urllib.error.HTTPError as exc:
        detail = None
        try:
            raw = exc.read()
            if raw:
                try:
                    payload = json.loads(raw.decode("utf-8", errors="replace"))
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    err = payload.get("error") if isinstance(payload.get("error"), dict) else payload
                    msg = err.get("message") if isinstance(err, dict) else None
                    code = err.get("code") if isinstance(err, dict) else None
                    msg_s = str(msg).lower() if msg is not None else ""
                    code_s = str(code).lower() if code is not None else ""
                    if code_s in {"insufficient_quota", "billing_hard_limit_reached"} or "insufficient_quota" in msg_s:
                        detail = "OpenAI quota/credits exhausted"
                    elif msg:
                        detail = str(msg)
        except Exception:
            detail = None

        suffix = f": {detail}" if detail else ""
        return False, (time.time() - start) * 1000.0, f"OpenAI HTTPError {exc.code}{suffix}"
    except Exception as exc:
        return False, (time.time() - start) * 1000.0, str(exc)


EVENT_LEVELS: dict[str, str] = {
    "job_created": "info",
    "job_enqueued": "info",
    "job_start": "info",
    "job_complete": "info",
    "phase_start": "info",
    "phase_complete": "info",
    "progress": "info",
    "email_sent": "info",
    "job_failed": "error",
    "job_enqueue_failed": "error",
    "phase_error": "error",
    "phase_warning": "warning",
}


def append_event(
    events: list[dict[str, Any]],
    event_type: str,
    phase: str,
    message: str,
    level: str | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    resolved_level = level or EVENT_LEVELS.get(event_type, "info")
    event = {
        "ts": now_iso(),
        "type": event_type,
        "level": resolved_level,
        "phase": phase,
        "message": message,
        **kwargs,
    }
    events.append(event)
    if len(events) > MAX_EVENTS:
        events = events[-MAX_EVENTS:]
    return events


def create_job(job_type: str, query: str, payload: dict[str, Any]) -> str | None:
    """Create a new job in Cosmos DB. Returns job_id on success, None on failure."""
    if not COSMOS_ENDPOINT or not COSMOS_KEY:
        logger.error("Cannot create job: Cosmos DB not configured")
        return None

    job_id = str(uuid.uuid4())
    now = now_iso()

    events: list[dict[str, Any]] = []
    events = append_event(
        events,
        "job_created",
        "init",
        "Job created",
        job_type=job_type,
        query=query,
    )

    job = {
        "id": job_id,
        "job_id": job_id,
        # Back-compat for Cosmos containers partitioned on /jobId (and enables point reads).
        "jobId": job_id,
        "job_type": job_type,
        "status": "queued",
        "query": query,
        "payload": payload,
        "created_at": now,
        "updated_at": now,
        "expires_at": expires_at(),
        "events": events,
        "progress": {
            "phase": "init",
            "step": 0,
            "message": "Waiting to start...",
            "current": 0,
            "total": 0,
        },
    }

    try:
        container = get_jobs_container()
        container.create_item(job)
        return job_id
    except Exception as exc:
        logger.error("Failed to create job in Cosmos DB: %s", exc)
        return None


def get_job(job_id: str) -> dict[str, Any] | None:
    """Retrieve a job from Cosmos DB. Returns None if not found or on error."""
    if not COSMOS_ENDPOINT or not COSMOS_KEY:
        logger.warning("Cannot get job: Cosmos DB not configured")
        return None

    try:
        container = get_jobs_container()
        pk_value = _get_jobs_partition_key_value(job_id)
        if pk_value is not None:
            try:
                return container.read_item(item=job_id, partition_key=pk_value)
            except Exception:
                # Fall back to query for older docs / unknown partition keys.
                pass
        # IMPORTANT: Avoid read_item(id, partition_key=...) because an incorrect partition_key
        # yields a 404 even when the document exists. We query by id/job_id instead.
        items = list(
            container.query_items(
                query="SELECT TOP 1 * FROM c WHERE c.id = @id OR c.job_id = @id",
                parameters=[{"name": "@id", "value": job_id}],
                enable_cross_partition_query=True,
            )
        )
        return items[0] if items else None
    except Exception as exc:
        logger.error("Failed to get job %s from Cosmos DB: %s", job_id, exc)
        return None


def update_job_document(job_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    """Update a job document in Cosmos DB. Returns updated job or None on failure."""
    if not COSMOS_ENDPOINT or not COSMOS_KEY:
        logger.warning("Cannot update job: Cosmos DB not configured")
        return None

    try:
        container = get_jobs_container()
        pk_value = _get_jobs_partition_key_value(job_id)

        # Fast path: patch without a read to avoid cross-partition queries and reduce latency.
        if pk_value is not None:
            try:
                patch_ops = [{"op": "set", "path": f"/{k}", "value": v} for k, v in updates.items()]
                return container.patch_item(item=job_id, partition_key=pk_value, patch_operations=patch_ops)
            except Exception:
                pass

        job = get_job(job_id)
        if not job:
            return None
        job.update(updates)
        container.upsert_item(job)
        return job
    except Exception as exc:
        logger.error("Failed to update job %s in Cosmos DB: %s", job_id, exc)
        return None


def append_job_event(
    job_id: str,
    event_type: str,
    phase: str,
    message: str,
    **kwargs,
) -> None:
    job = get_job(job_id)
    if not job:
        return

    events = job.get("events", []) or []
    events = append_event(events, event_type, phase, message, **kwargs)
    update_job_document(job_id, {"events": events, "updated_at": now_iso()})
    resolved_level = EVENT_LEVELS.get(event_type, "info")
    level = logging.INFO
    if resolved_level == "warning":
        level = logging.WARNING
    elif resolved_level == "error":
        level = logging.ERROR
    log_event(
        logger,
        level,
        event_type,
        job_id=job_id,
        phase=phase,
        event_message=message,
        **kwargs,
    )


def enqueue_job(job_id: str, job_type: str, payload: dict[str, Any]) -> bool:
    from azure.servicebus import ServiceBusMessage

    if not SERVICE_BUS_CONNECTION:
        msg = "Service Bus connection string not set; cannot enqueue job"
        logger.error(msg)
        append_job_event(job_id, "job_enqueue_failed", "error", msg)
        update_job_progress(job_id, "failed", "error", 0, msg, error=msg)
        return False

    message_body = json.dumps({
        "job_id": job_id,
        "job_type": job_type,
        "payload": payload,
    })

    try:
        with get_service_bus_client() as sb_client:
            with sb_client.get_queue_sender(QUEUE_NAME) as sender:
                sender.send_messages(ServiceBusMessage(message_body))
    except Exception as exc:
        msg = f"Failed to enqueue job: {exc}"
        logger.exception("Service Bus enqueue failed for job %s", job_id)
        append_job_event(job_id, "job_enqueue_failed", "error", msg)
        update_job_progress(job_id, "failed", "error", 0, msg, error=str(exc))
        return False

    append_job_event(
        job_id,
        "job_enqueued",
        "init",
        "Job enqueued to Service Bus",
        queue=QUEUE_NAME,
    )
    return True


def update_job_progress(
    job_id: str,
    status: str,
    phase: str,
    step: int,
    message: str,
    current: int = 0,
    total: int = 0,
    step_name: str | None = None,
    events: list[dict[str, Any]] | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    error_code: str | None = None,
) -> None:
    updates: dict[str, Any] = {
        "status": status,
        "updated_at": now_iso(),
        "progress": {
            "phase": phase,
            "step": step,
            "step_name": step_name,
            "message": message,
            "current": current,
            "total": total,
        },
    }

    if events is not None:
        updates["events"] = events
    if result is not None:
        updates["result"] = result
    if error is not None:
        updates["error_message"] = error
    if error_code is not None:
        updates["error_code"] = error_code

    updated = update_job_document(job_id, updates)
    if updated is None:
        logger.warning("Job %s not found while updating progress", job_id)
        return

    level = logging.INFO
    if status == "failed":
        level = logging.ERROR
    elif phase == "error":
        level = logging.ERROR

    log_event(
        logger,
        level,
        "job_progress",
        job_id=job_id,
        status=status,
        phase=phase,
        step=step,
        step_name=step_name,
        current=current,
        total=total,
        progress_message=message,
        error_message=error,
    )
