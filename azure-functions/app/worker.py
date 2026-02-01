"""Service Bus worker for processing jobs."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import azure.functions as func

from .config import QUEUE_NAME, logger
from .jobs import append_event, enqueue_job, get_job, update_job_progress
from .notifications import send_completion_email, send_failure_email
from .pipeline import run_search_job, run_ranking_stage, run_report_stage
from .utils import is_job_stale, load_openai_api_key

bp = func.Blueprint()


def _extract_dead_letter_details(msg: func.ServiceBusMessage) -> tuple[str | None, str | None]:
    reason = getattr(msg, "dead_letter_reason", None)
    description = getattr(msg, "dead_letter_error_description", None)

    if not reason or not description:
        props = getattr(msg, "user_properties", None) or getattr(msg, "application_properties", None) or {}
        if not reason:
            reason = props.get("DeadLetterReason") or props.get(b"DeadLetterReason")
        if not description:
            description = props.get("DeadLetterErrorDescription") or props.get(b"DeadLetterErrorDescription")

    return reason, description


def _mark_job_failed_from_dlq(job_id: str, reason: str | None, description: str | None, message_id: str | None) -> None:
    job = get_job(job_id)
    if not job:
        logger.warning("DLQ message for unknown job %s", job_id)
        return

    status = job.get("status")
    if status in ("completed", "failed"):
        logger.info("DLQ message for job %s already %s", job_id, status)
        return

    reason_text = reason or "DeadLettered"
    desc_text = f" {description}" if description else ""
    message = f"Job dead-lettered: {reason_text}.{desc_text}".strip()

    events = job.get("events", []) or []
    events = append_event(
        events,
        "job_failed",
        "error",
        message,
        dead_letter_reason=reason,
        dead_letter_error_description=description,
        message_id=message_id,
    )
    update_job_progress(job_id, "failed", "error", 0, message, events=events, error=message)


def process_job(job_id: str, job_type: str, payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], bool, bool]:
    """Process a job and return (result, events, was_processed, is_final).
    
    Returns was_processed=False if job was skipped due to idempotency check.
    """
    existing_job = get_job(job_id)
    stage = payload.get("stage") if isinstance(payload, dict) else None
    
    # Idempotency check: skip if job already completed or failed
    phase_order = {"search": 0, "ranking": 1, "report": 2}
    if existing_job:
        existing_status = existing_job.get("status")
        if existing_status in ("completed", "failed"):
            logger.info("Job %s already finished with status '%s', skipping re-execution", job_id, existing_status)
            return existing_job.get("result", {}), existing_job.get("events", []), False, True
        # Check if job is running but stale (stuck)
        if existing_status == "running":
            if is_job_stale(existing_job):
                logger.warning("Job %s is stale (running for too long), allowing retry", job_id)
            else:
                progress = existing_job.get("progress") or {}
                current_phase = (progress.get("phase") or "").lower()
                step_name = (progress.get("step_name") or "").lower()
                progress_message = (progress.get("message") or "").lower()
                is_queued_marker = "queued" in step_name or "queued" in progress_message
                stage_norm = stage.lower() if isinstance(stage, str) else ""

                if not stage_norm:
                    logger.warning("Job %s is already running and message has no stage, skipping", job_id)
                    return {}, existing_job.get("events", []), False, False

                # Enforce phase ordering to avoid running stages out of order.
                if current_phase in phase_order and stage_norm in phase_order:
                    if phase_order[stage_norm] < phase_order[current_phase]:
                        logger.warning(
                            "Job %s stage '%s' behind current phase '%s', skipping",
                            job_id,
                            stage_norm,
                            current_phase,
                        )
                        return {}, existing_job.get("events", []), False, False
                    if phase_order[stage_norm] > phase_order[current_phase]:
                        # Cosmos can briefly return a stale job document right after we update progress
                        # and enqueue the next stage. Re-enqueueing (and returning) can amplify cold-start
                        # delays, so prefer an in-process short wait + refresh. If still mismatched, allow
                        # execution only when the stage is the next sequential step.
                        deadline = time.time() + 2.0
                        refreshed = existing_job
                        while time.time() < deadline:
                            time.sleep(0.15)
                            refreshed = get_job(job_id) or refreshed
                            refreshed_progress = (refreshed.get("progress") or {}) if isinstance(refreshed, dict) else {}
                            refreshed_phase = (refreshed_progress.get("phase") or "").lower()
                            if refreshed_phase == stage_norm:
                                existing_job = refreshed
                                progress = refreshed_progress
                                current_phase = refreshed_phase
                                step_name = (progress.get("step_name") or "").lower()
                                progress_message = (progress.get("message") or "").lower()
                                is_queued_marker = "queued" in step_name or "queued" in progress_message
                                break

                        if current_phase != stage_norm:
                            diff = phase_order[stage_norm] - phase_order.get(current_phase, 0)
                            if diff == 1:
                                logger.warning(
                                    "Job %s stage '%s' ahead of current phase '%s' after refresh; proceeding (likely stale read)",
                                    job_id,
                                    stage_norm,
                                    current_phase,
                                )
                                # Allow execution for the next sequential stage to avoid dropping messages.
                                current_phase = stage_norm
                                is_queued_marker = True
                            else:
                                logger.warning(
                                    "Job %s stage '%s' ahead of current phase '%s' after refresh; skipping",
                                    job_id,
                                    stage_norm,
                                    current_phase,
                                )
                                return {}, existing_job.get("events", []), False, False

                # Only execute a running job when it's explicitly marked as queued for this stage.
                if stage_norm != current_phase:
                    logger.warning(
                        "Job %s stage '%s' does not match current phase '%s', skipping",
                        job_id,
                        stage_norm,
                        current_phase,
                    )
                    return {}, existing_job.get("events", []), False, False

                if not is_queued_marker:
                    logger.warning("Job %s stage '%s' already running, skipping duplicate message", job_id, stage_norm)
                    return {}, existing_job.get("events", []), False, False

                logger.info("Job %s stage '%s' is queued; allowing execution", job_id, stage_norm)
    
    events: list[dict[str, Any]] = existing_job.get("events", []) if existing_job else []
    events = append_event(events, "job_start", "init", f"Starting {job_type} job")
    update_job_progress(job_id, "running", "init", 0, "Initializing...", events=events)

    load_openai_api_key()

    if job_type == "pipeline":
        from papernavigator.openai_usage import merge_openai_usage

        def _merge_results(existing: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any]:
            base: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
            upd: dict[str, Any] = dict(update) if isinstance(update, dict) else {}

            for key, value in upd.items():
                if key == "openai_usage":
                    base["openai_usage"] = merge_openai_usage(base.get("openai_usage"), value)
                    continue
                if key == "artifacts":
                    existing_list = base.get("artifacts")
                    ex = existing_list if isinstance(existing_list, list) else []
                    nv = value if isinstance(value, list) else []
                    base["artifacts"] = list(dict.fromkeys([*ex, *nv]))
                    continue
                if isinstance(base.get(key), dict) and isinstance(value, dict):
                    base[key] = {**base[key], **value}
                    continue
                base[key] = value

            return base

        stage = payload.get("stage") or "search"
        if stage == "search":
            result = asyncio.run(run_search_job(job_id, payload, events))
            if result.get("papers_found", 0) <= 0:
                message = "Search produced 0 papers; cannot continue to ranking/report."
                current_events = (get_job(job_id) or {}).get("events", []) or events
                current_events = append_event(
                    current_events,
                    "job_failed",
                    "search",
                    message,
                )
                update_job_progress(
                    job_id,
                    "failed",
                    "search",
                    0,
                    message,
                    events=current_events,
                    result=result,
                    error=message,
                )
                return result, current_events, True, True
            next_payload = dict(payload)
            next_payload["stage"] = "ranking"
            refreshed = get_job(job_id)
            merged_result = _merge_results((refreshed or {}).get("result"), result)
            current_events = (refreshed or {}).get("events", []) or events
            current_events = append_event(
                current_events,
                "progress",
                "ranking",
                "Queued ranking stage",
                step=0,
                step_name="Queued",
            )
            update_job_progress(
                job_id,
                "running",
                "ranking",
                0,
                "Queued ranking stage",
                step_name="Queued",
                events=current_events,
                result=merged_result,
            )
            # Enqueue next stage after updating progress to avoid races with the next message.
            enqueue_job(job_id, "pipeline", next_payload)
            return merged_result, current_events, True, False
        if stage == "ranking":
            result = asyncio.run(run_ranking_stage(job_id, payload, events))
            next_payload = dict(payload)
            next_payload["stage"] = "report"
            refreshed = get_job(job_id)
            merged_result = _merge_results((refreshed or {}).get("result"), result)
            current_events = (refreshed or {}).get("events", []) or events
            current_events = append_event(
                current_events,
                "progress",
                "report",
                "Queued report stage",
                step=0,
                step_name="Queued",
            )
            update_job_progress(
                job_id,
                "running",
                "report",
                0,
                "Queued report stage",
                step_name="Queued",
                events=current_events,
                result=merged_result,
            )
            # Enqueue next stage after updating progress to avoid races with the next message.
            enqueue_job(job_id, "pipeline", next_payload)
            return merged_result, current_events, True, False
        if stage == "report":
            result = asyncio.run(run_report_stage(job_id, payload, events))
            refreshed = get_job(job_id)
            merged_result = _merge_results((refreshed or {}).get("result"), result)
            return merged_result, events, True, True
        raise ValueError(f"Unknown pipeline stage: {stage}")
    if job_type == "search":
        result = asyncio.run(run_search_job(job_id, payload, events))
        return result, events, True, True

    raise ValueError(f"Unknown job type: {job_type}")


@bp.service_bus_queue_trigger(
    arg_name="msg",
    queue_name=QUEUE_NAME,
    connection="AZURE_SERVICE_BUS_CONNECTION_STRING",
)
def process_job_message(msg: func.ServiceBusMessage):
    body = msg.get_body().decode("utf-8")
    try:
        from datetime import UTC, datetime

        enqueued_time = getattr(msg, "enqueued_time_utc", None) or getattr(msg, "enqueued_time", None)
        if enqueued_time:
            latency_ms = (datetime.now(UTC) - enqueued_time).total_seconds() * 1000.0
            logger.info(
                "Processing message (latency_ms=%.0f, message_id=%s): %s",
                latency_ms,
                getattr(msg, "message_id", None),
                body,
            )
        else:
            logger.info("Processing message: %s", body)
    except Exception:
        logger.info("Processing message: %s", body)

    job_id = None
    try:
        payload = json.loads(body)
        job_id = payload.get("job_id")
        job_type = payload.get("job_type")
        job_payload = payload.get("payload") or {}

        if not job_id or not job_type:
            logger.error("Invalid message: missing job_id or job_type")
            return

        result, events, was_processed, is_final = process_job(job_id, job_type, job_payload)
        
        # Only update status if job was actually processed (not skipped due to idempotency)
        if was_processed and is_final:
            events = append_event(events, "job_complete", "complete", "Job completed")
            update_job_progress(
                job_id,
                "completed",
                "complete",
                0,
                "Job completed",
                events=events,
                result=result,
            )

            # Send completion email notification if requested
            notification_email = job_payload.get("notification_email")
            if notification_email:
                query = job_payload.get("query", "")
                email_sent = send_completion_email(notification_email, query, job_id, result)
                if email_sent:
                    events = append_event(events, "email_sent", "complete", f"Notification sent to {notification_email}")
                    update_job_progress(job_id, "completed", "complete", 0, "Job completed", events=events, result=result)
        else:
            logger.info("Job %s was skipped (idempotency), not updating status", job_id)

    except Exception as exc:
        logger.exception("Error processing message for job %s: %s", job_id, exc)
        if job_id:
            existing_job = get_job(job_id)
            existing_events = existing_job.get("events", []) if existing_job else []

            try:
                from papernavigator.openai_usage import OpenAIInsufficientFundsError
                error_code = OpenAIInsufficientFundsError.error_code if isinstance(exc, OpenAIInsufficientFundsError) else None
            except Exception:
                error_code = None

            events = append_event(
                existing_events,
                "job_failed",
                "error",
                f"Job failed: {exc}",
            )
            update_job_progress(
                job_id,
                "failed",
                "error",
                0,
                f"Job failed: {exc}",
                events=events,
                error=str(exc),
                error_code=error_code,
            )

            # Send failure email notification if requested
            notification_email = job_payload.get("notification_email") if job_payload else None
            if notification_email:
                query = job_payload.get("query", "") if job_payload else ""
                send_failure_email(notification_email, query, job_id, str(exc))
        raise


@bp.service_bus_queue_trigger(
    arg_name="msg",
    queue_name=f"{QUEUE_NAME}/$DeadLetterQueue",
    connection="AZURE_SERVICE_BUS_CONNECTION_STRING",
)
def process_deadletter_message(msg: func.ServiceBusMessage):
    """Mark jobs as failed when their messages are dead-lettered."""
    try:
        body = msg.get_body().decode("utf-8")
        payload = json.loads(body)
    except Exception as exc:
        logger.exception("Failed to parse dead-letter message body: %s", exc)
        return

    job_id = payload.get("job_id")
    if not job_id:
        logger.warning("Dead-letter message missing job_id: %s", payload)
        return

    reason, description = _extract_dead_letter_details(msg)
    _mark_job_failed_from_dlq(job_id, reason, description, getattr(msg, "message_id", None))
