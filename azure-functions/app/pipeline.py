"""Pipeline execution and artifact upload helpers."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .clients import get_results_container_client
from .config import RESULTS_CONTAINER, REPORT_TIMEOUT_SECONDS, logger
from .jobs import append_event, get_job, update_job_progress
from .results import download_blob_to_path, get_blob_json, results_path
from .utils import now_iso, slugify


def upload_artifacts_to_blob(local_dir: Path, prefix: str) -> list[dict[str, Any]]:
    from azure.storage.blob import ContentSettings

    container = get_results_container_client()
    artifacts: list[dict[str, Any]] = []

    for file_path in local_dir.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{relative_path}"

        content_type = "application/json"
        if file_path.suffix == ".html":
            content_type = "text/html"
        elif file_path.suffix == ".txt":
            content_type = "text/plain"

        with open(file_path, "rb") as data:
            container.upload_blob(
                name=blob_name,
                data=data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )

        artifacts.append({
            "name": blob_name,
            "size": file_path.stat().st_size,
            "content_type": content_type,
        })

    return artifacts


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


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


async def run_pipeline(job_id: str, payload: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    from papernavigator.elo_ranker import EloRanker, RankerConfig
    from papernavigator.events import NullEventHandler
    from papernavigator.models import SnowballCandidate
    from papernavigator.profiler import generate_query_profile
    from papernavigator.report.generator import generate_report, report_to_dict, final_citation_check
    from papernavigator.service import run_search

    existing_job = get_job(job_id) or {}
    result_state: dict[str, Any] = {}
    if isinstance(existing_job.get("result"), dict):
        result_state.update(existing_job["result"])

    class RankingProgressHandler(NullEventHandler):
        def __init__(self, job_id: str, update_every: int = 1, top_k: int = 5) -> None:
            self.job_id = job_id
            self.update_every = max(1, update_every)
            self.top_k = top_k

        def on_elo_update(self, candidates, match_num: int, total_matches: int, **kwargs: Any) -> None:
            nonlocal events, result_state
            if match_num % self.update_every != 0 and match_num != total_matches:
                return

            leaderboard = sorted(candidates, key=lambda c: c.elo, reverse=True)[: self.top_k]
            top_papers = [
                {
                    "paper_id": c.candidate.paper_id,
                    "title": c.candidate.title,
                    "elo": round(c.elo, 1),
                    "wins": c.wins,
                    "losses": c.losses,
                }
                for c in leaderboard
            ]

            msg = f"Ranking match {match_num} / {total_matches}"
            events = append_event(
                events,
                "progress",
                "ranking",
                msg,
                step=1,
                step_name="Ranking Papers",
                current=match_num,
                total=total_matches,
            )
            result_state.update({
                "top_papers": top_papers,
                "matches_played": match_num,
            })
            update_job_progress(
                self.job_id,
                "running",
                "ranking",
                1,
                msg,
                current=match_num,
                total=total_matches,
                step_name="Ranking Papers",
                result={
                    **result_state,
                },
                events=events,
            )

    query = payload.get("query", "")
    num_results = payload.get("num_results", 15)
    max_iterations = payload.get("max_iterations", 5)
    max_accepted = payload.get("max_accepted", 200)
    top_n = payload.get("top_n", 50)
    k_factor = payload.get("k_factor", 32.0)
    pairing = payload.get("pairing", "swiss")
    early_stop = payload.get("early_stop", True)
    elo_concurrency = payload.get("elo_concurrency", 5)
    report_top_k = payload.get("report_top_k", 30)

    query_slug = slugify(query)

    workspace = Path(tempfile.mkdtemp(prefix=f"papernavigator_{job_id}_"))
    results_dir = workspace / query_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Phase 1: Search
        events = append_event(events, "phase_start", "search", "Starting search phase")
        update_job_progress(job_id, "running", "search", 0, "Starting search...", events=events)

        snowball_path = results_dir / "snowball.json"

        def search_progress_callback(step, step_name, current, total, message, curr_iter, total_iter):
            nonlocal events
            events = append_event(events, "progress", "search", message, step=step, step_name=step_name)
            update_job_progress(
                job_id,
                "running",
                "search",
                step,
                message,
                current=current,
                total=total,
                step_name=step_name,
                events=events,
            )

        try:
            accepted_papers = await run_search(
                query=query,
                num_results=num_results,
                output_file=str(snowball_path),
                max_iterations=max_iterations,
                max_accepted=max_accepted,
                top_n=top_n,
                progress_callback=search_progress_callback,
            )
        except Exception as exc:
            logger.exception("Search phase failed for job %s", job_id)
            events = append_event(
                events,
                "phase_error",
                "search",
                f"Search phase failed: {exc}",
                level="error",
                error=str(exc),
            )
            update_job_progress(
                job_id,
                "failed",
                "search",
                0,
                f"Search failed: {exc}",
                events=events,
                error=str(exc),
            )
            raise

        events = append_event(
            events,
            "phase_complete",
            "search",
            f"Search complete: found {len(accepted_papers)} papers",
        )
        update_job_progress(
            job_id,
            "running",
            "search",
            6,
            f"Search complete: {len(accepted_papers)} papers",
            events=events,
        )

        if not accepted_papers:
            message = "No papers found during search"
            events = append_event(events, "phase_error", "search", message, level="error")
            update_job_progress(job_id, "failed", "search", 6, message, events=events, error=message)
            raise ValueError(message)

        # Phase 2: Ranking
        events = append_event(events, "phase_start", "ranking", "Starting ELO ranking phase")

        candidates = [
            SnowballCandidate(
                paper_id=p.paper_id,
                title=p.title,
                abstract=p.abstract,
                year=p.year,
                citation_count=p.citation_count,
                influential_citation_count=0,
                discovered_from=p.discovered_from,
                edge_type=p.edge_type,
                depth=p.depth,
            )
            for p in accepted_papers
        ]

        ranker_config = RankerConfig(
            k_factor=k_factor,
            pairing_strategy=pairing,
            early_stop_enabled=early_stop,
            batch_size=elo_concurrency,
            concurrency=elo_concurrency,
            interactive=False,
        )
        expected_matches = ranker_config.max_matches or (len(candidates) * 3)
        update_job_progress(
            job_id,
            "running",
            "ranking",
            0,
            "Starting ranking...",
            current=0,
            total=expected_matches,
            step_name="Ranking Papers",
            events=events,
        )

        ranking_handler = RankingProgressHandler(
            job_id,
            update_every=elo_concurrency,
            top_k=5,
        )
        try:
            profile = await generate_query_profile(query)
            ranker = EloRanker(profile, candidates, ranker_config, event_handler=ranking_handler)
            heartbeat_seconds = int(os.getenv("RANKING_HEARTBEAT_SECONDS", "20"))
            heartbeat_seconds = max(5, min(120, heartbeat_seconds))

            ranking_task = asyncio.create_task(ranker.rank_candidates())
            last_heartbeat = time.monotonic()
            while not ranking_task.done():
                await asyncio.sleep(1)
                if time.monotonic() - last_heartbeat < heartbeat_seconds:
                    continue
                current = int(result_state.get("matches_played") or 0)
                msg = f"Ranking in progress... ({current} / {expected_matches})"
                events = append_event(
                    events,
                    "progress",
                    "ranking",
                    msg,
                    step=1,
                    step_name="Ranking Papers",
                    current=current,
                    total=expected_matches,
                )
                update_job_progress(
                    job_id,
                    "running",
                    "ranking",
                    1,
                    msg,
                    current=current,
                    total=expected_matches,
                    step_name="Ranking Papers",
                    events=events,
                    result={**result_state},
                )
                last_heartbeat = time.monotonic()

            ranked_candidates = await ranking_task
        except Exception as exc:
            logger.exception("Ranking phase failed for job %s", job_id)
            events = append_event(
                events,
                "phase_error",
                "ranking",
                f"Ranking phase failed: {exc}",
                level="error",
                error=str(exc),
            )
            update_job_progress(
                job_id,
                "failed",
                "ranking",
                0,
                f"Ranking failed: {exc}",
                events=events,
                error=str(exc),
            )
            raise

        elo_path = results_dir / f"elo_ranked_k{int(k_factor)}_p{pairing}.json"
        elo_data = {
            "query": query,
            "total_ranked": len(ranked_candidates),
            "matches_played": len(ranker.match_history),
            "papers": [
                {
                    "paper_id": c.candidate.paper_id,
                    "title": c.candidate.title,
                    "abstract": (c.candidate.abstract[:500] if c.candidate.abstract else None),
                    "year": c.candidate.year,
                    "citation_count": c.candidate.citation_count,
                    "elo_rating": round(c.elo, 1),
                    "wins": c.wins,
                    "losses": c.losses,
                    "draws": c.draws,
                }
                for c in ranked_candidates
            ],
        }
        with open(elo_path, "w", encoding="utf-8") as f:
            json.dump(elo_data, f, indent=2)

        events = append_event(
            events,
            "phase_complete",
            "ranking",
            f"Ranking complete: {len(ranker.match_history)} matches played",
        )
        update_job_progress(
            job_id,
            "running",
            "ranking",
            1,
            f"Ranking complete: {len(ranker.match_history)} matches",
            current=len(ranker.match_history),
            total=len(ranker.match_history),
            step_name="Ranking Papers",
            events=events,
        )

        # Phase 3: Report
        events = append_event(events, "phase_start", "report", "Starting report generation")
        update_job_progress(job_id, "running", "report", 0, "Starting report...", events=events)

        def report_progress_callback(step, step_name, current, total, message):
            nonlocal events
            level = None
            clean_message = message
            if message.startswith("WARNING:"):
                level = "warning"
                clean_message = message[len("WARNING:"):].strip()
            elif message.startswith("ERROR:"):
                level = "error"
                clean_message = message[len("ERROR:"):].strip()

            events = append_event(
                events,
                "progress",
                "report",
                clean_message,
                step=step,
                step_name=step_name,
                level=level,
            )
            update_job_progress(
                job_id,
                "running",
                "report",
                step,
                clean_message,
                current=current,
                total=total,
                step_name=step_name,
                events=events,
            )

        try:
            report = await asyncio.wait_for(
                generate_report(
                    snowball_file=snowball_path,
                    elo_file=elo_path,
                    top_k=report_top_k,
                    progress_callback=report_progress_callback,
                ),
                timeout=REPORT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            message = f"Report generation timed out after {REPORT_TIMEOUT_SECONDS}s"
            logger.error(message)
            events = append_event(
                events,
                "phase_error",
                "report",
                message,
                level="error",
                timeout_sec=REPORT_TIMEOUT_SECONDS,
            )
            update_job_progress(job_id, "failed", "report", 4, message, events=events, error=message)
            raise
        except Exception as exc:
            logger.exception("Report generation failed for job %s", job_id)
            events = append_event(
                events,
                "phase_error",
                "report",
                f"Report generation failed: {exc}",
                level="error",
                error=str(exc),
            )
            update_job_progress(
                job_id,
                "failed",
                "report",
                4,
                f"Report generation failed: {exc}",
                events=events,
                error=str(exc),
            )
            raise

        # Collect citation warnings for UI visibility (no report mutation).
        _, citation_warnings = final_citation_check(report)
        if citation_warnings:
            events = append_event(
                events,
                "phase_warning",
                "report",
                f"Final citation check raised {len(citation_warnings)} warnings",
                level="warning",
                warnings=citation_warnings[:5],
            )

        report_path = results_dir / f"report_top_k{report_top_k}.json"
        report_dict = report_to_dict(report)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        events = append_event(events, "phase_complete", "report", "Report generation complete")

        # Metadata
        metadata = {
            "snowball_file": "snowball.json",
            "snowball_count": len(accepted_papers),
            "created_at": now_iso(),
            "last_updated": now_iso(),
            "query": query,
            "elo_file": elo_path.name,
            "elo_matches": len(ranker.match_history),
            "elo_papers": len(ranked_candidates),
            "report_file": report_path.name,
            "report_papers_used": report_top_k,
            "report_sections": len(report.current_research),
            "report_generated_at": report.generated_at,
        }
        metadata_path = results_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Upload
        events = append_event(events, "phase_start", "upload", "Uploading artifacts to Blob")
        update_job_progress(job_id, "running", "upload", 0, "Uploading to Blob...", events=events)

        blob_prefix = results_path(query_slug, job_id)
        try:
            artifacts = upload_artifacts_to_blob(results_dir, blob_prefix)
        except Exception as exc:
            logger.exception("Upload phase failed for job %s", job_id)
            events = append_event(
                events,
                "phase_error",
                "upload",
                f"Upload failed: {exc}",
                level="error",
                error=str(exc),
            )
            update_job_progress(job_id, "failed", "upload", 0, f"Upload failed: {exc}", events=events, error=str(exc))
            raise
        events = append_event(events, "phase_complete", "upload", f"Uploaded {len(artifacts)} files")

        artifact_bytes_total = sum(
            a.get("size", 0) for a in artifacts if isinstance(a, dict) and isinstance(a.get("size"), int)
        )
        phase_durations_sec = _phase_durations_from_events(events)

        result = {
            "papers_found": len(accepted_papers),
            "papers_ranked": len(ranked_candidates),
            "matches_played": len(ranker.match_history),
            "report_sections": len(report.current_research),
            "results_container": RESULTS_CONTAINER,
            "results_prefix": blob_prefix,
            "artifacts": [a["name"] for a in artifacts],
            "artifact_count": len(artifacts),
            "artifact_bytes_total": artifact_bytes_total,
            "phase_durations_sec": phase_durations_sec,
            "top_papers": [
                {
                    "paper_id": c.candidate.paper_id,
                    "title": c.candidate.title,
                    "elo": round(c.elo, 1),
                    "wins": c.wins,
                    "losses": c.losses,
                }
                for c in ranked_candidates[:5]
            ],
        }

        return result

    finally:
        try:
            shutil.rmtree(workspace)
        except Exception as exc:
            logger.warning("Failed to cleanup workspace %s: %s", workspace, exc)


async def run_ranking_stage(job_id: str, payload: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Run only the ranking stage using existing search artifacts from blob."""
    from papernavigator.elo_ranker import EloRanker, RankerConfig
    from papernavigator.events import NullEventHandler
    from papernavigator.models import SnowballCandidate
    from papernavigator.openai_usage import OpenAIInsufficientFundsError, get_openai_usage_snapshot, start_openai_usage_tracking
    from papernavigator.profiler import generate_query_profile
    from papernavigator.report.generator import load_papers_from_file

    k_factor = payload.get("k_factor", 32.0)
    pairing = payload.get("pairing", "swiss")
    early_stop = payload.get("early_stop", True)
    elo_concurrency = payload.get("elo_concurrency", 5)

    query = payload.get("query", "")
    query_slug = slugify(query)

    workspace = Path(tempfile.mkdtemp(prefix=f"papernavigator_{job_id}_"))
    results_dir = workspace / query_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_openai_usage_tracking()

        existing_job = get_job(job_id) or {}
        result_state: dict[str, Any] = {}
        if isinstance(existing_job.get("result"), dict):
            result_state.update(existing_job["result"])

        metadata_blob = results_path(query_slug, job_id, "metadata.json")
        metadata = get_blob_json(metadata_blob) or {}
        if metadata.get("snowball_count", None) == 0:
            raise ValueError("Search produced 0 papers; cannot run ranking.")

        snowball_blob = results_path(query_slug, job_id, "snowball.json")
        snowball_path = results_dir / "snowball.json"
        if not download_blob_to_path(snowball_blob, snowball_path):
            raise FileNotFoundError(f"Missing snowball blob: {snowball_blob}")

        events = append_event(events, "phase_start", "ranking", "Starting ELO ranking phase")
        update_job_progress(
            job_id,
            "running",
            "ranking",
            0,
            "Starting ranking...",
            step_name="Ranking Papers",
            events=events,
        )

        papers, query = load_papers_from_file(snowball_path)
        candidates = [
            SnowballCandidate(
                paper_id=p.get("paper_id", ""),
                title=p.get("title", ""),
                abstract=p.get("abstract"),
                year=p.get("year"),
                citation_count=p.get("citation_count", 0),
                influential_citation_count=0,
                discovered_from=p.get("discovered_from"),
                edge_type=p.get("edge_type"),
                depth=p.get("depth", 0),
            )
            for p in papers
        ]

        ranker_config = RankerConfig(
            k_factor=k_factor,
            pairing_strategy=pairing,
            early_stop_enabled=early_stop,
            batch_size=elo_concurrency,
            concurrency=elo_concurrency,
            interactive=False,
        )
        expected_matches = ranker_config.max_matches or (len(candidates) * 3)
        update_job_progress(
            job_id,
            "running",
            "ranking",
            0,
            "Starting ranking...",
            current=0,
            total=expected_matches,
            step_name="Ranking Papers",
            events=events,
        )

        class RankingProgressHandler(NullEventHandler):
            def __init__(self, job_id: str, update_every: int = 1, top_k: int = 5) -> None:
                self.job_id = job_id
                self.update_every = max(1, update_every)
                self.top_k = top_k

            def on_elo_update(self, candidates, match_num: int, total_matches: int, **kwargs: Any) -> None:
                nonlocal events, result_state
                if match_num % self.update_every != 0 and match_num != total_matches:
                    return

                leaderboard = sorted(candidates, key=lambda c: c.elo, reverse=True)[: self.top_k]
                top_papers = [
                    {
                        "paper_id": c.candidate.paper_id,
                        "title": c.candidate.title,
                        "elo": round(c.elo, 1),
                        "wins": c.wins,
                        "losses": c.losses,
                    }
                    for c in leaderboard
                ]

                msg = f"Ranking match {match_num} / {total_matches}"
                events = append_event(
                    events,
                    "progress",
                    "ranking",
                    msg,
                    step=1,
                    step_name="Ranking Papers",
                    current=match_num,
                    total=total_matches,
                )
                result_state.update({
                    "top_papers": top_papers,
                    "matches_played": match_num,
                })
                update_job_progress(
                    self.job_id,
                    "running",
                    "ranking",
                    1,
                    msg,
                    current=match_num,
                    total=total_matches,
                    step_name="Ranking Papers",
                    events=events,
                    result={**result_state},
                )

        profile = await generate_query_profile(query)
        # Emit interim progress/leaderboard updates so the UI doesn't appear stuck on "Queued".
        ranking_handler = RankingProgressHandler(
            job_id,
            update_every=elo_concurrency,
            top_k=5,
        )
        ranker = EloRanker(profile, candidates, ranker_config, event_handler=ranking_handler)
        heartbeat_seconds = int(os.getenv("RANKING_HEARTBEAT_SECONDS", "20"))
        heartbeat_seconds = max(5, min(120, heartbeat_seconds))

        ranking_task = asyncio.create_task(ranker.rank_candidates())
        last_heartbeat = time.monotonic()
        while not ranking_task.done():
            await asyncio.sleep(1)
            if time.monotonic() - last_heartbeat < heartbeat_seconds:
                continue
            current = int(result_state.get("matches_played") or 0)
            msg = f"Ranking in progress... ({current} / {expected_matches})"
            events = append_event(
                events,
                "progress",
                "ranking",
                msg,
                step=1,
                step_name="Ranking Papers",
                current=current,
                total=expected_matches,
            )
            update_job_progress(
                job_id,
                "running",
                "ranking",
                1,
                msg,
                current=current,
                total=expected_matches,
                step_name="Ranking Papers",
                events=events,
                result={**result_state},
            )
            last_heartbeat = time.monotonic()

        ranked_candidates = await ranking_task

        elo_path = results_dir / f"elo_ranked_k{int(k_factor)}_p{pairing}.json"
        elo_data = {
            "query": query,
            "total_ranked": len(ranked_candidates),
            "matches_played": len(ranker.match_history),
            "papers": [
                {
                    "paper_id": c.candidate.paper_id,
                    "title": c.candidate.title,
                    "abstract": (c.candidate.abstract[:500] if c.candidate.abstract else None),
                    "year": c.candidate.year,
                    "citation_count": c.candidate.citation_count,
                    "elo_rating": round(c.elo, 1),
                    "wins": c.wins,
                    "losses": c.losses,
                    "draws": c.draws,
                }
                for c in ranked_candidates
            ],
        }
        with open(elo_path, "w", encoding="utf-8") as f:
            json.dump(elo_data, f, indent=2)

        # Update metadata if present
        metadata.update({
            "elo_file": elo_path.name,
            "elo_matches": len(ranker.match_history),
            "elo_papers": len(ranked_candidates),
            "last_updated": now_iso(),
            "query": query,
        })
        metadata_path = results_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Upload updated artifacts
        blob_prefix = results_path(query_slug, job_id)
        artifacts = upload_artifacts_to_blob(results_dir, blob_prefix)

        events = append_event(
            events,
            "phase_complete",
            "ranking",
            f"Ranking complete: {len(ranker.match_history)} matches played",
        )
        artifact_bytes_total = sum(
            a.get("size", 0) for a in artifacts if isinstance(a, dict) and isinstance(a.get("size"), int)
        )
        phase_durations_sec = _phase_durations_from_events(events)
        openai_usage = get_openai_usage_snapshot()
        result_state.update({
            "papers_ranked": len(ranked_candidates),
            "matches_played": len(ranker.match_history),
            "top_papers": [
                {
                    "paper_id": c.candidate.paper_id,
                    "title": c.candidate.title,
                    "elo": round(c.elo, 1),
                    "wins": c.wins,
                    "losses": c.losses,
                }
                for c in ranked_candidates[:5]
            ],
            "results_container": RESULTS_CONTAINER,
            "results_prefix": blob_prefix,
            "artifacts": [a["name"] for a in artifacts],
            "artifact_count": len(artifacts),
            "artifact_bytes_total": artifact_bytes_total,
            "phase_durations_sec": phase_durations_sec,
            "openai_usage": openai_usage,
        })
        update_job_progress(
            job_id,
            "running",
            "ranking",
            1,
            f"Ranking complete: {len(ranker.match_history)} matches",
            current=len(ranker.match_history),
            total=len(ranker.match_history),
            step_name="Ranking Papers",
            events=events,
            result={**result_state},
        )

        return {
            "papers_ranked": len(ranked_candidates),
            "matches_played": len(ranker.match_history),
            "artifacts": [a["name"] for a in artifacts],
            "artifact_count": len(artifacts),
            "artifact_bytes_total": artifact_bytes_total,
            "phase_durations_sec": phase_durations_sec,
            "openai_usage": openai_usage,
        }
    except Exception as exc:
        openai_usage = get_openai_usage_snapshot()
        error_code = OpenAIInsufficientFundsError.error_code if isinstance(exc, OpenAIInsufficientFundsError) else None
        events = append_event(
            events,
            "phase_error",
            "ranking",
            f"Ranking stage failed: {exc}",
            level="error",
            error=str(exc),
        )
        update_job_progress(
            job_id,
            "failed",
            "ranking",
            0,
            f"Ranking failed: {exc}",
            events=events,
            result={"openai_usage": openai_usage},
            error=str(exc),
            error_code=error_code,
        )
        raise
    finally:
        try:
            shutil.rmtree(workspace)
        except Exception as exc:
            logger.warning("Failed to cleanup workspace %s: %s", workspace, exc)


async def run_report_stage(job_id: str, payload: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Run only the report stage using existing search+ranking artifacts."""
    from papernavigator.report.generator import generate_report, report_to_dict, final_citation_check
    from papernavigator.openai_usage import OpenAIInsufficientFundsError, get_openai_usage_snapshot, start_openai_usage_tracking

    query = payload.get("query", "")
    report_top_k = payload.get("report_top_k", 30)
    query_slug = slugify(query)

    workspace = Path(tempfile.mkdtemp(prefix=f"papernavigator_{job_id}_"))
    results_dir = workspace / query_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_openai_usage_tracking()

        metadata_blob = results_path(query_slug, job_id, "metadata.json")
        metadata = get_blob_json(metadata_blob) or {}
        if metadata.get("snowball_count", None) == 0:
            raise ValueError("Search produced 0 papers; cannot generate a report.")

        snowball_blob = results_path(query_slug, job_id, "snowball.json")
        snowball_path = results_dir / "snowball.json"
        if not download_blob_to_path(snowball_blob, snowball_path):
            raise FileNotFoundError(f"Missing snowball blob: {snowball_blob}")

        # Use latest elo file from metadata if available
        elo_name = metadata.get("elo_file", "elo_ranked_k32_pswiss.json")
        elo_blob = results_path(query_slug, job_id, elo_name)
        elo_path = results_dir / elo_name
        if not download_blob_to_path(elo_blob, elo_path):
            raise FileNotFoundError(f"Missing elo blob: {elo_blob}")

        events = append_event(events, "phase_start", "report", "Starting report generation")
        update_job_progress(job_id, "running", "report", 0, "Starting report...", events=events)

        def report_progress_callback(step, step_name, current, total, message):
            nonlocal events
            level = None
            clean_message = message
            if message.startswith("WARNING:"):
                level = "warning"
                clean_message = message[len("WARNING:"):].strip()
            elif message.startswith("ERROR:"):
                level = "error"
                clean_message = message[len("ERROR:"):].strip()

            events = append_event(
                events,
                "progress",
                "report",
                clean_message,
                step=step,
                step_name=step_name,
                level=level,
            )
            update_job_progress(
                job_id,
                "running",
                "report",
                step,
                clean_message,
                current=current,
                total=total,
                step_name=step_name,
                events=events,
            )

        report = await asyncio.wait_for(
            generate_report(
                snowball_file=snowball_path,
                elo_file=elo_path,
                top_k=report_top_k,
                progress_callback=report_progress_callback,
            ),
            timeout=REPORT_TIMEOUT_SECONDS,
        )

        _, citation_warnings = final_citation_check(report)
        if citation_warnings:
            events = append_event(
                events,
                "phase_warning",
                "report",
                f"Final citation check raised {len(citation_warnings)} warnings",
                level="warning",
                warnings=citation_warnings[:5],
            )

        report_path = results_dir / f"report_top_k{report_top_k}.json"
        report_dict = report_to_dict(report)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        events = append_event(events, "phase_complete", "report", "Report generation complete")

        metadata.update({
            "report_file": report_path.name,
            "report_papers_used": report_top_k,
            "report_sections": len(report.current_research),
            "report_generated_at": report.generated_at,
            "last_updated": now_iso(),
            "query": report.query,
        })
        metadata_path = results_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        blob_prefix = results_path(query_slug, job_id)
        artifacts = upload_artifacts_to_blob(results_dir, blob_prefix)
        artifact_bytes_total = sum(
            a.get("size", 0) for a in artifacts if isinstance(a, dict) and isinstance(a.get("size"), int)
        )
        phase_durations_sec = _phase_durations_from_events(events)
        openai_usage = get_openai_usage_snapshot()

        return {
            "report_sections": len(report.current_research),
            "artifacts": [a["name"] for a in artifacts],
            "artifact_count": len(artifacts),
            "artifact_bytes_total": artifact_bytes_total,
            "phase_durations_sec": phase_durations_sec,
            "results_container": RESULTS_CONTAINER,
            "results_prefix": blob_prefix,
            "openai_usage": openai_usage,
        }
    except Exception as exc:
        openai_usage = get_openai_usage_snapshot()
        error_code = OpenAIInsufficientFundsError.error_code if isinstance(exc, OpenAIInsufficientFundsError) else None
        events = append_event(
            events,
            "phase_error",
            "report",
            f"Report stage failed: {exc}",
            level="error",
            error=str(exc),
        )
        update_job_progress(
            job_id,
            "failed",
            "report",
            0,
            f"Report failed: {exc}",
            events=events,
            result={"openai_usage": openai_usage},
            error=str(exc),
            error_code=error_code,
        )
        raise
    finally:
        try:
            shutil.rmtree(workspace)
        except Exception as exc:
            logger.warning("Failed to cleanup workspace %s: %s", workspace, exc)


async def run_search_job(job_id: str, payload: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    from papernavigator.service import export_results, run_search
    from papernavigator.openai_usage import OpenAIInsufficientFundsError, get_openai_usage_snapshot, start_openai_usage_tracking

    query = payload.get("query", "")
    num_results = payload.get("num_results", 15)
    max_iterations = payload.get("max_iterations", 5)
    max_accepted = payload.get("max_accepted", 200)
    top_n = payload.get("top_n", 50)

    query_slug = slugify(query)
    workspace = Path(tempfile.mkdtemp(prefix=f"papernavigator_{job_id}_"))
    results_dir = workspace / query_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_openai_usage_tracking()

        events = append_event(events, "phase_start", "search", "Starting search phase")
        update_job_progress(job_id, "running", "search", 0, "Starting search...", events=events)

        snowball_path = results_dir / "snowball.json"

        def search_progress_callback(step, step_name, current, total, message, curr_iter, total_iter):
            nonlocal events
            events = append_event(events, "progress", "search", message, step=step, step_name=step_name)
            update_job_progress(
                job_id,
                "running",
                "search",
                step,
                message,
                current=current,
                total=total,
                step_name=step_name,
                events=events,
            )

        try:
            accepted_papers = await run_search(
                query=query,
                num_results=num_results,
                output_file=str(snowball_path),
                max_iterations=max_iterations,
                max_accepted=max_accepted,
                top_n=top_n,
                progress_callback=search_progress_callback,
            )
        except Exception as exc:
            logger.exception("Search-only job failed for job %s", job_id)
            openai_usage = get_openai_usage_snapshot()
            error_code = OpenAIInsufficientFundsError.error_code if isinstance(exc, OpenAIInsufficientFundsError) else None
            events = append_event(
                events,
                "phase_error",
                "search",
                f"Search phase failed: {exc}",
                level="error",
                error=str(exc),
            )
            update_job_progress(
                job_id,
                "failed",
                "search",
                0,
                f"Search failed: {exc}",
                events=events,
                result={"openai_usage": openai_usage},
                error=str(exc),
                error_code=error_code,
            )
            raise

        # Safety net: ensure the expected artifact exists even if upstream short-circuits.
        if not snowball_path.exists():
            export_results([], query, str(snowball_path))

        events = append_event(
            events,
            "phase_complete",
            "search",
            f"Search complete: found {len(accepted_papers)} papers",
        )
        update_job_progress(
            job_id,
            "running",
            "search",
            6,
            f"Search complete: {len(accepted_papers)} papers",
            events=events,
        )

        metadata = {
            "snowball_file": "snowball.json",
            "snowball_count": len(accepted_papers),
            "created_at": now_iso(),
            "last_updated": now_iso(),
            "query": query,
        }
        metadata_path = results_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        blob_prefix = results_path(query_slug, job_id)
        artifacts = upload_artifacts_to_blob(results_dir, blob_prefix)

        artifact_bytes_total = sum(
            a.get("size", 0) for a in artifacts if isinstance(a, dict) and isinstance(a.get("size"), int)
        )
        phase_durations_sec = _phase_durations_from_events(events)
        openai_usage = get_openai_usage_snapshot()

        return {
            "papers_found": len(accepted_papers),
            "results_container": RESULTS_CONTAINER,
            "results_prefix": blob_prefix,
            "artifacts": [a["name"] for a in artifacts],
            "artifact_count": len(artifacts),
            "artifact_bytes_total": artifact_bytes_total,
            "phase_durations_sec": phase_durations_sec,
            "openai_usage": openai_usage,
        }

    finally:
        try:
            shutil.rmtree(workspace)
        except Exception as exc:
            logger.warning("Failed to cleanup workspace %s: %s", workspace, exc)
