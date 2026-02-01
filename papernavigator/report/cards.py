"""Paper card generation using LLM.

This module extracts structured paper cards from raw paper data using async
LLM calls with batch processing and rate limiting.
"""

import asyncio
import json
import os
import time
from collections.abc import Callable

from openai import AsyncOpenAI

from papernavigator.async_utils import get_loop_semaphore, validate_loop
from papernavigator.logging import get_logger
from papernavigator.openai_usage import OpenAIInsufficientFundsError, record_openai_response, raise_if_openai_insufficient_funds
from papernavigator.report.models import PaperCard

log = get_logger(__name__)

# Async OpenAI client
_async_client: AsyncOpenAI | None = None

# Concurrency limits
OPENAI_MAX_CONCURRENT = 20

# Timeout for OpenAI API calls (seconds)
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
# OpenAI retry attempts for transient errors (429/5xx)
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))


def _get_async_client() -> AsyncOpenAI:
    """Get or create the async OpenAI client."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )
    return _async_client


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the concurrency semaphore."""
    semaphore = get_loop_semaphore("report_cards", OPENAI_MAX_CONCURRENT)
    validate_loop(semaphore, "report_cards_semaphore")
    return semaphore


def _build_card_prompt(paper: dict) -> str:
    """Build the prompt for extracting a paper card."""
    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "") or "(No abstract available)"

    return f"""Given this academic paper's title and abstract, extract structured information.

Title: {title}

Abstract: {abstract}

Extract the following as JSON:
1. "claim": One sentence (max 30 words) describing the paper's main contribution or finding
2. "paradigm_tags": List of 2-4 research paradigm tags from this set: 
   [prompting, fine-tuning, retrieval, generative, evaluation, benchmark, 
    fairness, efficiency, multimodal, reasoning, agents, knowledge, 
    representation, training, inference, application, survey, dataset]
3. "data_benchmark": The primary dataset or benchmark used (null if not clearly stated)
4. "measured": What metrics or outcomes are measured/evaluated (null if not clearly stated)  
5. "limitation": One key limitation mentioned by the authors (null if not stated)
6. "key_quote": One important phrase (5-15 words) from the abstract that captures the essence

Return ONLY valid JSON with these exact keys. Use null for missing values."""


async def generate_paper_card(paper: dict) -> PaperCard | None:
    """Extract a structured card from a single paper using LLM.
    
    Args:
        paper: Paper dictionary with title, abstract, paper_id, etc.
        
    Returns:
        PaperCard if successful, None if extraction fails
    """
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "Unknown")[:50]

    log.debug("generating_card", paper_id=paper_id, title=title)

    prompt = _build_card_prompt(paper)
    client = _get_async_client()
    semaphore = _get_semaphore()
    start_time = time.monotonic()
    log.info("openai_request_start", operation="generate_card", paper_id=paper_id, model="gpt-4o-mini")

    try:
        async with semaphore:
            # Wrap API call with timeout to prevent indefinite hangs
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                    timeout=OPENAI_TIMEOUT_SECONDS,
                ),
                timeout=OPENAI_TIMEOUT_SECONDS
            )

        record_openai_response(response, model="gpt-4o-mini")
        content = response.choices[0].message.content.strip()
        log.info(
            "openai_request_complete",
            operation="generate_card",
            paper_id=paper_id,
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        data = json.loads(content)

        # Build the PaperCard with extracted data + original metadata
        card = PaperCard(
            id=paper_id,
            title=paper.get("title", "Unknown"),
            claim=data.get("claim", "No claim extracted"),
            paradigm_tags=data.get("paradigm_tags", []),
            data_benchmark=data.get("data_benchmark"),
            measured=data.get("measured"),
            limitation=data.get("limitation"),
            key_quote=data.get("key_quote"),
            year=paper.get("year"),
            citation_count=paper.get("citation_count", 0),
            elo_rating=paper.get("elo_rating"),
        )

        log.info("card_generated", paper_id=paper_id, tags=card.paradigm_tags)
        return card

    except asyncio.TimeoutError:
        log.error("card_generation_timeout", paper_id=paper_id, timeout_sec=OPENAI_TIMEOUT_SECONDS)
        log.info(
            "openai_request_timeout",
            operation="generate_card",
            paper_id=paper_id,
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        return None
    except json.JSONDecodeError as e:
        log.error("card_json_parse_error", paper_id=paper_id, error=str(e))
        log.info(
            "openai_request_failed",
            operation="generate_card",
            paper_id=paper_id,
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        return None
    except OpenAIInsufficientFundsError:
        raise
    except Exception as e:
        raise_if_openai_insufficient_funds(e)
        log.error("card_generation_failed", paper_id=paper_id, error=str(e))
        log.info(
            "openai_request_failed",
            operation="generate_card",
            paper_id=paper_id,
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        return None


async def generate_paper_cards(
    papers: list[dict],
    progress_callback: Callable[[int, int, str], None] | None = None
) -> list[PaperCard]:
    """Generate paper cards for a batch of papers concurrently.
    
    Args:
        papers: List of paper dictionaries
        progress_callback: Optional callback(current, total, message) for progress updates
        
    Returns:
        List of successfully generated PaperCards
    """
    if not papers:
        log.warning("no_papers_to_process")
        return []

    log.info("batch_card_generation_start", total_papers=len(papers))
    start_time = time.time()

    if progress_callback:
        progress_callback(0, len(papers), "Starting paper card generation...")

    # Process all papers concurrently
    tasks = [generate_paper_card(paper) for paper in papers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, OpenAIInsufficientFundsError):
            raise result

    # Filter successful results
    cards: list[PaperCard] = []
    failed = 0

    for i, result in enumerate(results):
        if isinstance(result, PaperCard):
            cards.append(result)
            if progress_callback:
                progress_callback(len(cards), len(papers), f"Generated card {len(cards)} of {len(papers)}")
        elif isinstance(result, Exception):
            log.error("card_task_exception", error=str(result))
            failed += 1
            if progress_callback:
                progress_callback(len(cards), len(papers), f"Processing... ({len(cards)}/{len(papers)} successful)")
        else:
            # None result from failed extraction
            failed += 1
            if progress_callback:
                progress_callback(len(cards), len(papers), f"Processing... ({len(cards)}/{len(papers)} successful)")

    duration = time.time() - start_time
    log.info(
        "batch_card_generation_complete",
        success=len(cards),
        failed=failed,
        duration_sec=round(duration, 2)
    )

    return cards
