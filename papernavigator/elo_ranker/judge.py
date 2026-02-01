"""LLM-based match judging with relevance-first prompts."""

import asyncio
import json
import os
import time

from openai import AsyncOpenAI

from papernavigator.logging import get_logger
from papernavigator.elo_ranker.models import MatchResult
from papernavigator.models import QueryProfile, SnowballCandidate
from papernavigator.openai_usage import OpenAIInsufficientFundsError, record_openai_response, raise_if_openai_insufficient_funds

# Timeout for OpenAI API calls (seconds)
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
# OpenAI retry attempts for transient errors (429/5xx)
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
# Limit abstract length (chars) for prompt efficiency
ABSTRACT_CHAR_LIMIT = int(os.getenv("RANKER_ABSTRACT_CHAR_LIMIT", "500"))

# Async OpenAI client
async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=OPENAI_TIMEOUT_SECONDS,
    max_retries=OPENAI_MAX_RETRIES,
)

log = get_logger(__name__)


async def judge_match(
    candidate1: SnowballCandidate,
    candidate2: SnowballCandidate,
    profile: QueryProfile
) -> tuple[int | None, str]:
    """Judge a single match between two candidates.
    
    Uses a relevance-first prompt that prioritizes citation usefulness
    over general quality/significance.
    
    Args:
        candidate1: First candidate paper
        candidate2: Second candidate paper
        profile: Query profile for relevance judgment
        
    Returns:
        Tuple of (winner, reason) where winner is 1, 2, or None for draw
    """
    # Build relevance-first prompt
    required_concepts_str = ", ".join(profile.required_concepts) if profile.required_concepts else "None"
    optional_concepts_str = ", ".join(profile.optional_concepts) if profile.optional_concepts else "None"

    abstract1 = candidate1.abstract or "(No abstract available)"
    abstract2 = candidate2.abstract or "(No abstract available)"

    prompt = f"""Decide which paper is MORE USEFUL TO CITE for:
"{profile.core_query}"

Domain: {profile.domain_description}
Required: {required_concepts_str}
Optional: {optional_concepts_str}

Priority:
1) Relevance to query.
2) If tied, prefer clearer method/evaluation.
Do NOT prefer by citation count, fame, or broadness.

Paper A: {candidate1.title}
Abstract: {abstract1[:ABSTRACT_CHAR_LIMIT]}

Paper B: {candidate2.title}
Abstract: {abstract2[:ABSTRACT_CHAR_LIMIT]}

Return JSON only: {{"winner":1|2|0, "reason":"max 20 words"}}"""

    try:
        # Wrap API call with timeout to prevent indefinite hangs
        start_time = time.monotonic()
        log.info(
            "openai_request_start",
            operation="ranker_judge_match",
            paper_a=candidate1.paper_id,
            paper_b=candidate2.paper_id,
            model="gpt-4o-mini",
        )
        response = await asyncio.wait_for(
            async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
                response_format={"type": "json_object"}
            ),
            timeout=OPENAI_TIMEOUT_SECONDS
        )

        record_openai_response(response, model="gpt-4o-mini")
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        log.info(
            "openai_request_complete",
            operation="ranker_judge_match",
            paper_a=candidate1.paper_id,
            paper_b=candidate2.paper_id,
            duration_sec=round(time.monotonic() - start_time, 2),
        )

        winner = data.get("winner")
        reason = data.get("reason", "")

        if winner == 1:
            return 1, reason
        elif winner == 2:
            return 2, reason
        elif winner == 0:
            return None, reason  # Draw
        else:
            # Invalid response, treat as draw
            return None, "Invalid response"

    except asyncio.TimeoutError:
        log.info(
            "openai_request_timeout",
            operation="ranker_judge_match",
            paper_a=candidate1.paper_id,
            paper_b=candidate2.paper_id,
        )
        # On timeout, treat as draw to avoid blocking the pipeline
        return None, "Timeout"
    except OpenAIInsufficientFundsError:
        raise
    except (json.JSONDecodeError, KeyError, AttributeError):
        log.info(
            "openai_request_failed",
            operation="ranker_judge_match",
            paper_a=candidate1.paper_id,
            paper_b=candidate2.paper_id,
        )
        # On error, treat as draw to avoid skewing ratings
        return None, "Parse error"
    except Exception as exc:
        raise_if_openai_insufficient_funds(exc)
        log.info(
            "openai_request_failed",
            operation="ranker_judge_match",
            paper_a=candidate1.paper_id,
            paper_b=candidate2.paper_id,
        )
        # On any other error, treat as draw
        return None, "API error"


async def judge_match_batch(
    pairs: list[tuple[SnowballCandidate, SnowballCandidate]],
    profile: QueryProfile,
    concurrency: int = 5
) -> list[MatchResult]:
    """Judge multiple matches concurrently.
    
    Args:
        pairs: List of (candidate1, candidate2) tuples
        profile: Query profile for relevance judgment
        concurrency: Maximum number of concurrent API calls
        
    Returns:
        List of MatchResult objects
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(pair: tuple[SnowballCandidate, SnowballCandidate]) -> MatchResult:
        async with semaphore:
            candidate1, candidate2 = pair
            winner, reason = await judge_match(candidate1, candidate2, profile)
            return MatchResult(
                paper1_title=candidate1.title,
                paper2_title=candidate2.title,
                winner=winner,
                reason=reason
            )

    results = await asyncio.gather(*[judge_one(pair) for pair in pairs])
    return list(results)
