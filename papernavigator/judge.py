"""LLM-based relevance judgment for papers (Async version).

This module provides async functions to judge paper relevance using OpenAI's API
with support for concurrent batch processing.
"""

import asyncio
import json
import os
import re

from openai import AsyncOpenAI

from papernavigator.async_utils import get_loop_semaphore, validate_loop
from papernavigator.models import (
    JudgmentResult,
    QueryProfile,
    ReducedArxivEntry,
    SnowballCandidate,
)
from papernavigator.openai_usage import OpenAIInsufficientFundsError, record_openai_response, raise_if_openai_insufficient_funds

# Async OpenAI client
async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Concurrency limits for OpenAI API
OPENAI_MAX_CONCURRENT = 50

# Timeout for OpenAI API calls (seconds)
OPENAI_TIMEOUT_SECONDS = 30


def _get_openai_semaphore() -> asyncio.Semaphore:
    """Get or create the OpenAI semaphore for rate limiting."""
    semaphore = get_loop_semaphore("openai_judge", OPENAI_MAX_CONCURRENT)
    validate_loop(semaphore, "openai_judge_semaphore")
    return semaphore


def _format_required_concept_groups(profile: QueryProfile) -> str:
    groups = profile.required_concept_groups or []
    if not groups:
        return "None specified"

    lines: list[str] = []
    for idx, group in enumerate(groups, 1):
        if not group:
            continue
        preview = ", ".join(group[:10])
        suffix = " ..." if len(group) > 10 else ""
        lines.append(f"- Group {idx}: {preview}{suffix}")
    return "\n".join(lines) if lines else "None specified"


def keyword_gate(profile: QueryProfile, title: str, summary: str, *, min_groups: int = 1) -> bool:
    """Fast pre-filter using dynamic keyword patterns from the profile.
    
    Returns True if the paper passes the keyword gate, False otherwise.
    If no patterns are defined, all papers pass.
    """
    if not profile.keyword_patterns:
        return True

    text = f"{title} {summary}"

    matches = 0
    for pattern_str in profile.keyword_patterns:
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(text):
                matches += 1
        except re.error:
            # Skip invalid patterns
            continue

    threshold = max(1, min_groups)
    return matches >= min(threshold, len(profile.keyword_patterns))


async def judge_result(
    profile: QueryProfile,
    source_query: str,
    result: ReducedArxivEntry
) -> bool:
    """Strict relevance judge using dynamic QueryProfile and JSON output.
    
    Args:
        profile: The QueryProfile with domain-specific filtering criteria
        source_query: The specific search variant that retrieved this paper
        result: The paper to evaluate
    
    Returns:
        True if the paper is relevant, False otherwise.
    """

    # 1. Cheap keyword gate (sync, no API call). Keep this permissive to avoid
    # over-filtering niche queries; the LLM does the heavy lifting.
    if not keyword_gate(profile, result.title, result.summary, min_groups=1):
        return False

    # 2. Build dynamic prompt from QueryProfile
    required_groups_str = _format_required_concept_groups(profile)
    optional_concepts_str = ", ".join(profile.optional_concepts) if profile.optional_concepts else "None specified"
    exclusion_concepts_str = ", ".join(profile.exclusion_concepts) if profile.exclusion_concepts else "None specified"

    prompt = f"""
You are a strict relevance judge for academic paper search.

You will be given:
(1) a CORE topic query (the survey topic)
(2) a SOURCE query (the specific search variant that retrieved this paper)
(3) a paper title + abstract/summary

Goal:
Return relevant=true if this paper belongs in a survey about the CORE topic.
The SOURCE query is a hint about why it was retrieved; it does NOT need to match perfectly.

Domain definition:
{profile.domain_description}

Domain boundaries:
{profile.domain_boundaries}

Required concept groups:
The CORE topic typically involves ALL groups below. Ideally, a relevant paper will clearly match
at least ONE term from EACH group. For niche topics, you may accept a paper that matches the CORE topic
strongly even if one group is only implicit (use lower confidence).
{required_groups_str}

Optional concepts (boost relevance if present):
{optional_concepts_str}

Exclusion signals (if the primary focus is one of these, mark as irrelevant):
{exclusion_concepts_str}

Relevance rules:
1) The paper must match the CORE topic domain as defined above.
2) The paper should usually align with the SOURCE query intent, but do not reject solely for mismatch.
3) Exclude papers where the primary focus is outside the defined domain boundaries.

Use ONLY title and summary. Do NOT use the link. If unsure, return relevant=false.

Output format:
Return ONLY valid JSON with keys:
- relevant: boolean
- confidence: number from 0 to 1
- reason: short string (max 20 words)

Given:
CORE query: {profile.core_query}
SOURCE query: {source_query}
Paper title: {result.title}
Paper summary: {result.summary}
"""

    semaphore = _get_openai_semaphore()

    try:
        async with semaphore:
            # Wrap API call with timeout to prevent indefinite hangs
            response = await asyncio.wait_for(
                async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=100,
                    response_format={"type": "json_object"}
                ),
                timeout=OPENAI_TIMEOUT_SECONDS
            )

        record_openai_response(response, model="gpt-4o-mini")
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        return bool(data.get("relevant", False))

    except asyncio.TimeoutError:
        # Timeout - bias toward recall so the pipeline doesn't collapse to 0 papers.
        return True
    except OpenAIInsufficientFundsError:
        raise
    except Exception as exc:
        raise_if_openai_insufficient_funds(exc)
        # API/parsing failure - bias toward recall so we still have seeds.
        return True


async def judge_candidate(
    profile: QueryProfile,
    candidate: SnowballCandidate,
    parent_context: str | None = None
) -> JudgmentResult:
    """Judge a snowball candidate and return structured result with provenance info.
    
    This function is used during snowballing expansion to evaluate candidates
    discovered through citation/reference traversal.
    
    Args:
        profile: The QueryProfile with domain-specific filtering criteria
        candidate: The SnowballCandidate to evaluate
        parent_context: Optional context about how this paper was discovered
        
    Returns:
        JudgmentResult with relevant, confidence, and reason fields
    """
    # Build dynamic prompt from QueryProfile
    required_groups_str = _format_required_concept_groups(profile)
    optional_concepts_str = ", ".join(profile.optional_concepts) if profile.optional_concepts else "None specified"
    exclusion_concepts_str = ", ".join(profile.exclusion_concepts) if profile.exclusion_concepts else "None specified"

    abstract = candidate.abstract or "(No abstract available)"

    # Determine if this might be a foundational paper based on discovery context
    is_foundational_candidate = (
        parent_context and
        ("foundation" in parent_context.lower() or
         "reference" in parent_context.lower() or
         "fallback" in (candidate.discovered_from or "").lower())
    )

    foundational_guidance = ""
    if is_foundational_candidate:
        foundational_guidance = """
IMPORTANT - Foundational Paper Consideration:
This paper was discovered as a potential foundational work (referenced by or related to core topic papers).
Foundational papers should be ACCEPTED if they:
- Introduce key methods, architectures, or algorithms used in the core topic
  (e.g., BERT/Transformer for LLM-based systems, BPR/Matrix Factorization for recommender systems)
- Are seminal works in ONE of the constituent domains that the core topic builds upon
- Have high citations and are likely referenced by papers in the core topic

For foundational papers, it is OK if they don't explicitly mention ALL required concepts,
as long as they provide essential building blocks for the core topic.
"""

    prompt = f"""
You are a relevance judge for academic paper search in a snowballing literature review.

You will be given:
(1) a CORE topic query (the survey topic)
(2) a paper title + abstract
(3) context about how this paper was discovered

Goal:
Return relevant=true if this paper belongs in a systematic literature review about the CORE topic.

Domain definition:
{profile.domain_description}

Domain boundaries:
{profile.domain_boundaries}

Required concept groups:
The CORE topic typically involves ALL groups below. A strong CORE paper matches at least one term from EACH group.
Foundational papers may primarily match one group but be essential building blocks for the CORE topic.
{required_groups_str}

Optional concepts (boost relevance if present):
{optional_concepts_str}

Exclusion signals (if the primary focus is one of these, mark as irrelevant):
{exclusion_concepts_str}
{foundational_guidance}
Relevance categories (accept papers in ANY of these):
1) CORE PAPERS: Directly address the intersection of all required concept domains
2) FOUNDATIONAL PAPERS: Seminal works that introduce methods/techniques used by core papers
   (e.g., for "LLM-based recommendation": BERT, Attention mechanism, BPR, Matrix Factorization)
3) METHODOLOGICAL PAPERS: Introduce evaluation methods, datasets, or benchmarks for the domain
4) SURVEY/REVIEW PAPERS: Comprehensive reviews of any of the constituent domains

Exclusion rules:
1) Completely unrelated domains (e.g., biology, physics unless applied to the topic)
2) Papers that only tangentially mention keywords without substantive contribution

Output format:
Return ONLY valid JSON with keys:
- relevant: boolean
- confidence: number from 0 to 1
- reason: short string (max 20 words) explaining why relevant or not

Given:
CORE query: {profile.core_query}
Paper title: {candidate.title}
Paper abstract: {abstract}
Discovery context: {parent_context or "Discovered through citation graph expansion"}
"""

    semaphore = _get_openai_semaphore()

    try:
        async with semaphore:
            # Wrap API call with timeout to prevent indefinite hangs
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

        return JudgmentResult(
            relevant=bool(data.get("relevant", False)),
            confidence=float(data.get("confidence", 0.0)),
            reason=str(data.get("reason", "No reason provided"))
        )

    except asyncio.TimeoutError:
        # Timeout - treat as not relevant to continue processing
        return JudgmentResult(
            relevant=False,
            confidence=0.0,
            reason="Judgment timed out"
        )
    except OpenAIInsufficientFundsError:
        raise
    except Exception as e:
        raise_if_openai_insufficient_funds(e)
        # Fallback for parsing errors or API issues
        return JudgmentResult(
            relevant=False,
            confidence=0.0,
            reason=f"Error during judgment: {str(e)[:50]}"
        )


# =============================================================================
# BATCH OPERATIONS FOR CONCURRENT PROCESSING
# =============================================================================


async def batch_judge_results(
    profile: QueryProfile,
    results: list[tuple[str, ReducedArxivEntry]]
) -> list[bool]:
    """Judge multiple arXiv results concurrently.
    
    Args:
        profile: The QueryProfile with domain-specific filtering criteria
        results: List of (source_query, result) tuples
        
    Returns:
        List of boolean relevance judgments
    """
    tasks = [
        judge_result(profile, source_query, result)
        for source_query, result in results
    ]
    return await asyncio.gather(*tasks)


async def batch_judge_candidates(
    profile: QueryProfile,
    candidates_with_context: list[tuple[SnowballCandidate, str | None]]
) -> list[JudgmentResult]:
    """Judge multiple snowball candidates concurrently.
    
    Args:
        profile: The QueryProfile with domain-specific filtering criteria
        candidates_with_context: List of (candidate, parent_context) tuples
        
    Returns:
        List of JudgmentResult objects
    """
    tasks = [
        judge_candidate(profile, candidate, context)
        for candidate, context in candidates_with_context
    ]
    return await asyncio.gather(*tasks)
