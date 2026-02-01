"""Query profile generation using LLM (Async version).

This module analyzes research queries and generates structured QueryProfiles
for relevance filtering.
"""

import asyncio
import json
import os
import re

from openai import AsyncOpenAI

from papernavigator.models import QueryProfile
from papernavigator.openai_usage import record_openai_response, raise_if_openai_insufficient_funds

# Timeout for OpenAI API calls (seconds)
OPENAI_TIMEOUT_SECONDS = 30

async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


async def generate_query_profile(query: str) -> QueryProfile:
    """Analyze a research query and generate a structured QueryProfile.
    
    Args:
        query: The user's research topic (e.g., "LLM Based Recommendation Systems")
    
    Returns:
        A QueryProfile with dynamic filtering criteria for relevance judgment.
    """
    prompt = f"""
You are an academic research domain analyzer. Given a research topic query,
extract structured criteria for filtering academic papers.

Your task: Analyze the query and return a JSON object with the following fields:

1. "domain_description": A 1-2 sentence description of this research domain.

2. "required_concepts": A list of concept groups where AT LEAST ONE term from 
   EACH group must appear in a relevant paper. Each group is a list of synonyms.
   Example for "LLM-based recommender systems":
   [["LLM", "large language model", "foundation model", "generative model", "GPT", "BERT", "transformer"],
    ["recommender", "recommendation", "personalized ranking", "user-item", "collaborative filtering"]]

3. "optional_concepts": Terms that increase relevance but aren't required.
   Example: ["cold start", "fairness", "explainability"]

4. "exclusion_concepts": Topics that indicate a paper is OUT of scope.
   Example: ["pure NLP evaluation", "machine translation", "speech recognition"]

5. "keyword_patterns": Regex patterns (case-insensitive) for fast pre-filtering.
   These should match the required concepts. Use \\b for word boundaries.
   Example: ["\\\\b(llm|large language model|foundation model)\\\\b",
             "\\\\b(recommender|recommendation|personalized)\\\\b"]

6. "domain_boundaries": A clear statement of what IS and IS NOT in scope.
   Example: "IN scope: papers about using LLMs for personalized recommendations.
   OUT of scope: generic LLM evaluation, RAG for QA, knowledge graphs without recommendations."

7. "fallback_queries": A list of 5-8 search queries to find FOUNDATIONAL papers 
   that are building blocks for this research topic. Include:
   - Queries for foundational papers in EACH constituent domain separately
   - Queries for seminal methods/architectures used in this domain
   - Survey/review queries for the overall topic
   
   Example for "LLM-based recommender systems":
   [
     "large language model recommendation survey",
     "transformer sequential recommendation",
     "BERT pre-training language representation",
     "collaborative filtering matrix factorization",
     "neural network embedding recommendation",
     "personalized ranking implicit feedback",
     "attention mechanism deep learning"
   ]

Return ONLY valid JSON. No markdown, no extra text.

Research topic query: {query}
"""

    # Wrap API call with timeout to prevent indefinite hangs
    try:
        response = await asyncio.wait_for(
            async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            ),
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise_if_openai_insufficient_funds(exc)
        raise

    record_openai_response(response, model="gpt-4o-mini")

    content = response.choices[0].message.content.strip()
    data = json.loads(content)

    # Parse required_concepts - preserve groups if nested
    required_raw = data.get("required_concepts", [])
    required_groups: list[list[str]] = []
    required_flat: list[str] = []

    if required_raw:
        if isinstance(required_raw[0], list):
            # Already grouped: [["LLM", "large language model"], ["recommender", ...]]
            required_groups = required_raw
            required_flat = [term for group in required_raw for term in group]
        else:
            # Flat list - treat as single group (backward compat)
            required_groups = [required_raw]
            required_flat = required_raw

    # Generate keyword patterns from groups: OR within group, AND across groups
    # This is more reliable than relying on LLM-generated patterns
    keyword_patterns = _build_keyword_patterns(required_groups)

    # Extract fallback queries, or generate defaults if not provided
    fallback_queries = data.get("fallback_queries", [])
    if not fallback_queries:
        # Generate basic fallback queries from concept groups
        fallback_queries = _generate_default_fallback_queries(query, required_groups)

    return QueryProfile(
        core_query=query,
        domain_description=data.get("domain_description", ""),
        required_concepts=required_flat,
        required_concept_groups=required_groups,
        optional_concepts=data.get("optional_concepts", []),
        exclusion_concepts=data.get("exclusion_concepts", []),
        keyword_patterns=keyword_patterns,
        domain_boundaries=data.get("domain_boundaries", ""),
        fallback_queries=fallback_queries
    )


def _build_keyword_patterns(groups: list[list[str]]) -> list[str]:
    """Build regex patterns from concept groups.
    
    Each group becomes one pattern with OR semantics inside.
    The keyword gate applies AND across all patterns.
    
    Args:
        groups: List of concept groups, e.g., [["LLM", "GPT"], ["recommender"]]
        
    Returns:
        List of regex patterns, one per group
    """
    patterns = []

    for group in groups:
        if not group:
            continue
        # Escape special regex chars and join with OR
        escaped = [re.escape(term) for term in group]
        # Case-insensitive matching will be applied at match time
        pattern = "(" + "|".join(escaped) + ")"
        patterns.append(pattern)

    return patterns


def _generate_default_fallback_queries(
    core_query: str,
    concept_groups: list[list[str]]
) -> list[str]:
    """Generate default fallback queries when LLM doesn't provide them.
    
    Creates queries for:
    1. Core query + survey
    2. Each concept group + survey/review
    3. Combinations of top terms from each group
    
    Args:
        core_query: The original search query
        concept_groups: List of concept groups
        
    Returns:
        List of fallback search queries
    """
    queries = [
        f"{core_query} survey",
        f"{core_query} review",
    ]

    # Add queries for each concept group separately
    for group in concept_groups:
        if group:
            # Use top 2 terms from each group
            top_terms = " ".join(group[:2])
            queries.append(f"{top_terms} survey")
            queries.append(f"{top_terms} deep learning")

    # Add cross-group combinations
    if len(concept_groups) >= 2:
        for term1 in concept_groups[0][:2]:
            for term2 in concept_groups[1][:2]:
                queries.append(f"{term1} {term2}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_queries: list[str] = []
    for q in queries:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)

    return unique_queries[:10]  # Limit to 10 queries
