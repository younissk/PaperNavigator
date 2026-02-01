"""Query augmentation using LLM (Async version).

This module expands a single search query into multiple variants
for better coverage of the research topic.
"""

import asyncio
import json
import os
import time

from openai import AsyncOpenAI
from pydantic import TypeAdapter, ValidationError

from papernavigator.openai_usage import record_openai_response, raise_if_openai_insufficient_funds

# Timeout for OpenAI API calls (seconds)
OPENAI_TIMEOUT_SECONDS = 30

async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


async def augment_search(query: str, k: int = 6) -> tuple[list[str], float]:
    """Expand a single query into multiple search variants.
    
    Args:
        query: The original search query
        k: Number of query variants to generate
        
    Returns:
        Tuple of (list of augmented queries including original, time taken in seconds)
    """
    prompt = f"""
    You are QueryExpander for academic paper search.

    Goal:
    Turn ONE input query into a small set of high coverage, high precision search queries.
    These expanded queries will be searched separately to retrieve papers.

    Output format:
    Return ONLY a JSON array of strings. No extra text, no markdown, no keys.

    Hard rules:
    - Return exactly {k} queries.
    - Each query must be meaningfully different (no minor rewording only).
    - Keep each query concise: 6 to 14 words.
    - Prefer scholarly terms, common acronyms, and method names.
    - Include the original query terms in at least 2 queries, but do not copy the input as-is.
    - Avoid adding names of companies, products, or random author names.

    Coverage requirements (must satisfy all):
    1) One query focused on surveys or reviews (include: survey OR review OR "systematic review").
    2) One query focused on benchmarks or datasets (include: benchmark OR dataset OR evaluation set).
    3) One query focused on methods or architectures (include: method OR architecture OR model OR algorithm).
    4) One query focused on evaluation or metrics (include: evaluation OR metric OR measurement).
    5) One query focused on failure modes or limitations (include: limitation OR failure mode OR error OR pitfall).
    6) The remaining queries should cover adjacent terms, synonyms, and closely related subtopics.

    Quality checks before returning:
    - Remove duplicates.
    - Remove overly broad queries (example: "AI research").
    - Make queries specific enough that they likely match academic papers.

    Example:
    Input query: "evaluation of retrieval augmented generation"
    Output:
    [
    "retrieval augmented generation evaluation metrics faithfulness grounding",
    "RAG citation correctness attribution errors verification methods",
    "benchmarks for RAG evaluation datasets human evaluation protocols",
    "hybrid retrieval BM25 dense retrieval for RAG performance",
    "systematic review survey of RAG evaluation and tooling",
    "RAG hallucination reduction methods and failure modes analysis"
    ]

    Now expand this input query:
    {query}
    """

    start_time = time.time()
    try:
        # Wrap API call with timeout to prevent indefinite hangs
        response = await asyncio.wait_for(
            async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=OPENAI_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        # On timeout, return just the original query
        return [query], 0.0
    except Exception as exc:
        raise_if_openai_insufficient_funds(exc)
        raise
    end_time = time.time()

    record_openai_response(response, model="gpt-4o-mini")

    content = response.choices[0].message.content.strip()

    try:
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()

        data = json.loads(content)

        validator = TypeAdapter(list[str])
        validated_data = validator.validate_python(data)
        augmented_queries = list(set(validated_data + [query]))

        time_taken = end_time - start_time

        return augmented_queries, time_taken

    except (json.JSONDecodeError, ValidationError) as e:
        # If it's still not right, we could try more aggressive cleaning
        # but for now, we'll raise an error or return a fallback
        raise ValueError(f"Failed to parse LLM response into List[str]: {e}\nRaw content: {content}")
