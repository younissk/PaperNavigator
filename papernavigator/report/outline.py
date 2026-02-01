"""Report outline generation using LLM.

This module generates a structured outline for the report by analyzing
paper cards and grouping them by research themes/paradigms.
"""

import asyncio
import json
import os
import time
from collections import defaultdict

from openai import AsyncOpenAI

from papernavigator.logging import get_logger
from papernavigator.openai_usage import record_openai_response, raise_if_openai_insufficient_funds
from papernavigator.report.models import PaperCard, ReportOutline, SectionPlan

log = get_logger(__name__)

# Timeout for OpenAI API calls (seconds)
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
# OpenAI retry attempts for transient errors (429/5xx)
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

# Async OpenAI client
_async_client: AsyncOpenAI | None = None


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


def group_by_tags(cards: list[PaperCard]) -> dict[str, list[PaperCard]]:
    """Group paper cards by their paradigm tags.
    
    Each paper may appear in multiple groups if it has multiple tags.
    
    Args:
        cards: List of paper cards
        
    Returns:
        Dictionary mapping tag -> list of papers with that tag
    """
    groups: dict[str, list[PaperCard]] = defaultdict(list)

    for card in cards:
        if not card.paradigm_tags:
            groups["general"].append(card)
        else:
            for tag in card.paradigm_tags:
                groups[tag].append(card)

    return dict(groups)




def _build_outline_prompt(query: str, cards: list[PaperCard]) -> str:
    """Build the prompt for generating the report outline."""
    # Group cards by tags for context
    tag_groups = group_by_tags(cards)

    # Compact tag distribution summary
    tag_summary = ", ".join(
        [f"{tag}:{len(tag_cards)}" for tag, tag_cards in sorted(tag_groups.items(), key=lambda x: -len(x[1]))]
    )

    # Build cards summary (claims only for brevity)
    cards_summary = "\n".join([
        f"[{c.id}] tags={','.join(c.paradigm_tags) if c.paradigm_tags else 'general'} | claim={c.claim}"
        for c in cards
    ])

    return f"""You are creating an outline for a research survey report.

Query: {query}

Papers (claims + tags):
{cards_summary}

Tag distribution: {tag_summary}

Create a report outline with 4-6 sections that organizes these papers into coherent themes.

For each section, provide:
1. "title": A descriptive section title (e.g., "Prompting-based Approaches", "Evaluation Methods")
2. "bullet_claims": 2-4 bullet points describing what this section should cover
3. "relevant_paper_ids": List of paper IDs that should be cited in this section

Guidelines:
- Group papers by methodology, application, or theme - NOT by individual paper
- Each paper should appear in at least one section
- Highly relevant papers may appear in multiple sections
- The outline should tell a coherent story about the research area
- Include a section on evaluation/benchmarks if applicable
- Include a section on limitations/open problems if the papers discuss them

Return ONLY valid JSON with this structure:
{{
  "sections": [
    {{
      "title": "Section Title",
      "bullet_claims": ["claim 1", "claim 2"],
      "relevant_paper_ids": ["W123", "W456"]
    }}
  ]
}}"""


async def generate_outline(query: str, cards: list[PaperCard]) -> ReportOutline:
    """Generate a report outline from paper cards.
    
    Args:
        query: Original research query
        cards: List of paper cards to include
        
    Returns:
        ReportOutline with section plans
    """
    if not cards:
        log.warning("no_cards_for_outline")
        return ReportOutline(sections=[])

    log.info("outline_planning_start", query=query, num_cards=len(cards))

    # Log tag distribution
    tag_groups = group_by_tags(cards)
    log.debug("tag_groups", groups=list(tag_groups.keys()), counts={
        k: len(v) for k, v in tag_groups.items()
    })

    prompt = _build_outline_prompt(query, cards)
    client = _get_async_client()
    start_time = time.monotonic()
    log.info("openai_request_start", operation="generate_outline", model="gpt-4o-mini")

    try:
        # Wrap API call with timeout to prevent indefinite hangs
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Slight creativity for organization
                max_tokens=2000,
                response_format={"type": "json_object"},
                timeout=OPENAI_TIMEOUT_SECONDS,
            ),
            timeout=OPENAI_TIMEOUT_SECONDS
        )

        record_openai_response(response, model="gpt-4o-mini")
        content = response.choices[0].message.content.strip()
        log.info(
            "openai_request_complete",
            operation="generate_outline",
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        data = json.loads(content)

        # Parse sections
        sections = []
        for section_data in data.get("sections", []):
            section = SectionPlan(
                title=section_data.get("title", "Untitled Section"),
                bullet_claims=section_data.get("bullet_claims", []),
                relevant_paper_ids=section_data.get("relevant_paper_ids", []),
            )
            sections.append(section)

        outline = ReportOutline(sections=sections)

        log.info(
            "outline_planning_complete",
            num_sections=len(outline.sections),
            section_titles=[s.title for s in outline.sections]
        )

        return outline

    except asyncio.TimeoutError:
        log.error("outline_generation_timeout", timeout_sec=OPENAI_TIMEOUT_SECONDS)
        log.info(
            "openai_request_timeout",
            operation="generate_outline",
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        # Return a fallback single-section outline
        return ReportOutline(sections=[
            SectionPlan(
                title="Research Overview",
                bullet_claims=["Overview of current research"],
                relevant_paper_ids=[c.id for c in cards]
            )
        ])
    except json.JSONDecodeError as e:
        log.error("outline_json_parse_error", error=str(e))
        log.info(
            "openai_request_failed",
            operation="generate_outline",
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        # Return a fallback single-section outline
        return ReportOutline(sections=[
            SectionPlan(
                title="Research Overview",
                bullet_claims=["Overview of current research"],
                relevant_paper_ids=[c.id for c in cards]
            )
        ])
    except Exception as e:
        raise_if_openai_insufficient_funds(e)
        log.error("outline_generation_failed", error=str(e))
        log.info(
            "openai_request_failed",
            operation="generate_outline",
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        raise
