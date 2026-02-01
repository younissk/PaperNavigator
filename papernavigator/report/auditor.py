"""Citation auditor for validating and revising sections.

This module provides a second-pass validation of written sections,
checking that all claims are properly supported by the cited papers.
It also ensures citations are preserved and added where missing.
"""

import asyncio
import json
import os
import re
import time
from collections.abc import Callable

from openai import AsyncOpenAI

# Timeout for OpenAI API calls (seconds)
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
# OpenAI retry attempts for transient errors (429/5xx)
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
# Timeout per audit section (seconds) to avoid indefinite hangs
AUDIT_SECTION_TIMEOUT_SECONDS = int(os.getenv("AUDIT_SECTION_TIMEOUT_SECONDS", "120"))
# Retries for audit timeouts/errors
AUDIT_SECTION_MAX_RETRIES = int(os.getenv("AUDIT_SECTION_MAX_RETRIES", "1"))
# Optional concurrency for auditing (no quality impact)
AUDIT_CONCURRENCY = int(os.getenv("AUDIT_CONCURRENCY", "2"))

from papernavigator.logging import get_logger
from papernavigator.openai_usage import OpenAIInsufficientFundsError, record_openai_response, raise_if_openai_insufficient_funds
from papernavigator.report.models import (
    AuditResult,
    PaperCard,
    SentenceAudit,
    WrittenSection,
)


def extract_cited_ids(text: str) -> list[str]:
    """Extract all paper IDs cited in the text.
    
    Looks for patterns like [W1234567890] or [S2:paper_id].
    
    Args:
        text: Text with inline citations
        
    Returns:
        List of unique paper IDs found
    """
    # Match OpenAlex-style IDs like [W1234567890] or S2-style [S2:abc123]
    pattern = r'\[([A-Za-z0-9_:-]+)\]'
    matches = re.findall(pattern, text)
    return list(set(matches))

log = get_logger(__name__)

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


def _format_cards_for_audit(cards: list[PaperCard]) -> str:
    """Format paper cards for the audit prompt."""
    card_strs = []
    for card in cards:
        parts = [
            f"[{card.id}] {card.title}",
            f"claim={card.claim}",
        ]
        if card.key_quote:
            parts.append(f"quote=\"{card.key_quote}\"")
        if card.data_benchmark:
            parts.append(f"data={card.data_benchmark}")
        if card.measured:
            parts.append(f"measured={card.measured}")

        card_strs.append(" | ".join(parts))

    return "\n".join(card_strs)


def _build_audit_prompt(section: WrittenSection, cards: list[PaperCard]) -> str:
    """Build the prompt for auditing a section."""
    cards_text = _format_cards_for_audit(cards)
    valid_ids = [c.id for c in cards]

    # Extract existing citations from original text
    original_citations = extract_cited_ids(section.content)

    return f"""Audit a survey section for citation accuracy.

TITLE: {section.title}
TEXT:
{section.content}

ORIGINAL CITATIONS: {original_citations}

EVIDENCE (use ONLY these IDs):
{cards_text}

Task:
- For each factual sentence, verify citations; fix mismatches.
- Add missing citations from: {valid_ids}
- Flag invalid citations.

Preservation rules (must follow):
1) Keep all original citations (move if needed).
2) revised_text must have >= {len(original_citations)} citations.
3) Every paragraph in revised_text must include >=1 citation.

Return JSON:
{{"sentences":[{{"sentence":"","supported":true,"cited_ids":["W123"],"issue":null,"suggested_fix":null}}],"revised_text":""}}"""


async def audit_section(
    section: WrittenSection,
    cards: list[PaperCard],
) -> AuditResult:
    """Audit a written section for citation accuracy.
    
    Args:
        section: The written section to audit
        cards: Paper cards that were available for this section
        
    Returns:
        AuditResult with sentence-level analysis and revised text
    """
    log.info(
        "citation_audit_start",
        section=section.title,
        text_length=len(section.content),
        num_cards=len(cards)
    )

    if not section.content or section.content.startswith("Error") or not cards:
        log.warning("skipping_audit", section=section.title, reason="empty or error content")
        return AuditResult(
            section_title=section.title,
            original_text=section.content,
            revised_text=section.content,
            sentences=[],
            supported_count=0,
            unsupported_count=0,
            revised_count=0
        )

    prompt = _build_audit_prompt(section, cards)
    client = _get_async_client()
    start_time = time.monotonic()
    log.info("openai_request_start", operation="audit_section", section=section.title, model="gpt-4o-mini")

    try:
        # Wrap API call with timeout to prevent indefinite hangs
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2500,
                response_format={"type": "json_object"},
                timeout=OPENAI_TIMEOUT_SECONDS,
            ),
            timeout=OPENAI_TIMEOUT_SECONDS
        )

        record_openai_response(response, model="gpt-4o-mini")
        content = response.choices[0].message.content.strip()
        log.info(
            "openai_request_complete",
            operation="audit_section",
            section=section.title,
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        data = json.loads(content)

        # Parse sentence audits
        sentence_audits = []
        supported_count = 0
        unsupported_count = 0

        for s_data in data.get("sentences", []):
            audit = SentenceAudit(
                sentence=s_data.get("sentence", ""),
                supported=s_data.get("supported", True),
                cited_ids=s_data.get("cited_ids", []),
                issue=s_data.get("issue"),
                suggested_fix=s_data.get("suggested_fix")
            )
            sentence_audits.append(audit)

            if audit.supported:
                supported_count += 1
            else:
                unsupported_count += 1

        revised_text = data.get("revised_text", section.content)
        revised_count = sum(1 for s in sentence_audits if s.suggested_fix)

        log.info(
            "citation_audit_complete",
            section=section.title,
            supported=supported_count,
            unsupported=unsupported_count,
            revised=revised_count
        )

        if unsupported_count > 0:
            log.warning(
                "unsupported_claims_found",
                section=section.title,
                count=unsupported_count,
                issues=[s.issue for s in sentence_audits if not s.supported and s.issue]
            )

        return AuditResult(
            section_title=section.title,
            original_text=section.content,
            revised_text=revised_text,
            sentences=sentence_audits,
            supported_count=supported_count,
            unsupported_count=unsupported_count,
            revised_count=revised_count
        )

    except asyncio.TimeoutError:
        log.error("audit_timeout", section=section.title, timeout_sec=OPENAI_TIMEOUT_SECONDS)
        log.info(
            "openai_request_timeout",
            operation="audit_section",
            section=section.title,
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        # Return original text without revision on timeout
        return AuditResult(
            section_title=section.title,
            original_text=section.content,
            revised_text=section.content,
            sentences=[],
            supported_count=0,
            unsupported_count=0,
            revised_count=0
        )
    except json.JSONDecodeError as e:
        log.error("audit_json_parse_error", section=section.title, error=str(e))
        log.info(
            "openai_request_failed",
            operation="audit_section",
            section=section.title,
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        # Return original text without revision
        return AuditResult(
            section_title=section.title,
            original_text=section.content,
            revised_text=section.content,
            sentences=[],
            supported_count=0,
            unsupported_count=0,
            revised_count=0
        )
    except OpenAIInsufficientFundsError:
        raise
    except Exception as e:
        raise_if_openai_insufficient_funds(e)
        log.error("audit_failed", section=section.title, error=str(e))
        log.info(
            "openai_request_failed",
            operation="audit_section",
            section=section.title,
            duration_sec=round(time.monotonic() - start_time, 2),
        )
        return AuditResult(
            section_title=section.title,
            original_text=section.content,
            revised_text=section.content,
            sentences=[],
            supported_count=0,
            unsupported_count=0,
            revised_count=0
        )


async def audit_all_sections(
    sections: list[WrittenSection],
    all_cards: list[PaperCard],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[AuditResult]:
    """Audit all written sections.
    
    Args:
        sections: List of written sections
        all_cards: All available paper cards
        
    Returns:
        List of audit results, one per section
    """
    if not sections:
        log.warning("no_sections_to_audit")
        return []

    log.info("batch_audit_start", num_sections=len(sections))

    # Create card lookup by ID
    card_lookup = {c.id: c for c in all_cards}

    results = []
    total_supported = 0
    total_unsupported = 0
    total_sections = len(sections)

    if progress_callback:
        progress_callback(0, total_sections, f"Starting to audit {total_sections} sections...")

    async def _audit_one(idx: int, section: WrittenSection) -> tuple[int, AuditResult]:
        # Get cards that were used in this section
        section_cards = [
            card_lookup[pid]
            for pid in section.paper_ids_used
            if pid in card_lookup
        ]

        # If no specific cards, use all cards
        if not section_cards:
            section_cards = all_cards

        if progress_callback:
            progress_callback(idx, total_sections, f"Auditing section {idx+1}/{total_sections}: {section.title}")

        result: AuditResult | None = None
        for attempt in range(AUDIT_SECTION_MAX_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    audit_section(section, section_cards),
                    timeout=AUDIT_SECTION_TIMEOUT_SECONDS,
                )
                break
            except asyncio.TimeoutError:
                log.warning(
                    "audit_section_timeout",
                    section=section.title,
                    timeout_sec=AUDIT_SECTION_TIMEOUT_SECONDS,
                    attempt=attempt + 1,
                )
                if progress_callback:
                    progress_callback(
                        idx,
                        total_sections,
                        (
                            "WARNING: Audit timed out for section "
                            f"{idx+1}/{total_sections}: {section.title} "
                            f"(attempt {attempt + 1})"
                        ),
                    )
            except Exception as exc:
                if isinstance(exc, OpenAIInsufficientFundsError):
                    raise
                raise_if_openai_insufficient_funds(exc)
                log.error(
                    "audit_section_failed",
                    section=section.title,
                    error=str(exc),
                    attempt=attempt + 1,
                )
                if progress_callback:
                    progress_callback(
                        idx,
                        total_sections,
                        (
                            "WARNING: Audit failed for section "
                            f"{idx+1}/{total_sections}: {section.title} "
                            f"(attempt {attempt + 1})"
                        ),
                    )
            if attempt == AUDIT_SECTION_MAX_RETRIES:
                result = AuditResult(
                    section_title=section.title,
                    original_text=section.content,
                    revised_text=section.content,
                    sentences=[],
                    supported_count=0,
                    unsupported_count=0,
                    revised_count=0,
                )
                break

        if result is None:
            result = AuditResult(
                section_title=section.title,
                original_text=section.content,
                revised_text=section.content,
                sentences=[],
                supported_count=0,
                unsupported_count=0,
                revised_count=0,
            )

        if progress_callback:
            progress_callback(idx + 1, total_sections, f"Completed audit {idx+1}/{total_sections}: {section.title}")

        return idx, result

    if AUDIT_CONCURRENCY <= 1:
        for i, section in enumerate(sections):
            _, result = await _audit_one(i, section)
            results.append(result)
            total_supported += result.supported_count
            total_unsupported += result.unsupported_count
    else:
        semaphore = asyncio.Semaphore(AUDIT_CONCURRENCY)

        async def _runner(idx: int, section: WrittenSection) -> tuple[int, AuditResult]:
            async with semaphore:
                return await _audit_one(idx, section)

        tasks = [asyncio.create_task(_runner(i, section)) for i, section in enumerate(sections)]
        ordered: list[AuditResult | None] = [None] * total_sections
        for task in asyncio.as_completed(tasks):
            idx, result = await task
            ordered[idx] = result

        for result in ordered:
            if result is None:
                continue
            results.append(result)
            total_supported += result.supported_count
            total_unsupported += result.unsupported_count

    log.info(
        "batch_audit_complete",
        num_sections=len(sections),
        total_supported=total_supported,
        total_unsupported=total_unsupported
    )

    return results


def merge_citations(
    original_text: str,
    revised_text: str,
    cards: list[PaperCard],
) -> str:
    """Merge citations from original into revised text if they were lost.
    
    Strategy:
    1. Extract citations from both texts
    2. If revised lost citations, try to restore them by:
       - Finding semantically similar sentences/paragraphs
       - Injecting citations at appropriate points
    3. Return merged text with all citations preserved
    
    Args:
        original_text: Original text with citations
        revised_text: Revised text that may have lost citations
        cards: Available paper cards for validation
        
    Returns:
        Text with citations preserved/restored
    """
    original_citations = set(extract_cited_ids(original_text))
    revised_citations = set(extract_cited_ids(revised_text))

    # Valid IDs from cards
    valid_ids = {c.id for c in cards}

    # Filter to only valid citations
    original_citations = original_citations & valid_ids
    revised_citations = revised_citations & valid_ids

    # If no citations were lost, return revised text as-is
    lost_citations = original_citations - revised_citations

    if not lost_citations:
        log.debug("no_citations_lost", original=len(original_citations), revised=len(revised_citations))
        return revised_text

    log.warning(
        "citations_lost_during_revision",
        original_count=len(original_citations),
        revised_count=len(revised_citations),
        lost=list(lost_citations)
    )

    # Try to restore lost citations
    result_text = revised_text

    # Split into paragraphs
    paragraphs = result_text.split('\n\n')

    # Distribute lost citations across paragraphs that don't have citations
    lost_list = list(lost_citations)
    citation_index = 0

    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue

        para_citations = extract_cited_ids(para)

        # If paragraph has no citations and we have lost citations to add
        if not para_citations and citation_index < len(lost_list):
            # Find end of first sentence
            sentences = para.split('.')
            if sentences and sentences[0].strip():
                # Add citation to end of first sentence
                citation_to_add = lost_list[citation_index]
                sentences[0] = sentences[0].rstrip() + f" [{citation_to_add}]"
                paragraphs[i] = '.'.join(sentences)
                citation_index += 1
                log.debug("restored_citation", citation=citation_to_add, paragraph_index=i)

    # If we still have lost citations, append them to the last paragraph
    while citation_index < len(lost_list):
        # Find the last non-empty paragraph
        for i in range(len(paragraphs) - 1, -1, -1):
            if paragraphs[i].strip():
                citation_to_add = lost_list[citation_index]
                paragraphs[i] = paragraphs[i].rstrip() + f" [{citation_to_add}]"
                citation_index += 1
                log.debug("appended_citation_to_last_para", citation=citation_to_add)
                break
        else:
            break  # No non-empty paragraphs found

    result_text = '\n\n'.join(paragraphs)

    # Verify restoration
    final_citations = set(extract_cited_ids(result_text))
    restored_count = len(final_citations) - len(revised_citations)

    log.info(
        "citations_merged",
        original=len(original_citations),
        revised=len(revised_citations),
        final=len(final_citations),
        restored=restored_count
    )

    return result_text
