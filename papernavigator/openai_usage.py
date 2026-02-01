"""OpenAI usage + cost tracking utilities.

This module provides a lightweight per-run usage accumulator that works across
async tasks (via contextvars) and sync calls.

Cost is an estimate computed from token usage and a simple pricing table.
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


class OpenAIInsufficientFundsError(RuntimeError):
    """Raised when the OpenAI API indicates the account has no remaining quota/credits."""

    error_code = "openai_insufficient_quota"

    def __init__(self, message: str, *, status_code: int | None = None, api_code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.api_code = api_code


def _get_default_pricing_per_1m_tokens_usd() -> dict[str, dict[str, float]]:
    # Defaults match OpenAI "Text tokens" and "Embeddings" pricing (standard) as of 2025-01-22.
    # You can override via OPENAI_PRICING_JSON, e.g.:
    # {"gpt-4o-mini":{"input":0.15,"output":0.60},"text-embedding-3-small":{"input":0.02}}
    return {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    }


def _load_pricing_per_1m_tokens_usd() -> dict[str, dict[str, float]]:
    raw = os.getenv("OPENAI_PRICING_JSON")
    if not raw:
        return _get_default_pricing_per_1m_tokens_usd()
    try:
        parsed = json.loads(raw)
    except Exception:
        return _get_default_pricing_per_1m_tokens_usd()

    if not isinstance(parsed, dict):
        return _get_default_pricing_per_1m_tokens_usd()

    pricing: dict[str, dict[str, float]] = {}
    for model, rates in parsed.items():
        if not isinstance(model, str) or not isinstance(rates, dict):
            continue
        inp = rates.get("input")
        out = rates.get("output", 0.0)
        if isinstance(inp, (int, float)) and inp >= 0 and isinstance(out, (int, float)) and out >= 0:
            pricing[model] = {"input": float(inp), "output": float(out)}

    return pricing or _get_default_pricing_per_1m_tokens_usd()


def _normalize_model(model: str | None) -> str:
    if not model:
        return ""
    m = model.strip()
    if not m:
        return ""
    # Collapse dated variants (e.g. "gpt-4o-mini-2024-07-18") to their base prefix.
    for prefix in ("gpt-4o-mini", "gpt-4o", "text-embedding-3-small"):
        if m == prefix or m.startswith(prefix + "-"):
            return prefix
    return m


def _extract_error_fields(body: object | None) -> tuple[str | None, str | None]:
    if not isinstance(body, dict):
        return None, None

    payload: dict[str, Any]
    if isinstance(body.get("error"), dict):
        payload = body["error"]
    else:
        payload = body

    msg = payload.get("message")
    code = payload.get("code")
    msg_s = str(msg) if msg is not None else None
    code_s = str(code) if code is not None else None
    return msg_s, code_s


def is_openai_insufficient_funds_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    message = str(exc)

    _, code = _extract_error_fields(body)

    if code in {"insufficient_quota", "billing_hard_limit_reached"}:
        return True

    msg = (message or "").lower()
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return True
    if "billing" in msg and ("limit" in msg or "quota" in msg or "credits" in msg):
        return True

    if status_code in {402, 429} and ("quota" in msg or "credits" in msg):
        return True

    return False


def raise_if_openai_insufficient_funds(exc: BaseException) -> None:
    if not is_openai_insufficient_funds_error(exc):
        return

    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    msg, code = _extract_error_fields(body)

    public = "OpenAI API quota/credits exhausted. Add billing or update your OpenAI plan, then retry."
    if msg:
        public = f"{public} (Provider: {msg})"

    raise OpenAIInsufficientFundsError(public, status_code=status_code, api_code=code) from exc


@dataclass
class _ModelUsage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = 0.0


@dataclass
class _Tracker:
    by_model: dict[str, _ModelUsage] = field(default_factory=dict)
    unpriced_models: set[str] = field(default_factory=set)

    def record(self, model: str, *, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        m = _normalize_model(model)
        if not m:
            m = "unknown"

        usage = self.by_model.setdefault(m, _ModelUsage())
        usage.requests += 1
        usage.prompt_tokens += max(0, int(prompt_tokens))
        usage.completion_tokens += max(0, int(completion_tokens))
        usage.total_tokens += max(0, int(total_tokens))

        cost = estimate_cost_usd(m, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if cost is None:
            self.unpriced_models.add(m)
            usage.estimated_cost_usd = None
        else:
            if usage.estimated_cost_usd is None:
                usage.estimated_cost_usd = 0.0
            usage.estimated_cost_usd += float(cost)

    def snapshot(self) -> dict[str, Any]:
        totals = _ModelUsage()
        totals.estimated_cost_usd = 0.0

        for u in self.by_model.values():
            totals.requests += u.requests
            totals.prompt_tokens += u.prompt_tokens
            totals.completion_tokens += u.completion_tokens
            totals.total_tokens += u.total_tokens
            if totals.estimated_cost_usd is not None:
                if u.estimated_cost_usd is None:
                    totals.estimated_cost_usd = None
                else:
                    totals.estimated_cost_usd += float(u.estimated_cost_usd)

        by_model_out: dict[str, Any] = {}
        for model, u in sorted(self.by_model.items()):
            by_model_out[model] = {
                "requests": u.requests,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
                "estimated_cost_usd": u.estimated_cost_usd,
            }

        return {
            "totals": {
                "requests": totals.requests,
                "prompt_tokens": totals.prompt_tokens,
                "completion_tokens": totals.completion_tokens,
                "total_tokens": totals.total_tokens,
                "estimated_cost_usd": totals.estimated_cost_usd,
            },
            "by_model": by_model_out,
            "unpriced_models": sorted(self.unpriced_models),
        }


_TRACKER: ContextVar[_Tracker | None] = ContextVar("papernavigator_openai_usage_tracker", default=None)


def start_openai_usage_tracking() -> None:
    _TRACKER.set(_Tracker())


def get_openai_usage_snapshot() -> dict[str, Any] | None:
    tracker = _TRACKER.get()
    if tracker is None:
        return None
    snap = tracker.snapshot()
    if snap["totals"]["requests"] <= 0:
        return None
    return snap


def _usage_int(value: Any) -> int:
    try:
        n = int(value)
    except Exception:
        return 0
    return max(0, n)


def record_openai_response(response: Any, *, model: str | None = None) -> None:
    tracker = _TRACKER.get()
    if tracker is None:
        return

    resolved_model = model
    if not resolved_model:
        resolved_model = getattr(response, "model", None)

    usage = getattr(response, "usage", None)
    if usage is None:
        return

    prompt_tokens = _usage_int(getattr(usage, "prompt_tokens", 0))
    completion_tokens = _usage_int(getattr(usage, "completion_tokens", 0))
    total_tokens = _usage_int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens))

    tracker.record(
        resolved_model or "unknown",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def estimate_cost_usd(model: str, *, prompt_tokens: int, completion_tokens: int) -> float | None:
    pricing = _load_pricing_per_1m_tokens_usd()
    m = _normalize_model(model)
    rates = pricing.get(m)
    if not rates:
        return None
    inp = float(rates.get("input", 0.0))
    out = float(rates.get("output", 0.0))
    return (max(0, int(prompt_tokens)) * inp + max(0, int(completion_tokens)) * out) / 1_000_000.0


def merge_openai_usage(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if not a and not b:
        return None
    if not a:
        return b
    if not b:
        return a

    merged: dict[str, Any] = {"totals": {}, "by_model": {}, "unpriced_models": []}

    a_tot = a.get("totals") if isinstance(a.get("totals"), dict) else {}
    b_tot = b.get("totals") if isinstance(b.get("totals"), dict) else {}

    def sum_int(k: str) -> int:
        av = a_tot.get(k)
        bv = b_tot.get(k)
        return _usage_int(av) + _usage_int(bv)

    merged_totals: dict[str, Any] = {
        "requests": sum_int("requests"),
        "prompt_tokens": sum_int("prompt_tokens"),
        "completion_tokens": sum_int("completion_tokens"),
        "total_tokens": sum_int("total_tokens"),
        "estimated_cost_usd": None,
    }

    a_cost = a_tot.get("estimated_cost_usd")
    b_cost = b_tot.get("estimated_cost_usd")
    if isinstance(a_cost, (int, float)) and isinstance(b_cost, (int, float)):
        merged_totals["estimated_cost_usd"] = float(a_cost) + float(b_cost)
    elif isinstance(a_cost, (int, float)) and b_cost is None:
        merged_totals["estimated_cost_usd"] = None
    elif a_cost is None and isinstance(b_cost, (int, float)):
        merged_totals["estimated_cost_usd"] = None
    elif isinstance(a_cost, (int, float)) and not b:
        merged_totals["estimated_cost_usd"] = float(a_cost)
    elif isinstance(b_cost, (int, float)) and not a:
        merged_totals["estimated_cost_usd"] = float(b_cost)
    elif a_cost is None or b_cost is None:
        merged_totals["estimated_cost_usd"] = None

    def model_map(x: dict[str, Any]) -> dict[str, Any]:
        m = x.get("by_model")
        return m if isinstance(m, dict) else {}

    combined_models = set(model_map(a).keys()) | set(model_map(b).keys())
    by_model: dict[str, Any] = {}
    for model in sorted(combined_models):
        am = model_map(a).get(model)
        bm = model_map(b).get(model)
        am = am if isinstance(am, dict) else {}
        bm = bm if isinstance(bm, dict) else {}

        def sum_m(k: str) -> int:
            return _usage_int(am.get(k)) + _usage_int(bm.get(k))

        cost = None
        a_mc = am.get("estimated_cost_usd")
        b_mc = bm.get("estimated_cost_usd")
        if isinstance(a_mc, (int, float)) and isinstance(b_mc, (int, float)):
            cost = float(a_mc) + float(b_mc)
        elif a_mc is None or b_mc is None:
            cost = None

        by_model[model] = {
            "requests": sum_m("requests"),
            "prompt_tokens": sum_m("prompt_tokens"),
            "completion_tokens": sum_m("completion_tokens"),
            "total_tokens": sum_m("total_tokens"),
            "estimated_cost_usd": cost,
        }

    u1 = a.get("unpriced_models") if isinstance(a.get("unpriced_models"), list) else []
    u2 = b.get("unpriced_models") if isinstance(b.get("unpriced_models"), list) else []
    merged["totals"] = merged_totals
    merged["by_model"] = by_model
    merged["unpriced_models"] = sorted({*(str(x) for x in u1), *(str(x) for x in u2)})

    return merged
