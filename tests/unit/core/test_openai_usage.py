"""Unit tests for OpenAI usage tracking helpers."""

import pytest

from papernavigator.openai_usage import (
    OpenAIInsufficientFundsError,
    estimate_cost_usd,
    get_openai_usage_snapshot,
    merge_openai_usage,
    record_openai_response,
    raise_if_openai_insufficient_funds,
    start_openai_usage_tracking,
)


pytestmark = pytest.mark.unit


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _Resp:
    def __init__(self, model: str, usage: _Usage):
        self.model = model
        self.usage = usage


def test_estimate_cost_usd_uses_default_pricing(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICING_JSON", raising=False)
    cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0)
    assert cost == pytest.approx(0.15)


def test_record_openai_response_accumulates_snapshot(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICING_JSON", raising=False)
    start_openai_usage_tracking()
    record_openai_response(_Resp("gpt-4o-mini", _Usage(10, 5, 15)))
    record_openai_response(_Resp("gpt-4o-mini", _Usage(1, 2, 3)))

    snap = get_openai_usage_snapshot()
    assert snap is not None
    assert snap["totals"]["requests"] == 2
    assert snap["totals"]["prompt_tokens"] == 11
    assert snap["totals"]["completion_tokens"] == 7
    assert snap["totals"]["total_tokens"] == 18
    assert snap["by_model"]["gpt-4o-mini"]["requests"] == 2


def test_merge_openai_usage_sums_totals(monkeypatch):
    monkeypatch.delenv("OPENAI_PRICING_JSON", raising=False)
    a = {
        "totals": {
            "requests": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "estimated_cost_usd": 0.01,
        },
        "by_model": {
            "gpt-4o-mini": {
                "requests": 1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "estimated_cost_usd": 0.01,
            }
        },
        "unpriced_models": [],
    }
    b = {
        "totals": {
            "requests": 2,
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "estimated_cost_usd": 0.02,
        },
        "by_model": {
            "gpt-4o-mini": {
                "requests": 2,
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "estimated_cost_usd": 0.02,
            }
        },
        "unpriced_models": [],
    }

    merged = merge_openai_usage(a, b)
    assert merged is not None
    assert merged["totals"]["requests"] == 3
    assert merged["totals"]["total_tokens"] == 18
    assert merged["totals"]["estimated_cost_usd"] == pytest.approx(0.03)


def test_raise_if_openai_insufficient_funds_raises():
    class DummyExc(Exception):
        def __init__(self):
            super().__init__("quota exceeded")
            self.status_code = 429
            self.body = {"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}}

    with pytest.raises(OpenAIInsufficientFundsError):
        raise_if_openai_insufficient_funds(DummyExc())

