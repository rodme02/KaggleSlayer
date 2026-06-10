"""Tests for kaggle_slayer.agent.cost_ledger."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.agent import cost_ledger as cl


def test_record_returns_usd_cost(tmp_path):
    ledger = cl.CostLedger(path=tmp_path / "cost.jsonl")
    cost = ledger.record(
        model="gemini-2.5-flash",
        input_tokens=1000,
        output_tokens=500,
        cached_tokens=0,
        competition="titanic",
    )
    assert cost > 0
    assert cost == pytest.approx(0.000075 + 0.0003 * 0.5, rel=1e-2)


def test_record_writes_one_jsonl_line(tmp_path):
    ledger = cl.CostLedger(path=tmp_path / "cost.jsonl")
    ledger.record(
        model="gemini-2.5-flash",
        input_tokens=100,
        output_tokens=50,
        cached_tokens=0,
        competition="titanic",
    )
    lines = (tmp_path / "cost.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["model"] == "gemini-2.5-flash"
    assert rec["input_tokens"] == 100
    assert rec["output_tokens"] == 50
    assert rec["cached_tokens"] == 0
    assert rec["competition"] == "titanic"
    assert "cost_usd" in rec
    assert "ts" in rec


def test_record_unknown_model_uses_default_rate(tmp_path):
    ledger = cl.CostLedger(path=tmp_path / "cost.jsonl")
    cost = ledger.record(
        model="gemini-future-model",
        input_tokens=1000,
        output_tokens=1000,
        cached_tokens=0,
        competition="x",
    )
    assert cost > 0  # falls back to a non-zero default rate


def test_total_for_competition(tmp_path):
    ledger = cl.CostLedger(path=tmp_path / "cost.jsonl")
    for _ in range(3):
        ledger.record(
            model="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=0,
            competition="titanic",
        )
    ledger.record(
        model="gemini-2.5-flash",
        input_tokens=1000,
        output_tokens=500,
        cached_tokens=0,
        competition="house-prices",
    )
    titanic_total = ledger.total_for(competition="titanic")
    other_total = ledger.total_for(competition="house-prices")
    assert titanic_total == pytest.approx(other_total * 3, rel=1e-6)


def test_total_for_all_competitions(tmp_path):
    ledger = cl.CostLedger(path=tmp_path / "cost.jsonl")
    ledger.record(model="gemini-2.5-flash", input_tokens=100, output_tokens=50, cached_tokens=0, competition="a")
    ledger.record(model="gemini-2.5-flash", input_tokens=100, output_tokens=50, cached_tokens=0, competition="b")
    grand = ledger.total_for()
    a = ledger.total_for(competition="a")
    b = ledger.total_for(competition="b")
    assert grand == pytest.approx(a + b, rel=1e-9)


def test_total_for_skips_truncated_trailing_line(tmp_path):
    """A partial line from a crash mid-write must not break totals
    (mirrors Journal.iter_records / calibration.read_history)."""
    ledger = cl.CostLedger(path=tmp_path / "cost.jsonl")
    ledger.record(
        model="gemini-2.5-flash",
        input_tokens=100, output_tokens=50, cached_tokens=0, competition="a",
    )
    good_total = ledger.total_for()
    with ledger.path.open("a") as f:
        f.write('{"ts": "2026-06-10T00:00:00+00:00", "cost_us')  # NO newline, truncated
    assert ledger.total_for() == pytest.approx(good_total)


def test_total_for_skips_null_cost_usd(tmp_path):
    """A row with cost_usd=null must be skipped, not raise TypeError."""
    ledger = cl.CostLedger(path=tmp_path / "cost.jsonl")
    with ledger.path.open("a") as f:
        f.write(json.dumps({"competition": "a", "cost_usd": None}) + "\n")
    ledger.record(
        model="gemini-2.5-flash",
        input_tokens=100, output_tokens=50, cached_tokens=0, competition="a",
    )
    assert ledger.total_for(competition="a") > 0


def test_cached_tokens_billed_at_reduced_rate(tmp_path):
    ledger = cl.CostLedger(path=tmp_path / "cost.jsonl")
    full_cost = ledger.record(
        model="gemini-2.5-flash",
        input_tokens=1000, output_tokens=0, cached_tokens=0, competition="x",
    )
    cached_cost = ledger.record(
        model="gemini-2.5-flash",
        input_tokens=0, output_tokens=0, cached_tokens=1000, competition="x",
    )
    # Cached tokens are billed at ~25% of the input rate; cost must be strictly less
    assert cached_cost < full_cost
    assert cached_cost > 0
