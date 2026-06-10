"""Cost ledger for LLM calls.

Writes one JSON line per call to a configurable JSONL file (default
~/.kaggle_slayer/cost_ledger.jsonl). Each row has timestamp, model,
input/output/cached tokens, computed USD cost, and the per-competition
attribution.

Prices are approximate per-million-token rates as of 2026-05. Update
_PRICE_TABLE when new models drop.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Approximate USD per 1M tokens. Format: (input, cached_input, output).
_PRICE_TABLE: dict[str, tuple[float, float, float]] = {
    "gemini-2.5-flash":   (0.075, 0.01875, 0.30),
    "gemini-2.5-pro":     (1.25,  0.3125,  10.00),
    "gemini-3-pro":       (1.25,  0.3125,  10.00),
    "gemini-3-pro-large": (2.50,  0.625,   20.00),
}
_DEFAULT_RATE: tuple[float, float, float] = (1.25, 0.3125, 10.00)  # = 2.5 Pro

DEFAULT_LEDGER_PATH = Path.home() / ".kaggle_slayer" / "cost_ledger.jsonl"


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _cost_usd(model: str, input_tokens: int, output_tokens: int, cached_tokens: int) -> float:
    in_rate, cached_rate, out_rate = _PRICE_TABLE.get(model, _DEFAULT_RATE)
    return (
        input_tokens * in_rate / 1_000_000
        + cached_tokens * cached_rate / 1_000_000
        + output_tokens * out_rate / 1_000_000
    )


@dataclass
class CostLedger:
    """Append-only ledger of LLM-call costs."""

    path: Path = DEFAULT_LEDGER_PATH

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        competition: str,
    ) -> float:
        cost = _cost_usd(model, input_tokens, output_tokens, cached_tokens)
        row = {
            "ts": _now_iso(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "competition": competition,
            "cost_usd": cost,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return cost

    def total_for(self, *, competition: str | None = None) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        with self.path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Partial trailing write from a crash; skip it (mirrors
                    # Journal.iter_records / calibration.read_history).
                    continue
                if competition is None or rec.get("competition") == competition:
                    try:
                        total += float(rec.get("cost_usd", 0.0))
                    except (TypeError, ValueError):
                        continue
        return total
