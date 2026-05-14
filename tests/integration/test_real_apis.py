"""Real-API smoke tests — slow tier, opt-in only.

These hit real Gemini + Kaggle endpoints. They will burn a tiny amount of
Gemini quota (≈ $0.0001 per run) and one Kaggle read. Skipped when the
credentials aren't present; marked `slow` so they don't run in default CI.

Run with:
    pytest -m slow tests/integration/test_real_apis.py -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# Load .env at module import so creds are available before any kaggle imports
load_dotenv()

# noqa: E402 — must load .env before importing kaggle-dependent modules
from kaggle_slayer.agent.cost_ledger import CostLedger  # noqa: E402
from kaggle_slayer.agent.llm_client import GeminiClient, Message  # noqa: E402, I001
from kaggle_slayer.harness.kaggle_client import KaggleClient  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture
def gemini_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("no GEMINI_API_KEY / GOOGLE_API_KEY in env or .env")
    return key


@pytest.fixture
def kaggle_creds_present():
    if not (
        os.environ.get("KAGGLE_API_TOKEN")
        or os.environ.get("KAGGLE_USERNAME")
        or (
            (p := os.path.expanduser("~/.kaggle/kaggle.json")) and os.path.exists(p)
        )
        or (
            (p := os.path.expanduser("~/.kaggle/access_token")) and os.path.exists(p)
        )
    ):
        pytest.skip("no Kaggle credentials available")


def test_real_gemini_one_token_smoke(tmp_path, gemini_key):
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    client = GeminiClient(
        api_key=gemini_key,
        ledger=ledger,
        competition="preflight",
        retry_max=1,
    )
    resp = client.call(messages=[Message(role="user", content="Reply with the single word: ok")])
    assert resp.text.lower().startswith("ok")
    assert resp.usage.input_tokens > 0
    assert resp.usage.output_tokens > 0
    assert ledger.total_for(competition="preflight") > 0


def test_real_kaggle_competitions_list(kaggle_creds_present):
    """One read-only call — list competitions."""
    # Don't auto-import kaggle at module top: we need creds in env first.
    from kaggle import api  # type: ignore[import-untyped]
    api.authenticate()
    resp = api.competitions_list(page=1)
    comps = getattr(resp, "competitions", resp)
    assert len(comps) > 0


def test_real_kaggle_view_competition(kaggle_creds_present):
    """List files in a well-known evergreen competition that should always exist."""
    client = KaggleClient()
    files = client.list_files("titanic")
    assert len(files) > 0
    assert any("train" in f.name.lower() for f in files)
