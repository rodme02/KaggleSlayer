# Week 2 — Workspace, Kaggle client, LLM client, context builder

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the harness plumbing the Solver agent will need in Week 3 — a per-competition workspace with append-before-return journalling, a Kaggle API client wrapper for the new v2.1 auth format, a `context.md` builder that assembles a competition brief from API + data profile, and a provider-agnostic `LLMClient` with a Gemini implementation, retry, and cost tracking. Plus the CV registry extensions (TimeSeriesSplit, GroupKFold) deferred from Week 1 and one small Week-1 follow-up (`metadata_extra` on `train_cv`).

**Architecture:** Each component is independently testable. The integration test wires them together via a *fake agent* (an `LLMClient` that returns scripted responses) so we exercise the full plumbing without burning real API tokens. One opt-in `slow`-marked test hits real Gemini + Kaggle to confirm the wrappers work against live APIs.

**Tech Stack:** Python 3.11+, pandas, kaggle>=2.1, google-genai, python-dotenv, jsonschema (already in deps), pytest. Mypy strict on `kaggle_slayer/harness` and `kaggle_slayer/agent`.

**Acceptance (from spec §14.3):** Fake agent runs a canned tool sequence on a synthetic competition; harness journals every step correctly into `run_log.jsonl`. Real-API smoke (slow tier) confirms wrappers work against live Gemini + Kaggle. Unit + integration tiers green; coverage ≥90% on the new code.

---

## File map

**Files created this week:**
- `kaggle_slayer/harness/workspace.py` — per-comp folder layout + journaller + resume helper
- `kaggle_slayer/harness/kaggle_client.py` — wrapper around kaggle v2.1 library (read-only ops + submit signature)
- `kaggle_slayer/harness/context.py` — assembles `context.md` from Kaggle metadata + data profile
- `kaggle_slayer/harness/context_template.md` — Jinja-like template (manual string interpolation, no template engine) for `context.md`
- `kaggle_slayer/agent/__init__.py` — new package
- `kaggle_slayer/agent/llm_client.py` — `LLMClient` protocol + dataclasses + `GeminiClient` implementation
- `kaggle_slayer/agent/cost_ledger.py` — JSONL cost tracker + aggregator
- `tests/unit/test_workspace.py`
- `tests/unit/test_workspace_journal.py`
- `tests/unit/test_workspace_resume.py`
- `tests/unit/test_kaggle_client.py`
- `tests/unit/test_context_builder.py`
- `tests/unit/test_llm_client.py`
- `tests/unit/test_cost_ledger.py`
- `tests/fixtures/fake_llm.py` — scripted-response `LLMClient` implementation
- `tests/integration/test_workspace_with_fake_agent.py` — Week-2 acceptance test
- `tests/integration/test_real_apis.py` — `@pytest.mark.slow`, opt-in real API hits

**Files modified:**
- `kaggle_slayer/harness/registry/cv_strategies.py` — add `time_series` (TimeSeriesSplit) and `group_kfold` (GroupKFold); extend `auto_select(date_col=None, group_col=None)`
- `kaggle_slayer/harness/cv.py` — accept `metadata_extra: dict | None = None` (Opus review carry-forward)
- `tests/unit/test_cv_strategies_registry.py` — coverage for new strategies + extended auto_select
- `tests/unit/test_cv_contract.py` — coverage for `metadata_extra` plumbing

---

## Task 1: CV registry — add `TimeSeriesSplit` and `GroupKFold`

Carry-forward from spec §6.3 + Opus review. Week 1 punted these; now they land.

**Files:**
- Modify: `kaggle_slayer/harness/registry/cv_strategies.py`
- Modify: `tests/unit/test_cv_strategies_registry.py`

- [ ] **Step 1: Write failing tests in `tests/unit/test_cv_strategies_registry.py`**

Append at the end of the existing file:

```python
def test_get_time_series_split():
    cv = cv_strategies.get("time_series", n_splits=5)
    assert cv.name == "time_series"
    assert cv.n_splits == 5


def test_time_series_split_yields_forward_only_folds(synthetic_time_series):
    train, target_col, _ = synthetic_time_series
    cv = cv_strategies.get("time_series", n_splits=4)
    for train_idx, val_idx in cv.split(train, target_col):
        # Every train index must be strictly less than every val index
        assert max(train_idx) < min(val_idx)


def test_get_group_kfold():
    cv = cv_strategies.get("group_kfold", n_splits=3, group_col="cat_a")
    assert cv.name == "group_kfold"
    assert cv.n_splits == 3
    assert cv.extra.get("group_col") == "cat_a"


def test_group_kfold_respects_group_boundaries(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.get("group_kfold", n_splits=3, group_col="cat_a")
    for train_idx, val_idx in cv.split(train, target_col):
        train_groups = set(train.iloc[train_idx]["cat_a"])
        val_groups = set(train.iloc[val_idx]["cat_a"])
        # No group appears in both train and val of the same fold
        assert train_groups.isdisjoint(val_groups)


def test_group_kfold_requires_group_col():
    with pytest.raises(ValueError, match="group_col"):
        cv_strategies.get("group_kfold", n_splits=3)  # no group_col


def test_auto_select_picks_time_series_when_date_col_given(synthetic_time_series):
    train, target_col, date_col = synthetic_time_series
    cv = cv_strategies.auto_select(
        problem_type="regression",
        train_df=train,
        target_col=target_col,
        date_col=date_col,
    )
    assert cv.name == "time_series"


def test_auto_select_picks_group_kfold_when_group_col_given(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.auto_select(
        problem_type="classification",
        train_df=train,
        target_col=target_col,
        group_col="cat_a",
    )
    assert cv.name == "group_kfold"
    assert cv.extra.get("group_col") == "cat_a"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/unit/test_cv_strategies_registry.py -v
```

Expected: 7 new failures (KeyError on `time_series`, `group_kfold`; TypeError on `date_col`/`group_col` kwargs).

- [ ] **Step 3: Extend `kaggle_slayer/harness/registry/cv_strategies.py`**

Replace the file with this content (it extends the existing structure; existing tests must still pass):

```python
"""Cross-validation strategy registry.

Wraps sklearn-style splitters under a CVStrategy dataclass. The harness's
train_cv contract consumes these by calling `cv.split(df, target_col)`
and iterating (train_idx, val_idx) tuples.

Week 1: kfold, stratified_kfold.
Week 2: + time_series (TimeSeriesSplit), group_kfold (GroupKFold).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import (  # type: ignore[import-untyped]
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


@dataclass
class CVStrategy:
    """Wraps a sklearn splitter for the harness CV contract."""

    name: str
    n_splits: int
    random_state: int | None = 42
    extra: dict = field(default_factory=dict)
    _splitter: object = None

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Iterator[tuple[list[int], list[int]]]:
        if self._splitter is None:
            raise RuntimeError("CVStrategy._splitter not initialized")
        y = df[target_col]
        groups = None
        if self.extra.get("group_col"):
            groups = df[self.extra["group_col"]]
        if groups is not None:
            it = self._splitter.split(df, y, groups=groups)
        else:
            it = self._splitter.split(df, y)
        for train_idx, val_idx in it:
            yield list(train_idx), list(val_idx)


def _make_kfold(
    n_splits: int, random_state: int | None = 42, shuffle: bool = True
) -> CVStrategy:
    rs = random_state if shuffle else None
    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=rs)
    return CVStrategy(
        name="kfold", n_splits=n_splits, random_state=rs, _splitter=splitter
    )


def _make_stratified_kfold(
    n_splits: int, random_state: int | None = 42, shuffle: bool = True
) -> CVStrategy:
    rs = random_state if shuffle else None
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=rs)
    return CVStrategy(
        name="stratified_kfold", n_splits=n_splits, random_state=rs, _splitter=splitter
    )


def _make_time_series(n_splits: int, **_: object) -> CVStrategy:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return CVStrategy(name="time_series", n_splits=n_splits, _splitter=splitter)


def _make_group_kfold(n_splits: int, group_col: str | None = None, **_: object) -> CVStrategy:
    if not group_col:
        raise ValueError("group_kfold requires a group_col kwarg")
    splitter = GroupKFold(n_splits=n_splits)
    return CVStrategy(
        name="group_kfold",
        n_splits=n_splits,
        random_state=None,
        extra={"group_col": group_col},
        _splitter=splitter,
    )


_FACTORIES = {
    "kfold": _make_kfold,
    "stratified_kfold": _make_stratified_kfold,
    "time_series": _make_time_series,
    "group_kfold": _make_group_kfold,
}


def get(
    name: str,
    *,
    n_splits: int = 5,
    random_state: int | None = 42,
    shuffle: bool = True,
    group_col: str | None = None,
) -> CVStrategy:
    if name not in _FACTORIES:
        raise KeyError(
            f"cv strategy '{name}' not in registry; known: {sorted(_FACTORIES)}"
        )
    factory = _FACTORIES[name]
    if name in ("kfold", "stratified_kfold"):
        return factory(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    if name == "time_series":
        return factory(n_splits=n_splits)
    if name == "group_kfold":
        return factory(n_splits=n_splits, group_col=group_col)
    raise KeyError(name)  # unreachable


def list_strategies() -> list[str]:
    return sorted(_FACTORIES)


def auto_select(
    *,
    problem_type: str,
    train_df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    date_col: str | None = None,
    group_col: str | None = None,
) -> CVStrategy:
    """Pick a sensible default CV strategy.

    Precedence: group_col > date_col > problem_type. The agent can override
    via the explicit `set_cv` tool when context demands it.
    """
    _ = train_df, target_col  # reserved for future heuristics
    if group_col:
        return get("group_kfold", n_splits=n_splits, group_col=group_col)
    if date_col:
        return get("time_series", n_splits=n_splits)
    if problem_type == "classification":
        return get("stratified_kfold", n_splits=n_splits)
    if problem_type == "regression":
        return get("kfold", n_splits=n_splits)
    raise ValueError(
        f"unsupported problem_type '{problem_type}'; expected 'classification' or 'regression'"
    )
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/unit/test_cv_strategies_registry.py -v
```

Expected: all tests pass (existing 9 + 7 new = 16).

- [ ] **Step 5: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/registry/cv_strategies.py tests/unit/test_cv_strategies_registry.py
git commit -m "$(cat <<'EOF'
feat(cv-strategies): add time_series and group_kfold

Registers TimeSeriesSplit as 'time_series' (forward-only folds; no
shuffle) and GroupKFold as 'group_kfold' (requires group_col kwarg;
ensures groups never appear in both train and val of the same fold).

auto_select() now considers date_col and group_col with precedence
group_col > date_col > problem_type. The agent will override via
set_cv() once that tool exists in Week 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `train_cv` accepts `metadata_extra` (Opus review follow-up)

**Files:**
- Modify: `kaggle_slayer/harness/cv.py`
- Modify: `tests/unit/test_cv_contract.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_cv_contract.py`:

```python
def test_train_cv_metadata_extra_merged_into_result(
    fe_pass_through, model_logreg, synthetic_binary
):
    """The caller can pass metadata_extra to enrich CVResult.metadata."""
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("accuracy")
    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
        metadata_extra={"fe_version": "v07", "agent_decision_id": "abc123"},
    )
    assert result.metadata["fe_version"] == "v07"
    assert result.metadata["agent_decision_id"] == "abc123"
    # Built-in metadata still present
    assert result.metadata["cv_strategy"] == "stratified_kfold"
    assert result.metadata["metric"] == "accuracy"


def test_train_cv_metadata_extra_does_not_overwrite_builtin(
    fe_pass_through, model_logreg, synthetic_binary
):
    """metadata_extra cannot clobber the harness-owned keys."""
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("accuracy")
    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
        metadata_extra={"cv_strategy": "evil_override", "fe_version": "v08"},
    )
    # Harness-owned key preserved
    assert result.metadata["cv_strategy"] == "stratified_kfold"
    # User-owned key accepted
    assert result.metadata["fe_version"] == "v08"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_cv_contract.py::test_train_cv_metadata_extra_merged_into_result -v
```

Expected: `TypeError: train_cv() got an unexpected keyword argument 'metadata_extra'`.

- [ ] **Step 3: Modify `kaggle_slayer/harness/cv.py`**

Find the `train_cv` function signature and result construction. Add `metadata_extra` parameter and merge logic.

In the function signature, add the new parameter:

```python
def train_cv(
    *,
    fe_path: str | Path,
    model_path: str | Path,
    train_df: pd.DataFrame,
    target_col: str,
    cv: CVStrategy,
    metric: Metric,
    metadata_extra: dict[str, Any] | None = None,
) -> CVResult:
```

Update the docstring to mention the new parameter (one line under Args). Then at the end of the function, before the `return CVResult(...)`, build the merged metadata:

```python
    duration_s = time.perf_counter() - started

    built_in_metadata = {
        "cv_strategy": cv.name,
        "n_splits": cv.n_splits,
        "metric": metric.name,
        "problem_type": problem_type,
    }
    # metadata_extra cannot clobber built-in keys
    extra = {k: v for k, v in (metadata_extra or {}).items() if k not in built_in_metadata}
    final_metadata = {**extra, **built_in_metadata}

    return CVResult(
        fold_scores=fold_scores,
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores)),
        oof=oof,
        duration_s=duration_s,
        metadata=final_metadata,
    )
```

- [ ] **Step 4: Run all cv_contract tests**

```bash
pytest tests/unit/test_cv_contract.py -v
```

Expected: all pass (existing 9 + 2 new = 11).

- [ ] **Step 5: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/cv.py tests/unit/test_cv_contract.py
git commit -m "$(cat <<'EOF'
feat(cv): accept metadata_extra param on train_cv

Adds a metadata_extra: dict|None param so callers (Week 3 agent loop,
Week 5 MLflow logger) can enrich CVResult.metadata with their own keys
(fe_version, model_version, agent_decision_id, subsample, ...) without
the harness needing to know about them.

Harness-owned keys (cv_strategy, n_splits, metric, problem_type) win in
the merge — extras cannot clobber them.

Carry-forward from Opus's Week 1 final review.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Workspace skeleton + `Workspace` dataclass

**Files:**
- Create: `kaggle_slayer/harness/workspace.py`
- Create: `tests/unit/test_workspace.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for kaggle_slayer.harness.workspace.Workspace."""

from __future__ import annotations

import pytest

from kaggle_slayer.harness import workspace as ws_mod


def test_workspace_create_makes_all_directories(tmp_path):
    w = ws_mod.Workspace.create(root=tmp_path / "competitions" / "titanic")
    assert w.root.is_dir()
    assert w.raw_dir.is_dir()
    assert w.agent_dir.is_dir()
    assert w.versions_dir.is_dir()
    assert w.scratch_dir.is_dir()
    assert w.artifacts_dir.is_dir()
    assert w.submissions_dir.is_dir()
    assert w.run_log_path.parent == w.root
    assert w.notes_path.parent == w.root


def test_workspace_create_is_idempotent(tmp_path):
    root = tmp_path / "competitions" / "titanic"
    w1 = ws_mod.Workspace.create(root=root)
    w2 = ws_mod.Workspace.create(root=root)
    assert w1.root == w2.root
    assert root.is_dir()


def test_workspace_load_existing(tmp_path):
    root = tmp_path / "competitions" / "titanic"
    ws_mod.Workspace.create(root=root)
    loaded = ws_mod.Workspace.load(root=root)
    assert loaded.root == root


def test_workspace_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no workspace at"):
        ws_mod.Workspace.load(root=tmp_path / "does_not_exist")


def test_workspace_context_path(tmp_path):
    w = ws_mod.Workspace.create(root=tmp_path / "comp")
    assert w.context_path == w.root / "context.md"


def test_workspace_fe_and_model_paths(tmp_path):
    w = ws_mod.Workspace.create(root=tmp_path / "comp")
    assert w.fe_path == w.agent_dir / "fe.py"
    assert w.model_path == w.agent_dir / "model.py"


def test_workspace_competition_name_derived_from_dir(tmp_path):
    w = ws_mod.Workspace.create(root=tmp_path / "competitions" / "house-prices")
    assert w.name == "house-prices"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_workspace.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.workspace`.

- [ ] **Step 3: Create `kaggle_slayer/harness/workspace.py`**

```python
"""Per-competition workspace.

Spec §10 layout:

    competitions/<name>/
        raw/                  Kaggle download (gitignored)
        context.md            auto-generated brief
        agent/                ALL agent-written code
            fe.py
            model.py
            versions/         fe_v01.py, model_v01.py, ...
            scratch/          one-off scripts via run_python
        artifacts/            pipeline.pkl, oof_preds.npy, ...
        submissions/          dated CSVs + leaderboard.jsonl
        notes.jsonl           agent's scratchpad
        run_log.jsonl         tool-call audit log

A `Workspace` is just typed paths plus a couple of create/load helpers.
Journalling and resume live in separate modules (journal.py, resume.py)
so this file stays small and focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    """Typed view of a per-competition workspace directory."""

    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def agent_dir(self) -> Path:
        return self.root / "agent"

    @property
    def versions_dir(self) -> Path:
        return self.agent_dir / "versions"

    @property
    def scratch_dir(self) -> Path:
        return self.agent_dir / "scratch"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def submissions_dir(self) -> Path:
        return self.root / "submissions"

    @property
    def context_path(self) -> Path:
        return self.root / "context.md"

    @property
    def fe_path(self) -> Path:
        return self.agent_dir / "fe.py"

    @property
    def model_path(self) -> Path:
        return self.agent_dir / "model.py"

    @property
    def run_log_path(self) -> Path:
        return self.root / "run_log.jsonl"

    @property
    def notes_path(self) -> Path:
        return self.root / "notes.jsonl"

    @classmethod
    def create(cls, root: Path) -> "Workspace":
        """Create the workspace directory structure (idempotent)."""
        root = Path(root)
        for sub in (
            root,
            root / "raw",
            root / "agent",
            root / "agent" / "versions",
            root / "agent" / "scratch",
            root / "artifacts",
            root / "submissions",
        ):
            sub.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    @classmethod
    def load(cls, root: Path) -> "Workspace":
        """Open an existing workspace; raises if not present."""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"no workspace at {root}")
        return cls(root=root)
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/unit/test_workspace.py -v
```

Expected: 7 passes.

- [ ] **Step 5: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/workspace.py tests/unit/test_workspace.py
git commit -m "$(cat <<'EOF'
feat(workspace): add per-competition Workspace dataclass

Typed view of the per-comp directory layout from spec §10. create() is
idempotent (safe to call on existing workspaces); load() raises if the
root is missing.

Journalling and resume helpers land in subsequent tasks; this file stays
small and focused on the path layout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Workspace journaller

**Files:**
- Create: `kaggle_slayer/harness/journal.py`
- Create: `tests/unit/test_workspace_journal.py`

The journaller writes one JSON object per line to `run_log.jsonl` (tool-call audit) or `notes.jsonl` (agent scratchpad). Critical property: the write is flushed before the function returns, so a crash after the call doesn't lose the entry.

- [ ] **Step 1: Write failing tests**

```python
"""Tests for kaggle_slayer.harness.journal.Journal."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.harness import journal as journal_mod
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def fresh_workspace(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_log_tool_call_appends_one_line(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.log_tool_call(
        tool="load_competition",
        args={"name": "titanic"},
        result_summary="loaded 891 train rows, 418 test rows",
    )
    lines = fresh_workspace.run_log_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "tool_call"
    assert rec["tool"] == "load_competition"
    assert rec["args"] == {"name": "titanic"}
    assert rec["result_summary"].startswith("loaded 891")
    assert "ts" in rec  # ISO timestamp


def test_log_multiple_tool_calls_are_appended(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    for i in range(3):
        j.log_tool_call(tool="probe", args={"i": i}, result_summary=f"r{i}")
    lines = fresh_workspace.run_log_path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["args"]["i"] for line in lines] == [0, 1, 2]


def test_log_tool_error_records_failure(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.log_tool_error(
        tool="submit_kaggle",
        args={"comp": "titanic"},
        error="403 Forbidden: rules not accepted",
    )
    rec = json.loads(fresh_workspace.run_log_path.read_text().splitlines()[0])
    assert rec["kind"] == "tool_error"
    assert rec["error"] == "403 Forbidden: rules not accepted"


def test_take_note_writes_to_notes_jsonl(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.take_note(category="observation", content="target column is heavily imbalanced (10% positive)")
    lines = fresh_workspace.notes_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["category"] == "observation"
    assert rec["content"].startswith("target column")


def test_take_note_rejects_unknown_category(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    with pytest.raises(ValueError, match="unknown category"):
        j.take_note(category="random", content="x")


def test_list_notes_filters_by_category(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.take_note(category="observation", content="a")
    j.take_note(category="decision", content="b")
    j.take_note(category="observation", content="c")
    obs = j.list_notes(category="observation")
    assert len(obs) == 2
    assert [n["content"] for n in obs] == ["a", "c"]


def test_iter_tool_calls_returns_dicts(fresh_workspace):
    j = journal_mod.Journal(fresh_workspace)
    j.log_tool_call(tool="a", args={}, result_summary="x")
    j.log_tool_call(tool="b", args={}, result_summary="y")
    records = list(j.iter_records())
    assert [r["tool"] for r in records] == ["a", "b"]


def test_append_before_return_is_durable(fresh_workspace, monkeypatch):
    """If we crash immediately after a journal call, the record must be on disk."""
    j = journal_mod.Journal(fresh_workspace)
    j.log_tool_call(tool="profile_data", args={}, result_summary="ok")
    # Read the file back from a fresh Journal — should see the entry
    j2 = journal_mod.Journal(fresh_workspace)
    records = list(j2.iter_records())
    assert len(records) == 1
    assert records[0]["tool"] == "profile_data"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_workspace_journal.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.journal`.

- [ ] **Step 3: Create `kaggle_slayer/harness/journal.py`**

```python
"""Append-only journaller for per-competition workspaces.

run_log.jsonl  — tool-call audit log, machine-readable
notes.jsonl    — agent's scratchpad, structured by category

Every append flushes and fsyncs before returning, so a crash immediately
after a `log_tool_call(...)` call does not lose the entry. This is the
durability property the spec §12 resume mechanism depends on.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from typing import Any

from kaggle_slayer.harness.workspace import Workspace

NOTE_CATEGORIES: frozenset[str] = frozenset(
    {"observation", "decision", "hypothesis", "todo"}
)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Journal:
    """Append-only journaller bound to a Workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    # --- run_log.jsonl ---

    def log_tool_call(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        result_summary: str,
    ) -> None:
        self._append(
            self.workspace.run_log_path,
            {
                "ts": _now_iso(),
                "kind": "tool_call",
                "tool": tool,
                "args": args,
                "result_summary": result_summary,
            },
        )

    def log_tool_error(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        error: str,
    ) -> None:
        self._append(
            self.workspace.run_log_path,
            {
                "ts": _now_iso(),
                "kind": "tool_error",
                "tool": tool,
                "args": args,
                "error": error,
            },
        )

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield every record from run_log.jsonl, in order."""
        path = self.workspace.run_log_path
        if not path.exists():
            return
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    # --- notes.jsonl ---

    def take_note(self, *, category: str, content: str) -> None:
        if category not in NOTE_CATEGORIES:
            raise ValueError(
                f"unknown category '{category}'; allowed: {sorted(NOTE_CATEGORIES)}"
            )
        self._append(
            self.workspace.notes_path,
            {"ts": _now_iso(), "category": category, "content": content},
        )

    def list_notes(self, *, category: str | None = None) -> list[dict[str, Any]]:
        path = self.workspace.notes_path
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if category is None or rec.get("category") == category:
                    records.append(rec)
        return records

    # --- internal: durable append ---

    @staticmethod
    def _append(path, record: dict[str, Any]) -> None:
        """Append a JSON line and flush+fsync before returning."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            import os
            os.fsync(f.fileno())
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/unit/test_workspace_journal.py -v
```

Expected: 8 passes.

- [ ] **Step 5: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/journal.py tests/unit/test_workspace_journal.py
git commit -m "$(cat <<'EOF'
feat(workspace): add append-only Journal for run_log + notes

Journal writes one JSON object per line to run_log.jsonl (tool-call
audit) or notes.jsonl (agent scratchpad). Every append flushes and
fsyncs before returning, so a crash immediately after a log call does
not lose the entry — the durability property spec §12 resume depends on.

NOTE_CATEGORIES are observation/decision/hypothesis/todo; unknown
categories raise ValueError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Workspace resume helper

**Files:**
- Create: `kaggle_slayer/harness/resume.py`
- Create: `tests/unit/test_workspace_resume.py`

The resume helper reconstructs a "what happened" summary from `run_log.jsonl`. For Week 2 this is a simple replay — Week 4 will extend it to rebuild the agent's conversation history for re-invocation.

- [ ] **Step 1: Write failing tests**

```python
"""Tests for kaggle_slayer.harness.resume."""

from __future__ import annotations

import pytest

from kaggle_slayer.harness import journal as journal_mod, resume as resume_mod
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def populated_workspace(tmp_path):
    w = Workspace.create(root=tmp_path / "comp")
    j = journal_mod.Journal(w)
    j.log_tool_call(tool="load_competition", args={"name": "titanic"}, result_summary="loaded")
    j.log_tool_call(tool="profile_data", args={}, result_summary="891 rows, 12 cols")
    j.log_tool_error(tool="submit_kaggle", args={}, error="403 Forbidden")
    j.log_tool_call(tool="train_cv", args={}, result_summary="cv=0.823")
    return w


def test_resume_summary_empty_workspace(tmp_path):
    w = Workspace.create(root=tmp_path / "empty")
    summary = resume_mod.summarize(w)
    assert summary.total_calls == 0
    assert summary.tool_counts == {}
    assert summary.last_call is None
    assert summary.error_count == 0


def test_resume_summary_counts_per_tool(populated_workspace):
    summary = resume_mod.summarize(populated_workspace)
    assert summary.total_calls == 4
    assert summary.tool_counts == {
        "load_competition": 1,
        "profile_data": 1,
        "submit_kaggle": 1,
        "train_cv": 1,
    }
    assert summary.error_count == 1


def test_resume_summary_last_call(populated_workspace):
    summary = resume_mod.summarize(populated_workspace)
    assert summary.last_call is not None
    assert summary.last_call["tool"] == "train_cv"
    assert summary.last_call["kind"] == "tool_call"


def test_resume_summary_detects_stuck_loop(tmp_path):
    """5+ identical (tool, args) in a 10-call window indicates a stuck loop."""
    w = Workspace.create(root=tmp_path / "stuck")
    j = journal_mod.Journal(w)
    for _ in range(6):
        j.log_tool_call(tool="train_cv", args={"fe": "agent/fe.py"}, result_summary="failed")
    summary = resume_mod.summarize(w)
    assert summary.stuck_loop is not None
    assert summary.stuck_loop["tool"] == "train_cv"
    assert summary.stuck_loop["repeats"] >= 5


def test_resume_summary_no_stuck_loop_when_args_vary(tmp_path):
    w = Workspace.create(root=tmp_path / "ok")
    j = journal_mod.Journal(w)
    for i in range(6):
        j.log_tool_call(tool="train_cv", args={"fe": f"agent/fe_v{i}.py"}, result_summary=f"cv={0.8 + i * 0.01}")
    summary = resume_mod.summarize(w)
    assert summary.stuck_loop is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_workspace_resume.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.resume`.

- [ ] **Step 3: Create `kaggle_slayer/harness/resume.py`**

```python
"""Workspace resume / inspection helpers.

For Week 2 this is a read-only summary: count tool calls, detect stuck
loops, return the last call. Week 4 will extend it to rebuild the
Solver's conversation history so an aborted run can pick up where it
left off.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class ResumeSummary:
    total_calls: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    last_call: dict[str, Any] | None = None
    stuck_loop: dict[str, Any] | None = None


def summarize(workspace: Workspace, *, stuck_window: int = 10, stuck_threshold: int = 5) -> ResumeSummary:
    """Read run_log.jsonl and return a high-level summary.

    stuck_loop detection: if the same (tool, args) tuple appears
    `stuck_threshold` times within the last `stuck_window` calls, flag it.
    """
    j = Journal(workspace)
    records = list(j.iter_records())

    summary = ResumeSummary(total_calls=len(records))
    if not records:
        return summary

    counts: Counter[str] = Counter()
    for r in records:
        counts[r["tool"]] += 1
        if r["kind"] == "tool_error":
            summary.error_count += 1
    summary.tool_counts = dict(counts)
    summary.last_call = records[-1]

    # Stuck loop: tally (tool, hash(args)) over the trailing window
    window = records[-stuck_window:]
    sigs: Counter[tuple[str, str]] = Counter()
    for r in window:
        sig = (r["tool"], json.dumps(r.get("args", {}), sort_keys=True))
        sigs[sig] += 1
    for (tool, args_repr), count in sigs.most_common(1):
        if count >= stuck_threshold:
            summary.stuck_loop = {
                "tool": tool,
                "args": json.loads(args_repr),
                "repeats": count,
                "window": stuck_window,
            }
    return summary
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/unit/test_workspace_resume.py -v
```

Expected: 5 passes.

- [ ] **Step 5: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/resume.py tests/unit/test_workspace_resume.py
git commit -m "$(cat <<'EOF'
feat(workspace): add resume.summarize() inspecting run_log

ResumeSummary captures total_calls, tool_counts, error_count, last_call,
and a stuck_loop detector (same (tool, args) ≥5x within trailing window
of 10 — same detector the spec §11.2 telemetry layer will surface).

This is the read-only foundation for Week 4's full conversation-replay
resume mechanism.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Kaggle client wrapper

**Files:**
- Create: `kaggle_slayer/harness/kaggle_client.py`
- Create: `tests/unit/test_kaggle_client.py`

Wraps kaggle library v2.1 with the new response shapes. Most tests use a mock kaggle module; one real-API test lives in `test_real_apis.py` (Task 13).

- [ ] **Step 1: Write failing tests**

```python
"""Tests for kaggle_slayer.harness.kaggle_client.

Most tests mock the underlying kaggle library; live-API tests live in
tests/integration/test_real_apis.py (slow tier, skipped by default).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kaggle_slayer.harness import kaggle_client as kc_mod


@pytest.fixture
def mock_api(monkeypatch):
    """Patch kaggle_client._get_api() to return a MagicMock."""
    api = MagicMock(name="kaggle_api")
    monkeypatch.setattr(kc_mod, "_get_api", lambda: api)
    return api


def test_client_view_competition(mock_api):
    resp = MagicMock()
    resp.title = "Titanic - Machine Learning from Disaster"
    resp.description = "Predict survival on the Titanic..."
    resp.evaluation_metric = "accuracy"
    mock_api.competition_view.return_value = resp

    client = kc_mod.KaggleClient()
    info = client.view_competition("titanic")

    mock_api.competition_view.assert_called_once_with("titanic")
    assert info.title.startswith("Titanic")
    assert info.description.startswith("Predict")
    assert info.metric == "accuracy"


def test_client_view_competition_handles_missing_metric(mock_api):
    resp = MagicMock(spec=["title", "description"])
    resp.title = "Untitled Comp"
    resp.description = "..."
    mock_api.competition_view.return_value = resp

    client = kc_mod.KaggleClient()
    info = client.view_competition("foo")
    assert info.metric is None


def test_client_list_files(mock_api):
    file_a = MagicMock()
    file_a.name = "train.csv"
    file_a.size = 60302
    file_b = MagicMock()
    file_b.name = "test.csv"
    file_b.size = 28629
    resp = MagicMock()
    resp.files = [file_a, file_b]
    mock_api.competition_list_files.return_value = resp

    client = kc_mod.KaggleClient()
    files = client.list_files("titanic")
    assert [f.name for f in files] == ["train.csv", "test.csv"]
    assert files[0].size == 60302


def test_client_get_leaderboard(mock_api):
    e1 = MagicMock()
    e1.team_name = "team_a"
    e1.score = "1.00000"
    e2 = MagicMock()
    e2.team_name = "team_b"
    e2.score = "0.99999"
    resp = MagicMock()
    resp.submissions = [e1, e2]
    mock_api.competition_view_leaderboard.return_value = resp

    client = kc_mod.KaggleClient()
    lb = client.get_leaderboard("titanic", top_n=2)
    assert len(lb) == 2
    assert lb[0].team_name == "team_a"
    assert lb[0].score == 1.0


def test_client_get_leaderboard_truncates_to_top_n(mock_api):
    entries = []
    for i in range(20):
        e = MagicMock()
        e.team_name = f"team_{i}"
        e.score = f"{1.0 - i * 0.01:.5f}"
        entries.append(e)
    resp = MagicMock()
    resp.submissions = entries
    mock_api.competition_view_leaderboard.return_value = resp

    client = kc_mod.KaggleClient()
    lb = client.get_leaderboard("titanic", top_n=5)
    assert len(lb) == 5
    assert lb[0].team_name == "team_0"


def test_client_download_returns_target_dir(mock_api, tmp_path):
    target = tmp_path / "raw"
    client = kc_mod.KaggleClient()
    result = client.download("titanic", dest=target)
    mock_api.competition_download_files.assert_called_once()
    args, kwargs = mock_api.competition_download_files.call_args
    assert args[0] == "titanic" or kwargs.get("competition") == "titanic"
    assert result == target


def test_client_submit(mock_api, tmp_path):
    csv = tmp_path / "submission.csv"
    csv.write_text("id,target\n1,0\n")
    mock_api.competition_submit.return_value = MagicMock(spec=[])

    client = kc_mod.KaggleClient()
    client.submit("titanic", csv_path=csv, message="cv=0.842")
    mock_api.competition_submit.assert_called_once()


def test_client_submit_rejects_missing_csv(mock_api, tmp_path):
    client = kc_mod.KaggleClient()
    with pytest.raises(FileNotFoundError):
        client.submit("titanic", csv_path=tmp_path / "nope.csv", message="x")
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_kaggle_client.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.kaggle_client`.

- [ ] **Step 3: Create `kaggle_slayer/harness/kaggle_client.py`**

```python
"""Wrapper around the kaggle library (v2.1+ structured-response shape).

Provides typed return values (CompetitionInfo, CompetitionFile, LBEntry)
so the rest of the harness doesn't depend on kaggle's internal dataclasses.

The kaggle library authenticates on first use from KAGGLE_API_TOKEN or
~/.kaggle/access_token (new format) or KAGGLE_USERNAME+KAGGLE_KEY or
~/.kaggle/kaggle.json (legacy). The wrapper does not duplicate that
logic; if no creds are present, the underlying library raises and we
propagate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompetitionInfo:
    title: str
    description: str
    metric: str | None


@dataclass(frozen=True)
class CompetitionFile:
    name: str
    size: int


@dataclass(frozen=True)
class LBEntry:
    team_name: str
    score: float


def _get_api() -> Any:  # noqa: ANN401 — kaggle library is untyped
    """Lazy import + authenticate; kept as a function so tests can patch it."""
    from kaggle import api  # type: ignore[import-untyped]

    api.authenticate()
    return api


def _safe_attr(obj: object, name: str, default: object = None) -> Any:  # noqa: ANN401
    return getattr(obj, name, default)


class KaggleClient:
    """Read-only-by-default wrapper. submit() is the one write op."""

    def view_competition(self, name: str) -> CompetitionInfo:
        api = _get_api()
        resp = api.competition_view(name)
        return CompetitionInfo(
            title=_safe_attr(resp, "title", ""),
            description=_safe_attr(resp, "description", ""),
            metric=_safe_attr(resp, "evaluation_metric"),
        )

    def list_files(self, name: str) -> list[CompetitionFile]:
        api = _get_api()
        resp = api.competition_list_files(name)
        files = _safe_attr(resp, "files", []) or []
        return [
            CompetitionFile(
                name=_safe_attr(f, "name", ""),
                size=int(_safe_attr(f, "size", 0) or 0),
            )
            for f in files
        ]

    def download(self, name: str, *, dest: Path) -> Path:
        """Download all competition files into `dest`. Returns dest."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        api = _get_api()
        api.competition_download_files(name, path=str(dest))
        return dest

    def get_leaderboard(self, name: str, *, top_n: int = 50) -> list[LBEntry]:
        api = _get_api()
        resp = api.competition_view_leaderboard(name)
        entries = _safe_attr(resp, "submissions", []) or []
        result: list[LBEntry] = []
        for entry in entries[:top_n]:
            raw_score = _safe_attr(entry, "score", "0")
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            result.append(LBEntry(team_name=_safe_attr(entry, "team_name", ""), score=score))
        return result

    def submit(self, name: str, *, csv_path: Path, message: str) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"submission CSV not found: {csv_path}")
        api = _get_api()
        api.competition_submit(str(csv_path), message, name)
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/unit/test_kaggle_client.py -v
```

Expected: 8 passes.

- [ ] **Step 5: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/kaggle_client.py tests/unit/test_kaggle_client.py
git commit -m "$(cat <<'EOF'
feat(kaggle): add KaggleClient wrapper for v2.1 library

Wraps the kaggle library's v2.1 structured-response shape into typed
dataclasses (CompetitionInfo, CompetitionFile, LBEntry) so the rest of
the harness doesn't depend on kaggle's internals.

Read ops: view_competition, list_files, get_leaderboard, download.
Write op: submit (one method, file-validated, no autonomous fallback).

Auth is delegated to the kaggle library which reads KAGGLE_API_TOKEN /
access_token / legacy formats. _get_api() is a lazy seam tests can patch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Cost ledger

**Files:**
- Create: `kaggle_slayer/agent/__init__.py`
- Create: `kaggle_slayer/agent/cost_ledger.py`
- Create: `tests/unit/test_cost_ledger.py`

JSONL append, one row per LLM call. Per-model price table for USD calculation. Aggregator for per-comp / per-day rollups.

- [ ] **Step 1: Create the agent package skeleton**

```python
# kaggle_slayer/agent/__init__.py
"""LLM-side of the harness: LLMClient, cost ledger, future solver loop."""
```

- [ ] **Step 2: Write failing tests**

```python
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
```

- [ ] **Step 3: Run to confirm failure**

```bash
pytest tests/unit/test_cost_ledger.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.agent.cost_ledger`.

- [ ] **Step 4: Create `kaggle_slayer/agent/cost_ledger.py`**

```python
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
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


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
                rec = json.loads(line)
                if competition is None or rec.get("competition") == competition:
                    total += float(rec.get("cost_usd", 0.0))
        return total
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/test_cost_ledger.py -v
```

Expected: 6 passes.

- [ ] **Step 6: Lint + type-check**

mypy is configured for `kaggle_slayer/harness` only. We need to extend it to also cover `kaggle_slayer/agent`. Edit `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.11"
files = ["kaggle_slayer/harness", "kaggle_slayer/agent"]
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
exclude = ["legacy/.*", "tests/.*"]
```

Then run:

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add kaggle_slayer/agent/__init__.py kaggle_slayer/agent/cost_ledger.py tests/unit/test_cost_ledger.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat(agent): add cost ledger + agent package skeleton

CostLedger writes one JSONL row per LLM call with computed USD cost
based on _PRICE_TABLE (gemini-2.5-flash, 2.5-pro, 3-pro, 3-pro-large).
Cached tokens billed at the reduced cached-input rate. Unknown models
fall back to the 2.5-pro rate (conservative).

total_for(competition=...) aggregates per-comp; total_for() rolls up
everything.

Also extends mypy strict to cover kaggle_slayer/agent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: LLMClient protocol + dataclasses

**Files:**
- Create: `kaggle_slayer/agent/llm_client.py`
- Create: `tests/unit/test_llm_client.py`

We split this into two tasks: this task defines the protocol + dataclasses + the abstract structure; the next task (Task 9) implements the Gemini concrete class. Splitting keeps each commit small.

- [ ] **Step 1: Write failing tests for the protocol + dataclasses**

```python
"""Tests for kaggle_slayer.agent.llm_client.

This task covers the protocol + dataclasses. Gemini implementation tests
land in Task 9.
"""

from __future__ import annotations

from kaggle_slayer.agent import llm_client as llm


def test_message_dataclass():
    m = llm.Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_response_dataclass_defaults():
    r = llm.Response(text="ok", tool_calls=[], usage=llm.Usage(0, 0, 0))
    assert r.text == "ok"
    assert r.tool_calls == []
    assert r.usage.input_tokens == 0


def test_usage_dataclass():
    u = llm.Usage(input_tokens=10, output_tokens=5, cached_tokens=2)
    assert u.input_tokens == 10
    assert u.output_tokens == 5
    assert u.cached_tokens == 2
    assert u.total == 15  # input + output (cached is included in input but tracked separately)


def test_tool_call_dataclass():
    tc = llm.ToolCall(id="call_1", name="train_cv", args={"fe": "agent/fe.py"})
    assert tc.id == "call_1"
    assert tc.name == "train_cv"
    assert tc.args == {"fe": "agent/fe.py"}


def test_llm_client_protocol_exposes_call():
    """LLMClient is a Protocol with a call method.

    Structural conformance is verified end-to-end in test_fake_llm.py
    (FakeLLMClient is isinstance(LLMClient)). Here we just confirm the
    protocol object exists and exposes the expected name.
    """
    assert hasattr(llm.LLMClient, "call")
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_llm_client.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.agent.llm_client`.

- [ ] **Step 3: Create `kaggle_slayer/agent/llm_client.py`**

```python
"""Provider-agnostic LLMClient protocol + dataclasses.

The protocol is intentionally tiny: a single `call(messages, tools)` method
returning a Response. Concrete implementations live alongside (GeminiClient
in this same module; Claude/OpenAI clients later if needed).

ToolCall captures one function-call request from the model. Usage captures
token counts so the harness can compute cost via the CostLedger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "model" | "system" | "tool"
    content: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Response:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    raw: Any = field(default=None, repr=False)  # provider-specific response object


@runtime_checkable
class LLMClient(Protocol):
    """A provider-agnostic LLM interface."""

    def call(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Response:
        ...
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_llm_client.py -v
```

Expected: 5 passes.

- [ ] **Step 5: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/agent/llm_client.py tests/unit/test_llm_client.py
git commit -m "$(cat <<'EOF'
feat(agent): add LLMClient protocol + Message/ToolCall/Response dataclasses

Provider-agnostic interface — one call() method, typed dataclasses for
Message, ToolCall, Usage, Response. The Protocol is runtime_checkable so
test fakes can be validated via isinstance() if needed.

Gemini concrete impl lands in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: GeminiClient implementation + retry + cost tracking

**Files:**
- Modify: `kaggle_slayer/agent/llm_client.py` — add `GeminiClient`
- Modify: `tests/unit/test_llm_client.py` — add Gemini tests with mock client

- [ ] **Step 1: Add failing tests at end of `tests/unit/test_llm_client.py`**

```python
from unittest.mock import MagicMock, patch

from kaggle_slayer.agent.cost_ledger import CostLedger


def _fake_genai_response(text: str, in_tok: int = 10, out_tok: int = 5, cached_tok: int = 0):
    resp = MagicMock()
    resp.text = text
    candidate = MagicMock()
    candidate.content.parts = []  # no tool calls
    resp.candidates = [candidate]
    usage = MagicMock()
    usage.prompt_token_count = in_tok
    usage.candidates_token_count = out_tok
    usage.cached_content_token_count = cached_tok
    resp.usage_metadata = usage
    return resp


def test_gemini_client_call_returns_response(tmp_path):
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as mock_factory:
        client_impl = MagicMock()
        client_impl.models.generate_content.return_value = _fake_genai_response("hello", 100, 50)
        mock_factory.return_value = client_impl

        client = llm.GeminiClient(api_key="fake", ledger=ledger, competition="test-comp")
        resp = client.call(
            messages=[llm.Message(role="user", content="hi")],
            model="gemini-2.5-flash",
        )

    assert resp.text == "hello"
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 50
    # Ledger should have one entry attributed to test-comp
    assert ledger.total_for(competition="test-comp") > 0


def test_gemini_client_retries_on_transient_error(tmp_path):
    """Transient errors (rate limit, 5xx) should retry up to 3 times."""
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as mock_factory:
        client_impl = MagicMock()
        # Two transient errors, then success
        good = _fake_genai_response("ok", 5, 2)
        client_impl.models.generate_content.side_effect = [
            llm.TransientLLMError("rate limit"),
            llm.TransientLLMError("temporarily unavailable"),
            good,
        ]
        mock_factory.return_value = client_impl

        client = llm.GeminiClient(
            api_key="fake", ledger=ledger, competition="x",
            retry_max=3, retry_base_delay_s=0.0,
        )
        resp = client.call(messages=[llm.Message(role="user", content="hi")])

    assert resp.text == "ok"
    assert client_impl.models.generate_content.call_count == 3


def test_gemini_client_gives_up_after_retry_max(tmp_path):
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as mock_factory:
        client_impl = MagicMock()
        client_impl.models.generate_content.side_effect = llm.TransientLLMError("nope")
        mock_factory.return_value = client_impl

        client = llm.GeminiClient(
            api_key="fake", ledger=ledger, competition="x",
            retry_max=2, retry_base_delay_s=0.0,
        )
        with pytest.raises(llm.TransientLLMError):
            client.call(messages=[llm.Message(role="user", content="hi")])
    assert client_impl.models.generate_content.call_count == 3  # initial + 2 retries


def test_gemini_client_does_not_retry_on_permanent_error(tmp_path):
    """Auth errors, malformed requests, etc. should NOT retry."""
    ledger = CostLedger(path=tmp_path / "cost.jsonl")

    class _AuthError(Exception):
        pass

    with patch("kaggle_slayer.agent.llm_client._make_genai_client") as mock_factory:
        client_impl = MagicMock()
        client_impl.models.generate_content.side_effect = _AuthError("invalid key")
        mock_factory.return_value = client_impl

        client = llm.GeminiClient(
            api_key="fake", ledger=ledger, competition="x",
            retry_max=3, retry_base_delay_s=0.0,
        )
        with pytest.raises(_AuthError):
            client.call(messages=[llm.Message(role="user", content="hi")])
    assert client_impl.models.generate_content.call_count == 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_llm_client.py -v
```

Expected: 4 new tests fail (`AttributeError: module ... has no attribute 'GeminiClient'`).

- [ ] **Step 3: Extend `kaggle_slayer/agent/llm_client.py`** — append after the protocol:

```python
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaggle_slayer.agent.cost_ledger import CostLedger


class TransientLLMError(Exception):
    """Retryable error — rate limit, timeout, transient 5xx, etc."""


def _make_genai_client(api_key: str) -> Any:  # noqa: ANN401 — google-genai is loosely typed
    """Construct a google-genai Client. Factored out so tests can patch it."""
    from google import genai

    return genai.Client(api_key=api_key)


def _messages_to_genai_contents(messages: list[Message]) -> str:
    """Flatten messages to the simple string form Gemini accepts for now.

    Week 3 will replace this with the structured contents-array form once we
    have multi-turn conversations and tool messages to encode.
    """
    parts = []
    for m in messages:
        prefix = {"user": "USER", "model": "MODEL", "system": "SYSTEM", "tool": "TOOL"}.get(
            m.role, m.role.upper()
        )
        parts.append(f"{prefix}: {m.content}")
    return "\n\n".join(parts)


def _is_transient(err: Exception) -> bool:
    if isinstance(err, TransientLLMError):
        return True
    msg = str(err).lower()
    return any(s in msg for s in ("rate limit", "timeout", "temporarily", "503", "429", "500", "502", "504"))


class GeminiClient:
    """Concrete LLMClient for Google Gemini via google-genai."""

    def __init__(
        self,
        *,
        api_key: str,
        ledger: "CostLedger",
        competition: str,
        default_model: str = "gemini-2.5-flash",
        retry_max: int = 3,
        retry_base_delay_s: float = 1.0,
    ) -> None:
        self._client = _make_genai_client(api_key)
        self._ledger = ledger
        self._competition = competition
        self._default_model = default_model
        self._retry_max = retry_max
        self._retry_base_delay_s = retry_base_delay_s

    def call(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Response:
        _ = tools  # Tool-use schema translation lands in Week 3
        chosen_model = model or self._default_model
        contents = _messages_to_genai_contents(messages)

        last_err: Exception | None = None
        for attempt in range(self._retry_max + 1):
            try:
                raw = self._client.models.generate_content(
                    model=chosen_model,
                    contents=contents,
                )
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if not _is_transient(e) or attempt == self._retry_max:
                    raise
                delay = self._retry_base_delay_s * (2 ** attempt)
                time.sleep(delay)
        else:  # unreachable but mypy is happier
            raise last_err or RuntimeError("unreachable")

        usage = raw.usage_metadata
        u = Usage(
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            cached_tokens=int(getattr(usage, "cached_content_token_count", 0) or 0),
        )
        self._ledger.record(
            model=chosen_model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cached_tokens=u.cached_tokens,
            competition=self._competition,
        )
        return Response(
            text=(raw.text or "").strip(),
            tool_calls=[],  # parsing tool-call parts lands in Week 3
            usage=u,
            raw=raw,
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_llm_client.py -v
```

Expected: 9 passes (5 from Task 8 + 4 new).

- [ ] **Step 5: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/agent/llm_client.py tests/unit/test_llm_client.py
git commit -m "$(cat <<'EOF'
feat(agent): add GeminiClient with retry + cost tracking

Concrete LLMClient implementation for Google Gemini via google-genai.
Wraps responses into the typed Response dataclass, records per-call
usage into the CostLedger attributed to the bound competition.

Retry policy: exponential backoff (base * 2^attempt seconds) on
TransientLLMError or messages matching common transient patterns
(rate limit, 5xx, timeout). Permanent errors (auth, malformed) raise
immediately.

Tool-call parsing is deferred to Week 3 — the `tools` param is accepted
for forward-compat but currently ignored.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Context.md builder

**Files:**
- Create: `kaggle_slayer/harness/context.py`
- Create: `tests/unit/test_context_builder.py`

Assembles a `context.md` from Kaggle metadata + data profile. Heavy reliance on mock `KaggleClient`. The template is inlined (not a separate file) for now — simple enough that a Jinja-style template would be overkill.

- [ ] **Step 1: Write failing tests**

```python
"""Tests for kaggle_slayer.harness.context.build_context."""

from __future__ import annotations

import pandas as pd
import pytest

from kaggle_slayer.harness import context as ctx_mod
from kaggle_slayer.harness.kaggle_client import CompetitionFile, CompetitionInfo, LBEntry
from kaggle_slayer.harness.workspace import Workspace


class FakeKaggleClient:
    def __init__(
        self,
        info: CompetitionInfo,
        files: list[CompetitionFile],
        lb: list[LBEntry],
    ):
        self.info = info
        self.files = files
        self.lb = lb

    def view_competition(self, name: str) -> CompetitionInfo:
        return self.info

    def list_files(self, name: str) -> list[CompetitionFile]:
        return self.files

    def get_leaderboard(self, name: str, *, top_n: int = 10) -> list[LBEntry]:
        return self.lb[:top_n]


@pytest.fixture
def workspace(tmp_path):
    return Workspace.create(root=tmp_path / "titanic")


@pytest.fixture
def kaggle_fake():
    return FakeKaggleClient(
        info=CompetitionInfo(
            title="Titanic - Machine Learning from Disaster",
            description="Predict survival on the Titanic.",
            metric="accuracy",
        ),
        files=[
            CompetitionFile(name="train.csv", size=60302),
            CompetitionFile(name="test.csv", size=28629),
            CompetitionFile(name="gender_submission.csv", size=3258),
        ],
        lb=[
            LBEntry(team_name="alpha", score=1.0),
            LBEntry(team_name="beta", score=0.99999),
        ],
    )


@pytest.fixture
def sample_train_csv(workspace):
    df = pd.DataFrame({
        "PassengerId": range(1, 11),
        "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 2.0, 27.0, 14.0],
        "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
    })
    train_path = workspace.raw_dir / "train.csv"
    df.to_csv(train_path, index=False)
    return train_path


def test_build_context_writes_context_md(workspace, kaggle_fake, sample_train_csv):
    path = ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    assert path == workspace.context_path
    assert path.exists()


def test_build_context_includes_title_and_metric(workspace, kaggle_fake, sample_train_csv):
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    assert "Titanic" in body
    assert "accuracy" in body


def test_build_context_lists_data_files(workspace, kaggle_fake, sample_train_csv):
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    assert "train.csv" in body
    assert "test.csv" in body


def test_build_context_includes_leaderboard_summary(workspace, kaggle_fake, sample_train_csv):
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    # Top score is shown, even if team name redacted
    assert "1.0" in body or "1.00000" in body


def test_build_context_profiles_train_data(workspace, kaggle_fake, sample_train_csv):
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    # Columns appear in the data summary
    assert "Survived" in body
    assert "Age" in body
    # Row count
    assert "10" in body


def test_build_context_suggests_target_column(workspace, kaggle_fake, sample_train_csv):
    """A column named 'Survived' or 'target' should be flagged as the likely target."""
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text().lower()
    assert "target" in body
    assert "survived" in body


def test_build_context_works_without_train_csv(workspace, kaggle_fake):
    """If raw/train.csv doesn't exist, the data section says so but the file is still written."""
    path = ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = path.read_text()
    assert "no train.csv" in body.lower() or "train data not yet downloaded" in body.lower()


def test_build_context_overwrites_existing(workspace, kaggle_fake, sample_train_csv):
    workspace.context_path.write_text("stale content")
    ctx_mod.build_context(workspace=workspace, kaggle_client=kaggle_fake)
    body = workspace.context_path.read_text()
    assert "stale content" not in body
    assert "Titanic" in body
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_context_builder.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.context`.

- [ ] **Step 3: Create `kaggle_slayer/harness/context.py`**

```python
"""Build context.md for a per-competition workspace.

The agent reads context.md as its system message. The file is regenerated
from scratch on each call to build_context (overwrites stale content).

For Week 2 the structure is intentionally simple — a markdown file with
named sections. Future weeks can add: parsed competition rules, public-LB
calibration history, learned patterns from prior comps.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd  # type: ignore[import-untyped]

from kaggle_slayer.harness.kaggle_client import CompetitionFile, CompetitionInfo, LBEntry
from kaggle_slayer.harness.workspace import Workspace


class _KaggleClientLike(Protocol):
    """Minimal interface build_context needs — easy to fake in tests."""

    def view_competition(self, name: str) -> CompetitionInfo: ...
    def list_files(self, name: str) -> list[CompetitionFile]: ...
    def get_leaderboard(self, name: str, *, top_n: int = 10) -> list[LBEntry]: ...


# Common target column names we'll surface as a hint to the agent. The agent
# remains responsible for confirming the actual target.
_TARGET_HINTS: tuple[str, ...] = (
    "target", "Target", "TARGET",
    "label", "Label", "LABEL",
    "y",
    "Survived", "SalePrice", "Class", "outcome",
)


def build_context(
    *,
    workspace: Workspace,
    kaggle_client: _KaggleClientLike,
    leaderboard_top_n: int = 5,
) -> "pathlib.Path":  # noqa: F821 — Path is referenced via Workspace
    name = workspace.name
    info = kaggle_client.view_competition(name)
    files = kaggle_client.list_files(name)
    leaderboard = kaggle_client.get_leaderboard(name, top_n=leaderboard_top_n)

    sections: list[str] = [
        f"# Competition: {info.title or name}",
        "",
        "## Description",
        info.description.strip() or "(no description available)",
        "",
        "## Evaluation metric",
        f"`{info.metric}`" if info.metric else "(metric not provided by Kaggle API)",
        "",
        "## Data files",
        _files_section(files),
        "",
        "## Data profile (train.csv)",
        _data_profile(workspace),
        "",
        "## Public leaderboard (top scores for reference)",
        _lb_section(leaderboard),
    ]

    body = "\n".join(sections) + "\n"
    workspace.context_path.write_text(body)
    return workspace.context_path


def _files_section(files: list[CompetitionFile]) -> str:
    if not files:
        return "(no files listed)"
    lines = []
    for f in files:
        size_mb = f.size / 1024 / 1024
        lines.append(f"- `{f.name}` ({size_mb:.1f} MB)")
    return "\n".join(lines)


def _lb_section(lb: list[LBEntry]) -> str:
    if not lb:
        return "(no leaderboard data available)"
    lines = ["| Rank | Team | Score |", "|---|---|---|"]
    for i, entry in enumerate(lb, start=1):
        lines.append(f"| {i} | {entry.team_name or '(redacted)'} | {entry.score} |")
    return "\n".join(lines)


def _data_profile(workspace: Workspace) -> str:
    train_csv = workspace.raw_dir / "train.csv"
    if not train_csv.exists():
        return "*Train data not yet downloaded (no train.csv in raw/).*"

    try:
        df = pd.read_csv(train_csv, nrows=5000)
    except Exception as e:  # noqa: BLE001
        return f"*Could not read train.csv: {e!r}*"

    target_candidates = [c for c in df.columns if c in _TARGET_HINTS]

    lines = [
        f"- **Rows (sampled, first 5000):** {len(df)}",
        f"- **Columns:** {len(df.columns)}",
    ]
    if target_candidates:
        lines.append(f"- **Likely target column(s):** {', '.join(f'`{c}`' for c in target_candidates)}")
    else:
        lines.append("- **Target column:** none of the standard names matched; the agent should infer.")
    lines.append("")
    lines.append("### Column schema")
    lines.append("| Column | dtype | non-null | unique |")
    lines.append("|---|---|---|---|")
    for col in df.columns:
        nn = df[col].notna().sum()
        nu = df[col].nunique()
        lines.append(f"| `{col}` | {df[col].dtype} | {nn}/{len(df)} | {nu} |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_context_builder.py -v
```

Expected: 8 passes.

- [ ] **Step 5: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/context.py tests/unit/test_context_builder.py
git commit -m "$(cat <<'EOF'
feat(harness): add context.md builder

build_context(workspace, kaggle_client) writes a markdown brief with
sections: title, description, evaluation metric, data files, train.csv
profile (column dtypes + non-null + unique counts), public leaderboard
top-N for reference.

If raw/train.csv hasn't been downloaded yet the profile section says so
but the file is still written — the agent can read what's available and
download later.

_TARGET_HINTS pre-flags common target column names (target, Survived,
SalePrice, etc.). The agent remains responsible for final identification.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Fake LLMClient fixture

**Files:**
- Create: `tests/fixtures/fake_llm.py`
- Create: `tests/unit/test_fake_llm.py`

Implements the `LLMClient` protocol with scripted responses. Used by the integration tier to exercise the harness without real API calls.

- [ ] **Step 1: Write failing tests**

```python
"""Tests for tests/fixtures/fake_llm.py.

This is test-support code, but it gets a real test file because the
integration tier and Week 3 agent loop will depend on it being correct.
"""

from __future__ import annotations

import pytest

from kaggle_slayer.agent.llm_client import LLMClient, Message
from tests.fixtures.fake_llm import FakeLLMClient, ScriptedResponse


def test_fake_llm_returns_next_scripted_response():
    fake = FakeLLMClient(
        script=[
            ScriptedResponse(text="hello"),
            ScriptedResponse(text="world"),
        ]
    )
    r1 = fake.call(messages=[Message(role="user", content="say hi")])
    r2 = fake.call(messages=[Message(role="user", content="say more")])
    assert r1.text == "hello"
    assert r2.text == "world"


def test_fake_llm_records_messages():
    fake = FakeLLMClient(script=[ScriptedResponse(text="ok")])
    fake.call(messages=[Message(role="user", content="probe")])
    assert len(fake.calls) == 1
    assert fake.calls[0].messages[-1].content == "probe"


def test_fake_llm_raises_on_script_exhaustion():
    fake = FakeLLMClient(script=[ScriptedResponse(text="only one")])
    fake.call(messages=[Message(role="user", content="first")])
    with pytest.raises(RuntimeError, match="exhausted"):
        fake.call(messages=[Message(role="user", content="second")])


def test_fake_llm_implements_protocol():
    """Confirm FakeLLMClient is a structural match for LLMClient."""
    fake = FakeLLMClient(script=[])
    assert isinstance(fake, LLMClient)


def test_fake_llm_default_usage_is_zero():
    fake = FakeLLMClient(script=[ScriptedResponse(text="x")])
    resp = fake.call(messages=[Message(role="user", content="probe")])
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/unit/test_fake_llm.py -v
```

Expected: `ModuleNotFoundError: tests.fixtures.fake_llm`.

- [ ] **Step 3: Create `tests/fixtures/fake_llm.py`**

```python
"""FakeLLMClient — scripted responses for integration tests.

Implements the LLMClient protocol without burning real API quota. Each
call() pops the next ScriptedResponse off the script and returns it.
The captured `calls` list lets tests assert what the harness sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.agent.llm_client import Message, Response, Usage


@dataclass
class ScriptedResponse:
    text: str
    usage: Usage = field(default_factory=lambda: Usage(0, 0, 0))


@dataclass
class CapturedCall:
    messages: list[Message]
    tools: list[dict[str, Any]] | None
    model: str | None


class FakeLLMClient:
    def __init__(self, *, script: list[ScriptedResponse]) -> None:
        self._script = list(script)
        self.calls: list[CapturedCall] = []

    def call(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> Response:
        self.calls.append(CapturedCall(messages=list(messages), tools=tools, model=model))
        if not self._script:
            raise RuntimeError("FakeLLMClient script exhausted")
        scripted = self._script.pop(0)
        return Response(text=scripted.text, tool_calls=[], usage=scripted.usage)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_fake_llm.py -v
```

Expected: 5 passes.

- [ ] **Step 5: Lint + type-check**

mypy excludes `tests/*` so the fake won't be type-checked, but ruff still applies. Run:

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/fake_llm.py tests/unit/test_fake_llm.py
git commit -m "$(cat <<'EOF'
test: add FakeLLMClient fixture with scripted responses

FakeLLMClient implements the LLMClient protocol without API calls. Pops
ScriptedResponse off the script per call; records every CapturedCall so
tests can assert what the harness sent.

Used by Week 2's integration tier (next task) and Week 3's agent loop
tests to exercise the harness end-to-end without burning quota.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Integration test — fake agent end-to-end (Week 2 acceptance)

**Files:**
- Create: `tests/integration/test_workspace_with_fake_agent.py`

The acceptance test: a `FakeLLMClient` runs through a scripted sequence; the harness wires together `Workspace`, `Journal`, `KaggleClient` (mocked), and `build_context`; we assert `run_log.jsonl` reflects the journey correctly.

- [ ] **Step 1: Write the acceptance test**

```python
"""Week 2 acceptance integration test.

Threads together Workspace + Journal + KaggleClient (mocked) + context
builder + FakeLLMClient. The fake "agent" makes one LLM call, then the
harness journals it; we verify the journal is correct and complete.

This is plumbing-level, not yet the agent loop (which lands in Week 3).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from kaggle_slayer.agent.llm_client import Message
from kaggle_slayer.harness.context import build_context
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.kaggle_client import CompetitionFile, CompetitionInfo, LBEntry
from kaggle_slayer.harness.workspace import Workspace
from tests.fixtures.fake_llm import FakeLLMClient, ScriptedResponse


pytestmark = pytest.mark.integration


class _FakeKaggle:
    def view_competition(self, name):
        return CompetitionInfo(
            title="Fake Comp", description="A synthetic competition.",
            metric="accuracy",
        )

    def list_files(self, name):
        return [CompetitionFile(name="train.csv", size=12345)]

    def get_leaderboard(self, name, *, top_n=10):
        return [LBEntry(team_name="alpha", score=0.95)]


def test_fake_agent_loop_journals_each_step(tmp_path):
    # --- setup ---
    workspace = Workspace.create(root=tmp_path / "competitions" / "fake")

    # Write a tiny train.csv so context builder can profile it
    pd.DataFrame({
        "x1": range(10),
        "Survived": [0, 1] * 5,
    }).to_csv(workspace.raw_dir / "train.csv", index=False)

    kaggle = _FakeKaggle()
    journal = Journal(workspace)

    fake_llm = FakeLLMClient(script=[
        ScriptedResponse(text="I have read the context. My plan: train LightGBM."),
        ScriptedResponse(text="train_cv returned cv=0.82. Submitting."),
    ])

    # --- run ---
    # Step 1: build context, log it as a tool call
    ctx_path = build_context(workspace=workspace, kaggle_client=kaggle)
    journal.log_tool_call(
        tool="build_context",
        args={"competition": "fake"},
        result_summary=f"wrote {ctx_path.name}",
    )

    # Step 2: pretend "agent" reads the context and makes a planning call
    resp1 = fake_llm.call(messages=[
        Message(role="system", content=ctx_path.read_text()),
        Message(role="user", content="Plan a solution."),
    ])
    journal.log_tool_call(
        tool="llm_call",
        args={"role": "planner"},
        result_summary=resp1.text[:80],
    )

    # Step 3: pretend an action happened
    journal.log_tool_call(
        tool="train_cv",
        args={"fe": "agent/fe.py", "model": "agent/model.py"},
        result_summary="cv=0.82",
    )

    # Step 4: a second LLM call reading the result
    resp2 = fake_llm.call(messages=[
        Message(role="user", content="train_cv returned cv=0.82. What now?"),
    ])
    journal.log_tool_call(
        tool="llm_call",
        args={"role": "post_train"},
        result_summary=resp2.text[:80],
    )

    # --- asserts ---
    # context.md exists and has the right structure
    body = ctx_path.read_text()
    assert "Fake Comp" in body
    assert "Survived" in body

    # run_log has exactly 4 entries in order
    records = [json.loads(line) for line in workspace.run_log_path.read_text().splitlines()]
    assert len(records) == 4
    assert [r["tool"] for r in records] == [
        "build_context", "llm_call", "train_cv", "llm_call",
    ]
    assert all(r["kind"] == "tool_call" for r in records)

    # FakeLLMClient was called exactly twice
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[0].messages[-1].content == "Plan a solution."


def test_fake_agent_loop_resume_summary(tmp_path):
    """After a partial run, resume.summarize() should describe what happened."""
    workspace = Workspace.create(root=tmp_path / "competitions" / "fake")
    journal = Journal(workspace)

    journal.log_tool_call(tool="build_context", args={}, result_summary="ok")
    journal.log_tool_call(tool="train_cv", args={"v": 1}, result_summary="cv=0.7")
    journal.log_tool_error(tool="submit_kaggle", args={}, error="rules not accepted")

    from kaggle_slayer.harness.resume import summarize
    summary = summarize(workspace)
    assert summary.total_calls == 3
    assert summary.error_count == 1
    assert summary.last_call["tool"] == "submit_kaggle"
    assert summary.tool_counts == {
        "build_context": 1,
        "train_cv": 1,
        "submit_kaggle": 1,
    }
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/integration/test_workspace_with_fake_agent.py -v -m integration
```

Expected: 2 passes. **This is the Week 2 acceptance gate.**

- [ ] **Step 3: Run the whole suite + coverage**

```bash
pytest -m "not slow" --cov=kaggle_slayer/harness --cov=kaggle_slayer/agent --cov-report=term -v
```

Expected: all tests pass; coverage on the new code ≥ 90%.

- [ ] **Step 4: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_workspace_with_fake_agent.py
git commit -m "$(cat <<'EOF'
test: add Week 2 acceptance integration test

Wires together Workspace + Journal + (fake) KaggleClient + context
builder + FakeLLMClient through a small scripted sequence. Asserts:
  - context.md is written with the right content
  - run_log.jsonl has exactly the expected 4 entries in order
  - resume.summarize() correctly describes the partial run

This is the Week 2 acceptance gate. The actual Solver agent loop lands
in Week 3 — for now we've proven the plumbing wires up correctly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Real-API smoke test (slow tier)

**Files:**
- Create: `tests/integration/test_real_apis.py`

A handful of tests that actually call Gemini and Kaggle. Marked `slow`, skipped in CI by default, runnable on demand to verify the wrappers work against live APIs.

- [ ] **Step 1: Write the slow tests**

```python
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

from kaggle_slayer.agent.cost_ledger import CostLedger
from kaggle_slayer.agent.llm_client import GeminiClient, Message
from kaggle_slayer.harness.kaggle_client import KaggleClient


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
    """View a well-known evergreen competition that should always exist."""
    client = KaggleClient()
    info = client.view_competition("titanic")
    assert "titanic" in (info.title or "").lower()
```

- [ ] **Step 2: Run the slow tests (opt-in)**

```bash
pytest -m slow tests/integration/test_real_apis.py -v
```

Expected: 3 passes (assuming both creds are configured per `scripts/preflight.py`).

- [ ] **Step 3: Confirm `-m "not slow"` still excludes them**

```bash
pytest -m "not slow" -v 2>&1 | tail -3
```

Expected: the slow tests show up as deselected/not run.

- [ ] **Step 4: Lint + type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness kaggle_slayer/agent
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_real_apis.py
git commit -m "$(cat <<'EOF'
test: add slow-tier real-API smoke tests

Three opt-in tests that hit live Gemini + Kaggle:
  - real_gemini_one_token_smoke: 1-token "ok" response, cost ≈ $0.0001
  - real_kaggle_competitions_list: read-only competitions listing
  - real_kaggle_view_competition: KaggleClient.view_competition("titanic")

Skipped when credentials are missing; @pytest.mark.slow so default CI
doesn't run them. Run on demand with `pytest -m slow`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Summary of Week 2 acceptance

When all 13 tasks complete, the following must hold:

- ✅ `kaggle_slayer/harness/workspace.py` — per-comp Workspace dataclass with all spec §10 paths.
- ✅ `kaggle_slayer/harness/journal.py` — append-before-return Journal for run_log + notes.
- ✅ `kaggle_slayer/harness/resume.py` — `summarize()` with tool counts + stuck-loop detection.
- ✅ `kaggle_slayer/harness/kaggle_client.py` — KaggleClient over v2.1 library (view, files, leaderboard, download, submit).
- ✅ `kaggle_slayer/harness/context.py` — build_context() writes context.md with description, metric, files, train profile, LB summary.
- ✅ CV registry — `time_series` + `group_kfold` added; `auto_select` extended with `date_col` + `group_col`.
- ✅ `train_cv` accepts `metadata_extra` (Opus carry-forward).
- ✅ `kaggle_slayer/agent/cost_ledger.py` — JSONL cost ledger with per-model price table.
- ✅ `kaggle_slayer/agent/llm_client.py` — LLMClient Protocol + Message/ToolCall/Usage/Response + GeminiClient with retry.
- ✅ `tests/fixtures/fake_llm.py` — FakeLLMClient implementing the protocol.
- ✅ `tests/integration/test_workspace_with_fake_agent.py` — Week 2 acceptance gate, 2 passes.
- ✅ `tests/integration/test_real_apis.py` — 3 slow-tier real-API smoke tests.
- ✅ Full suite (`-m "not slow"`) green; coverage on new code ≥ 90%; ruff + mypy strict clean.

**Week 3 starts with:** the Solver agent loop (`solver.py`), tool JSON schemas (`tools.py`), system prompt, the actual reason-act loop, and the first real Gemini-driven CV pass on a synthetic micro-comp.
