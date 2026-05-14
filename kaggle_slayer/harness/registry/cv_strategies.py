"""Cross-validation strategy registry.

A CVStrategy wraps a sklearn-style splitter and carries a name + the
config the harness needs to log it (n_splits, random_state, etc.).
Week 1 strategies: kfold, stratified_kfold.

Time-indexed and grouped strategies land in Week 2 when context-driven
CV selection is built (the agent and the data shape decide which).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore[import-untyped]


@dataclass
class CVStrategy:
    """Wraps a sklearn splitter for the harness CV contract."""

    name: str
    n_splits: int
    random_state: int | None = 42
    extra: dict[str, Any] = field(default_factory=dict)
    _splitter: Any = None

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Yield (train_idx, val_idx) tuples for each fold."""
        if self._splitter is None:
            raise RuntimeError("CVStrategy._splitter not initialized")
        y = df[target_col]
        # sklearn splitters all take (X, y); we pass df so the splitter can
        # access any extra columns it might need (e.g., groups in future).
        for train_idx, val_idx in self._splitter.split(df, y):
            yield list(train_idx), list(val_idx)


def _make_kfold(
    n_splits: int, random_state: int | None = 42, shuffle: bool = True
) -> CVStrategy:
    # sklearn requires random_state=None when shuffle=False
    effective_random_state = random_state if shuffle else None
    splitter = KFold(
        n_splits=n_splits, shuffle=shuffle, random_state=effective_random_state
    )
    return CVStrategy(
        name="kfold",
        n_splits=n_splits,
        random_state=effective_random_state,
        _splitter=splitter,
    )


def _make_stratified_kfold(
    n_splits: int, random_state: int | None = 42, shuffle: bool = True
) -> CVStrategy:
    # sklearn requires random_state=None when shuffle=False
    effective_random_state = random_state if shuffle else None
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=effective_random_state
    )
    return CVStrategy(
        name="stratified_kfold",
        n_splits=n_splits,
        random_state=effective_random_state,
        _splitter=splitter,
    )


_FACTORIES = {
    "kfold": _make_kfold,
    "stratified_kfold": _make_stratified_kfold,
}


def get(
    name: str,
    *,
    n_splits: int = 5,
    random_state: int | None = 42,
    shuffle: bool = True,
) -> CVStrategy:
    """Construct a CVStrategy by name."""
    if name not in _FACTORIES:
        raise KeyError(f"cv strategy '{name}' not in registry; known: {sorted(_FACTORIES)}")
    return _FACTORIES[name](n_splits=n_splits, random_state=random_state, shuffle=shuffle)


def list_strategies() -> list[str]:
    """Return the names of all registered strategies."""
    return sorted(_FACTORIES)


def auto_select(
    *, problem_type: str, train_df: pd.DataFrame, target_col: str, n_splits: int = 5
) -> CVStrategy:
    """Pick a sensible default CV strategy from the problem type.

    Week-1 logic is intentionally minimal: classification → stratified,
    regression → plain. Time-indexed and grouped detection is added in
    Week 2 along with the strategies themselves.
    """
    if problem_type == "classification":
        return get("stratified_kfold", n_splits=n_splits)
    if problem_type == "regression":
        return get("kfold", n_splits=n_splits)
    raise ValueError(
        f"unsupported problem_type '{problem_type}'; expected 'classification' or 'regression'"
    )
