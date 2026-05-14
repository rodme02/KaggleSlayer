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
from typing import Any

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
    extra: dict[str, Any] = field(default_factory=dict)
    _splitter: Any = None

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Iterator[tuple[list[int], list[int]]]:
        if self._splitter is None:
            raise RuntimeError("CVStrategy._splitter not initialized")
        y = df[target_col]
        groups = None
        if self.extra.get("group_col"):
            groups = df[self.extra["group_col"]]
        if groups is not None:  # noqa: SIM108
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


_FACTORIES: dict[str, Any] = {
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
        return factory(n_splits=n_splits, random_state=random_state, shuffle=shuffle)  # type: ignore[no-any-return]
    if name == "time_series":
        return factory(n_splits=n_splits)  # type: ignore[no-any-return]
    if name == "group_kfold":
        return factory(n_splits=n_splits, group_col=group_col)  # type: ignore[no-any-return]
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
