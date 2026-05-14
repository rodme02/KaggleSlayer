"""Minimal valid feature-engineering module.

Demonstrates the agent's contract: fit_feature_transformer(train_df, target_col)
returns an object with .transform(df) -> df. Statistics (means for imputation)
are fit on the train fold ONLY — see spec §6.
"""

from __future__ import annotations

import pandas as pd


class _StubTransformer:
    def __init__(self, numeric_cols: list[str], cat_cols: list[str],
                 numeric_means: dict[str, float], cat_modes: dict[str, str]):
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.numeric_means = numeric_means
        self.cat_modes = cat_modes

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.numeric_cols:
            if col in df.columns:
                out[col] = df[col].fillna(self.numeric_means.get(col, 0.0))
            else:
                out[col] = self.numeric_means.get(col, 0.0)
        for col in self.cat_cols:
            if col in df.columns:
                series = df[col].fillna(self.cat_modes.get(col, "MISSING"))
                # One-hot encode against the training categories.
                dummies = pd.get_dummies(series, prefix=col, drop_first=False)
                out = pd.concat([out, dummies], axis=1)
        return out


def fit_feature_transformer(train_df: pd.DataFrame, target_col: str) -> _StubTransformer:
    """Fit on train fold only.

    Imputes numerics with train-fold mean; one-hot encodes categoricals using
    train-fold categories. Returns an object with .transform(df) for val/test.
    """
    feature_df = train_df.drop(columns=[target_col])
    numeric_cols = [c for c in feature_df.columns if feature_df[c].dtype.kind in "fiub"]
    cat_cols = [c for c in feature_df.columns if c not in numeric_cols]
    numeric_means = {c: float(feature_df[c].mean()) for c in numeric_cols}
    cat_modes = {
        c: str(feature_df[c].mode(dropna=True).iloc[0])
        if not feature_df[c].mode(dropna=True).empty
        else "MISSING"
        for c in cat_cols
    }
    return _StubTransformer(numeric_cols, cat_cols, numeric_means, cat_modes)
