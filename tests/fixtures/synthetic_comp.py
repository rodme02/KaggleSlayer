"""Programmatic synthetic Kaggle micro-competition.

Creates a tiny binary-classification dataset (500 train rows, 100 test rows)
inside a temporary workspace. Used by integration tests so we exercise the
full Solver loop without depending on real Kaggle data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_slayer.harness.workspace import Workspace


def make_synthetic_comp(root: Path, *, seed: int = 0) -> Workspace:
    """Create a workspace at `root` with raw/train.csv, raw/test.csv,
    and a context.md flagging Survived as target + accuracy as metric.

    The target is a noisy function of x1 + x2 — learnable by LR.
    """
    rng = np.random.default_rng(seed)
    n_train, n_test = 500, 100

    train = pd.DataFrame({
        "id": range(n_train),
        "x1": rng.normal(size=n_train),
        "x2": rng.normal(size=n_train),
        "x3": rng.normal(size=n_train),
        "Sex": rng.choice(["male", "female"], size=n_train),
    })
    logits = 1.5 * train["x1"] - 0.8 * train["x2"] + rng.normal(scale=0.5, size=n_train)
    train["Survived"] = (logits > 0).astype(int)

    test = pd.DataFrame({
        "id": range(n_train, n_train + n_test),
        "x1": rng.normal(size=n_test),
        "x2": rng.normal(size=n_test),
        "x3": rng.normal(size=n_test),
        "Sex": rng.choice(["male", "female"], size=n_test),
    })

    workspace = Workspace.create(root=root)
    train.to_csv(workspace.raw_dir / "train.csv", index=False)
    test.to_csv(workspace.raw_dir / "test.csv", index=False)
    workspace.context_path.write_text(
        "# Synthetic Comp\n\n"
        "## Description\nBinary classification on 5 features.\n\n"
        "## Evaluation metric\n`accuracy`\n\n"
        "## Data profile (train.csv)\n"
        "- **Rows:** 500\n"
        "- **Likely target column(s):** `Survived`\n"
        "- **ID column:** `id`\n\n"
        "## Public leaderboard (top scores for reference)\n"
        "*synthetic; no real LB*\n"
    )
    return workspace
