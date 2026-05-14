"""Tests for kaggle_slayer.harness.sandbox.lint_module."""

from __future__ import annotations

import textwrap

from kaggle_slayer.harness import sandbox


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def test_lint_passes_minimal_valid_fe(tmp_path):
    p = _write(tmp_path, "fe.py", """
        import pandas as pd

        class T:
            def transform(self, df):
                return df

        def fit_feature_transformer(train_df, target_col):
            return T()
    """)
    result = sandbox.lint_module(p)
    assert result.ok, result.violations


def test_lint_passes_stub_fe(tmp_path):
    """The shipped fe_stub.py must lint clean."""
    from pathlib import Path
    stub = Path(__file__).resolve().parents[1] / "fixtures" / "fe_stub.py"
    result = sandbox.lint_module(stub)
    assert result.ok, result.violations


def test_lint_rejects_os_remove(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import os
        def fit_feature_transformer(train_df, target_col):
            os.remove("/tmp/x")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("os.remove" in v for v in result.violations)


def test_lint_rejects_shutil_rmtree(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import shutil
        def fit_feature_transformer(train_df, target_col):
            shutil.rmtree("/tmp")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("shutil.rmtree" in v for v in result.violations)


def test_lint_rejects_subprocess(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import subprocess
        def fit_feature_transformer(train_df, target_col):
            subprocess.run(["rm", "-rf", "/tmp"])
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("subprocess" in v for v in result.violations)


def test_lint_rejects_os_system(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import os
        def fit_feature_transformer(train_df, target_col):
            os.system("echo bad")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("os.system" in v for v in result.violations)


def test_lint_rejects_eval(tmp_path):
    p = _write(tmp_path, "bad.py", """
        def fit_feature_transformer(train_df, target_col):
            return eval("None")
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("eval" in v for v in result.violations)


def test_lint_rejects_exec(tmp_path):
    p = _write(tmp_path, "bad.py", """
        def fit_feature_transformer(train_df, target_col):
            exec("pass")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("exec" in v for v in result.violations)


def test_lint_rejects_requests(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import requests
        def fit_feature_transformer(train_df, target_col):
            requests.get("https://evil.com")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("requests" in v for v in result.violations)


def test_lint_rejects_urllib(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import urllib.request
        def fit_feature_transformer(train_df, target_col):
            urllib.request.urlopen("https://evil.com")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("urllib" in v for v in result.violations)


def test_lint_rejects_read_of_raw_path(tmp_path):
    """The agent must not read competition raw data directly — only what
    the harness passes to fit_feature_transformer."""
    p = _write(tmp_path, "bad.py", """
        import pandas as pd
        def fit_feature_transformer(train_df, target_col):
            extra = pd.read_csv("raw/train.csv")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("raw/" in v for v in result.violations)


def test_lint_rejects_open_write_outside_workspace(tmp_path):
    p = _write(tmp_path, "bad.py", """
        def fit_feature_transformer(train_df, target_col):
            open("/etc/passwd", "w")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("open" in v for v in result.violations)


def test_lint_aggregates_multiple_violations(tmp_path):
    """All violations are reported, not just the first."""
    p = _write(tmp_path, "bad.py", """
        import os
        import subprocess
        def fit_feature_transformer(train_df, target_col):
            os.remove("/tmp/x")
            subprocess.run(["true"])
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert len(result.violations) >= 2


def test_lint_catches_os_remove_even_when_os_path_was_imported(tmp_path):
    """import os.path must not defeat the os.* denylist."""
    p = _write(tmp_path, "bad.py", """
        import os.path
        def fit_feature_transformer(train_df, target_col):
            os.remove("/tmp/x")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("os.remove" in v for v in result.violations)


def test_lint_catches_os_system_even_when_os_path_was_imported(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import os.path
        def fit_feature_transformer(train_df, target_col):
            os.system("echo bad")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("os.system" in v for v in result.violations)
