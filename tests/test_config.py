"""Tests for utils.config: yaml loading, dotted-path lookups, defaults."""

from __future__ import annotations

import yaml

from utils.config import ConfigManager


def test_dotted_lookup(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml.safe_dump({"pipeline": {"cv_folds": 7}, "data": {}}))
    cm = ConfigManager(str(cfg_file))
    assert cm.get("pipeline.cv_folds") == 7
    assert cm.get("pipeline.missing", default=42) == 42
    assert cm.get("data.also_missing", default="x") == "x"


def test_missing_file_uses_defaults(tmp_path):
    cm = ConfigManager(str(tmp_path / "does_not_exist.yaml"))
    assert cm.get("anything", default="fallback") == "fallback"
