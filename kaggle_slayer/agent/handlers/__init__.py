"""Tool handlers — pure Python callables registered into a ToolRegistry.

make_builtin_registry() returns a ToolRegistry pre-loaded with the
13 builtin tools the Solver uses in Week 4.
"""

from __future__ import annotations

from kaggle_slayer.agent.handlers import files as fh
from kaggle_slayer.agent.handlers import ml as ml_h
from kaggle_slayer.agent.handlers import python as ph_python
from kaggle_slayer.agent.tools import Tool, ToolRegistry


def make_builtin_registry() -> ToolRegistry:
    """Build a ToolRegistry pre-loaded with the Week-3 tool set."""
    reg = ToolRegistry()
    reg.register(Tool(
        name="read_context",
        description="Read the competition's context.md (problem brief + data profile).",
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=fh.read_context,
    ))
    reg.register(Tool(
        name="read_file",
        description="Read a file from inside the workspace (e.g., agent/fe.py).",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Workspace-relative path"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=fh.read_file,
    ))
    reg.register(Tool(
        name="write_file",
        description=(
            "Write a file inside the workspace. Use this to create agent/fe.py "
            "and agent/model.py with your feature-engineering and model code. "
            "Overwrites any existing content."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path (e.g., 'agent/fe.py')"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=fh.write_file,
    ))
    reg.register(Tool(
        name="sample_rows",
        description="Return a sample of rows from raw/<table>.csv as a formatted string.",
        schema={
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name without extension, e.g., 'train' or 'test'"},
                "n": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "random": {"type": "boolean", "default": False},
            },
            "required": ["table"],
            "additionalProperties": False,
        },
        handler=fh.sample_rows,
    ))
    reg.register(Tool(
        name="take_note",
        description="Append a structured note to notes.jsonl for later reference.",
        schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["observation", "decision", "hypothesis", "todo"],
                },
                "content": {"type": "string"},
            },
            "required": ["category", "content"],
            "additionalProperties": False,
        },
        handler=fh.take_note,
    ))
    reg.register(Tool(
        name="set_cv",
        description=(
            "Override the CV strategy. By default the harness auto-selects "
            "(stratified_kfold for classification, kfold for regression). "
            "Use this when the data is time-indexed (kind='time_series') or "
            "grouped (kind='group_kfold', group_col='<col>')."
        ),
        schema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["kfold", "stratified_kfold", "time_series", "group_kfold"],
                },
                "n_splits": {"type": "integer", "minimum": 2, "maximum": 20, "default": 5},
                "group_col": {"type": "string"},
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
        handler=ml_h.set_cv,
    ))
    reg.register(Tool(
        name="train_cv",
        description=(
            "Run leak-free K-fold cross-validation using the current "
            "agent/fe.py and agent/model.py. Archives both into "
            "agent/versions/ before the run and returns a CV summary "
            "(mean, std, per-fold scores)."
        ),
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=ml_h.train_cv,
    ))
    reg.register(Tool(
        name="submit_local",
        description=(
            "Fit fe.py + model.py on the full training set, predict the test "
            "data, and write a submission CSV under submissions/. The CSV "
            "is local-only (this does NOT push to Kaggle)."
        ),
        schema={
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Short label used in the CSV filename"},
            },
            "required": ["label"],
            "additionalProperties": False,
        },
        handler=ml_h.submit_local,
    ))
    reg.register(Tool(
        name="done",
        description=(
            "Signal that you are finished. The harness loop exits after this. "
            "Pass a one-sentence summary of what you accomplished."
        ),
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        handler=ml_h.done,
    ))
    reg.register(Tool(
        name="run_python",
        description=(
            "Run a Python snippet in a sandboxed subprocess (resource-limited, "
            "cwd=workspace root). Use for plotting, peeks, quick debug. Returns "
            "stdout/stderr/returncode as a string. Do NOT use for CV — train_cv "
            "is the only path to a valid CV score."
        ),
        schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute"},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 600, "default": 60},
                "memory_mb": {"type": "integer", "minimum": 64, "maximum": 16384, "default": 2048},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        handler=ph_python.run_python,
    ))
    reg.register(Tool(
        name="set_metric",
        description=(
            "Change the scoring metric. Checkpoint-gated. Use when the parsed "
            "metric is wrong (e.g., comp uses weighted F1 but the harness picked "
            "accuracy). Pick from accuracy/auc/logloss/rmse/mae/r2."
        ),
        schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=ml_h.set_metric,
    ))
    reg.register(Tool(
        name="submit_kaggle",
        description=(
            "Push a submission CSV to Kaggle. Checkpoint-gated (always for the "
            "first submission of a comp; on score regression for subsequent ones). "
            "csv_path is workspace-relative."
        ),
        schema={
            "type": "object",
            "properties": {
                "csv_path": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["csv_path", "message"],
            "additionalProperties": False,
        },
        handler=ml_h.submit_kaggle,
    ))
    reg.register(Tool(
        name="request_human_approval",
        description=(
            "Pause and ask the human for explicit approval before proceeding. "
            "Use when you're about to take an action whose stakes you can't fully "
            "judge (e.g., a non-obvious metric override, an unusual leak risk). "
            "evidence_json is a JSON-encoded dict of context the human should see."
        ),
        schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "evidence_json": {"type": "string", "default": "{}"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        handler=ml_h.request_human_approval,
    ))
    return reg
