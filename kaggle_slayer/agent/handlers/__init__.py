"""Tool handlers — pure Python callables registered into a ToolRegistry.

make_builtin_registry() returns a ToolRegistry pre-loaded with the
9 builtin tools the Solver uses in Week 3.
"""

from __future__ import annotations

from kaggle_slayer.agent.handlers import files as fh
from kaggle_slayer.agent.handlers import ml as ml_h
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
    return reg
