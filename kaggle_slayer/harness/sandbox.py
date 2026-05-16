"""Sandbox for agent-written Python modules.

Week 1 scope: AST lint that scans a Python file before the harness loads
it and rejects forbidden patterns. The lint is the leak-prevention
mechanism for the in-process CV contract (see spec §6.5).

Threat model: 'agent typos itself into something destructive', not 'agent
is adversarial'. We catch obvious patterns (os.remove, shutil.rmtree,
subprocess, etc.) including aliased forms (`import os as o; o.remove(...)`)
but do NOT defend against deliberate bypasses like:
  - getattr(os, 'remove')(...)
  - globals()['os'].remove(...)
  - __import__('os').remove(...)
For an adversarial threat model, upgrade to subprocess-isolated execution
with a real sandbox (Docker, gVisor, or seccomp filter).

Resource limits (subprocess + setrlimit) ship in `run_subprocess` below —
used by the `run_python` tool. Note: macOS does not reliably enforce
RLIMIT_AS; the memory cap is a best-effort on Darwin and a hard cap on
Linux.
"""

from __future__ import annotations

import ast
import contextlib
import datetime as _dt
import resource as _resource  # POSIX-only; we don't support Windows
import subprocess as _subprocess
import sys as _sys
from dataclasses import dataclass, field
from pathlib import Path

from kaggle_slayer.harness.workspace import Workspace as _Workspace

# Module-prefix denylist: any `X.Y.Z(...)` whose dotted attribute chain
# starts with one of these tuples is rejected. We match against the
# *attribute chain* (e.g., `os.remove`), not the imported alias —
# `import os as o; o.remove(...)` is still flagged because we track aliases.
_FORBIDDEN_ATTR_CALLS: tuple[tuple[str, ...], ...] = (
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "removedirs"),
    ("os", "system"),
    ("os", "popen"),
    ("shutil", "rmtree"),
    ("shutil", "move"),
    ("subprocess",),  # any subprocess.* call
    ("requests",),
    ("urllib",),
    ("socket",),
    ("http", "client"),
)

_FORBIDDEN_BUILTINS: frozenset[str] = frozenset({"eval", "exec", "compile", "__import__"})

# Path literals starting with these prefixes are forbidden as arguments to
# file-reading calls — agent code must not directly touch competition data.
_FORBIDDEN_PATH_PREFIXES: tuple[str, ...] = ("raw/", "raw\\", "/raw/", "./raw/")

# Calls whose first string-literal argument we check for forbidden path prefixes.
_PATH_OPEN_CALLS: frozenset[tuple[str, ...]] = frozenset({
    ("pd", "read_csv"),
    ("pd", "read_parquet"),
    ("pd", "read_feather"),
    ("pd", "read_json"),
    ("pd", "read_excel"),
    ("pandas", "read_csv"),
    ("pandas", "read_parquet"),
    ("pandas", "read_feather"),
    ("pandas", "read_json"),
    ("pandas", "read_excel"),
    ("np", "loadtxt"),
    ("np", "load"),
    ("numpy", "loadtxt"),
    ("numpy", "load"),
})

# Absolute filesystem paths the agent must never try to open (any mode).
# /Users/, /home/, /private/ are intentionally excluded: the workspace lives
# under one of these on macOS (/Users/) and Linux (/home/), and macOS also
# maps /tmp → /private/var/... Agent code may legitimately reference workspace
# paths under those prefixes. /etc/, /var/, /usr/, /root/ remain forbidden.
_FORBIDDEN_ABS_PATHS: tuple[str, ...] = (
    "/etc/", "/var/", "/usr/", "/root/",
)


@dataclass(frozen=True)
class LintResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def lint_module(path: str | Path) -> LintResult:
    """Scan a Python file's AST for forbidden patterns.

    Returns LintResult.ok=True if no violations; otherwise a list of
    human-readable violation messages keyed by line number.
    """
    path = Path(path)
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))

    aliases = _collect_import_aliases(tree)
    visitor = _ForbidVisitor(aliases=aliases, filename=str(path))
    visitor.visit(tree)

    return LintResult(ok=not visitor.violations, violations=visitor.violations)


def _collect_import_aliases(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """Map local names to their underlying dotted module path.

    `import os` → {"os": ("os",)}
    `import os as o` → {"o": ("os",)}
    `from os import path` → {"path": ("os", "path")}
    `from os import path as p` → {"p": ("os", "path")}
    """
    aliases: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.asname:
                    aliases[n.asname] = tuple(n.name.split("."))
                else:
                    # `import x.y.z` makes `x` available locally; the full module
                    # is loaded but `x.y.z` is accessed as `x.y.z`. We register the
                    # root so subsequent `x.foo(...)` resolves to ("x", "foo") not
                    # ("x", "y", "z", "foo").
                    root = n.name.split(".")[0]
                    aliases[root] = (root,)
        elif isinstance(node, ast.ImportFrom) and node.module:
            base = tuple(node.module.split("."))
            for n in node.names:
                local = n.asname or n.name
                aliases[local] = base + (n.name,)
    return aliases


class _ForbidVisitor(ast.NodeVisitor):
    def __init__(self, *, aliases: dict[str, tuple[str, ...]], filename: str) -> None:
        self.aliases = aliases
        self.filename = filename
        self.violations: list[str] = []

    def _attr_chain(self, node: ast.AST) -> tuple[str, ...]:
        """Resolve an AST attribute/name chain to a dotted tuple,
        de-aliased against imports."""
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        else:
            return ()
        chain = tuple(reversed(parts))
        if chain and chain[0] in self.aliases:
            return self.aliases[chain[0]] + chain[1:]
        return chain

    def _violate(self, msg: str, node: ast.AST) -> None:
        lineno = getattr(node, "lineno", "?")
        self.violations.append(f"{self.filename}:{lineno}: {msg}")

    # --- Calls ---
    def visit_Call(self, node: ast.Call) -> None:
        chain = self._attr_chain(node.func)

        # Forbidden builtins (eval, exec, compile, __import__)
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_BUILTINS:
            self._violate(f"forbidden builtin call: {node.func.id}", node)
        elif chain and chain[-1] == "open":
            # builtin open: check path arg for absolute denylist
            self._check_open_path(node)
        else:
            # Module-prefix denylist
            for forbidden in _FORBIDDEN_ATTR_CALLS:
                if chain[: len(forbidden)] == forbidden:
                    self._violate(
                        f"forbidden call: {'.'.join(chain)}", node
                    )
                    break

            # pd.read_csv / np.load et al. against raw/ paths
            if chain in _PATH_OPEN_CALLS:
                self._check_raw_path_arg(chain, node)

        # plain `open(...)`
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self._check_open_path(node)

        self.generic_visit(node)

    def _check_raw_path_arg(self, chain: tuple[str, ...], node: ast.Call) -> None:
        if not node.args:
            return
        arg = node.args[0]
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and any(arg.value.startswith(p) for p in _FORBIDDEN_PATH_PREFIXES)
        ):
            self._violate(
                f"forbidden {'.'.join(chain)} read of competition raw/ path: {arg.value!r}",
                node,
            )

    def _check_open_path(self, node: ast.Call) -> None:
        if not node.args:
            return
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            value = arg.value
            if any(value.startswith(p) for p in _FORBIDDEN_PATH_PREFIXES):
                self._violate(
                    f"forbidden open of competition raw/ path: {value!r}", node
                )
            if any(value.startswith(p) for p in _FORBIDDEN_ABS_PATHS):
                self._violate(
                    f"forbidden open of absolute system path: {value!r}", node
                )


@dataclass(frozen=True)
class SubprocessResult:
    """Outcome of run_subprocess: captured streams + classification.

    killed_reason is one of:
      None      — clean exit (success or non-zero rc with normal termination)
      "timeout" — wall-clock cap hit
      "memory"  — RLIMIT_AS triggered the kill (heuristic: SIGKILL on Linux)
    """

    returncode: int
    stdout: str
    stderr: str
    killed_reason: str | None
    script_path: Path


def _preexec_setrlimit(memory_bytes: int, cpu_seconds: int) -> None:
    """preexec_fn payload — POSIX-only. Sets per-process limits before exec.

    Errors from RLIMIT_AS are swallowed because Darwin rejects the call with
    "current limit exceeds maximum limit" (the inherited soft limit on macOS
    can exceed any value setrlimit will accept); we still try the call so
    Linux gets a hard cap. RLIMIT_CPU works on both platforms.
    """
    # Memory cap (address space). macOS often rejects RLIMIT_AS; degrade silently.
    with contextlib.suppress(OSError, ValueError):
        _resource.setrlimit(_resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    # CPU-time cap as a backstop against busy loops the wall-clock timeout might
    # race with (e.g., kernel scheduling slop). Same value as wall-clock for now.
    _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))


def run_subprocess(
    *,
    code: str | None = None,
    script_path: Path | None = None,
    workspace: _Workspace,
    timeout_s: int = 60,
    memory_mb: int = 2048,
) -> SubprocessResult:
    """Run Python in an isolated subprocess scoped to the workspace.

    Two input modes (exactly one required):
      - `code`: write to a fresh `workspace.scratch_dir/run_<ts>.py`,
        then execute. Backwards-compatible default.
      - `script_path`: execute an existing file as-is (F7). The caller
        owns the file — `run_subprocess` does NOT rewrite or rename it.
        This lets `run_python` lint and run the SAME on-disk artifact
        instead of writing it twice.

    Invokes `python <script>` with `cwd=workspace.root`, applies RLIMIT_AS,
    RLIMIT_CPU, RLIMIT_NPROC, RLIMIT_FSIZE via preexec_fn, and enforces
    wall-clock via subprocess `timeout`. Returns a SubprocessResult
    capturing stdout/stderr/returncode and a `killed_reason` string for
    timeout/memory kills.

    The caller is responsible for AST-linting first (e.g., by writing the
    code to a file and calling `lint_module`). This function does not lint.
    """
    if (code is None) == (script_path is None):
        raise ValueError("run_subprocess requires exactly one of `code` or `script_path`")

    workspace.scratch_dir.mkdir(parents=True, exist_ok=True)
    if script_path is None:
        assert code is not None  # narrowed for mypy
        stamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d_%H%M%S_%f")
        script_path = workspace.scratch_dir / f"run_{stamp}.py"
        script_path.write_text(code)

    memory_bytes = max(1, memory_mb) * 1024 * 1024
    preexec = None
    if _sys.platform != "win32":
        # Bind the two arguments at closure-construction time.
        def preexec() -> None:  # noqa: ANN202 — used only by subprocess
            _preexec_setrlimit(memory_bytes, timeout_s)

    try:
        completed = _subprocess.run(
            [_sys.executable, str(script_path)],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=preexec,
            check=False,
        )
        killed_reason = None
        # Heuristic: SIGKILL (-9) on Linux when RLIMIT_AS fires.
        if completed.returncode == -9:
            killed_reason = "memory"
        return SubprocessResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            killed_reason=killed_reason,
            script_path=script_path,
        )
    except _subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return SubprocessResult(
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            killed_reason="timeout",
            script_path=script_path,
        )
