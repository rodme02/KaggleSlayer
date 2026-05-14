"""Sandbox for agent-written Python modules.

Week 1 scope: AST lint that scans a Python file before the harness loads
it and rejects forbidden patterns. The lint is the leak-prevention
mechanism for the in-process CV contract (see spec §6.5): if an agent
tries to read `raw/...` directly, the file fails lint and never executes.

Resource limits (subprocess + setrlimit) are added in Week 4 alongside the
broader sandbox hardening when the agent gets a generic `run_python` tool.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

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
_FORBIDDEN_ABS_PATHS: tuple[str, ...] = (
    "/etc/", "/var/", "/usr/", "/private/", "/Users/", "/root/", "/home/",
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
