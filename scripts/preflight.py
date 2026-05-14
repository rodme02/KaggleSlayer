"""Preflight check: verify Gemini API + Kaggle API credentials are working.

Usage:
    python scripts/preflight.py

Exits 0 if all checks pass, 1 otherwise. Re-run after fixing any issues.
This script is the seed of the future health-check telemetry (spec §11.2).
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def check_kaggle_credentials_file() -> tuple[bool, str]:
    if not KAGGLE_JSON.exists():
        return False, (
            f"missing {KAGGLE_JSON}\n"
            "  → Create one at https://www.kaggle.com/settings (API section, "
            "'Create New API Token'), then:\n"
            "      mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "      chmod 600 ~/.kaggle/kaggle.json"
        )
    mode = stat.S_IMODE(KAGGLE_JSON.stat().st_mode)
    if mode & 0o077:
        return False, (
            f"{KAGGLE_JSON} is world/group readable (mode {oct(mode)})\n"
            "  → Fix: chmod 600 ~/.kaggle/kaggle.json"
        )
    return True, f"{KAGGLE_JSON} present, mode {oct(mode)}"


def check_kaggle_api_call() -> tuple[bool, str]:
    """Try a read-only Kaggle API call. Authenticates implicitly on import."""
    try:
        from kaggle import api  # type: ignore[import-untyped]

        api.authenticate()
        comps = api.competitions_list(page=1)
    except Exception as e:  # noqa: BLE001
        return False, f"Kaggle API call failed: {e!r}"
    if not comps:
        return False, "Kaggle API returned an empty competition list (unexpected)"
    return True, f"listed {len(comps)} competitions (first: '{comps[0].ref}')"


def check_gemini_env_var() -> tuple[bool, str]:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return True, f"{var} is set (length {len(os.environ[var])})"
    return False, (
        "neither GEMINI_API_KEY nor GOOGLE_API_KEY is set\n"
        "  → Get a key at https://aistudio.google.com/app/apikey, then add to ~/.zshrc:\n"
        "      export GEMINI_API_KEY=\"...\"\n"
        "      source ~/.zshrc"
    )


def check_gemini_api_call() -> tuple[bool, str]:
    """Try a single cheap call to gemini-2.5-flash."""
    try:
        from google import genai
    except ImportError as e:
        return False, f"google-genai not installed: {e!r}"

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return False, "no API key in env (caught earlier)"

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with the single word: ok",
        )
    except Exception as e:  # noqa: BLE001
        return False, f"Gemini API call failed: {e!r}"

    text = (resp.text or "").strip()
    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", "?") if usage else "?"
    out_tok = getattr(usage, "candidates_token_count", "?") if usage else "?"
    return True, f"reply={text!r} (tokens: in={in_tok}, out={out_tok})"


def main() -> int:
    console.rule("[bold cyan]KaggleSlayer preflight")

    checks = [
        ("Kaggle credentials file", check_kaggle_credentials_file),
        ("Kaggle API call",         check_kaggle_api_call),
        ("Gemini env var",          check_gemini_env_var),
        ("Gemini API call",         check_gemini_api_call),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", width=28)
    table.add_column("Status", width=6, justify="center")
    table.add_column("Detail")

    all_ok = True
    blocked_by_prereq = False
    for label, fn in checks:
        # Skip the API call if its prereq failed
        if label == "Kaggle API call" and blocked_by_prereq:
            table.add_row(label, "[yellow]skip", "(credentials file missing)")
            continue
        if label == "Gemini API call" and blocked_by_prereq:
            table.add_row(label, "[yellow]skip", "(env var missing)")
            continue

        ok, detail = fn()
        if not ok:
            all_ok = False
            if label in ("Kaggle credentials file", "Gemini env var"):
                blocked_by_prereq = True
            else:
                blocked_by_prereq = False
            table.add_row(label, "[red]✗", detail)
        else:
            blocked_by_prereq = False
            table.add_row(label, "[green]✓", detail)

    console.print(table)

    if all_ok:
        console.print(Panel.fit("[bold green]All preflight checks passed.[/]", border_style="green"))
        return 0
    console.print(Panel.fit(
        "[bold red]Preflight failed.[/] Fix the issues above and rerun.",
        border_style="red",
    ))
    return 1


if __name__ == "__main__":
    sys.exit(main())
