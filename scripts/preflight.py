"""Preflight check: verify Gemini API + Kaggle API credentials are working.

Usage:
    python scripts/preflight.py

Loads .env automatically (if present). Exits 0 if all checks pass, 1 otherwise.
This script is the seed of the future health-check telemetry (spec §11.2).

Credential sources accepted:
  - Gemini: GEMINI_API_KEY or GOOGLE_API_KEY (env or .env)
  - Kaggle (new format, v2.1+):
        KAGGLE_API_TOKEN env var       (preferred for .env workflows)
        ~/.kaggle/access_token file
  - Kaggle (legacy format, pre-v2):
        KAGGLE_USERNAME + KAGGLE_KEY env vars
        ~/.kaggle/kaggle.json file
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load .env from the repo root (one level up from this script).
ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = ROOT / ".env"
DOTENV_LOADED = load_dotenv(DOTENV_PATH)

console = Console()

KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
KAGGLE_ACCESS_TOKEN_FILE = Path.home() / ".kaggle" / "access_token"


def _check_file_mode(path: Path) -> tuple[bool, str]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        return False, (
            f"{path} is world/group readable (mode {oct(mode)})\n"
            f"  → Fix: chmod 600 {path}"
        )
    return True, f"{path} (mode {oct(mode)})"


def check_kaggle_credentials() -> tuple[bool, str]:
    """Accept any of: KAGGLE_API_TOKEN (new), ~/.kaggle/access_token (new),
    KAGGLE_USERNAME+KAGGLE_KEY (legacy), ~/.kaggle/kaggle.json (legacy)."""
    if token := os.environ.get("KAGGLE_API_TOKEN"):
        return True, f"using env var KAGGLE_API_TOKEN (len {len(token)})"

    if KAGGLE_ACCESS_TOKEN_FILE.exists():
        return _check_file_mode(KAGGLE_ACCESS_TOKEN_FILE)

    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return True, f"using legacy env vars KAGGLE_USERNAME='{env_user}', KAGGLE_KEY (len {len(env_key)})"

    if KAGGLE_JSON.exists():
        return _check_file_mode(KAGGLE_JSON)

    return False, (
        "no Kaggle credentials found.\n"
        "  → Option A (recommended, new format) — single env var in .env:\n"
        "      KAGGLE_API_TOKEN=KGAT_...        (paste the whole token from kaggle.com)\n"
        "  → Option B — new-format file:\n"
        "      mkdir -p ~/.kaggle && echo 'KGAT_...' > ~/.kaggle/access_token\n"
        "      chmod 600 ~/.kaggle/access_token\n"
        "  → Option C (legacy) — split env vars:  KAGGLE_USERNAME + KAGGLE_KEY\n"
        "  → Option D (legacy) — ~/.kaggle/kaggle.json with {username, key} JSON\n"
        "  (Get the token from https://www.kaggle.com/settings → API → 'Create New API Token')"
    )


def check_kaggle_api_call() -> tuple[bool, str]:
    """Try a read-only Kaggle API call. Library auto-detects creds at import time."""
    try:
        from kaggle import api  # type: ignore[import-untyped]

        api.authenticate()
        resp = api.competitions_list(page=1)
    except Exception as e:  # noqa: BLE001
        return False, f"Kaggle API call failed: {e!r}"

    # kaggle>=2.x returns an ApiListCompetitionsResponse with .competitions list;
    # older versions returned a plain list.
    comps = getattr(resp, "competitions", resp)
    if not comps:
        return False, "Kaggle API returned an empty competition list (unexpected)"
    first = comps[0]
    ref = getattr(first, "ref", None) or getattr(first, "url", None) or str(first)[:60]
    return True, f"listed {len(comps)} competitions (first: '{ref}')"


def check_gemini_env_var() -> tuple[bool, str]:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return True, f"{var} is set (length {len(os.environ[var])})"
    return False, (
        "neither GEMINI_API_KEY nor GOOGLE_API_KEY is set\n"
        "  → Add to .env (or ~/.zshrc):\n"
        "      GEMINI_API_KEY=your-key-here\n"
        "  (Get a key at https://aistudio.google.com/app/apikey)"
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
    if DOTENV_LOADED:
        console.print(f"[dim]loaded {DOTENV_PATH.relative_to(ROOT)}[/dim]")
    elif DOTENV_PATH.exists():
        console.print(f"[yellow]warning:[/] {DOTENV_PATH} exists but failed to load")
    else:
        console.print(f"[dim]no {DOTENV_PATH.relative_to(ROOT)} found (using only shell env)[/dim]")

    checks = [
        ("Kaggle credentials", check_kaggle_credentials),
        ("Kaggle API call",    check_kaggle_api_call),
        ("Gemini env var",     check_gemini_env_var),
        ("Gemini API call",    check_gemini_api_call),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", width=28)
    table.add_column("Status", width=6, justify="center")
    table.add_column("Detail")

    all_ok = True
    blocked_by_prereq = False
    for label, fn in checks:
        if label == "Kaggle API call" and blocked_by_prereq:
            table.add_row(label, "[yellow]skip", "(credentials missing)")
            continue
        if label == "Gemini API call" and blocked_by_prereq:
            table.add_row(label, "[yellow]skip", "(env var missing)")
            continue

        ok, detail = fn()
        if not ok:
            all_ok = False
            if label in ("Kaggle credentials", "Gemini env var"):
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
