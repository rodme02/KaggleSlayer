#!/usr/bin/env python3
"""
Kaggle Tabular Harvester — Unified Script (no premature folders; delete non-tabular)
- Scans competitions with tabular-related terms
- Maintains competitions.csv + blacklist.csv
- Tries downloads via staging; only creates final competition folder on success+tabular
- Updates has_access = True if (joined) OR (download works)
- If non-tabular after download, deletes folder and blacklists the competition
"""

import argparse
import csv
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    KaggleApi = None  # type: ignore

SEARCH_TERMS = [
    "tabular", "regression", "classification",
    "time series", "forecasting", "structured", "numerical",
]

CSV_FIELDS = [
    "ref", "title", "category", "deadline", "reward", "url",
    "has_access", "is_tabular", "terms_hit",
    "last_checked_at", "last_download_status",
]

BLACKLIST_FIELDS = ["ref", "reason", "timestamp"]


@dataclass
class CompRow:
    ref: str
    title: str
    category: str
    deadline: str
    reward: str
    url: str
    has_access: str = "false"
    is_tabular: str = ""
    terms_hit: str = ""
    last_checked_at: str = ""
    last_download_status: str = ""


# ---------- Utils ----------

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_ref(v: str) -> str:
    from urllib.parse import urlparse
    if not v:
        return ""
    v = v.strip()
    if "/" not in v and " " not in v:
        return v
    try:
        parsed = urlparse(v if v.startswith("http") else f"https://{v}")
        parts = [p for p in parsed.path.split("/") if p and p not in {"competitions", "c"}]
        if parts:
            return parts[0]
    except Exception:
        pass
    return v.rstrip("/").split("/")[-1]


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict], fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_csv(path: Path, row: dict, fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_blacklist(path: Path) -> Set[str]:
    return {normalize_ref(r.get("ref", "")) for r in read_csv(path) if r.get("ref")}


def add_blacklist(path: Path, ref: str, reason: str):
    append_csv(path, {"ref": ref, "reason": reason, "timestamp": now_iso()}, BLACKLIST_FIELDS)


def ensure_removed(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except Exception:
                pass


# ---------- Kaggle API helpers ----------

def kaggle_auth() -> KaggleApi:
    if KaggleApi is None:
        print("Install kaggle first: pip install kaggle", file=sys.stderr)
        sys.exit(2)
    api = KaggleApi()
    api.authenticate()
    return api


def crawl(api: KaggleApi, group: str, terms: List[str], max_pages=-1) -> Dict[str, CompRow]:
    found: Dict[str, CompRow] = {}
    for term in terms:
        page = 1
        while True:
            batch = api.competitions_list(search=term or None, group=group, page=page)
            if not batch:
                break
            for c in batch:
                ref = normalize_ref(getattr(c, "ref", ""))
                if not ref:
                    continue
                if ref not in found:
                    found[ref] = CompRow(
                        ref=ref,
                        title=getattr(c, "title", ""),
                        category=getattr(c, "category", ""),
                        deadline=str(getattr(c, "deadline", "")),
                        reward=getattr(c, "reward", ""),
                        url=f"https://www.kaggle.com/competitions/{ref}",
                        terms_hit=term,
                        last_checked_at=now_iso(),
                    )
                else:
                    terms_set = set(t for t in found[ref].terms_hit.split(",") if t)
                    terms_set.add(term)
                    found[ref].terms_hit = ",".join(sorted(terms_set))
            page += 1
            if 0 < max_pages < page:
                break
    return found


def crawl_joined(api: KaggleApi, max_pages=-1) -> Set[str]:
    joined = set()
    page = 1
    while True:
        batch = api.competitions_list(group="entered", page=page)
        if not batch:
            break
        for c in batch:
            ref = normalize_ref(getattr(c, "ref", ""))
            if ref:
                joined.add(ref)
        page += 1
        if 0 < max_pages < page:
            break
    return joined


# ---------- Files & validation ----------

def list_all_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def unzip_all(root: Path, max_passes=3):
    import zipfile
    for _ in range(max_passes):
        zips = [p for p in list_all_files(root) if p.suffix.lower() == ".zip"]
        if not zips:
            break
        for z in zips:
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(z.parent)
            except Exception as e:
                print(f"    Could not unzip {z.name}: {e}")


def has_tabular(root: Path) -> bool:
    return any(p.suffix.lower() in (".csv", ".parquet") for p in list_all_files(root))


# ---------- Download via staging (no final folder on failure) ----------

def try_download_to_staging(api: KaggleApi, slug: str, staging_root: Path) -> Tuple[bool, str, Path]:
    """
    Attempts download into staging/<slug>. Returns (ok, status, staging_path).
    - On any failure, staging folder is cleaned up by the caller.
    - On success, caller will validate and then move to final location.
    """
    staging_path = staging_root / slug
    ensure_removed(staging_path)  # in case of leftovers
    staging_path.mkdir(parents=True, exist_ok=True)

    try:
        api.competition_download_files(slug, path=str(staging_path), quiet=True)
        unzip_all(staging_path, max_passes=3)
        return True, "downloaded", staging_path
    except Exception as e:
        # don't keep staging on failure
        msg = str(e).lower()
        if "403" in msg or "forbidden" in msg or "unauthorized" in msg:
            return False, "skipped_not_authorized", staging_path
        return False, f"error: {e}", staging_path


# ---------- Orchestration ----------

def orchestrate(out_csv: Path, blacklist_csv: Path, out_dir: Path, max_pages: int, terms: List[str]) -> int:
    api = kaggle_auth()

    print("Scanning competitions (general)...")
    comps = crawl(api, "general", terms, max_pages)
    joined = crawl_joined(api)

    blacklist = load_blacklist(blacklist_csv)
    rows = read_csv(out_csv)
    existing = {normalize_ref(r.get("ref", "")): r for r in rows}

    # Prepare CSV rows (do NOT create any competition folders here)
    final_rows: List[dict] = []
    for slug, comp in comps.items():
        if slug in blacklist:
            continue
        prev = existing.get(slug, {})
        row = {**{k: "" for k in CSV_FIELDS}, **asdict(comp)}
        # carry over previous known states
        if prev.get("is_tabular"):
            row["is_tabular"] = prev["is_tabular"]
        if prev.get("last_download_status"):
            row["last_download_status"] = prev["last_download_status"]
        # has_access: true if joined (may be upgraded later if download succeeds)
        row["has_access"] = "true" if slug in joined else "false"
        final_rows.append(row)

    final_rows = sorted(final_rows, key=lambda r: r["ref"])
    write_csv(out_csv, final_rows, CSV_FIELDS)
    print(f"Wrote {len(final_rows)} competitions to {out_csv}")

    # Download phase (use hidden staging; only move on success+tabular)
    staging_root = out_dir / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    updated: List[dict] = []
    for i, row in enumerate(final_rows, 1):
        slug = row["ref"]
        final_path = out_dir / slug

        print(f"[{i}/{len(final_rows)}] {slug}: evaluating...")

        # If final folder already exists, decide based on its contents
        if final_path.exists():
            if has_tabular(final_path):
                # Already valid; ensure flags are correct
                row["is_tabular"] = "true"
                row["has_access"] = "true"  # we must have succeeded before
                row["last_download_status"] = row.get("last_download_status") or "already_present"
                row["last_checked_at"] = now_iso()
                updated.append(row)
                continue
            else:
                # Existing non-tabular -> delete and blacklist
                ensure_removed(final_path)
                add_blacklist(blacklist_csv, slug, "non_tabular")
                row["is_tabular"] = "false"
                row["last_download_status"] = "removed_non_tabular"
                row["last_checked_at"] = now_iso()
                # do NOT keep in competitions.csv
                continue

        # Try download to staging (no final folder yet)
        ok, status, staging_path = try_download_to_staging(api, slug, staging_root)
        row["last_download_status"] = status
        row["last_checked_at"] = now_iso()

        if ok:
            # Successful download; validate tabularity in staging
            if has_tabular(staging_path):
                # mark access true (download worked), move to final location
                row["has_access"] = "true"
                row["is_tabular"] = "true"
                final_path.parent.mkdir(parents=True, exist_ok=True)
                # move staging/<slug> -> out_dir/<slug>
                ensure_removed(final_path)
                shutil.move(str(staging_path), str(final_path))
                updated.append(row)
            else:
                # Non-tabular -> delete staging and blacklist
                ensure_removed(staging_path)
                row["is_tabular"] = "false"
                add_blacklist(blacklist_csv, slug, "non_tabular")
                # do NOT keep in competitions.csv
        else:
            # Download failure -> never create final folder; remove staging (if created)
            ensure_removed(staging_path)
            # has_access stays as-is (true if joined; else false). Keep row so we can retry later.
            if slug in joined:
                row["has_access"] = "true"
            # We don't know tabularity yet.
            updated.append(row)

        time.sleep(0.2)

    # Rewrite competitions.csv with the rows we kept
    write_csv(out_csv, sorted(updated, key=lambda r: r["ref"]), CSV_FIELDS)
    print(f"Summary — kept: {len(updated)}, removed (non-tabular): {len(final_rows) - len(updated)}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scan Kaggle for tabular comps, download via staging, blacklist non-tabular.")
    p.add_argument("--out-csv", default="competition_data/competitions.csv")
    p.add_argument("--blacklist", default="competition_data/blacklist.csv")
    p.add_argument("--out-dir", default="downloaded_datasets")
    p.add_argument("--max-pages", type=int, default=-1)
    p.add_argument("--terms", nargs="*", default=SEARCH_TERMS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_csv = Path(args.out_csv)
    blacklist_csv = Path(args.blacklist)
    out_dir = Path(args.out_dir)
    try:
        return orchestrate(out_csv, blacklist_csv, out_dir, args.max_pages, args.terms)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
