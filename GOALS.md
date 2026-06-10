# GOALS — KaggleSlayer

Single source of truth for project status, the v1 scope boundary, and what comes
after. README.md and CLAUDE.md point here so status lives in exactly one place.

## Where we are today

KaggleSlayer is an LLM-agent harness for tabular Kaggle competitions: a Gemini-driven
Solver writes `agent/fe.py` + `agent/model.py`, and the trusted harness owns leak-free
cross-validation, the tool surface, the checkpoint gate on Kaggle submissions, and the
journal that makes any run resumable.

What a clean clone can do **today**, with no API keys:

- `pip install -e ".[dev,dashboard]"` then `pytest -m "not slow"` → **408 tests pass**
  (~5s). This is exactly what CI enforces on Linux 3.11 + 3.12.
- `ruff check kaggle_slayer tests` is clean; `mypy kaggle_slayer/harness` is clean
  (CI type-checks the **harness only** — see CLAUDE.md).
- `kaggle-slayer-dashboard` launches the read-only Streamlit dashboard over disk.
- Docs suite shipped 2026-06-10 (pulled forward from the roadmap at user request):
  `docs/architecture.md`, six ADRs under `docs/adr/`, and `.claude/` scaffolding
  (`/gates`, `/solve`, `/harness-review`, `/new-adr` + a `harness-reviewer` agent).

What needs credentials today: a **real solve** (`kaggle-slayer competitions/<name>
--target <col>`) requires a Gemini API key and Kaggle credentials. There is **no
credential-free demo wired yet** — exposing one is the headline v1 goal (see below).

## v1 scope boundary

v1 = make the existing harness clean-clone-runnable and demonstrable. We are NOT
adding new agent capabilities.

### In scope for v1

- **Credential-free demo (the headline goal).** A clean clone runs the agent loop
  end-to-end with NO API keys via a documented fake-LLM / synthetic-comp path. A
  generator already exists at `tests/fixtures/synthetic_comp.py`; v1 decides how to
  expose it given `competitions/`, `*.csv`, and `*.parquet` are gitignored. *(Not yet
  wired — do not claim it works until it does.)*
- **One fully worked real example** for the Gemini + Kaggle path, reproducible, plus
  the `.env.example` (shipped).
- **Honest, drift-free docs** — every README claim matches what CI enforces and what a
  clean clone can do; one coherent story across README / CLAUDE.md / GOALS.md.
- **Hero media** in `docs/media/` wired into the README (file capture is a user action).
- **Airtight hygiene** — no secrets or run artifacts tracked; stale per-comp workspaces
  stay out of the tree (`competitions/` is gitignored).
- **Promote to a real v1** — drop the "Week 5 of 6" framing; frame deferred work as
  roadmap (this file).

### Out of scope for v1 (roadmap, below)

- Live leaderboard / benchmark table
- MLflow artifact logging (`fe.py` / `model.py` / `oof_preds.npy`)
- CV↔LB backfill (writing the `lb_score` side of the calibration log)
- Dashboard diff page (`fe_v01`↔`fe_v02`) and cross-comp page
- Phase 2 / Phase 3 features (multi-agent, cloud-burst, NLP / CV / audio tracks)

(ADRs and `.claude/` scaffolding were originally deferred here; both shipped
2026-06-10.)

## Post-v1 roadmap

| Item | Notes |
| --- | --- |
| **Live leaderboard / benchmark table** | Run the harness across real Kaggle Playground comps and publish a results table. |
| **MLflow artifact logging** | Log `fe.py`, `model.py`, and `oof_preds.npy` as run artifacts (today only params + metrics + tags land). |
| **CV↔LB backfill** | Backfill `lb_score` into `~/.kaggle_slayer/calibration.jsonl` after Kaggle scores a submission; today only the CV side is written. |
| **Dashboard diff page** | Compare `fe_v01`↔`fe_v02` across archived agent versions. |
| **Dashboard cross-comp page** | Aggregate metrics across competitions. |
| **Phase 2 / Phase 3** | Multi-agent, cloud-burst, and NLP / CV / audio tracks — explicitly deferred. |

## Invariants v1 must never regress

These are the load-bearing contracts (mirrored as Hard Rules in CLAUDE.md):

1. **Leak-free CV contract** — `train_cv` only ever passes one fold's training data to
   the agent's `fit_feature_transformer` (`harness/cv.py`).
2. **AST sandbox lint** — agent code is linted before import; rejected files never run
   (`harness/sandbox.py`).
3. **Checkpoint gate** — Kaggle submissions only flow through the gated `submit_kaggle`
   (`harness/checkpoints.py`).
4. **Append-only journal + resume** — every tool call is journalled; `--resume` rebuilds
   from `run_log.jsonl` (`harness/journal.py`, `harness/resume.py`).
5. **Telemetry never crashes the agent** — MLflow / OTel / calibration failures are
   caught and logged; the loop continues (`harness/telemetry/`).
6. **Four-tier test structure** — unit / integration / chaos / slow, kept intact.
