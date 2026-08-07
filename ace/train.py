"""Offline Generator/Reflector/Curator loop that produces `rulebook.json`.

Usage (from the repo root, with EM_API_KEY and DEEPSEEK_API_KEY set in .env):

    uv run python -m ace.train
    uv run python -m ace.train --sample-size 200 --epochs 5

Split is by QUARTER, not by event: `predicted_percentile` is a cross-sectional
rank *within* a quarter (examples.scoring), so mixing train/validation rows
from the same quarter would leak that quarter's cross-section composition into
the signal we're optimizing against. TRAIN_QUARTERS feed the Generator/
Reflector; VALIDATION_QUARTER is the epoch-level accept/reject gate; the
contest's most recent quarter is deliberately left untouched here -- evaluate
the final rulebook against it once, afterward, via the existing
examples/notebooks/01_historical_archive.ipynb Section 6-8 flow (same
delta_r2_log.csv you already have), for a clean, lookahead-free comparison.

This is entirely an OFFLINE loop: nothing here runs at prediction time. The
live `predict()` path always reads whatever `rulebook.json` currently says on
disk -- there is no online adaptation, because the 5-minute webhook deadline
documented in predict.py leaves no room for a live Reflector/Curator pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import Client, load_config
from examples.archive import download_archive, load_archive
from examples.scoring import add_percentiles, outcomes_frame, score_submission

import predict
from ace.curator import merge
from ace.reflector import MissRecord, reflect
from ace.store import Bullet, load_rulebook, render, save_rulebook

ARCHIVE_DIR = REPO_ROOT / "examples" / "data" / "archive"
RULEBOOK_PATH = REPO_ROOT / "rulebook.json"
TRAIN_LOG_PATH = REPO_ROOT / "logs" / "train_log.csv"

TRAIN_QUARTERS = ["2025Q4", "2026Q1"]
VALIDATION_QUARTER = "2026Q2"
# The most recent quarter (2026Q3 as of writing) is deliberately absent from
# this file -- see module docstring. Check examples/data/README.md if the
# archive's available quarters have moved on since.

DEFAULT_SAMPLE_SIZE = 150
DEFAULT_VAL_SAMPLE_SIZE = 80
DEFAULT_EPOCHS = 5
DEFAULT_WORST_N = 20
DEFAULT_SEED = 0
DEFAULT_VAL_DRAWS = 3
MAX_WORKERS = 10


def facts_to_summary(disclosure: dict) -> str:
    """Rebuild an information_url-shaped summary string from an archive disclosure.

    Mirrors examples/notebooks/01_historical_archive.ipynb's helper of the same
    name -- duplicated rather than imported since it's notebook-local, not part
    of the examples package.
    """
    lines: list[str] = []
    for item in (disclosure or {}).get("items", []):
        if item.get("kind") == "facts":
            lines.extend(item["content"])
    return "\n".join(f"- {f}" for f in lines)


def _ensure_downloaded(quarters: list[str]) -> None:
    config = load_config()
    config.require_api_key()
    with Client.from_env() as client:
        manifest = client.archive_manifest()
        download_archive(
            manifest,
            ARCHIVE_DIR,
            client=client,
            only=lambda f: f.event_type == "EARNINGS_RELEASE" and f.quarter in quarters,
        )


def _quarter_frame(all_events: pd.DataFrame, quarter: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw records df, scored (event,asset) df with ground-truth `y`) for one quarter."""
    quarter_df = all_events[all_events["quarter"] == quarter].copy()
    records = quarter_df.to_dict("records")
    scored = add_percentiles(outcomes_frame(records))
    return quarter_df, scored


def _sample_rows(quarter_df: pd.DataFrame, scored: pd.DataFrame, n: int, seed: int) -> list[dict]:
    """Build Generator-ready rows: one per sampled (event, ticker) with ground truth `y`."""
    event_by_id = {r["event_id"]: r for r in quarter_df.to_dict("records")}
    sample = scored.sample(n=min(n, len(scored)), random_state=seed)
    rows = []
    for _, r in sample.iterrows():
        event = event_by_id.get(r["event_id"])
        if event is None:
            continue
        rows.append(
            {
                "event_id": r["event_id"],
                "ticker": r["identifier_value"],
                "event_type": event.get("event_type"),
                "as_of": event.get("knowledge_cutoff"),
                "summary_text": facts_to_summary(event.get("disclosure")),
                "y": float(r["y"]),
            }
        )
    return rows


def _run_generator(rows: list[dict], rulebook_text: str) -> list[tuple[dict, predict.Prediction]]:
    """Run predict._ask_llm over `rows` concurrently -- the Generator step."""

    def _one(row: dict) -> tuple[dict, predict.Prediction]:
        pred = predict._ask_llm(
            summary={"summary": row["summary_text"]},
            ticker=row["ticker"],
            event_type=row["event_type"],
            as_of=row["as_of"],
            rulebook_text=rulebook_text,
        )
        return row, pred

    results: list[tuple[dict, predict.Prediction]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_one, row) for row in rows]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 25 == 0 or i == len(futures):
                print(f"    generator: {i}/{len(futures)}")
    return results


@dataclass
class _AveragedPrediction:
    predicted_percentile: float


def _run_generator_averaged(
    rows: list[dict], rulebook_text: str, n_draws: int = DEFAULT_VAL_DRAWS
) -> list[tuple[dict, _AveragedPrediction]]:
    """Score `rows` with `n_draws` independent Generator passes, averaged per row.

    deepseek-v4-flash runs at temperature=1: a single pass is noisy enough that
    the accept/reject gate can't reliably tell "this rulebook change helped"
    from "the model just rolled differently this time" -- confirmed in practice
    (baseline delta_r_squared_imputed swung 0.0044-0.0056 across reruns with an
    identical, unchanged empty rulebook). The reference DeepSeek V4 Flash entry
    hits the same problem and works around it with 16 parallel draws, averaged;
    this does the same at a smaller multiple for the validation gate only --
    training-pass Generator calls (used just to surface candidate misses for
    the Reflector) stay single-draw, since averaging there isn't what's noisy
    for the decision that actually matters (accept/reject).
    """
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for draw in range(1, n_draws + 1):
        print(f"    validation draw {draw}/{n_draws}:")
        for row, pred in _run_generator(rows, rulebook_text):
            key = (row["event_id"], row["ticker"])
            sums[key] = sums.get(key, 0.0) + pred.predicted_percentile
            counts[key] = counts.get(key, 0) + 1

    by_key = {(r["event_id"], r["ticker"]): r for r in rows}
    return [
        (by_key[key], _AveragedPrediction(total / counts[key]))
        for key, total in sums.items()
    ]


def _score_against(
    scored: pd.DataFrame,
    results: list[tuple[dict, predict.Prediction]] | list[tuple[dict, "_AveragedPrediction"]],
) -> dict:
    """Score Generator `results` against a quarter's `scored` ground-truth frame."""
    pred_df = pd.DataFrame(
        [
            {
                "event_id": row["event_id"],
                "identifier_value": row["ticker"],
                "predicted": pred.predicted_percentile,
            }
            for row, pred in results
        ]
    )
    merged = scored.merge(pred_df, on=["event_id", "identifier_value"], how="left")
    return score_submission(merged, "predicted")


def _log_epoch(row: dict) -> None:
    TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not TRAIN_LOG_PATH.exists()
    with TRAIN_LOG_PATH.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def train(
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    val_sample_size: int = DEFAULT_VAL_SAMPLE_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    worst_n: int = DEFAULT_WORST_N,
    seed: int = DEFAULT_SEED,
    val_draws: int = DEFAULT_VAL_DRAWS,
) -> list[Bullet]:
    print(f"Downloading/checking archive for {TRAIN_QUARTERS + [VALIDATION_QUARTER]}...")
    _ensure_downloaded(TRAIN_QUARTERS + [VALIDATION_QUARTER])
    all_events = load_archive(ARCHIVE_DIR)

    train_rows: list[dict] = []
    for q in TRAIN_QUARTERS:
        qdf, scored = _quarter_frame(all_events, q)
        train_rows.extend(_sample_rows(qdf, scored, sample_size // len(TRAIN_QUARTERS), seed))
    print(f"Train pool: {len(train_rows)} (event, ticker) rows across {TRAIN_QUARTERS}")

    val_qdf, val_scored = _quarter_frame(all_events, VALIDATION_QUARTER)
    val_rows = _sample_rows(val_qdf, val_scored, val_sample_size, seed)
    print(f"Validation pool: {len(val_rows)} rows in {VALIDATION_QUARTER} (fixed across epochs)")

    current_bullets = load_rulebook(RULEBOOK_PATH)
    print(f"Starting from {len(current_bullets)} existing bullet(s) in {RULEBOOK_PATH.name}")

    print(f"Scoring baseline (current rulebook) on validation quarter ({val_draws} draws, averaged)...")
    baseline_results = _run_generator_averaged(val_rows, render(current_bullets), val_draws)
    best_score = _score_against(val_scored, baseline_results)["delta_r_squared_imputed"] or 0.0
    print(f"  baseline delta_r_squared_imputed = {best_score:.4f}")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_results = _run_generator(train_rows, render(current_bullets))

        misses = sorted(
            (
                MissRecord(
                    event_id=row["event_id"],
                    ticker=row["ticker"],
                    event_type=row["event_type"],
                    summary_text=row["summary_text"],
                    predicted_percentile=pred.predicted_percentile,
                    rationale=pred.rationale,
                    actual_percentile=row["y"],
                )
                for row, pred in train_results
            ),
            key=lambda m: m.abs_error,
            reverse=True,
        )[:worst_n]

        proposed = reflect(misses, epoch, current_bullets)
        print(f"  reflector proposed {len(proposed)} bullet(s)")
        for b in proposed:
            print(f"    [{b.category}] {b.text}")

        log_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "n_train": len(train_rows),
            "n_proposed": len(proposed),
            "proposed_bullets": json.dumps([{"category": b.category, "text": b.text} for b in proposed]),
            "best_score_before": best_score,
            "candidate_score": None,
            "accepted": False,
            "rulebook_size": len(current_bullets),
        }

        if not proposed:
            _log_epoch(log_row)
            continue

        candidate_bullets = merge(current_bullets, proposed)
        candidate_results = _run_generator_averaged(val_rows, render(candidate_bullets), val_draws)
        candidate_score = _score_against(val_scored, candidate_results)["delta_r_squared_imputed"]
        candidate_score = candidate_score if candidate_score is not None else -1.0
        accepted = candidate_score >= best_score

        print(f"  candidate delta_r_squared_imputed = {candidate_score:.4f} "
              f"({'accepted' if accepted else 'rejected'}, best so far {best_score:.4f})")

        log_row["candidate_score"] = candidate_score
        log_row["accepted"] = accepted
        _log_epoch(log_row)

        if accepted:
            current_bullets = candidate_bullets
            best_score = candidate_score
            # Checkpoint immediately, not just at the end -- a crash or Ctrl-C
            # partway through a long run (10 epochs x 400+100 events is not
            # fast) used to lose every accepted bullet, not just the pending one.
            save_rulebook(RULEBOOK_PATH, current_bullets)
            print(f"  checkpointed {len(current_bullets)} bullet(s) to {RULEBOOK_PATH.name}")

    save_rulebook(RULEBOOK_PATH, current_bullets)
    print(f"\nWrote {len(current_bullets)} bullet(s) to {RULEBOOK_PATH} "
          f"(best validation delta_r_squared_imputed = {best_score:.4f})")
    return current_bullets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--val-sample-size", type=int, default=DEFAULT_VAL_SAMPLE_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--worst-n", type=int, default=DEFAULT_WORST_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-draws", type=int, default=DEFAULT_VAL_DRAWS,
                         help="Independent Generator passes per validation event, averaged "
                              "before scoring -- reduces temperature=1 noise in the accept/reject gate.")
    args = parser.parse_args()
    train(
        sample_size=args.sample_size,
        val_sample_size=args.val_sample_size,
        epochs=args.epochs,
        worst_n=args.worst_n,
        seed=args.seed,
        val_draws=args.val_draws,
    )


if __name__ == "__main__":
    main()
