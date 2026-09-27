"""Controlled A/B backtest: predict_02.py vs predict.py (Frankfurt-MC-02).

Runs the SAME event sample from a held-out quarter through both variants,
several independent trials each, so you can see whether the live leaderboard
gap between them holds up under repeated draws or is within noise. Both now
run on DeepSeek -- predict_02.py is predict_kimi.py's static 3-bullet,
single-call, no-extra-signals setup ported to DeepSeek, so the only variables
left vs predict.py are rulebook (static 3-bullet vs dynamic 8+ bullet), extra
signals (market cap/SEC filings), and ensembling -- the model itself is no
longer a confound. This at least tells you whether the *overall* gap is
bigger than run-to-run noise.

2026Q3 is used by default because it's the one quarter never touched by
ace/train.py (2025Q4+2026Q1 are the training quarters, 2026Q2 is the
validation quarter) -- a clean, lookahead-free comparison point.

Usage:
    uv run python3 validate.py
    uv run python3 validate.py --sample-size 100 --trials 3 --quarter 2026Q3
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# This script lives one level below the repo root; predict.py sits at the
# root and predict_02_submitted.py in submissions/, neither importable by
# default from here.
REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (REPO_ROOT, REPO_ROOT / "submissions"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from examples import Client, load_config
from examples.archive import download_archive, load_archive
from examples.scoring import add_percentiles, outcomes_frame, score_submission

import predict as predict_deepseek
import predict_02_submitted

ARCHIVE_DIR = REPO_ROOT / "examples" / "data" / "archive"

DEFAULT_QUARTER = "2026Q3"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_TRIALS = 3
MAX_WORKERS = 10  # Both variants are DeepSeek now (ace/train.py's Generator
                  # already runs 10 concurrent DeepSeek calls without issue).


def facts_to_summary(disclosure: dict) -> str:
    """Rebuild an information_url-shaped summary string from an archive disclosure.

    Mirrors examples/notebooks/01_historical_archive.ipynb's helper of the same
    name and ace/train.py's copy of it.
    """
    lines: list[str] = []
    for item in (disclosure or {}).get("items", []):
        if item.get("kind") == "facts":
            lines.extend(item["content"])
    return "\n".join(f"- {f}" for f in lines)


def _ensure_downloaded(quarter: str) -> None:
    config = load_config()
    config.require_api_key()
    with Client.from_env() as client:
        manifest = client.archive_manifest()
        download_archive(
            manifest,
            ARCHIVE_DIR,
            client=client,
            only=lambda f: f.event_type == "EARNINGS_RELEASE" and f.quarter == quarter,
        )


def _build_sample(quarter: str, n: int, seed: int) -> tuple[list[dict], pd.DataFrame]:
    all_events = load_archive(ARCHIVE_DIR)
    qdf = all_events[all_events["quarter"] == quarter].copy()
    records = qdf.to_dict("records")
    scored = add_percentiles(outcomes_frame(records))
    event_by_id = {r["event_id"]: r for r in records}

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
            }
        )
    return rows, scored


def _run_predict_02(rows: list[dict]) -> list[dict]:
    """predict_02.py: DeepSeek, static rulebook baked into SYSTEM_PROMPT."""

    def _one(row: dict) -> dict:
        pct = predict_02_submitted._ask_llm(
            summary={"summary": row["summary_text"]},
            ticker=row["ticker"],
            event_type=row["event_type"],
            as_of=row["as_of"],
        )
        return {"event_id": row["event_id"], "identifier_value": row["ticker"], "predicted": pct}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_one, row) for row in rows]
        results = []
        for i, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            print(f"    {i}/{len(futures)}")
        return results


def _run_deepseek(rows: list[dict]) -> list[dict]:
    """Frankfurt-MC-02's model: DeepSeek, dynamic rulebook.json, market cap, SEC filings, ensembled."""

    def _one(row: dict) -> dict:
        pred = predict_deepseek._ask_llm(
            summary={"summary": row["summary_text"]},
            ticker=row["ticker"],
            event_type=row["event_type"],
            as_of=row["as_of"],
        )
        return {
            "event_id": row["event_id"],
            "identifier_value": row["ticker"],
            "predicted": pred.predicted_percentile,
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_one, row) for row in rows]
        results = []
        for i, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            print(f"    {i}/{len(futures)}")
        return results


def _score(results: list[dict], scored: pd.DataFrame) -> dict:
    pred_df = pd.DataFrame(results)
    merged = scored.merge(pred_df, on=["event_id", "identifier_value"], how="left")
    return score_submission(merged, "predicted")


def run(*, quarter: str, sample_size: int, trials: int, seed: int) -> None:
    print(f"Downloading/checking archive for {quarter}...")
    _ensure_downloaded(quarter)
    rows, scored = _build_sample(quarter, sample_size, seed)
    print(f"Sample: {len(rows)} (event, ticker) rows from {quarter} -- same rows for every trial and both variants.\n")

    variants = {
        "predict_02 (DeepSeek, static rulebook)": _run_predict_02,
        "Frankfurt-MC-02 (DeepSeek, dynamic rulebook + signals + ensembling)": _run_deepseek,
    }

    summary = []
    for name, run_fn in variants.items():
        print(f"=== {name} ===")
        trial_scores = []
        for t in range(1, trials + 1):
            results = run_fn(rows)
            score = _score(results, scored)
            # Unimputed, not delta_r_squared_imputed: both variants always predict
            # all `rows` here (100/100 coverage), so there's nothing for the
            # official imputation rule to protect against -- it would just dilute
            # this sample's real signal with ~1000+ mean-filled common-sample rows
            # neither variant ever touches, crushing the gap into trial-to-trial
            # noise. Imputation matters for comparing *live* submissions with
            # genuinely different coverage, not this fixed-sample local A/B.
            delta = score["delta_r_squared"]
            trial_scores.append(delta if delta is not None else float("nan"))
            r2, r2_s = score["r_squared"], score["r_squared_surprise"]
            print(
                f"  trial {t}/{trials}: r_squared={r2:.4f}  r_squared_surprise={r2_s:.4f}"
                f"  delta_r_squared={r2 - r2_s:.4f}"
            )
        valid = [s for s in trial_scores if s == s]  # drop NaN
        mean = sum(valid) / len(valid) if valid else float("nan")
        spread = (max(valid) - min(valid)) if valid else float("nan")
        print(f"  mean = {mean:.4f}   range across trials = {spread:.4f}\n")
        summary.append({"variant": name, "trials": trial_scores, "mean": mean, "range": spread})

    print("=== Summary ===")
    for r in summary:
        trials_str = ", ".join(f"{x:.4f}" for x in r["trials"])
        print(f"{r['variant']}")
        print(f"    trials: [{trials_str}]   mean: {r['mean']:.4f}   range: {r['range']:.4f}")

    if len(summary) == 2:
        gap = summary[1]["mean"] - summary[0]["mean"]
        worst_range = max(summary[0]["range"], summary[1]["range"])
        print(f"\nMean gap (MC-02 - 02): {gap:.4f}   Largest single-variant trial-to-trial range: {worst_range:.4f}")
        if abs(gap) < worst_range:
            print("-> Gap is SMALLER than the noise seen within a single variant's own trials -- not distinguishable from run-to-run noise at this sample size.")
        else:
            print("-> Gap is LARGER than either variant's own trial-to-trial noise -- more likely a real difference, though still worth a bigger sample to be sure.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarter", default=DEFAULT_QUARTER)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(quarter=args.quarter, sample_size=args.sample_size, trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()
