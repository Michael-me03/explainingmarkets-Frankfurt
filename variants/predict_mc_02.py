"""predict_mc_02.py — frozen snapshot of the Frankfurt-MC-02 strategy, for the
validate.py / notebook A/B comparison against predict_02.py.

`predict.py` is the file that's actually deployed and actively trained by
ace/train.py -- it keeps moving. This file is a stable copy of it, so the A/B
comparison has a fixed target instead of shifting under you mid-training.

Only intended difference vs predict_02.py: this one loads its rulebook
dynamically from rulebook.json (ace/store.render) and keeps the `rationale`
tool field for ace/reflector.py's diagnostics; predict_02.py has the rulebook
baked into SYSTEM_PROMPT as a fixed string and no `rationale` field. No
market-cap/SEC-filing signals in either right now -- add them back manually
later if you want to test them again.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field
from ace.store import load_rulebook, render as render_rulebook
from data_provider.sec_edgar import get_insider_activity_summary, get_recent_filings_summary


_deepseek: OpenAI | None = None  # lazy: importing this file must not require a key
_deepseek_warned = False         # one-shot warning when no key is configured
_prompt_log_count = 0            # print every PROMPT_LOG_EVERY-th system/user prompt, for debugging
_prompt_log_lock = threading.Lock()
# Override per-run with `PROMPT_LOG_EVERY=25 uv run ...` -- defaults high so a
# large notebook backtest (hundreds of calls) isn't flooded; ace/train.py's
# much smaller sample sizes are where a lower value is actually useful.
PROMPT_LOG_EVERY = int(os.environ.get("PROMPT_LOG_EVERY", 100))

# Timeouts, sized against the 5-minute prediction window that opens when your
# handler ACKs the webhook. Worst case is 15 + (120 x 2) + 15 = 270s, which
# fits with ~30s to spare. Nothing upstream retries a failed prediction — once
# the delivery is ACKed the platform considers it done — so the one retry here
# is the only one you get. Raising either value can push you past the deadline.
SUMMARY_TIMEOUT_SECONDS = 15.0
LLM_TIMEOUT_SECONDS = 120.0
LLM_MAX_RETRIES = 1

# `temperature=1` makes a single draw noisy -- the accept/reject gate in
# ace/train.py already averages multiple draws to see past that noise, and
# the same noise hits every live submission just as much. Averaging
# N_ENSEMBLE_DRAWS concurrent draws per asset applies that same fix live.
# Concurrent, not sequential: worst case is still bounded by one call's
# timeout (120s x 2 retries = 240s), not N x that, so this doesn't change the
# 270s budget math above.
N_ENSEMBLE_DRAWS = 3

# DeepSeek is OpenAI-compatible; only the base_url + model differ.
# deepseek-v4-flash is the cheap/fast tier; deepseek-v4-pro is the stronger,
# pricier reasoning tier. Swap the active line to A/B test.
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"
#DEEPSEEK_MODEL = "deepseek-v4-pro"

# Rulebook produced offline by `ace/train.py` (Generator/Reflector/Curator loop).
# Loaded once per process -- it's only ever written between deploys, never at
# request time, so there is no online adaptation and no per-call disk read.
RULEBOOK_PATH = Path(__file__).with_name("rulebook.json")
_rulebook_text_cache: str | None = None


def _rulebook_block(rulebook_text: str | None) -> str:
    """Resolve the rulebook text to inject: explicit override, else the cached file."""
    global _rulebook_text_cache
    if rulebook_text is not None:
        return rulebook_text
    if _rulebook_text_cache is None:
        _rulebook_text_cache = render_rulebook(load_rulebook(RULEBOOK_PATH))
    return _rulebook_text_cache


def predict(event: dict) -> list[dict]:
    """Return predictions for one Explaining Markets event.

    `event` is the verified webhook payload. Useful fields:
      event["event_type"]          e.g. "EARNINGS_RELEASE"
      event["focal_assets"]        list of {"identifier_type", "identifier_value"}
      event["information_url"]     short-lived signed URL with the event summary JSON
      event["prediction_deadline"] ISO timestamp; submit before this fires

    Required return: a list of dicts, one per focal asset:
      [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]

    `predicted_percentile` is a float in [0, 1] — where you predict the asset's
    next-day abnormal (market-adjusted) return will rank across all of the
    quarter's event outcomes: 0 = the quarter's most negative reaction,
    0.50 = median, 1 = its most positive. It's a cross-sectional rank across the
    quarter's events, not a percentile within the asset's own history.
    """
    summary = httpx.get(event["information_url"], timeout=SUMMARY_TIMEOUT_SECONDS)
    summary.raise_for_status()
    summary_json = summary.json()

    # One ensembled prediction per focal asset. Today every event carries a
    # single asset, so this is N_ENSEMBLE_DRAWS calls total, all concurrent;
    # if multi-asset events show up, the per-asset ensembles would need to run
    # concurrently with each other too, rather than serially, to stay in budget.
    return [
        {
            "identifier_value": asset["identifier_value"],
            "predicted_percentile": _ensembled_percentile(
                summary=summary_json,
                ticker=asset["identifier_value"],
                event_type=event["event_type"],
                as_of=event.get("knowledge_cutoff"),
            ),
        }
        for asset in event["focal_assets"]
    ]


PREDICT_LOG_PATH = Path(__file__).with_name("logs") / "predict_log.jsonl"


def _log_prediction(**record) -> None:
    """Append one compact JSON line per prediction to logs/predict_log.jsonl.

    Local-only: useful for local/backtest runs, but the deployed Modal
    container's filesystem is ephemeral -- each predict_and_submit call gets
    its own fresh container, so this does not persist across live webhook
    calls without a Modal Volume attached.
    """
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        PREDICT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICT_LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[WARN] could not write {PREDICT_LOG_PATH.name}: {e}")


def _ensembled_percentile(
    *, summary: dict, ticker: str, event_type: str, as_of: str | None
) -> float:
    """Average `N_ENSEMBLE_DRAWS` concurrent `_ask_llm` draws into one percentile.

    Each draw is an independent DeepSeek call at temperature=1 -- averaging
    smooths out per-call sampling noise, the same fix already applied to the
    ace/train.py validation gate, now applied to what actually gets submitted.
    """
    def _one_draw(_):
        return _ask_llm(summary=summary, ticker=ticker, event_type=event_type, as_of=as_of).predicted_percentile

    with ThreadPoolExecutor(max_workers=N_ENSEMBLE_DRAWS) as pool:
        draws = list(pool.map(_one_draw, range(N_ENSEMBLE_DRAWS)))
    averaged = sum(draws) / len(draws)
    _log_prediction(ticker=ticker, event_type=event_type, draws=draws, predicted_percentile=averaged)
    return averaged


# ----------------------------------------------------------------------
# Default strategy: a single calibrated LLM call per asset, using a forced
# tool call to get a structured, schema-validated response.
# Swap this out, or rewrite `predict` entirely, to enter your own model.
# ----------------------------------------------------------------------


class Prediction(BaseModel):
    """Structured response shape for the LLM call.

    The `Field(ge=0, le=1)` constraint is enforced by us via Pydantic after
    parsing the tool call arguments — DeepSeek's tool-calling doesn't guarantee
    numeric bounds the way a JSON Schema `minimum`/`maximum` might suggest,
    so we validate on our side rather than trust it blindly.

    `rationale` is never submitted to the competition -- it exists so the ACE
    training loop (ace/reflector.py) can diagnose *why* a prediction missed,
    not just by how much. The live path ignores it entirely.
    """

    predicted_percentile: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


SYSTEM_PROMPT = """\
You are a senior equity analyst predicting how a stock will react to an event.

Predict a single percentile in [0, 1] for how the focal asset's next-day
abnormal return will rank across all of the quarter's event outcomes:
0 = the quarter's most negative reaction, 0.50 = median, 1 = its most positive.
The relevant return is the *unexpected*, market-adjusted return — a
great-but-fully-priced-in beat is not a top-decile event.

Calibration discipline:
- Long-run base rates: about 25% of events land "up" (>0.75), 50% "neutral"
  (0.25-0.75), 25% "down" (<0.25). Default toward 0.40-0.60 when signals are
  mixed or modest.
- Reserve values above 0.80 or below 0.20 for cases with unambiguous,
  multi-signal evidence. Do not exceed 0.90 or fall below 0.10 without
  overwhelming, lopsided evidence.
- Tone alone (confident vs hedging language) should move you no more than
  ~0.10 absent quantitative confirmation.

You must respond by calling the submit_prediction tool — do not answer in plain text.
"""

# Tool definition used to force DeepSeek to return a structured percentile via
# tool_calls, instead of relying on prose or loose JSON-mode output.
_PREDICTION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_prediction",
        "description": "Submit the calibrated percentile prediction for this event.",
        "parameters": {
            "type": "object",
            "required": ["predicted_percentile"],
            "properties": {
                "predicted_percentile": {
                    "type": "number",
                    "description": (
                        "Cross-sectional percentile in [0, 1] for the asset's "
                        "next-day abnormal return this quarter. 0 = most "
                        "negative reaction of the quarter, 0.5 = median, "
                        "1 = most positive."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "2-3 sentence justification citing the specific facts that "
                        "drove the percentile. Not submitted -- used only to "
                        "diagnose misses during offline rulebook training."
                    ),
                },
            },
        },
    },
}


def _ask_llm(
    *,
    summary: dict,
    ticker: str,
    event_type: str,
    as_of: str | None = None,
    peer_context: str | None = None,
    surprise_signal: str | None = None,
    rulebook_text: str | None = None,
) -> Prediction:
    """Ask the configured DeepSeek model for a calibrated percentile via a forced tool call.

    `as_of` is the event's `knowledge_cutoff` (ISO-8601 string) — it bounds the
    yfinance price window so historical backtests never see price data from
    after what would have been known at prediction time. Live calls pass the
    real `knowledge_cutoff` too; omitting it just defaults to "now", which is
    equivalent since the event hasn't happened yet.

    `peer_context` is optional: the mean next-day abnormal return of same-sector
    events that reported strictly earlier in the same quarter, as a formatted
    string (or a "no peers yet" message). Only the notebook backtest currently
    computes and passes this — the live `predict()` path doesn't have cheap
    access to the rest of the quarter's sibling events, so it defaults to None.

    `surprise_signal` is optional and EXPERIMENTAL / BACKTEST-ONLY: a formatted
    quarter-to-date percentile rank of this event's EPS/revenue surprise vs.
    analyst consensus. The archive has this pre-computed (`metrics.earnings_
    surprise.surprise`) for backtesting, but the live webhook event does NOT
    include it -- there's no consensus-estimate source wired up yet (FMP's
    analyst-estimates endpoint is paywalled on the free tier). So this stays
    None in the live `predict()` path until that's solved; it exists purely so
    the notebook backtest can measure whether this signal is worth the effort
    of finding a live consensus source at all -- and so `ace/train.py` can feed
    it to the Reflector, which can only turn it into a *qualitative* rulebook
    heuristic (no raw numeric rank reaches the live model).

    `rulebook_text` overrides the rulebook block normally loaded from
    `rulebook.json` -- used by `ace/train.py` to test a candidate rulebook
    before it's written to disk. Live `predict()` never passes this, so a
    deploy always runs whatever is currently checked in (no online adaptation;
    the 5-minute webhook deadline doesn't leave room for a live Reflector pass).

    Returns the model's `Prediction` (percentile + optional rationale). Falls
    back to a 0.5 placeholder if no `DEEPSEEK_API_KEY` is configured, the model
    doesn't return the tool call, or the arguments fail validation.
    """
    global _deepseek, _deepseek_warned
    if not os.environ.get("DEEPSEEK_API_KEY"):
        if not _deepseek_warned:
            print(
                "[WARN] DEEPSEEK_API_KEY not set — submitting 0.5 placeholder. "
                "Set the key (or edit predict.py) for real predictions."
            )
            _deepseek_warned = True
        return Prediction(predicted_percentile=0.5)
    if _deepseek is None:
        _deepseek = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=DEEPSEEK_BASE_URL,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )

    summary_text = summary.get("summary") if isinstance(summary, dict) else None
    if not summary_text:
        summary_text = json.dumps(summary)
    summary_text = summary_text[:8000]


    rulebook_section = _rulebook_block(rulebook_text)

    # Point-in-time-safe SEC filing signal: `as_of` bounds the list to filings
    # already public at the knowledge cutoff, so backtests never leak future
    # filings into the prompt.
    filings = get_recent_filings_summary(ticker, as_of=as_of)
    filings_text = (
        f"Recent SEC filings (as known at {as_of or 'now'}): {filings}"
        if filings
        else "No recent SEC filing information available."
    )

    # Insider (Form 4) activity in the 90 days before the event -- presence-only
    # signal, separate from the general recent-filings line above since Form 4s
    # otherwise flood out the more material 8-K/10-Q/10-K entries at limit=3.
    insider_summary = get_insider_activity_summary(ticker, as_of=as_of)
    insider_text = insider_summary or "Insider Form 4 activity unavailable."

    user_prompt = (
        f"Event type: {event_type}\n"
        f"Ticker: {ticker}\n\n"
        f"Event summary:\n{summary_text}\n\n"
        # f"SEC filings:\n{filings_text}\n\n"
        f"Insider activity (Form 4, 90 days pre-event):\n{insider_text}\n\n"
        "Weigh, in roughly this order:\n"
        "  1. Quantitative surprise vs expectations — revenue, EPS, segment metrics.\n"
        "  2. Guidance / outlook — raises, holds, cuts vs the prior trajectory.\n"
        "  3. Strategic shifts — product launches, M&A, capital allocation, leadership.\n"
        "  4. Tone and confidence in management commentary (small weight).\n"
        "  5. Risks called out — regulatory, supply chain, demand, competition.\n\n"
        #"If a same-sector reference is given, use it only as a mild anchor — sector-wide "
        #"drift this quarter, not a substitute for this event's own facts.\n\n"
        #"If a quantitative surprise rank is given, treat it as a strong, reliable signal "
        #"(backtesting shows it's one of the best single predictors of next-day reaction) — "
        #"weigh it more heavily than tone or qualitative framing, though it can still be "
        #"outweighed by a clear guidance change in the opposite direction.\n\n"
        f"Predict the next-day unexpected-return percentile for {ticker}. "
        "Call the submit_prediction tool with your answer."
    )

    system_prompt = SYSTEM_PROMPT + (f"\n{rulebook_section}" if rulebook_section else "")

    global _prompt_log_count
    with _prompt_log_lock:
        _prompt_log_count += 1
        call_number = _prompt_log_count
    if (call_number - 1) % PROMPT_LOG_EVERY == 0:
        print(f"\n{'='*20} predict_mc_02.py -- SYSTEM PROMPT (call #{call_number}) {'='*20}\n{system_prompt}")
        print(f"{'='*20} predict_mc_02.py -- USER PROMPT (call #{call_number}, {ticker}) {'='*20}\n{user_prompt}\n")

    try:
        resp = _deepseek.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.2,
            messages=[
                 {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[_PREDICTION_TOOL],
            # Must stay "auto": deepseek-v4-flash serves this call in "thinking mode"
            # by default, and its API rejects a forced tool_choice outright (400:
            # "Thinking mode does not support this tool_choice") -- confirmed by
            # actually hitting the error while testing the ace/ training loop. Not
            # an oversight carried over from predict_kimi.py; auto is required here.
            tool_choice="auto",
        )
    except Exception as e:
        # Network/timeout/API errors (e.g. openai.APITimeoutError) used to
        # propagate uncaught -- fine for a single live prediction (modal_app.py
        # catches it at the top level), but fatal for ace/train.py: one flaky
        # call among hundreds of concurrent Generator calls would crash the
        # entire training run, losing all progress since the last checkpoint.
        print(f"[WARN] DeepSeek call failed for {ticker}: {e} -- submitting 0.5 placeholder.")
        return Prediction(predicted_percentile=0.5)

    message = resp.choices[0].message
    tool_calls = message.tool_calls
    if not tool_calls:
        return Prediction(predicted_percentile=0.5)  # model didn't call the tool

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
        return Prediction.model_validate(arguments)
    except Exception:
        return Prediction(predicted_percentile=0.5)  # malformed/refused tool arguments
