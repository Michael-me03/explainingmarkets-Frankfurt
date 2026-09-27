"""predict_02.py — the DeepSeek, static-rulebook variant used by validate.py.

`predict(event)` is called once per competition event, after the webhook has
already been verified for you. Return one prediction per focal asset. Everything
else in this repo (webhook verification, dedupe, submission) is plumbing.

The default implementation asks a DeepSeek model for a calibrated percentile
via a forced tool call. If `DEEPSEEK_API_KEY` is not set, it returns a
0.5 baseline so the full deploy → receive → submit round-trip still works
without burning credits. Replace the body of `predict` with whatever strategy
you like — the only contract is the return shape documented below.

Copy of predict_kimi.py with only the model call swapped to DeepSeek --
static 3-bullet rulebook, single call per asset, no market cap/SEC/ensembling
-- kept otherwise identical so validate.py can isolate the model variable
(Kimi vs DeepSeek) from the rulebook/signals/ensembling variables that
predict.py (Frankfurt-MC-02) also differs on.
"""

from __future__ import annotations

import json
import os
import threading

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field
from data_provider.fmp import get_market_context
from data_provider.yfinance import get_market_trend, get_analyst_expectations
from data_provider.sec_edgar import get_recent_filings_summary


_deepseek: OpenAI | None = None  # lazy: importing this file must not require a key
_deepseek_warned = False         # one-shot warning when no key is configured
_prompt_logged = False           # print the full system/user prompt once, for debugging
_prompt_log_lock = threading.Lock()

# Timeouts, sized against the 5-minute prediction window that opens when your
# handler ACKs the webhook. Worst case is 15 + (120 x 2) + 15 = 270s, which
# fits with ~30s to spare. Nothing upstream retries a failed prediction — once
# the delivery is ACKed the platform considers it done — so the one retry here
# is the only one you get. Raising either value can push you past the deadline.
SUMMARY_TIMEOUT_SECONDS = 15.0
LLM_TIMEOUT_SECONDS = 120.0
LLM_MAX_RETRIES = 1

# DeepSeek is OpenAI-compatible; only the base_url + model differ.
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"
#DEEPSEEK_MODEL = "deepseek-v4-pro"


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

    # One model call per focal asset, in series — so the LLM budget below is
    # per asset, not per event. Today every event carries a single asset; if
    # that changes and you need several, run them concurrently rather than
    # raising the timeout.
    return [
        {
            "identifier_value": asset["identifier_value"],
            "predicted_percentile": _ask_llm(
                summary=summary_json,
                ticker=asset["identifier_value"],
                event_type=event["event_type"],
                as_of=event.get("knowledge_cutoff"),
            ),
        }
        for asset in event["focal_assets"]
    ]


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
    """

    predicted_percentile: float = Field(ge=0.0, le=1.0)


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

Rulebook (learned heuristics from past events):
- High-expectation momentum names: For stocks that have already re-rated sharply
  (high multiple, large run-up into the print, crowded momentum), the reaction
  bar is set by the valuation, not the reported numbers: a big beat that merely
  validates the embedded trajectory is frequently sold off. Treat high-flying
  momentum names as negatively skewed even on genuine beats -- keep at or below
  neutral unless the surprise is unprecedented relative to the run-up, rather
  than merely consistent with the stock's acceleration.
- Priced-in downside / relief rallies: When a stock has already de-rated sharply
  on well-known weakness (sector downturn, prior guidance cuts, depressed
  valuation), a bad print that merely confirms the known deterioration often
  rallies -- the market prices 'less bad than feared' plus forward positives
  such as cost cuts, new growth lines, and capital returns. Do not predict
  bottom-quartile reactions for already-priced negatives unless the print
  carries genuinely new, large negative information; the asymmetry favors
  relief.
- Forward path dominates in small caps: In small/micro-caps and launch-stage
  companies, the market prices the forward trajectory (guidance, margins, cash
  runway) more heavily than the backward reported quarter: a beat that comes
  with in-line or soft guidance, margin compression, or delayed growth is
  routinely sold hard below neutral. For these names, anchor on the forward
  signals and do not let a backward-looking beat carry the prediction above
  neutral.
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
) -> float:
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
    of finding a live consensus source at all.

    Returns the model's `predicted_percentile`. Falls back to 0.5 if no
    `DEEPSEEK_API_KEY` is configured, the model doesn't return the tool call, or
    the arguments fail validation.
    """
    global _deepseek, _deepseek_warned
    if not os.environ.get("DEEPSEEK_API_KEY"):
        if not _deepseek_warned:
            print(
                "[WARN] DEEPSEEK_API_KEY not set — submitting 0.5 placeholder. "
                "Set the key (or edit predict_02.py) for real predictions."
            )
            _deepseek_warned = True
        return 0.5
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

    # Point-in-time-safe SEC filing signal -- same as predict_mc_02.py.
    filings = get_recent_filings_summary(ticker, as_of=as_of)
    filings_text = (
        f"Recent SEC filings (as known at {as_of or 'now'}): {filings}"
        if filings
        else "No recent SEC filing information available."
    )

    #fmp_context = get_market_context(ticker)
    # market_context_text = json.dumps(fmp_context) if fmp_context else "No market context available."

    # --- everything below is commented out for now -- back to summary-only,
    # like the very first version. Re-enable individually to A/B test again. ---
    # trend = get_market_trend(ticker, as_of=as_of)
    # if trend:
    #     print("success YF", ticker)
    #     price_history_text = (
    #         f"90-day realized volatility (annualized): {trend['realized_volatility_annualized']!r}\n"
    #         f"90-day price trend: {trend['trend_pct']!r}%\n"
    #         f"Last 10 closes: {trend['closes'][-10:]}"
    #     )
    # else:
    #     price_history_text = "No price history available."

    #analyst = get_analyst_expectations(ticker, as_of=as_of)
    #analyst_text = (
    #    json.dumps(analyst) if analyst
     #   else "No analyst price targets available (backtests never get this — live-only, current-targets data)."
    #)

    # peer_section = (
    #     f"Same-sector reference (mean next-day abnormal return of same-sector "
    #     f"peers that already reported this quarter): {peer_context}\n\n"
    #     if peer_context else ""
    # )

    # surprise_section = (
    #     f"Quantitative surprise rank (this event's EPS/revenue surprise vs. "
    #     f"analyst consensus, ranked against all events so far this quarter, "
    #     f"0=most negative surprise, 1=most positive): {surprise_signal}\n\n"
    #     if surprise_signal else ""
    # )

    user_prompt = (
        f"Event type: {event_type}\n"
        f"Ticker: {ticker}\n\n"
        f"Event summary:\n{summary_text}\n\n"
        "Weigh, in roughly this order:\n"
        "  1. Quantitative surprise vs expectations — revenue, EPS, segment metrics.\n"
        "  2. Guidance / outlook — raises, holds, cuts vs the prior trajectory.\n"
        "  3. Strategic shifts — product launches, M&A, capital allocation, leadership.\n"
        "  4. Tone and confidence in management commentary (small weight).\n"
        "  5. Risks called out — regulatory, supply chain, demand, competition.\n\n"
        # "If a same-sector reference is given, use it only as a mild anchor — sector-wide "
        # "drift this quarter, not a substitute for this event's own facts.\n\n"
        # "If a quantitative surprise rank is given, treat it as a strong, reliable signal "
        # "(backtesting shows it's one of the best single predictors of next-day reaction) — "
        # "weigh it more heavily than tone or qualitative framing, though it can still be "
        # "outweighed by a clear guidance change in the opposite direction.\n\n"
        f"Predict the next-day unexpected-return percentile for {ticker}. "
        "Call the submit_prediction tool with your answer."
    )

    global _prompt_logged
    if not _prompt_logged:
        with _prompt_log_lock:
            if not _prompt_logged:
                _prompt_logged = True
                print(f"\n{'='*20} predict_02.py -- SYSTEM PROMPT {'='*20}\n{SYSTEM_PROMPT}")
                print(f"{'='*20} predict_02.py -- USER PROMPT ({ticker}) {'='*20}\n{user_prompt}\n")

    try:
        resp = _deepseek.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=[_PREDICTION_TOOL],
            # Must stay "auto": deepseek-v4-flash serves this call in "thinking mode"
            # by default, and its API rejects a forced tool_choice outright (400:
            # "Thinking mode does not support this tool_choice").
            tool_choice="auto",
        )
    except Exception as e:
        # Network/timeout/API errors used to propagate uncaught -- same fix
        # already applied to predict.py's DeepSeek call: one flaky call
        # shouldn't crash an entire validate.py/backtest run.
        print(f"[WARN] DeepSeek call failed for {ticker}: {e} -- submitting 0.5 placeholder.")
        return 0.5

    message = resp.choices[0].message
    tool_calls = message.tool_calls
    if not tool_calls:
        return 0.5  # model didn't call the tool; competition expects a number

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
        parsed = Prediction.model_validate(arguments)
    except Exception:
        return 0.5  # malformed/refused tool arguments; competition expects a number

    return parsed.predicted_percentile
