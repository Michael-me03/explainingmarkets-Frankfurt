"""predict_first.py — bare-minimum DeepSeek baseline for the validation notebook.

No rulebook, no insider/SEC signals, no ensembling — just the base system prompt,
the event summary, and a single calibrated DeepSeek call per asset. Used as the
"null" reference in the 3-way A/B against predict.py (full stack) and
predict_02_submitted.py (static-rulebook, the submitted version).
"""

from __future__ import annotations

import json
import os

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field


_deepseek: OpenAI | None = None
_deepseek_warned = False

SUMMARY_TIMEOUT_SECONDS = 15.0
LLM_TIMEOUT_SECONDS = 120.0
LLM_MAX_RETRIES = 1

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"


def predict(event: dict) -> list[dict]:
    summary = httpx.get(event["information_url"], timeout=SUMMARY_TIMEOUT_SECONDS)
    summary.raise_for_status()
    summary_json = summary.json()

    return [
        {
            "identifier_value": asset["identifier_value"],
            "predicted_percentile": _ask_llm(
                summary=summary_json,
                ticker=asset["identifier_value"],
                event_type=event["event_type"],
            ),
        }
        for asset in event["focal_assets"]
    ]


class Prediction(BaseModel):
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
"""

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
) -> float:
    global _deepseek, _deepseek_warned
    if not os.environ.get("DEEPSEEK_API_KEY"):
        if not _deepseek_warned:
            print(
                "[WARN] DEEPSEEK_API_KEY not set — submitting 0.5 placeholder. "
                "Set the key for real predictions."
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
        f"Predict the next-day unexpected-return percentile for {ticker}. "
        "Call the submit_prediction tool with your answer."
    )

    try:
        resp = _deepseek.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=[_PREDICTION_TOOL],
            tool_choice="auto",
        )
    except Exception as e:
        print(f"[WARN] DeepSeek call failed for {ticker}: {e} — submitting 0.5 placeholder.")
        return 0.5

    message = resp.choices[0].message
    tool_calls = message.tool_calls
    if not tool_calls:
        return 0.5

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
        parsed = Prediction.model_validate(arguments)
        return parsed.predicted_percentile
    except Exception:
        return 0.5
