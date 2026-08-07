"""Reflector: diagnoses miss patterns and proposes a rulebook bullet.

Takes a batch of the training epoch's worst misses (predicted vs. actual
percentile, plus the Generator's own rationale) and asks the model to propose
ONE heuristic bullet per epoch -- one hypothesis at a time, not a batch of 3,
so the epoch-level accept/reject gate in ace/train.py can attribute credit/
blame to a single change instead of conflating three at once. The model is
allowed to propose from a single strong signal, not just cross-event patterns
-- more exploratory, at the cost of a higher chance any given proposal doesn't
generalize (which is exactly what the accept/reject gate exists to catch).
Structured output via a forced tool call, same pattern `predict.py` uses for
`submit_prediction`.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, Field

from ace.store import Bullet
from predict import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, SYSTEM_PROMPT as GENERATOR_SYSTEM_PROMPT

REFLECTOR_TIMEOUT_SECONDS = 120.0
MAX_BULLETS_PER_BATCH = 1

_client: OpenAI | None = None


@dataclass
class MissRecord:
    """One training-epoch prediction, joined against its realized outcome."""

    event_id: str
    ticker: str
    event_type: str
    summary_text: str
    predicted_percentile: float
    rationale: str | None
    actual_percentile: float  # `y` from examples.scoring.add_percentiles

    @property
    def abs_error(self) -> float:
        return abs(self.predicted_percentile - self.actual_percentile)


class BulletProposal(BaseModel):
    category: str = Field(description='Short tag, e.g. "guidance", "sector:telecom", "surprise".')
    text: str = Field(description="One heuristic sentence, general enough to apply beyond this batch.")


class ReflectorOutput(BaseModel):
    bullets: list[BulletProposal] = Field(
        default_factory=list,
        description="At most 1 bullet -- your single best hypothesis this batch. Empty if truly nothing stands out.",
    )


_REFLECT_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_bullets",
        "description": "Propose rulebook heuristics distilled from recurring prediction misses.",
        "parameters": {
            "type": "object",
            "required": ["bullets"],
            "properties": {
                "bullets": {
                    "type": "array",
                    "maxItems": MAX_BULLETS_PER_BATCH,
                    "items": {
                        "type": "object",
                        "required": ["category", "text"],
                        "properties": {
                            "category": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

REFLECTOR_SYSTEM_PROMPT = """\
You are diagnosing a batch of an equity-prediction model's worst misses: cases
where its predicted percentile (0=quarter's most negative reaction, 1=most
positive) was far from the realized outcome.

Your job: propose your SINGLE best hypothesis for what's causing these misses
-- one bullet, not several. A pattern that recurs across multiple events below
is stronger evidence, but you don't need to wait for that: if one event shows
an especially clear, generalizable signal, propose it. Experiment -- a
hypothesis that turns out not to generalize will be caught and discarded by
validation scoring downstream, so err toward proposing your best guess rather
than staying silent. Propose nothing only if you genuinely see no signal worth
testing.

You will be shown the base instructions the predicting model already follows,
and the rulebook bullets already in effect on top of them. Do not repeat or
rephrase anything already covered by either -- if the base instructions or an
existing bullet already handle a pattern you see, propose nothing for it. Only
propose a bullet for a genuinely new pattern not already covered, and don't
propose anything that contradicts the base instructions' calibration
discipline (e.g. its base-rate/extremes guidance).

Write one general heuristic (not a restatement of one event's facts) that
would help a future prediction avoid the same class of error. Tag it with a
short category.

Call the propose_bullets tool with your answer (0 or 1 bullet).
"""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=DEEPSEEK_BASE_URL,
            timeout=REFLECTOR_TIMEOUT_SECONDS,
            max_retries=1,
        )
    return _client


def _format_miss(m: MissRecord) -> str:
    return (
        f"- event={m.event_id} ticker={m.ticker} type={m.event_type}\n"
        f"  predicted={m.predicted_percentile:.2f} actual={m.actual_percentile:.2f} "
        f"error={m.abs_error:.2f}\n"
        f"  rationale given: {m.rationale or '(none)'}\n"
        f"  summary: {m.summary_text[:600]}"
    )


def reflect(
    misses: list[MissRecord],
    epoch: int,
    existing_bullets: list[Bullet] | None = None,
) -> list[Bullet]:
    """Diagnose `misses` and return newly proposed bullets (may be empty).

    `misses` should already be filtered to the epoch's worst errors -- this
    function doesn't do that selection itself, callers (ace/train.py) do, so
    the batch size / selection strategy stays visible and tunable there.

    `existing_bullets` is the rulebook's current state -- shown to the model so
    it doesn't repropose something already covered (or contradict it). Without
    this the Reflector has no way to know what's already in the rulebook it's
    supposedly extending.
    """
    if not misses:
        return []

    if existing_bullets:
        existing_text = "\n".join(f"- [{b.category}] {b.text}" for b in existing_bullets)
    else:
        existing_text = "(none yet -- rulebook is currently empty)"

    batch_text = "\n\n".join(_format_miss(m) for m in misses)
    user_prompt = (
        f"Base instructions the predicting model already follows:\n"
        f"{GENERATOR_SYSTEM_PROMPT}\n\n"
        f"Rulebook bullets already in effect on top of those instructions:\n{existing_text}\n\n"
        f"Misses this batch ({len(misses)} events):\n\n{batch_text}"
    )

    resp = _get_client().chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=1,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=[_REFLECT_TOOL],
        # "auto", not forced -- deepseek-v4-flash's thinking mode rejects a
        # forced tool_choice (400: "Thinking mode does not support this
        # tool_choice"), same constraint as predict.py's _ask_llm.
        tool_choice="auto",
    )

    tool_calls = resp.choices[0].message.tool_calls
    if not tool_calls:
        return []

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
        parsed = ReflectorOutput.model_validate(arguments)
    except Exception:
        return []

    return [
        Bullet(
            id=uuid.uuid4().hex[:12],
            category=b.category.strip(),
            text=b.text.strip(),
            created_epoch=epoch,
        )
        for b in parsed.bullets[:MAX_BULLETS_PER_BATCH]
        if b.text.strip()
    ]
