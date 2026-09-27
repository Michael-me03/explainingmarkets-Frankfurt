"""The rulebook: a small, checked-in JSON file of learned heuristic bullets.

Deliberately dependency-free (stdlib only) -- `predict.py` imports this module
directly into the live deploy, and the live path must not need pandas/openai/
the `examples` workspace just to render a text block into the system prompt.
Training (`ace/train.py`, `ace/reflector.py`, `ace/curator.py`) is where the
heavier dependencies live.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Bullet:
    """One heuristic in the rulebook.

    `category` is a short tag (e.g. "guidance", "sector:telecom", "surprise")
    used by the curator to find related bullets to merge against or cap --
    not shown to the model, just internal bookkeeping.
    """

    id: str
    category: str
    text: str
    created_epoch: int


def load_rulebook(path: Path) -> list[Bullet]:
    """Load bullets from `path`. Missing or empty file -> no bullets (not an error)."""
    if not path.exists():
        return []
    raw = path.read_text().strip()
    if not raw:
        return []
    data = json.loads(raw)
    return [Bullet(**b) for b in data.get("bullets", [])]


def save_rulebook(path: Path, bullets: list[Bullet]) -> None:
    """Write bullets to `path`, atomically (temp file + replace)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"bullets": [asdict(b) for b in bullets]}, indent=2))
    tmp.replace(path)


def render(bullets: list[Bullet]) -> str:
    """Render bullets into the text block `predict.py` appends to SYSTEM_PROMPT.

    Empty input renders to `""` so callers can skip appending anything.
    """
    if not bullets:
        return ""
    # width=80, subsequent_indent="  " -- matches predict_02.py's hand-wrapped
    # static rulebook text exactly, so the two prompts differ only in content.
    lines = "\n".join(
        textwrap.fill(f"- {b.category}: {b.text}", width=80, subsequent_indent="  ")
        for b in bullets
    )
    return f"Rulebook (learned heuristics from past events):\n{lines}\n"
