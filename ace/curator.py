"""Curator: deterministic (non-LLM) merge of proposed bullets into the rulebook.

No embedding dependency -- dedup is category + token-overlap based, which is
enough at the scale this runs at (dozens of bullets, not thousands of
agent-memory entries). The harder question -- "did this epoch's bullets
actually help?" -- is NOT decided here: it's an epoch-level accept/reject gate
in ace/train.py against the real `delta_r_squared_imputed` metric, not a
per-bullet helpful/harmful counter guessed by an LLM. This module only handles
structural bookkeeping: dedup, per-category caps, and the total size cap that
protects the live 120s-per-call timeout budget from unbounded prompt growth.
"""

from __future__ import annotations

from ace.store import Bullet

MAX_TOTAL_BULLETS = 25
MAX_PER_CATEGORY = 3
DUPLICATE_OVERLAP_THRESHOLD = 0.6  # Jaccard similarity on normalized tokens


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def _is_duplicate(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / len(ta | tb)
    return overlap >= DUPLICATE_OVERLAP_THRESHOLD


def merge(
    existing: list[Bullet],
    proposed: list[Bullet],
    *,
    max_total: int = MAX_TOTAL_BULLETS,
    max_per_category: int = MAX_PER_CATEGORY,
) -> list[Bullet]:
    """Merge `proposed` bullets into `existing`, returning the new full list.

    A proposed bullet is dropped if: it's a near-duplicate of one already kept
    (by category + token overlap), or its category is already at
    `max_per_category`. If the result would exceed `max_total`, the oldest
    bullets (lowest `created_epoch`) are evicted first -- a simple, documented
    simplification rather than a quality-ranked eviction, since quality is
    already gated at the epoch level before bullets reach this function.
    """
    result = list(existing)

    for bullet in proposed:
        # Dedup check runs against ALL kept bullets, not just same-category --
        # the Reflector invents free-text category names every call, so two
        # near-identical bullets almost never land in the same category
        # string (confirmed: two overlapping bullets in the live rulebook
        # were filed under completely different category names and never
        # got compared). The per-category cap below is still scoped to
        # category on purpose -- that's a growth-rate limiter per topic
        # label, not a duplicate check.
        if any(_is_duplicate(b.text, bullet.text) for b in result):
            continue
        same_category = [b for b in result if b.category == bullet.category]
        if len(same_category) >= max_per_category:
            continue
        result.append(bullet)

    if len(result) > max_total:
        result = sorted(result, key=lambda b: b.created_epoch, reverse=True)[:max_total]

    return result
