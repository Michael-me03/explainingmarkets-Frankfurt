"""Finnhub earnings surprise -- EPS actual vs. consensus estimate, per quarter.

Free-tier alternative to FMP's `/stable/earnings` (see data_provider/fmp.py's
`get_earnings_surprise_summary`), which turned out to only serve a fixed
whitelist of ~15 blue-chip tickers on the free plan -- confirmed live: BLK,
DPZ, FBK, and LEVI all returned "402 Payment Required" there. Finnhub's
`/stock/earnings` endpoint has no such restriction: verified live against
those same four tickers below, all returned real data (see `__main__`).

No revenue surprise here -- this endpoint only reports EPS actual/estimate.
That happens to match the earnings-surprise definition the Explaining Markets
benchmark itself uses (Koijen & Levy (2026), Sec. 2.3.2): EPS surprise scaled
by price, not a blended EPS+revenue score.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.getenv("FINNHUB_API_KEY")
BASE_URL = "https://finnhub.io/api/v1"

# Free tier is 60 calls/minute -- a little slack under the exact limit, same
# throttle-lock pattern as fmp.py/sec_edgar.py since ace/train.py hits this
# from several concurrent ThreadPoolExecutor workers.
_MIN_REQUEST_INTERVAL_SECONDS = 1.1
_rate_limit_lock = threading.Lock()
_last_request_at = 0.0


def _throttled_get(params: dict) -> list:
    global _last_request_at
    with _rate_limit_lock:
        wait = _MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        resp = requests.get(f"{BASE_URL}/stock/earnings", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _as_of_date(value) -> str | None:
    """Reduce an `as_of` (datetime or ISO-8601) to a 'YYYY-MM-DD' cutoff string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def get_earnings_surprise_summary(ticker: str, *, as_of=None) -> str | None:
    """EPS actual vs. consensus estimate for the ticker's most recent reported quarter.

    `as_of` bounds the result to quarters whose fiscal period end (`period`)
    is on or before that day.

    CAVEAT -- unlike `sec_edgar.get_insider_activity_summary` (real SEC filing
    dates) or `fmp.get_earnings_surprise_summary` (a real report `date`),
    Finnhub's `/stock/earnings` only gives the fiscal period *end*, not the
    actual report/filing date. Filtering on `period <= as_of` rules out
    picking a future fiscal quarter, but can't catch a quarter that ended
    before `as_of` yet wasn't actually reported until after it (reports
    typically lag period-end by 3-8 weeks). This is safe for live `predict()`
    (`as_of` is effectively "now", called right after the report happened,
    so the latest row is always the one that just printed) but is NOT
    point-in-time-guaranteed for backtests with a historical `as_of` --
    don't wire this into `ace/train.py` without a real report-date check
    first (e.g. cross-referencing an earnings calendar endpoint).

    Returns `None` (never raises) if no `FINNHUB_API_KEY` is set, the ticker
    has no data on or before `as_of`, or the request fails for any reason.
    """
    if not API_KEY:
        return None
    try:
        cutoff = _as_of_date(as_of)
        rows = _throttled_get({"symbol": ticker, "token": API_KEY})
        candidates = [
            r for r in rows
            if r.get("actual") is not None
            and r.get("estimate") is not None
            and (cutoff is None or r.get("period", "") <= cutoff)
        ]
        if not candidates:
            return None
        row = max(candidates, key=lambda r: r["period"])

        actual, estimate = row["actual"], row["estimate"]
        pct = row.get("surprisePercent")
        if pct is None:
            pct = (actual - estimate) / abs(estimate) * 100 if estimate else None
        if pct is None:
            return None
        return f"Quarter ended {row['period']} -- EPS {actual:.2f} actual vs {estimate:.2f} est ({pct:+.1f}% surprise)."
    except Exception as e:
        print(f"[Finnhub] earnings surprise unavailable for {ticker}: {e}")
        return None


if __name__ == "__main__":
    # The exact four tickers FMP's free plan rejected with 402 -- confirms
    # Finnhub's free tier has no equivalent whitelist restriction.
    for t in ["AAPL", "BLK", "DPZ", "FBK", "LEVI"]:
        print(t, "->", get_earnings_surprise_summary(t, as_of="2026-08-15"))
