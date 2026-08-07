"""SEC EDGAR filing history -- minimal signal: recent filing types/dates.

No API key -- EDGAR is free, but requires a descriptive User-Agent identifying
the requester (not optional; unidentified requests get blocked harder than
identified ones). Same throttle + disk-cache pattern already used for FMP
(data_provider/fmp.py) and yfinance (data_provider/yfinance.py), since this
runs under the same concurrent ace/train.py Generator calls.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import requests

USER_AGENT = "starter-modal research contact:michi20031029@gmail.com"
_HEADERS = {"User-Agent": USER_AGENT}

_MIN_REQUEST_INTERVAL_SECONDS = 0.3
_rate_limit_lock = threading.Lock()
_last_request_at = 0.0


def _throttled_get(url: str) -> dict:
    global _last_request_at
    with _rate_limit_lock:
        wait = _MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        resp = requests.get(url, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


_TICKER_MAP_CACHE_PATH = Path(__file__).resolve().parent / "sec_ticker_cik_cache.json"
_ticker_cik_map: dict | None = None
_map_lock = threading.Lock()


def _load_ticker_cik_map() -> dict:
    """ticker -> zero-padded CIK, disk-cached -- EDGAR's own mapping rarely changes."""
    if _TICKER_MAP_CACHE_PATH.exists() and _TICKER_MAP_CACHE_PATH.stat().st_size > 0:
        try:
            return json.loads(_TICKER_MAP_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            pass

    data = _throttled_get("https://www.sec.gov/files/company_tickers.json")
    mapping = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}

    tmp = _TICKER_MAP_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(mapping))
    os.replace(tmp, _TICKER_MAP_CACHE_PATH)
    return mapping


def _get_cik(ticker: str) -> str | None:
    global _ticker_cik_map
    with _map_lock:
        if _ticker_cik_map is None:
            _ticker_cik_map = _load_ticker_cik_map()
    return _ticker_cik_map.get(ticker.upper())


def get_recent_filings_summary(ticker: str, limit: int = 3) -> str | None:
    """Short summary of the ticker's most recent SEC filings: form type + date only.

    Deliberately minimal -- no filing content, just what was filed and when
    (e.g. "8-K (2026-07-30); 10-Q (2026-07-15)"). Not cached: filing lists
    change daily, unlike the ticker->CIK map above.
    """
    try:
        cik = _get_cik(ticker)
        if not cik:
            return None
        data = _throttled_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        recent = data["filings"]["recent"]
        forms = recent["form"][:limit]
        dates = recent["filingDate"][:limit]
        return "; ".join(f"{f} ({d})" for f, d in zip(forms, dates))
    except Exception as e:
        print(f"[SEC EDGAR] filings unavailable for {ticker}: {e}")
        return None
