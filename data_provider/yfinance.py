"""Historical price context via yfinance — closing prices, trend, volatility.

Unlike FMP's `/quote` (a real-time snapshot), this pulls a window of daily
closes ending at a given cutoff. That makes it safe for backtesting: pass the
event's `knowledge_cutoff` as `as_of` and the window never includes data from
after what would actually have been known at prediction time. For live use,
omit `as_of` (defaults to now) — the event hasn't happened yet, so "now" is
naturally the right cutoff.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yfinance as yf

# `.info` calls (get_sector, get_market_cap_bucket) rate-limit hard under
# concurrency -- ace/train.py runs up to MAX_WORKERS Generator calls at once,
# each potentially hitting this. A lock + minimum interval serializes `.info`
# access across all threads; `.history()` (get_market_trend) and
# `.analyst_price_targets` (get_analyst_expectations) haven't shown the same
# issue empirically, so they stay unthrottled.
_MIN_INFO_INTERVAL_SECONDS = 0.35
_info_rate_limit_lock = threading.Lock()
_last_info_request_at = 0.0


def _get_info(ticker: str) -> dict:
    global _last_info_request_at
    with _info_rate_limit_lock:
        wait = _MIN_INFO_INTERVAL_SECONDS - (time.monotonic() - _last_info_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_info_request_at = time.monotonic()
        return yf.Ticker(ticker).info


_SECTOR_CACHE_PATH = Path(__file__).resolve().parent / "sector_cache.json"


def _load_sector_cache() -> dict:
    if not _SECTOR_CACHE_PATH.exists() or _SECTOR_CACHE_PATH.stat().st_size == 0:
        return {}
    try:
        return json.loads(_SECTOR_CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _save_sector_cache(cache: dict) -> None:
    tmp = _SECTOR_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    os.replace(tmp, _SECTOR_CACHE_PATH)


_sector_cache = _load_sector_cache()


def get_sector(ticker: str) -> str | None:
    """GICS-ish sector classification via yfinance, disk-cached.

    Sectors don't change day to day, so once a ticker is resolved it's cached
    to `data_provider/sector_cache.json` permanently -- a notebook restart or
    a widened sample never re-fetches it.

    Safe to call from a `ThreadPoolExecutor` now -- `_get_info` throttles all
    `.info` access globally to one request per `_MIN_INFO_INTERVAL_SECONDS`,
    regardless of how many threads call in. (Previously required calling this
    sequentially; concurrent `.info` calls rate-limited hard and threw
    spurious 404s even for valid, actively-traded tickers.)
    """
    if ticker in _sector_cache:
        return _sector_cache[ticker]
    try:
        sector = _get_info(ticker).get("sector")
    except Exception as e:
        print(f"[yfinance] sector unavailable for {ticker}: {e}")
        sector = None
    _sector_cache[ticker] = sector
    _save_sector_cache(_sector_cache)
    return sector


def _parse_as_of(value) -> datetime:
    if value is None:
        return datetime.utcnow()
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    # ISO-8601 string, e.g. "2025-10-08T20:00:00+00:00" or with a trailing "Z".
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def get_market_trend(ticker: str, *, as_of=None, lookback_days: int = 90) -> dict | None:
    """Last `lookback_days` daily closes for `ticker`, ending strictly before `as_of`.

    Returns ``None`` if no price data is available (bad ticker, delisted,
    provider hiccup). On success:

      - ``closes``: list of ``{"date": "YYYY-MM-DD", "close": float}``,
        oldest first
      - ``realized_volatility_annualized``: stdev of daily log returns,
        annualized (``* sqrt(252)``) — ``None`` if fewer than 2 return points
      - ``trend_pct``: total % change from the window's first to last close —
        ``None`` if fewer than 2 closes
    """
    end = _parse_as_of(as_of)
    start = end - timedelta(days=int(lookback_days * 1.6) + 10)  # buffer for weekends/holidays

    try:
        hist = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
        )
    except Exception as e:
        print(f"[yfinance] error fetching history for {ticker}: {e}")
        return None

    if hist is None or hist.empty:
        return None

    hist = hist.tail(lookback_days)
    closes = hist["Close"]

    if len(closes) < 2:
        return {
            "closes": [{"date": idx.strftime("%Y-%m-%d"), "close": float(v)} for idx, v in closes.items()],
            "realized_volatility_annualized": None,
            "trend_pct": None,
        }

    log_returns = np.log(closes / closes.shift(1)).dropna()
    realized_vol = float(log_returns.std() * (252**0.5)) if len(log_returns) > 1 else None
    trend_pct = float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)

    return {
        "closes": [{"date": idx.strftime("%Y-%m-%d"), "close": float(v)} for idx, v in closes.items()],
        "realized_volatility_annualized": realized_vol,
        "trend_pct": trend_pct,
    }


_MARKET_CAP_CACHE_PATH = Path(__file__).resolve().parent / "market_cap_cache.json"


def _load_market_cap_cache() -> dict:
    if not _MARKET_CAP_CACHE_PATH.exists() or _MARKET_CAP_CACHE_PATH.stat().st_size == 0:
        return {}
    try:
        return json.loads(_MARKET_CAP_CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _save_market_cap_cache(cache: dict) -> None:
    tmp = _MARKET_CAP_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    os.replace(tmp, _MARKET_CAP_CACHE_PATH)


_market_cap_cache = _load_market_cap_cache()


def get_market_cap_bucket(ticker: str) -> dict | None:
    """Market cap and a coarse small/mid/large-cap bucket, via yfinance.

    Live-only, like `get_analyst_expectations` -- yfinance has no point-in-time
    market cap API, so this always reflects *today's* value. Three plain
    thresholds, not a scored model: < $2B small, $2B-$10B mid, > $10B large.

    Disk-cached, same rationale as `get_sector`: a bucket rarely flips day to
    day, and caching also means a repeated ticker (across events, or across
    ace/train.py epochs) doesn't re-hit `.info` at all -- on top of the
    request-level throttle in `_get_info`.
    """
    if ticker in _market_cap_cache:
        return _market_cap_cache[ticker]

    try:
        market_cap = _get_info(ticker).get("marketCap")
    except Exception as e:
        print(f"[yfinance] market cap unavailable for {ticker}: {e}")
        market_cap = None

    if not market_cap:
        _market_cap_cache[ticker] = None
        _save_market_cap_cache(_market_cap_cache)
        return None

    if market_cap < 2_000_000_000:
        bucket = "small-cap"
    elif market_cap < 10_000_000_000:
        bucket = "mid-cap"
    else:
        bucket = "large-cap"

    result = {"market_cap": market_cap, "bucket": bucket}
    _market_cap_cache[ticker] = result
    _save_market_cap_cache(_market_cap_cache)
    return result


def get_analyst_expectations(ticker: str, *, as_of=None) -> dict | None:
    """Current analyst price targets — LIVE ONLY, intentionally.

    yfinance's `analyst_price_targets` has no historical / point-in-time API —
    it always reflects analysts' targets as of *today*, not what they looked
    like around a past event. There is no safe way to backdate it. So:

    - `as_of` is None (live use, e.g. the deployed webhook handler) -> fetch
      and return the current targets.
    - `as_of` is set (backtesting on the archive) -> always return None,
      on purpose, to avoid feeding a 2026 analyst target into a 2025 event
      as if it were known at the time (look-ahead bias).
    """
    if as_of is not None:
        return None  # no historical data available; refuse rather than leak

    try:
        targets = yf.Ticker(ticker).analyst_price_targets
    except Exception as e:
        print(f"[yfinance] error fetching analyst targets for {ticker}: {e}")
        return None

    if not targets:
        return None

    return {
        "target_mean": targets.get("mean"),
        "target_median": targets.get("median"),
        "target_low": targets.get("low"),
        "target_high": targets.get("high"),
        "current_price": targets.get("current"),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(get_market_trend("AAPL"), indent=2)[:1000])
    print(json.dumps(get_analyst_expectations("AAPL"), indent=2))
