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
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import requests

USER_AGENT = "starter-modal research contact:michi20031029@gmail.com"
_HEADERS = {"User-Agent": USER_AGENT}

_MIN_REQUEST_INTERVAL_SECONDS = 0.1  # SEC's published limit is 10 req/s -- this was 0.3 (~3.3 req/s), 3x slower than allowed
_rate_limit_lock = threading.Lock()
_last_request_at = 0.0


def _throttled_wait() -> None:
    global _last_request_at
    with _rate_limit_lock:
        wait = _MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _throttled_get(url: str) -> dict:
    _throttled_wait()
    resp = requests.get(url, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _throttled_get_text(url: str) -> str:
    _throttled_wait()
    resp = requests.get(url, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.text


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


def _as_of_date(value) -> str | None:
    """Reduce an `as_of` (datetime or ISO-8601) to a 'YYYY-MM-DD' cutoff string.

    EDGAR only reports filing dates at day granularity, so a day-level cutoff
    is the finest comparison we can make anyway.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def get_recent_filings_summary(ticker: str, limit: int = 3, *, as_of=None) -> str | None:
    """Short summary of the ticker's most recent SEC filings: form type + date only.

    Deliberately minimal -- no filing content, just what was filed and when
    (e.g. "8-K (2026-07-30); 10-Q (2026-07-15)"). Not cached: filing lists
    change daily, unlike the ticker->CIK map above.

    `as_of` (datetime or ISO-8601) optionally bounds the summary to filings
    dated on or before that day -- pass the event's `knowledge_cutoff` in
    backtests so the list never includes a filing that wasn't public yet at
    prediction time (no look-ahead leak). Defaults to now, i.e. live behavior.
    """
    try:
        cik = _get_cik(ticker)
        if not cik:
            return None
        cutoff = _as_of_date(as_of)
        data = _throttled_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        recent = data["filings"]["recent"]
        pairs = []
        for form, date in zip(recent["form"], recent["filingDate"]):
            if cutoff is not None and date > cutoff:
                continue
            pairs.append((form, date))
            if len(pairs) >= limit:
                break
        if not pairs:
            return None
        return "; ".join(f"{f} ({d})" for f, d in pairs)
    except Exception as e:
        print(f"[SEC EDGAR] filings unavailable for {ticker}: {e}")
        return None


# P/S = open-market buy/sell (discretionary). Everything else (grant, option
# exercise, tax withholding, gift, ...) is administrative -- tallied as "other".
_BUY_CODES = {"P"}
_SELL_CODES = {"S"}


def _xtag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _xfind(el: ET.Element, *path: str) -> str | None:
    """Text of the first descendant matching each tag in `path`, in order (namespace-stripped)."""
    for name in path:
        el = next((c for c in el.iter() if _xtag(c) == name), None)
        if el is None:
            return None
    return el.text.strip() if el.text else None


def _form4_transactions(xml_text: str) -> list[dict]:
    """Each buy/sell/other transaction in a Form 4: code, shares, price, filer role.

    Role (officer title / director / 10%-owner) and dollar size (shares x
    price) matter because a CEO's six-figure buy and a director's routine one
    read identically as a bare "1 buy" -- which is why the count-only version
    of this signal tested worse than no signal at all.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    rel = next((e for e in root.iter() if _xtag(e) == "reportingOwnerRelationship"), None)
    role = "insider"
    if rel is not None:
        if _xfind(rel, "isOfficer") in ("1", "true"):
            role = _xfind(rel, "officerTitle") or "officer"
        elif _xfind(rel, "isDirector") in ("1", "true"):
            role = "director"
        elif _xfind(rel, "isTenPercentOwner") in ("1", "true"):
            role = "10%+ owner"

    out = []
    for tx in (e for e in root.iter() if _xtag(e) in ("nonDerivativeTransaction", "derivativeTransaction")):
        code = _xfind(tx, "transactionCode")
        if not code:
            continue
        shares = _xfind(tx, "transactionShares", "value")
        price = _xfind(tx, "transactionPricePerShare", "value")
        out.append({
            "code": code,
            "shares": float(shares) if shares else None,
            "price": float(price) if price else None,
            "role": role,
        })
    return out


def get_insider_activity_summary(
    ticker: str, *, as_of=None, window_days: int = 90, max_filings: int = 10
) -> str | None:
    """Summarize insider (Form 4) buy/sell activity in the `window_days` days before `as_of`.

    `as_of` bounds the window point-in-time-safely, same as `get_recent_filings_summary`.
    `max_filings` caps how many Form 4 documents get fetched per call (each is a
    separate throttled request). Always returns a non-empty string.
    """
    try:
        cik = _get_cik(ticker)
        if not cik:
            return None
        cutoff = _as_of_date(as_of) or datetime.now().strftime("%Y-%m-%d")
        window_start = (
            datetime.strptime(cutoff, "%Y-%m-%d") - timedelta(days=window_days)
        ).strftime("%Y-%m-%d")

        data = _throttled_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        recent = data["filings"]["recent"]
        filings = [
            (date, accn, doc)
            for form, date, accn, doc in zip(
                recent["form"],
                recent["filingDate"],
                recent["accessionNumber"],
                recent["primaryDocument"],
            )
            if form in ("4", "4/A") and window_start <= date <= cutoff
        ]
        if not filings:
            return f"No insider Form 4 filings in the {window_days} days before {cutoff}."
        filings.sort(key=lambda f: f[0], reverse=True)

        buy_sell: list[dict] = []
        other = 0
        cik_int = str(int(cik))
        for date, accn, doc in filings[:max_filings]:
            # `doc` (e.g. "xslF345X06/form4.xml") points at EDGAR's HTML-rendered
            # viewer page, not the raw machine-readable XML -- fetching it as-is
            # silently returns HTML, which _form4_transactions can't parse (always
            # []). The raw XML sits in the same accession folder, just without
            # that viewer subdirectory prefix.
            filename = doc.rsplit("/", 1)[-1]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn.replace('-', '')}/{filename}"
            try:
                xml_text = _throttled_get_text(url)
            except Exception:
                continue
            for t in _form4_transactions(xml_text):
                if t["code"] in _BUY_CODES or t["code"] in _SELL_CODES:
                    buy_sell.append(t)
                else:
                    other += 1

        def _describe(t: dict) -> str:
            if t["shares"] is not None and t["price"] is not None:
                return f"{t['role']} (~${t['shares'] * t['price']:,.0f})"
            if t["shares"] is not None:
                return f"{t['role']} ({t['shares']:,.0f} shares)"
            return t["role"]

        buys = [t for t in buy_sell if t["code"] in _BUY_CODES]
        sells = [t for t in buy_sell if t["code"] in _SELL_CODES]

        parts = []
        if buys:
            parts.append("open-market buy(s) by " + ", ".join(_describe(t) for t in buys))
        if sells:
            parts.append("open-market sell(s) by " + ", ".join(_describe(t) for t in sells))
        if not buys and not sells:
            parts.append("no open-market buy/sell transactions")
        if other:
            parts.append(f"{other} other (grant/exercise/tax-withholding/gift, not buy or sell)")

        return (
            f"{len(filings)} insider Form 4 filing(s) in the {window_days} days before "
            f"{cutoff} (most recent: {filings[0][0]}) -- " + "; ".join(parts) + "."
        )
    except Exception as e:
        print(f"[SEC EDGAR] insider activity unavailable for {ticker}: {e}")
        return None
