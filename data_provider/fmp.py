import json
import os
import threading
import time

import requests
from dotenv import load_dotenv
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(ENV_PATH)

API_KEY = os.getenv("FMP_API_KEY")

BASE_URL = "https://financialmodelingprep.com/stable"

# ace/train.py runs up to MAX_WORKERS Generator calls concurrently, each of
# which can call fmp_get -- without a global throttle that's a burst of
# simultaneous requests against FMP's free-tier rate limit (observed: 429 Too
# Many Requests within seconds of starting a training run). A lock + minimum
# interval serializes calls across ALL threads, regardless of how many
# ThreadPoolExecutor workers are calling in at once.
_MIN_REQUEST_INTERVAL_SECONDS = 0.35
_rate_limit_lock = threading.Lock()
_last_request_at = 0.0


def fmp_get(endpoint, params=None):
    if params is None:
        params = {}

    params["apikey"] = API_KEY

    url = f"{BASE_URL}/{endpoint}"

    global _last_request_at
    with _rate_limit_lock:
        wait = _MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        response = requests.get(url, params=params, timeout=10)

    print("Request:", response.url)

    response.raise_for_status()

    return response.json()


def get_profile(ticker):
    return fmp_get(
        "profile",
        {
            "symbol": ticker
        }
    )


def get_quote(ticker):
    return fmp_get(
        "quote",
        {
            "symbol": ticker
        }
    )


def get_income_statement(ticker):
    return fmp_get(
        "income-statement",
        {
            "symbol": ticker
        }
    )


def get_price_history(ticker):
    return fmp_get(
        "historical-price-eod/full",
        {
            "symbol": ticker
        } 
    )

"""
def get_analyst_estimates(ticker):
    return fmp_get(
        "analyst-estimates",
        {
            "symbol": ticker,
            "period": "quarter",
            "page": 0,
            "limit": 10
        }
    )
"""


_PROFILE_CACHE_PATH = Path(__file__).resolve().parent / "profile_cache.json"
_profile_cache_lock = threading.Lock()


def _load_profile_cache() -> dict:
    if not _PROFILE_CACHE_PATH.exists() or _PROFILE_CACHE_PATH.stat().st_size == 0:
        return {}
    try:
        return json.loads(_PROFILE_CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _save_profile_cache(cache: dict) -> None:
    tmp = _PROFILE_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    os.replace(tmp, _PROFILE_CACHE_PATH)


_profile_cache = _load_profile_cache()


def get_company_profile_summary(ticker):
    """Short one-line company blurb (name, sector, industry) for the LLM prompt.

    Deliberately excludes FMP's long `description` field -- just enough to
    identify the kind of company, not a paragraph. Disk-cached like
    yfinance's `get_sector` -- name/sector/industry don't change day to day,
    so a repeated ticker (across events, or across training epochs) never
    re-hits FMP, on top of the request-level throttle in `fmp_get`.
    """
    if ticker in _profile_cache:
        return _profile_cache[ticker]

    try:
        profile = get_profile(ticker)
        p = profile[0]
        name = p.get("companyName") or ticker
        bits = [b for b in (p.get("sector"), p.get("industry")) if b]
        summary = f"{name} ({', '.join(bits)})" if bits else name
    except Exception as e:
        print(f"[FMP] company profile unavailable for {ticker}: {e}")
        summary = None

    with _profile_cache_lock:
        _profile_cache[ticker] = summary
        _save_profile_cache(_profile_cache)
    return summary


def get_market_context(ticker):
    # Each source is fetched independently — a paywalled/failing endpoint
    # (e.g. /quote or /analyst-estimates on the free FMP plan) should not
    # throw away data that other endpoints already returned successfully.
    context = {}

    try:
        profile = get_profile(ticker)
        context["sector"] = profile[0]["sector"]
        context["market_cap"] = profile[0]["marketCap"]
        context["profile"] = profile[0]
    except Exception as e:
        print(f"[FMP] profile unavailable for {ticker}: {e}")

    try:
        quote = get_quote(ticker)
        context["price"] = quote[0]["price"]
        context["volume"] = quote[0]["volume"]
    except Exception as e:
        print(f"[FMP] quote unavailable for {ticker}: {e}")

    try:
        income = get_income_statement(ticker)
        context["actual_revenue"] = income[0]["revenue"]
        context["actual_eps"] = income[0]["eps"]
    except Exception as e:
        print(f"[FMP] income statement unavailable for {ticker}: {e}")

    #try:
    #    estimates = get_analyst_estimates(ticker)
    #    context["expected_revenue"] = estimates[0]["estimatedRevenueAvg"]
    #    context["expected_eps"] = estimates[0]["estimatedEpsAvg"]
    #except Exception as e:
    #    print(f"[FMP] analyst estimates unavailable for {ticker}: {e}")

    return context or None

if __name__ == "__main__":

    ticker = "AAPL"

    print("\n=== PROFILE ===")
    profile = get_profile(ticker)
    print(profile)

    print("\n=== QUOTE ===")
    quote = get_quote(ticker)
    print(quote)

    print("\n=== INCOME STATEMENT ===")
    income = get_income_statement(ticker)
    print(income[:1] if isinstance(income, list) else income)

    print("\n=== Price History ===")
    price_history = get_price_history(ticker)
    print(price_history)

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(price_history)

    # Last 90 days
    df = df.head(90)

    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df['date']), df['close'], label='Close Price')
    plt.title(f'{ticker} Price History')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.show()
