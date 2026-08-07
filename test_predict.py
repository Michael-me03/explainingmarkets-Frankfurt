from dotenv import load_dotenv
from predict_kimi import _ask_llm

# .env laden
load_dotenv()

# Eingaben
summary = (
    "Apple reported Q3 revenue of $95B, beating estimates of $92B. "
    "Guidance raised for Q4."
)

ticker = "AAPL"
event_type = "EARNINGS_RELEASE"

# LLM aufrufen
result = _ask_llm(
    summary={"summary": summary},
    ticker=ticker,
    event_type=event_type,
)

print(f"Summary: {summary}")
print(f"Ticker: {ticker}")
print(f"Event: {event_type}")
print(f"Predicted percentile: {result}")