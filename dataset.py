
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

# Stock tickers and friendly names
tickers = {
    "RELIANCE.NS": "Reliance",
    "TCS.NS": "TCS",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "HINDUNILVR.NS": "HUL",
    "ITC.NS": "ITC",
    "SBIN.NS": "SBI",
    "BHARTIARTL.NS": "Airtel",
    "KOTAKBANK.NS": "Kotak Bank",
    "LT.NS": "L&T",
    "BAJFINANCE.NS": "Bajaj Finance",
    "MARUTI.NS": "Maruti",
    "AXISBANK.NS": "Axis Bank",
    "HCLTECH.NS": "HCL Tech"
}

# Headline templates
headline_templates = [
    "{} reports record profits",
    "{} misses quarterly revenue targets",
    "{} announces new strategic partnership",
    "{} under government scrutiny",
    "{} launches new product line",
    "{} wins major overseas contract",
    "{} to invest in green energy",
    "{} sees strong domestic growth",
    "{} faces legal challenges",
    "{} stock upgraded by analysts"
]

# Dates (5 years)
start_date = datetime.now() - timedelta(days=3 * 365)
end_date = datetime.now() - timedelta(days=1)
rows = []

# Helper function
def is_weekend(dt):
    return dt.weekday() >= 5  # Saturday or Sunday

print("📈 Fetching 5 years of data...")

# Main loop
for ticker, name in tickers.items():
    try:
        hist = yf.Ticker(ticker).history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        hist = hist[~hist.index.duplicated(keep='first')]
        hist['MA5'] = hist['Close'].rolling(window=5).mean()
        hist['MA10'] = hist['Close'].rolling(window=10).mean()
        hist['Volatility'] = hist['Close'].rolling(window=5).std()

        # Drop rows with NaNs
        hist = hist.dropna(subset=['MA5', 'MA10', 'Volatility'])
        hist.index = pd.to_datetime(hist.index)

        dates = hist.index

        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]

            if is_weekend(current_date) or is_weekend(next_date):
                continue

            price_before = hist.loc[current_date, 'Close']
            price_after = hist.loc[next_date, 'Close']
            change = ((price_after - price_before) / price_before) * 100
            label = 'Up' if change > 1 else 'Down' if change < -1 else 'Stable'
            headline = random.choice(headline_templates).format(name)

            # Extract indicators
            ma5 = hist.loc[current_date, 'MA5']
            ma10 = hist.loc[current_date, 'MA10']
            vol = hist.loc[current_date, 'Volatility']

            # Ensure scalars (fixes the ndarray error)
            if isinstance(ma5, (np.ndarray, pd.Series)): ma5 = ma5.item()
            if isinstance(ma10, (np.ndarray, pd.Series)): ma10 = ma10.item()
            if isinstance(vol, (np.ndarray, pd.Series)): vol = vol.item()

            rows.append([
                headline, ticker, current_date.strftime("%Y-%m-%d"),
                round(price_before, 2), round(price_after, 2), round(change, 2),
                round(ma5, 2), round(ma10, 2), round(vol, 4), label
            ])

    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")

# Save to CSV
df = pd.DataFrame(rows, columns=[
    "Headline", "Ticker", "Date", "PriceBefore", "PriceAfter", "ChangePercent",
    "MA5", "MA10", "Volatility", "Label"
])

df.to_csv("nnindian_stock_news.csv", index=False)
print(f"\n✅ Full 5-year dataset created with {len(df)} rows → saved as 'indian_stock_news.csv'")

