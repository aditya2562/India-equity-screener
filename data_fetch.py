import yfinance as yf
import pandas as pd
import os

# ─── NSE Mid-Cap Stock Universe ───────────────────────────────────────────────
# yfinance uses ".NS" suffix for NSE-listed stocks
MIDCAP_STOCKS = [
    "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS", "MINDTREE.NS", "MPHASIS.NS",
    "VOLTAS.NS", "CROMPTON.NS", "WHIRLPOOL.NS", "HAVELLS.NS", "POLYCAB.NS",
    "METROPOLIS.NS", "LALPATHLAB.NS", "THYROCARE.NS", "IPCALAB.NS", "ALKEM.NS",
    "FEDERALBNK.NS", "KARURVYSYA.NS", "CITYUNIONB.NS", "DCBBANK.NS", "EQUITAS.NS",
    "RELAXO.NS", "BATAINDIA.NS", "PAGEIND.NS", "VMART.NS", "TRENT.NS"
]

def fetch_stock_data(tickers: list, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch historical price + volume data for a list of NSE tickers.
    
    Args:
        tickers: list of NSE ticker symbols (with .NS suffix)
        period:  how far back to fetch — "6mo", "1y", "2y" etc.
    
    Returns:
        A cleaned DataFrame with one row per ticker
    """
    records = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period=period)

            if hist.empty:
                print(f"[SKIP] No data for {ticker}")
                continue

            info = stock.info  # fundamentals

            # ── Price & Momentum ──────────────────────────────────────────────
            current_price = hist["Close"].iloc[-1]
            price_1m_ago  = hist["Close"].iloc[-21] if len(hist) >= 21 else None
            price_3m_ago  = hist["Close"].iloc[-63] if len(hist) >= 63 else None

            return_1m = ((current_price - price_1m_ago) / price_1m_ago * 100
                         if price_1m_ago else None)
            return_3m = ((current_price - price_3m_ago) / price_3m_ago * 100
                         if price_3m_ago else None)

            avg_volume_30d = hist["Volume"].tail(30).mean()

            # ── Fundamentals ──────────────────────────────────────────────────
            records.append({
                "Ticker":        ticker.replace(".NS", ""),
                "Company":       info.get("shortName", ticker),
                "Sector":        info.get("sector", "N/A"),
                "Industry":      info.get("industry", "N/A"),
                "Current Price": round(current_price, 2),
                "1M Return %":   round(return_1m, 2) if return_1m else None,
                "3M Return %":   round(return_3m, 2) if return_3m else None,
                "Avg Volume 30D": int(avg_volume_30d),
                "Market Cap":    info.get("marketCap", None),
                "P/E Ratio":     info.get("trailingPE", None),
                "EPS":           info.get("trailingEps", None),
                "ROE %":         info.get("returnOnEquity", None),
                "Debt/Equity":   info.get("debtToEquity", None),
                "52W High":      info.get("fiftyTwoWeekHigh", None),
                "52W Low":       info.get("fiftyTwoWeekLow", None),
            })

            print(f"[OK] Fetched {ticker}")

        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    df = pd.DataFrame(records)
    return df

def save_data(df: pd.DataFrame, path: str = "data/stocks.csv") -> None:
    """Save the DataFrame to CSV."""
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n✅ Data saved to {path} — {len(df)} stocks fetched.")

if __name__ == "__main__":
    df = fetch_stock_data(MIDCAP_STOCKS)
    save_data(df)
    print(df.head())