import pandas as pd
import numpy as np


def load_data(path: str = "data/stocks.csv") -> pd.DataFrame:
    """Load raw stock data from CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw stock data:
    - Drop rows with no price
    - Fix data types
    - Handle missing fundamentals gracefully
    """
    # Must have a price — drop if missing
    df = df.dropna(subset=["Current Price"])

    # Convert numeric columns (some come in as strings or objects)
    numeric_cols = [
        "Current Price", "1M Return %", "3M Return %",
        "Market Cap", "P/E Ratio", "EPS", "ROE %",
        "Debt/Equity", "52W High", "52W Low", "Avg Volume 30D"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ROE comes as decimal from yfinance (0.25 = 25%) — convert to percentage
    df["ROE %"] = df["ROE %"] * 100

    # Market Cap: convert to Crores (easier to read for Indian context)
    df["Market Cap (Cr)"] = (df["Market Cap"] / 1e7).round(0)
    df = df.drop(columns=["Market Cap"])

    # Fill missing fundamentals with NaN (don't fill with 0 — misleading)
    df = df.where(pd.notnull(df), other=None)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features useful for screening:
    - 52W Position %: where is price between 52W low and high
    - Momentum Signal: combined 1M + 3M return score
    - Value Score: simple score based on P/E and ROE
    """

    # ── 52W Position % ───────────────────────────────────────────────────────
    # 100% = at 52W high, 0% = at 52W low
    df["52W Position %"] = (
        (df["Current Price"] - df["52W Low"]) /
        (df["52W High"] - df["52W Low"]) * 100
    ).round(1)

    # ── Momentum Score (0–100) ────────────────────────────────────────────────
    # Ranks stocks by combined 1M + 3M returns relative to the universe
    df["Momentum Score"] = (
        df["1M Return %"].rank(pct=True) * 50 +
        df["3M Return %"].rank(pct=True) * 50
    ).round(1)

    # ── Value Score (0–100) ───────────────────────────────────────────────────
    # Low P/E = good (inverted rank), High ROE = good (normal rank)
    pe_score  = (1 - df["P/E Ratio"].rank(pct=True)) * 50   # lower P/E → higher score
    roe_score = df["ROE %"].rank(pct=True) * 50              # higher ROE → higher score
    df["Value Score"] = (pe_score + roe_score).round(1)

    # ── Overall Score ─────────────────────────────────────────────────────────
    df["Overall Score"] = (
        df["Momentum Score"] * 0.5 +
        df["Value Score"]    * 0.5
    ).round(1)

    return df


def get_screened_data() -> pd.DataFrame:
    """Full pipeline: load → clean → engineer → return."""
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = get_screened_data()
    print(df[["Ticker", "Company", "Current Price",
              "1M Return %", "3M Return %",
              "Momentum Score", "Value Score", "Overall Score"]].to_string())
    print(f"\n✅ {len(df)} stocks processed.")