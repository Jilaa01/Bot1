# Crypto AI Paper-Trading Bot

An educational Streamlit web dashboard using Binance public Spot market candles.
It calculates RSI, MACD and EMA indicators, trains a simple logistic-regression
classifier, displays hypothetical signals, and runs a walk-forward paper backtest.

## Safety and scope
- Paper/analysis only. No order-placement code exists.
- No Binance API key is required.
- No deposits, withdrawals, account access, margin, or leverage.
- Signals and historical results are not financial advice or a guarantee.

## Run locally

1. Install Python 3.10 or newer.
2. Extract this folder.
3. Open a terminal in the extracted folder.
4. Create and activate a virtual environment:

   Windows:
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

7. Open the local address Streamlit prints (usually http://localhost:8501).

## Notes
- Market data comes from Binance's public market-data endpoint:
  https://data-api.binance.vision/api/v3/klines
- Data is cached for 60 seconds; use Refresh market data to clear the cache.
- The app uses up to 1,000 recent candles. The walk-forward backtest can take
  time because it repeatedly trains a model using only earlier observations.
- The backtest assumes a one-candle holding period and a configurable fee.
  It does not model spread, slippage, partial fills, taxes, or outages.
