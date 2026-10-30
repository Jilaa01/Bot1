import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

st.set_page_config(page_title="Crypto Paper Bot", page_icon="📈", layout="wide")

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

@st.cache_data(ttl=60)
def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    response = requests.get(
        BINANCE_URL,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise ValueError("Binance returned no candles for this selection.")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]].dropna()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()

    out["return_1"] = close.pct_change()
    out["return_3"] = close.pct_change(3)
    out["volatility_10"] = out["return_1"].rolling(10).std()
    out["ema_gap"] = (out["ema20"] / out["ema50"]) - 1
    out["macd_pct"] = out["macd"] / close
    out["macd_hist_pct"] = out["macd_hist"] / close
    out["target"] = np.where(close.shift(-1).notna(), (close.shift(-1) > close).astype(float), np.nan)
    return out

FEATURES = [
    "rsi", "return_1", "return_3", "volatility_10",
    "ema_gap", "macd_pct", "macd_hist_pct",
]

def train_and_score(data: pd.DataFrame):
    clean = data.dropna(subset=FEATURES + ["target"]).copy()
    if len(clean) < 150:
        raise ValueError("Not enough valid candles. Choose a larger history window.")

    # Chronological holdout: the newest 20% is used only for evaluation.
    split = int(len(clean) * 0.8)
    train = clean.iloc[:split]
    test = clean.iloc[split:]
    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        raise ValueError("Not enough examples from both classes to train/evaluate.")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(train[FEATURES], train["target"])
    test_prob = model.predict_proba(test[FEATURES])[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(test["target"], test_pred),
        "precision": precision_score(test["target"], test_pred, zero_division=0),
        "test_rows": len(test),
    }

    # Refit on all labelled candles before scoring the latest candle.
    model.fit(clean[FEATURES], clean["target"])
    latest = data.dropna(subset=FEATURES).iloc[[-1]]
    probability_up = float(model.predict_proba(latest[FEATURES])[0, 1])
    return model, metrics, latest, probability_up

def make_signal(row: pd.Series, probability_up: float, threshold: float):
    bullish_stack = row["ema20"] > row["ema50"]
    momentum_up = row["macd"] > row["macd_signal"]
    rsi_ok = 45 <= row["rsi"] <= 70
    if probability_up >= threshold and bullish_stack and momentum_up and rsi_ok:
        return "PAPER BUY"
    if probability_up <= (1 - threshold):
        return "WAIT / NO ENTRY"
    return "NEUTRAL"

def run_backtest(data: pd.DataFrame, threshold: float, fee_pct: float):
    # Walk-forward model predictions: each prediction uses only earlier candles.
    clean = data.dropna(subset=FEATURES + ["target"]).copy()
    start = max(150, int(len(clean) * 0.5))
    if len(clean) - start < 20:
        return pd.DataFrame()

    records = []
    for i in range(start, len(clean)):
        train = clean.iloc[:i]
        current = clean.iloc[[i]]
        if train["target"].nunique() < 2:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        model.fit(train[FEATURES], train["target"])
        p_up = float(model.predict_proba(current[FEATURES])[0, 1])
        row = current.iloc[0]
        enter = (
            p_up >= threshold
            and row["ema20"] > row["ema50"]
            and row["macd"] > row["macd_signal"]
            and 45 <= row["rsi"] <= 70
        )
        next_return = float(row["target_return"])
        # Fee is charged on entry and exit; no leverage, one-candle holding period.
        strategy_return = next_return - (2 * fee_pct) if enter else 0.0
        records.append({
            "time": row["time"],
            "probability_up": p_up,
            "paper_entry": enter,
            "next_candle_return": next_return,
            "strategy_return": strategy_return,
        })
    result = pd.DataFrame(records)
    if not result.empty:
        result["equity"] = (1 + result["strategy_return"]).cumprod()
    return result

st.title("📈 Crypto AI Paper-Trading Dashboard")
st.caption("Educational simulator • Binance public market data • No real orders or account access")

with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Trading pair", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"])
    interval = st.selectbox("Candle interval", ["5m", "15m", "1h", "4h", "1d"], index=2)
    limit = st.slider("Historical candles", min_value=300, max_value=1000, value=600, step=50)
    threshold = st.slider("Model probability threshold", 0.50, 0.80, 0.60, 0.01)
    fee_pct = st.number_input("Assumed fee per side (%)", min_value=0.0, max_value=1.0, value=0.10, step=0.01) / 100
    refresh = st.button("Refresh market data")

if refresh:
    fetch_klines.clear()

try:
    raw = fetch_klines(symbol, interval, limit)
    data = add_indicators(raw)
    # For backtesting, use next-candle close-to-close return.
    data["target_return"] = data["close"].shift(-1) / data["close"] - 1
    model, metrics, latest, p_up = train_and_score(data)
    last = latest.iloc[0]
    signal = make_signal(last, p_up, threshold)

    a, b, c, d = st.columns(4)
    a.metric("Latest close", f"${last['close']:,.4f}")
    b.metric("Model probability ↑", f"{p_up:.1%}")
    c.metric("RSI (14)", f"{last['rsi']:.1f}")
    d.metric("Signal", signal)

    st.subheader("Market overview")
    chart_data = data.set_index("time")[["close", "ema20", "ema50"]].tail(200)
    st.line_chart(chart_data, height=320)
    st.dataframe(
        data[["time", "close", "rsi", "macd", "macd_signal", "ema20", "ema50"]].tail(12),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Model evaluation")
    m1, m2, m3 = st.columns(3)
    m1.metric("Chronological holdout accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Holdout precision", f"{metrics['precision']:.1%}")
    m3.metric("Holdout candles", f"{metrics['test_rows']}")
    st.caption("These are historical classification metrics, not a forecast guarantee. Accuracy alone does not establish profitability.")

    st.subheader("Walk-forward paper backtest")
    with st.spinner("Evaluating historical signals. This may take a little while..."):
        bt = run_backtest(data, threshold, fee_pct)
    if bt.empty:
        st.info("Not enough data for the selected walk-forward test.")
    else:
        total_return = bt["equity"].iloc[-1] - 1
        entries = int(bt["paper_entry"].sum())
        x, y, z = st.columns(3)
        x.metric("Simulated strategy return", f"{total_return:.2%}")
        y.metric("Paper entries", f"{entries}")
        z.metric("Assumed round-trip fee", f"{2*fee_pct:.2%}")
        st.line_chart(bt.set_index("time")[["equity"]], height=250)
        st.dataframe(bt.tail(20), use_container_width=True, hide_index=True)

    st.info(
        "Paper-only mode: the app reads public candles and calculates hypothetical signals. "
        "It does not connect to a trading account, place orders, or handle deposits/withdrawals."
    )
except requests.RequestException as exc:
    st.error(f"Could not reach Binance public market data: {exc}")
except Exception as exc:
    st.error(f"App could not complete this run: {exc}")

with st.expander("How the signal works"):
    st.markdown(
        """
        - **RSI (14):** a momentum measure; this example accepts values from 45 to 70 for a bullish setup.
        - **MACD:** requires MACD to be above its signal line.
        - **EMA:** requires EMA 20 to be above EMA 50.
        - **AI model:** logistic regression learns from RSI, returns, volatility, EMA gap and MACD features to estimate whether the next candle closes higher.
        - **Paper BUY:** all three indicator filters pass and model probability meets the selected threshold.

        This is a learning example, not investment advice. Backtests can be misleading due to fees, slippage,
        changing market conditions, and overfitting. Do not use it to make real-money decisions.
        """
    )
