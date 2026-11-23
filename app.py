import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="5m Trader | 15m->45m AI Predictor", layout="wide")
st.title("⚡ Pro Intraday Setup: 15-Min Analysis ➔ Agle 45-Min Outlook (5-Min Chart)")

# Sidebar Controls
index_choice = st.sidebar.selectbox("Index Chunein", ["Nifty 50", "Sensex", "Bank Nifty"])
auto_refresh = st.sidebar.checkbox("Har 30 sec mein Live Update karein", value=True)

ticker_map = {"Nifty 50": "^NSEI", "Sensex": "^BSESN", "Bank Nifty": "^NSEBANK"}
ticker = ticker_map[index_choice]

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def get_nse_pcr():
    try:
        session = requests.Session()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        session.get("https://www.nseindia.com", headers=headers, timeout=4)
        res = session.get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY", headers=headers, timeout=4).json()
        ce_oi = res["filtered"]["CE"]["totOI"]
        pe_oi = res["filtered"]["PE"]["totOI"]
        return round(pe_oi / ce_oi, 2) if ce_oi else None
    except:
        return None

def detect_candlestick_pattern(c1, c2):
    body1 = abs(c1["Close"] - c1["Open"])
    upper_wick1 = c1["High"] - max(c1["Open"], c1["Close"])
    lower_wick1 = min(c1["Open"], c1["Close"]) - c1["Low"]
    
    if c2["Close"] < c2["Open"] and c1["Close"] > c1["Open"] and c1["Close"] > c2["Open"] and c1["Open"] <= c2["Close"]:
        return "Bullish Engulfing (Strong Reversal Up)", 35
    if c2["Close"] > c2["Open"] and c1["Close"] < c1["Open"] and c1["Close"] < c2["Open"] and c1["Open"] >= c2["Close"]:
        return "Bearish Engulfing (Strong Reversal Down)", -35
    if lower_wick1 > (2 * body1) and upper_wick1 < body1 and body1 > 0:
        return "Bullish Hammer (Buyers Active at Bottom)", 25
    if upper_wick1 > (2 * body1) and lower_wick1 < body1 and body1 > 0:
        return "Shooting Star (Sellers Rejection at Top)", -25
    if c1["Close"] > c2["High"]:
        return "15m High Breakout Candle", 25
    if c1["Close"] < c2["Low"]:
        return "15m Low Breakdown Candle", -25
    return "Normal Price Action", 0

df_15 = yf.download(ticker, period="5d", interval="15m", progress=False)
df_5 = yf.download(ticker, period="2d", interval="5m", progress=False)

if not df_15.empty and not df_5.empty:
    for d in [df_15, df_5]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
            
    df_15 = df_15.reset_index()
    df_5 = df_5.reset_index()
    
    # 15-Min Indicators
    df_15["EMA9"] = df_15["Close"].ewm(span=9, adjust=False).mean()
    df_15["EMA21"] = df_15["Close"].ewm(span=21, adjust=False).mean()
    df_15["RSI"] = calc_rsi(df_15["Close"], 14)
    df_15["ATR"] = calc_atr(df_15, 14)
    
    # 5-Min Indicators
    df_5["EMA9"] = df_5["Close"].ewm(span=9, adjust=False).mean()
    df_5["EMA21"] = df_5["Close"].ewm(span=21, adjust=False).mean()
    
    last15 = df_15.iloc[-1]
    prev15 = df_15.iloc[-2]
    spot = round(float(df_5.iloc[-1]["Close"]), 2)
    rsi_val = round(float(last15["RSI"]), 1)
    atr_val = float(last15["ATR"])
    pcr = get_nse_pcr() if index_choice == "Nifty 50" else None

    score = 0
    reasons = []
    
    pattern_name, pat_score = detect_candlestick_pattern(last15, prev15)
    score += pat_score
    if pat_score != 0:
        reasons.append(f"Pattern: {pattern_name}")
        
    if last15["Close"] > last15["EMA9"] > last15["EMA21"]:
        score += 30; reasons.append("15m Uptrend (Above 9 & 21 EMA)")
    elif last15["Close"] < last15["EMA9"] < last15["EMA21"]:
        score -= 30; reasons.append("15m Downtrend (Below 9 & 21 EMA)")
        
    trap_warning = ""
    if rsi_val >= 75:
        trap_warning = "⚠️ OVERBOUGHT ZONE (RSI 75+): Chhota pullback aa sakta hai, fresh Call lene se pehle 5m dip ka wait karein!"
        score -= 15
    elif rsi_val <= 25:
        trap_warning = "⚠️ OVERSOLD ZONE (RSI 25-): Short covering bounce aa sakta hai, fresh Put mein savdhan rahein!"
        score += 15
    elif rsi_val > 55:
        score += 20; reasons.append(f"RSI Bullish Momentum ({rsi_val})")
    elif rsi_val < 45:
        score -= 20; reasons.append(f"RSI Bearish Momentum ({rsi_val})")
        
    if pcr:
        if pcr > 1.05:
            score += 20; reasons.append(f"Put Writers Active (PCR {pcr})")
        elif pcr < 0.90:
            score -= 20; reasons.append(f"Call Writers Active (PCR {pcr})")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{index_choice} Live", f"{spot}")
    m2.metric("15m Pattern", pattern_name.split("(")[0].strip())
    m3.metric("15m RSI", f"{rsi_val}")
    m4.metric("Nifty PCR", f"{pcr if pcr else 'N/A'}")
    m5.metric("Volatility (ATR)", f"{round(atr_val, 1)} pts")

    if trap_warning:
        st.info(trap_warning)
        
    if score >= 45:
        t1 = round(spot + (atr_val * 0.8), 1)
        t2 = round(spot + (atr_val * 1.4), 1)
        sl = round(spot - (atr_val * 0.6), 1)
        st.success(f"🟢 AGLE 45 MIN OUTLOOK: **BULLISH (UPAR JANE KE HIGH CHANCES)** | Confidence Score: {min(score, 95)}%")
        st.markdown(f"**🎯 5-Min Trade Plan (Call / Buy Side):** Entry: `{spot}` | **Target 1:** `{t1}` | **Target 2:** `{t2}` | **Strict SL:** `{sl}`")
    elif score <= -45:
        t1 = round(spot - (atr_val * 0.8), 1)
        t2 = round(spot - (atr_val * 1.4), 1)
        sl = round(spot + (atr_val * 0.6), 1)
        st.error(f"🔴 AGLE 45 MIN OUTLOOK: **BEARISH (NICHE GIRNE KE HIGH CHANCES)** | Confidence Score: {min(abs(score), 95)}%")
        st.markdown(f"**🎯 5-Min Trade Plan (Put / Sell Side):** Entry: `{spot}` | **Target 1:** `{t1}` | **Target 2:** `{t2}` | **Strict SL:** `{sl}`")
    else:
        st.warning(f"🟡 AGLE 45 MIN OUTLOOK: **SIDEWAYS / RANGE-BOUND** (Score: {score}) — Abhi premium decay hoga, clear breakout ka wait karein.")
        
    st.caption(f"📊 **Study Logic:** {'  •  '.join(reasons) if reasons else 'Market abhi consolidation phase mein hai.'}")

    plot_5 = df_5.tail(65).copy()
    time_col = "Datetime" if "Datetime" in plot_5.columns else "Date"
    plot_5["TimeStr"] = pd.to_datetime(plot_5[time_col]).dt.strftime("%d %b %H:%M")
    
    fig = go.Figure(data=[go.Candlestick(
        x=plot_5["TimeStr"],
        open=plot_5["Open"],
        high=plot_5["High"],
        low=plot_5["Low"],
        close=plot_5["Close"],
        name="5m Candles"
    )])
    fig.add_trace(go.Scatter(x=plot_5["TimeStr"], y=plot_5["EMA9"], line=dict(color="#00f2ff", width=1.5), name="9 EMA"))
    fig.add_trace(go.Scatter(x=plot_5["TimeStr"], y=plot_5["EMA21"], line=dict(color="#ff9900", width=1.5), name="21 EMA"))
    
    day_high = plot_5["High"].max()
    day_low = plot_5["Low"].min()
    fig.add_hline(y=day_high, line_dash="dot", line_color="green", annotation_text=f"Resistance: {round(day_high,1)}")
    fig.add_hline(y=day_low, line_dash="dot", line_color="red", annotation_text=f"Support: {round(day_low,1)}")
    
    fig.update_xaxes(type="category", nticks=12)
    fig.update_layout(xaxis_rangeslider_visible=False, height=540, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

if st.button("🔄 Instant Refresh"):
    st.rerun()

if auto_refresh:
    time.sleep(30)
    st.rerun()
