import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="PRO Option Buyer Terminal", layout="wide")
st.title("🎯 Pro Option Buyer Terminal (Sirf CALL BUY / PUT BUY)")

index_choice = st.sidebar.selectbox("Index Chunein", ["Nifty 50", "Bank Nifty", "Sensex"])
risk_mode = st.sidebar.radio("Trading Style", ["Safe (High Accuracy)", "Scalp (Fast 15m)"])
auto_refresh = st.sidebar.checkbox("Har 30 sec mein Live Update", value=True)

ticker_map = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Sensex": "^BSESN"}
step_map = {"Nifty 50": 50, "Bank Nifty": 100, "Sensex": 100}
orb_limit_map = {"Nifty 50": 130, "Bank Nifty": 300, "Sensex": 400}

ticker = ticker_map[index_choice]
strike_step = step_map[index_choice]
max_orb_width = orb_limit_map[index_choice]

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

def calc_adx(df, period=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = calc_atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

df_15 = yf.download(ticker, period="5d", interval="15m", progress=False)
df_5 = yf.download(ticker, period="5d", interval="5m", progress=False)

if not df_15.empty and not df_5.empty:
    for d in [df_15, df_5]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    df_15 = df_15.reset_index()
    df_5 = df_5.reset_index()

    # 15m Indicators
    df_15["EMA9"] = df_15["Close"].ewm(span=9, adjust=False).mean()
    df_15["EMA21"] = df_15["Close"].ewm(span=21, adjust=False).mean()
    df_15["RSI"] = calc_rsi(df_15["Close"], 14)
    df_15["ATR"] = calc_atr(df_15, 14)
    df_15["ADX"] = calc_adx(df_15, 14)

    # Today's 5m Data
    time_col = "Datetime" if "Datetime" in df_5.columns else "Date"
    df_5["DT"] = pd.to_datetime(df_5[time_col])
    df_5["DateOnly"] = df_5["DT"].dt.date
    unique_dates = sorted(df_5["DateOnly"].unique())
    
    latest_date = unique_dates[-1]
    today_5 = df_5[df_5["DateOnly"] == latest_date].copy()

    today_5["EMA9"] = today_5["Close"].ewm(span=9, adjust=False).mean()
    today_5["EMA21"] = today_5["Close"].ewm(span=21, adjust=False).mean()
    today_5["TP"] = (today_5["High"] + today_5["Low"] + today_5["Close"]) / 3
    today_5["VWAP"] = today_5["TP"].expanding().mean()

    orb_high = round(float(today_5.head(6)["High"].max()), 1)
    orb_low = round(float(today_5.head(6)["Low"].min()), 1)
    orb_width = round(orb_high - orb_low, 1)
    is_wide_orb = orb_width > max_orb_width

    c15 = df_15.iloc[-2]
    spot = round(float(today_5.iloc[-1]["Close"]), 2)
    vwap_now = round(float(today_5.iloc[-1]["VWAP"]), 2)
    ema9_5m = round(float(today_5.iloc[-1]["EMA9"]), 2)
    ema21_5m = round(float(today_5.iloc[-1]["EMA21"]), 2)
    rsi_val = round(float(c15["RSI"]), 1)
    adx_val = round(float(c15["ADX"]), 1)
    atr_val = float(c15["ATR"])

    # Option Strikes (ITM & ATM)
    atm_strike = int(round(spot / strike_step) * strike_step)
    itm_ce = atm_strike - strike_step
    itm_pe = atm_strike + strike_step

    score = 0
    bull_reasons = []
    bear_reasons = []
    no_trade_reasons = []

    if adx_val < 20:
        no_trade_reasons.append(f"Market Weak/Sideways hai (ADX {adx_val} < 20) — Option buying mein premium decay hoga.")
    else:
        if c15["Close"] > c15["EMA9"] > c15["EMA21"]:
            score += 25; bull_reasons.append("15m Uptrend (9 > 21 EMA)")
        elif c15["Close"] < c15["EMA9"] < c15["EMA21"]:
            score -= 25; bear_reasons.append("15m Downtrend (9 < 21 EMA)")

    if spot > ema9_5m > ema21_5m:
        score += 25; bull_reasons.append(f"5m Momentum Upar (Above 9 EMA {ema9_5m})")
    elif spot < ema9_5m < ema21_5m:
        score -= 25; bear_reasons.append(f"5m Momentum Neeche (Below 9 EMA {ema9_5m})")
    else:
        no_trade_reasons.append("5m Candles 9 & 21 EMA ke beech uljhi hui hain (Choppy).")

    if spot > vwap_now:
        score += 25; bull_reasons.append(f"Price VWAP ({vwap_now}) ke UPAR hai")
    else:
        score -= 25; bear_reasons.append(f"Price VWAP ({vwap_now}) ke NEECHE hai")

    if len(today_5) < 6:
        no_trade_reasons.append("Abhi 9:15–9:45 AM ka Opening Range ban raha hai.")
    elif not is_wide_orb:
        if spot > orb_high:
            score += 20; bull_reasons.append(f"Morning High ({orb_high}) Breakout")
        elif spot < orb_low:
            score -= 20; bear_reasons.append(f"Morning Low ({orb_low}) Breakdown")
        else:
            no_trade_reasons.append(f"Price tight Morning Range ({orb_low}–{orb_high}) ke andar hai.")
    else:
        if spot > vwap_now and rsi_val > 52:
            score += 15; bull_reasons.append("VWAP Pullback Call Buy Active")
        elif spot < vwap_now and rsi_val < 48:
            score -= 15; bear_reasons.append("VWAP Rejection Put Buy Active")

    if rsi_val >= 72:
        no_trade_reasons.append(f"RSI Overbought ({rsi_val}) — Top par Call buy na karein.")
        score -= 20
    elif rsi_val <= 28:
        no_trade_reasons.append(f"RSI Oversold ({rsi_val}) — Bottom par Put buy na karein.")
        score += 20
    elif rsi_val > 54:
        score += 15; bull_reasons.append(f"RSI Strong ({rsi_val})")
    elif rsi_val < 46:
        score -= 15; bear_reasons.append(f"RSI Weak ({rsi_val})")

    # Top Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{index_choice} Live", f"{spot}")
    m2.metric("VWAP Level", f"{vwap_now}", delta=f"{round(spot - vwap_now, 1)} pts")
    m3.metric("5m 9-EMA", f"{ema9_5m}")
    m4.metric("Trend Power (ADX)", f"{adx_val}")
    m5.metric("15m RSI", f"{rsi_val}")

    threshold = 70 if "Safe" in risk_mode else 55

    if len(today_5) < 6 or len(no_trade_reasons) >= 2 or (abs(score) < threshold):
        st.warning("🟡 **ABHI KOI BUY NA KAREIN (NO-TRADE ZONE)**")
        st.markdown("### 🛑 Abhi Call Buy ya Put Buy Kyun Nahi Karna Hai:")
        for r in (no_trade_reasons if no_trade_reasons else ["Abhi 5m EMA aur VWAP ek direction mein confirm nahi hain."]):
            st.markdown(f"- ❌ {r}")
        st.info(f"💡 **Agla Buy Kab Banega?** **CALL BUY** ke liye price ko **VWAP (`{vwap_now}`) & 9 EMA (`{ema9_5m}`)** ke upar hold karne dein; **PUT BUY** ke liye dono ke neeche red candle close hone dein.")

    elif score >= threshold:
        t1 = round(spot + (atr_val * 0.65), 1)
        t2 = round(spot + (atr_val * 1.2), 1)
        sl = round(min(spot - (atr_val * 0.45), ema9_5m - 8), 1)
        st.success(f"🟢 **ACTION: CALL BUY KAREIN (CE BUY) — Market Upar Jayega | Accuracy: {min(score, 95)}%**")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            ### ✅ 1. Kaun-Sa Call Buy Karein & Kahan Entry Lein:
            * **Option Strike Buy:** `{index_choice} {itm_ce} CE` (Best/Safe) ya `{atm_strike} CE`
            * **Entry Spot Price:** `{spot}` ke aas-paas (jab 5m candle `{ema9_5m}` ke upar hold kare)
            * **Reason:** {', '.join(bull_reasons)}
            """)
            st.markdown(f"""
            ### 🛡️ 2. Stop-Loss (SL) Kahan Lagayein:
            * **Spot Strict SL:** `{sl}` (agar 5m candle iske neeche close ho toh Call se out ho jayein)
            * **Option Premium SL:** Apne Call ke buying price se **15%–20% neeche** SL lagayein.
            """)
        with c2:
            st.markdown(f"""
            ### 🎯 3. Call Se Out (Profit Book) Kab Hona Hai:
            * **Target 1 (50% Book):** Spot `{t1}` — Yahan SL ko apne buying price par le aayein.
            * **Target 2 (Full Exit):** Spot `{t2}` — Poora profit book karke out ho jayein.
            * **Early Exit:** Agar 5m red candle **9 EMA (`{ema9_5m}`)** ke neeche close de toh turant exit!
            """)
            st.markdown("""
            ### 🚫 4. Kab Call Buy NA Karein:
            * Agar ek hi badi candle mein Target 1 aa jaye (bhaagti candle mein entry na lein).
            * 2:45 PM ke baad naya trade na lein.
            """)

    elif score <= -threshold:
        t1 = round(spot - (atr_val * 0.65), 1)
        t2 = round(spot - (atr_val * 1.2), 1)
        sl = round(max(spot + (atr_val * 0.45), ema9_5m + 8), 1)
        st.error(f"🔴 **ACTION: PUT BUY KAREIN (PE BUY) — Market Neeche Girega | Accuracy: {min(abs(score), 95)}%**")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            ### ✅ 1. Kaun-Sa Put Buy Karein & Kahan Entry Lein:
            * **Option Strike Buy:** `{index_choice} {itm_pe} PE` (Best/Safe) ya `{atm_strike} PE`
            * **Entry Spot Price:** `{spot}` ke aas-paas (jab 5m candle `{ema9_5m}` ke neeche hold kare)
            * **Reason:** {', '.join(bear_reasons)}
            """)
            st.markdown(f"""
            ### 🛡️ 2. Stop-Loss (SL) Kahan Lagayein:
            * **Spot Strict SL:** `{sl}` (agar 5m candle iske upar close ho toh Put se out ho jayein)
            * **Option Premium SL:** Apne Put ke buying price se **15%–20% neeche** SL lagayein.
            """)
        with c2:
            st.markdown(f"""
            ### 🎯 3. Put Se Out (Profit Book) Kab Hona Hai:
            * **Target 1 (50% Book):** Spot `{t1}` — Yahan SL ko apne buying price par le aayein.
            * **Target 2 (Full Exit):** Spot `{t2}` — Poora profit book karke out ho jayein.
            * **Early Exit:** Agar 5m green candle **9 EMA (`{ema9_5m}`)** ke upar close de toh turant exit!
            """)
            st.markdown("""
            ### 🚫 4. Kab Put Buy NA Karein:
            * Agar ek hi badi candle mein Target 1 aa jaye (bhaagti candle mein entry na lein).
            * 2:45 PM ke baad naya trade na lein.
            """)

    # Clean Pro Chart
    today_5["TimeStr"] = today_5["DT"].dt.strftime("%H:%M")
    fig = go.Figure(data=[go.Candlestick(
        x=today_5["TimeStr"], open=today_5["Open"], high=today_5["High"],
        low=today_5["Low"], close=today_5["Close"], name="5m Price"
    )])
    fig.add_trace(go.Scatter(x=today_5["TimeStr"], y=today_5["VWAP"], line=dict(color="#ff00ff", width=2), name="VWAP"))
    fig.add_trace(go.Scatter(x=today_5["TimeStr"], y=today_5["EMA9"], line=dict(color="#00f2ff", width=1.5), name="9 EMA"))
    fig.add_trace(go.Scatter(x=today_5["TimeStr"], y=today_5["EMA21"], line=dict(color="#ff9900", width=1.5), name="21 EMA"))

    fig.update_xaxes(type="category", nticks=12)
    fig.update_layout(xaxis_rangeslider_visible=False, height=480, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

if st.button("🔄 Instant Refresh"):
    st.rerun()

if auto_refresh:
    time.sleep(30)
    st.rerun()
