from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    build_signal_table,
    historical_risk,
    optimize_strategy,
    portfolio_positions,
    portfolio_summary,
    rolling_stress,
)
from database import (
    get_trades,
    initialize_database,
    place_order,
    portfolio_state,
)
from market_data import ASSETS, build_market_data, market_timestamp
from live_data import fetch_historical_data

st.set_page_config(
    page_title="Atlas AI Trading",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #080d17;
        --panel: #101827;
        --panel-soft: #121e2e;
        --line: #223047;
        --text: #edf3fb;
        --muted: #8291a7;
        --mint: #64e0b4;
        --red: #ff7785;
        --blue: #8bb9ff;
    }
    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stHeader"] { background: rgba(8,13,23,.88); }
    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.3rem; }
    .block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { color: var(--text) !important; letter-spacing: -.035em; }
    p, label, [data-testid="stMarkdownContainer"] { color: #c6d1e0; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #111c2b, #0e1724);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 17px 18px;
    }
    [data-testid="stMetricLabel"] { color: #91a1b7; }
    [data-testid="stMetricValue"] { color: #f2f6fc; }
    [data-testid="stMetricDelta"] { font-size: .88rem; }
    [data-testid="stTabs"] button { color: #aebbd0; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
    div[data-testid="stForm"] {
        background: #101827; border: 1px solid var(--line);
        padding: 18px; border-radius: 14px;
    }
    .brand-lockup { display:flex; align-items:center; gap:11px; padding: 2px 4px 22px; }
    .brand-mark {
        display:grid; place-items:center; width:37px; height:37px; border-radius:12px;
        color:#08121b; background:linear-gradient(145deg,#8cf0cd,#43c69b);
        font-weight:900; font-size:19px;
    }
    .brand-name { color:#f1f6fc; font-weight:750; font-size:15px; letter-spacing:-.025em; }
    .brand-sub { color:#77869d; font-size:10px; letter-spacing:.14em; margin-top:2px; }
    .eyebrow { color:#7f91aa; font-size:11px; letter-spacing:.15em; text-transform:uppercase; }
    .page-title { font-size:34px; line-height:1.1; font-weight:750; color:#f2f6fc; margin:4px 0 7px; letter-spacing:-.05em; }
    .page-subtitle { color:#91a0b5; font-size:14px; margin-bottom:24px; }
    .demo-banner {
        border:1px solid #4f4b2c; background:#211e14; color:#e9d797;
        padding:10px 13px; border-radius:10px; font-size:12px; margin: 4px 0 19px;
    }
    .section-title { color:#e9f0f8; font-size:17px; font-weight:650; margin: 7px 0 12px; }
    .subtle { color:#8090a7; font-size:12px; }
    .signal-pill {
        display:inline-block; padding:5px 9px; border-radius:20px; font-size:11px;
        font-weight:700; letter-spacing:.04em;
    }
    .signal-buy { background:#12372d; color:#77e3b8; }
    .signal-sell { background:#3a2029; color:#ff8792; }
    .signal-hold { background:#282c34; color:#bbc5d3; }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius:9px; border:1px solid #3b806b; background:#64e0b4;
        color:#07140f; font-weight:700;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        border-color:#94f1d0; background:#83ebc5; color:#07140f;
    }
    hr { border-color: var(--line) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def chart_layout(fig: go.Figure, *, height: int = 330) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aebbd0", family="Inter, sans-serif", size=11),
        height=height,
        margin=dict(l=10, r=14, t=20, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(showgrid=False, linecolor="#263449", tickfont=dict(color="#7f90a7")),
        yaxis=dict(gridcolor="#1b2738", zerolinecolor="#263449", tickfont=dict(color="#7f90a7")),
        hoverlabel=dict(bgcolor="#111c2b", bordercolor="#31445c", font_color="#edf3fb"),
    )
    return fig


def page_header(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">ATLAS RESEARCH TERMINAL</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def demo_banner(message: str = "DEMO DATA — simulated market history. Not live prices or real investment performance.") -> None:
    st.markdown(f'<div class="demo-banner">{message}</div>', unsafe_allow_html=True)


def money(value: float) -> str:
    return f"${value:,.2f}"


def render_overview(market_data: dict[str, pd.DataFrame], account: dict) -> None:
    page_header("Portfolio overview", "A clear view of your simulated account, positions, and research watchlist.")
    data_notice("PAPER ACCOUNT • $100,000 virtual starting balance. No real trades.")

    summary = portfolio_summary(account["cash"], account["positions"], market_data)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Account equity", money(summary["equity"]), "paper portfolio")
    metric_columns[1].metric("Available cash", money(summary["cash"]))
    metric_columns[2].metric("Open positions", f"{len(account['positions'])}")
    pnl = summary["total_pnl"]
    metric_columns[3].metric("Paper P&L", money(pnl), delta=f"{pnl / 100000 * 100:+.2f}% of start")

    left, right = st.columns([1.45, 1])
    with left:
        st.markdown('<div class="section-title">Portfolio allocation</div>', unsafe_allow_html=True)
        position_rows = portfolio_positions(account["positions"], market_data)
        if position_rows:
            allocation = pd.DataFrame(position_rows)
            fig = go.Figure(
                go.Pie(
                    labels=allocation["Symbol"],
                    values=allocation["Market value"],
                    hole=0.69,
                    sort=False,
                    marker=dict(colors=["#64e0b4", "#86aef8", "#a78bfa", "#f4bd70", "#ff7785", "#4fd1c5", "#d2a8ff", "#a6b5c8"]),
                    textinfo="label+percent",
                    textfont=dict(size=11),
                    hovertemplate="%{label}<br>%{percent}<br>$%{value:,.2f}<extra></extra>",
                )
            )
            chart_layout(fig, height=315)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Your portfolio is all cash. Place a paper order to see allocation here.")

    with right:
        st.markdown('<div class="section-title">Watchlist</div>', unsafe_allow_html=True)
        rows = []
        for symbol, asset in ASSETS.items():
            frame = market_data[symbol]
            price = float(frame["Close"].iloc[-1])
            day_return = float(frame["Close"].pct_change().iloc[-1] * 100)
            rows.append({"Asset": symbol, "Name": asset["name"], "Last": money(price), "1D": f"{day_return:+.2f}%"})
        watch = pd.DataFrame(rows)
        st.dataframe(watch, use_container_width=True, hide_index=True, height=315)
        st.caption(f"Demo series generated through {market_timestamp(market_data)}.")

    st.markdown('<div class="section-title">Current positions</div>', unsafe_allow_html=True)
    rows = portfolio_positions(account["positions"], market_data)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions yet. Paper orders are available from the Paper Trading page.")


def render_markets(market_data: dict[str, pd.DataFrame]) -> None:
    page_header("Markets", "Explore historical or clearly labeled simulated daily candles.")
    data_notice()
    col_asset, col_range = st.columns([1, 1])
    symbols = list(market_data.keys())
    with col_asset:
        symbol = st.selectbox("Instrument", symbols, format_func=lambda item: f"{item} · {ASSETS[item]['name']}")
    with col_range:
        period = st.selectbox("History", ["3 months", "6 months", "1 year", "2 years"], index=2)
    days = {"3 months": 65, "6 months": 130, "1 year": 252, "2 years": 500}[period]
    frame = market_data[symbol].tail(days)
    change = float(frame["Close"].pct_change().iloc[-1] * 100)
    cols = st.columns(4)
    cols[0].metric("Last close", money(float(frame["Close"].iloc[-1])))
    cols[1].metric("Daily move", f"{change:+.2f}%")
    cols[2].metric("High · latest bar", money(float(frame["High"].iloc[-1])))
    cols[3].metric("Low · latest bar", money(float(frame["Low"].iloc[-1])))
    candles = go.Figure(
        data=[
            go.Candlestick(
                x=frame.index,
                open=frame["Open"],
                high=frame["High"],
                low=frame["Low"],
                close=frame["Close"],
                increasing_line_color="#64e0b4",
                decreasing_line_color="#ff7785",
                name=symbol,
            )
        ]
    )
    chart_layout(candles, height=440)
    candles.update_layout(xaxis_rangeslider_visible=False, title=f"{symbol} · daily candles")
    st.plotly_chart(candles, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"Source: {st.session_state.get('atlas_source', 'DEMO')} · Last bar: {frame.index[-1].strftime('%Y-%m-%d')} · Historical closing prices are not live executable quotes."
    )
    with st.expander("View sample OHLCV rows"):
        display = frame.tail(12).reset_index(names="Date")
        display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True, hide_index=True)


def render_signals(market_data: dict[str, pd.DataFrame]) -> None:
    page_header("AI signals", "A small, transparent classifier for research exploration — not an instruction to trade.")
    data_notice("Research model probabilities are not calibrated forecasts or trading advice.")
    with st.spinner("Training lightweight research models on the selected historical series…"):
        signals = build_signal_table(market_data)
    st.markdown('<div class="section-title">Latest research signals</div>', unsafe_allow_html=True)
    st.dataframe(signals, use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect an instrument", signals["Symbol"].tolist(), key="signal_symbol")
    signal = signals.loc[signals["Symbol"] == selected].iloc[0]
    probability_columns = ["P(BUY)", "P(HOLD)", "P(SELL)"]
    probabilities = [float(str(signal[key]).rstrip("%")) / 100 for key in probability_columns]
    chart = go.Figure(
        go.Bar(
            x=["BUY", "HOLD", "SELL"],
            y=probabilities,
            marker_color=["#64e0b4", "#7b8ba4", "#ff7785"],
            text=[f"{p:.1%}" for p in probabilities],
            textposition="outside",
        )
    )
    chart.update_yaxes(range=[0, 1], tickformat=".0%", title="Model class probability")
    chart_layout(chart, height=275)
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        """
        **How to read this:** the model uses a chronological train/test split and recent price-derived
        features to classify a five-session synthetic return as BUY, HOLD, or SELL. It is a learning
        example, not investment advice.

        **Limitations:** every price series here is simulated; probabilities are not calibrated;
        no fundamentals, news, liquidity, taxes, or portfolio constraints are modeled. A historical
        holdout score on synthetic data does not imply future performance.
        """
    )
    holdout_accuracy = float(str(signal["Holdout accuracy"]).rstrip("%")) / 100
    st.caption(f"Temporal holdout accuracy for {selected}: {holdout_accuracy:.1%} on synthetic observations.")


def render_risk(market_data: dict[str, pd.DataFrame], account: dict) -> None:
    page_header("Risk engine", "Historical risk measures for current holdings, plus stress tests drawn from demo history.")
    data_notice("RISK PROXY — constant holdings applied to historical prices; not realized account performance.")
    rows = portfolio_positions(account["positions"], market_data)
    equity = portfolio_summary(account["cash"], account["positions"], market_data)["equity"]
    risk = historical_risk(account["positions"], market_data)
    if not rows:
        st.info("Add a paper position to calculate portfolio-weighted risk. Market risk tables remain available below.")
    metrics = st.columns(4)
    metrics[0].metric("Annualized volatility", f"{risk['volatility']:.1%}" if risk else "—")
    metrics[1].metric("Maximum drawdown", f"{risk['max_drawdown']:.1%}" if risk else "—")
    metrics[2].metric("Invested exposure", money(equity - account["cash"]))
    metrics[3].metric("Positions / equity", f"{(equity - account['cash']) / equity:.1%}" if equity else "0.0%")

    if risk and risk["series"] is not None:
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=risk["series"].index, y=risk["series"].values, mode="lines", fill="tozeroy", line=dict(color="#64e0b4", width=2), name="Risk proxy"))
        chart.update_yaxes(title="Growth of $1 (rebased)")
        chart_layout(chart, height=300)
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
        st.caption("Proxy assumes today's asset weights were held constant throughout the displayed synthetic period.")

    st.markdown('<div class="section-title">Rolling stress windows</div>', unsafe_allow_html=True)
    stress = rolling_stress(account["positions"], market_data)
    if stress:
        st.dataframe(pd.DataFrame(stress), use_container_width=True, hide_index=True)
        st.caption("Worst observed rolling 1-, 5-, and 20-session returns in each synthetic price series, weighted by current holdings.")
    else:
        st.caption("Stress windows will be portfolio-weighted when holdings are available.")

    st.markdown('<div class="section-title">Exposure by instrument</div>', unsafe_allow_html=True)
    if rows:
        exposure = pd.DataFrame(rows)[["Symbol", "Quantity", "Market value", "Weight"]]
        st.dataframe(exposure, use_container_width=True, hide_index=True)
    else:
        st.caption("No invested exposure. Cash is not at market risk.")


def render_strategy(market_data: dict[str, pd.DataFrame]) -> None:
    page_header("Strategy lab", "Compare a moving-average strategy with a time-ordered holdout and explicit trading costs.")
    data_notice("BACKTEST — hypothetical results, with costs; not live trading performance.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        symbol = st.selectbox("Instrument", list(market_data), key="strategy_symbol")
    with col_b:
        split_pct = st.slider("Training window", 60, 80, 70, 5, format="%d%%")
    with col_c:
        fee_bps = st.number_input("Fee per side (bps)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    slip_bps = st.number_input("Slippage per side (bps)", min_value=0.0, max_value=100.0, value=3.0, step=1.0)

    result = optimize_strategy(market_data[symbol]["Close"], split_pct / 100, fee_bps, slip_bps)
    st.markdown(
        f"Training selected **{result['short']} / {result['long']} session** moving averages "
        f"from {result['train_start']} to {result['train_end']}. The selected parameters are frozen "
        f"for the holdout from {result['test_start']} to {result['test_end']}."
    )
    metrics = st.columns(4)
    metrics[0].metric("Holdout return", f"{result['test_return']:+.2%}")
    metrics[1].metric("Holdout volatility", f"{result['test_volatility']:.2%}")
    metrics[2].metric("Holdout max drawdown", f"{result['test_drawdown']:.2%}")
    metrics[3].metric("Trades · holdout", str(result["test_trades"]))
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=result["equity"].index, y=result["equity"], name="Strategy", line=dict(color="#64e0b4", width=2)))
    chart.add_trace(go.Scatter(x=result["benchmark"].index, y=result["benchmark"], name="Buy & hold", line=dict(color="#86aef8", width=1.5, dash="dot")))
    chart.update_yaxes(title="Growth of $1")
    chart_layout(chart, height=340)
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"Costs: {fee_bps:.1f} bps fees + {slip_bps:.1f} bps slippage on each side. "
        "Signals use prior-session information; the holdout period is not used to choose the parameters."
    )
    with st.expander("Parameter search (training period only)"):
        st.dataframe(pd.DataFrame(result["grid"]), use_container_width=True, hide_index=True)


def render_paper_trading(market_data: dict[str, pd.DataFrame], account: dict) -> None:
    page_header("Paper trading", "Practice order entry with a persistent virtual account. No broker connection or real-money trades.")
    data_notice("SIMULATION ONLY • Virtual balance $100,000. Fills use the latest available closing bar, NOT an executable market quote.")
    summary = portfolio_summary(account["cash"], account["positions"], market_data)
    metrics = st.columns(4)
    metrics[0].metric("Virtual cash", money(account["cash"]))
    metrics[1].metric("Position value", money(summary["market_value"]))
    metrics[2].metric("Paper equity", money(summary["equity"]))
    metrics[3].metric("Unrealized P&L", money(summary["unrealized_pnl"]))
    left, right = st.columns([1, 1.35])
    with left:
        st.markdown('<div class="section-title">Place a simulated order</div>', unsafe_allow_html=True)
        with st.form("paper_order", clear_on_submit=False):
            side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
            symbol = st.selectbox("Instrument", list(market_data), key="paper_symbol")
            price = float(market_data[symbol]["Close"].iloc[-1])
            st.caption(f"Reference close: {money(price)} · fractional quantity allowed")
            quantity = st.number_input("Quantity", min_value=0.0001, value=1.0, step=0.25, format="%.4f")
            submitted = st.form_submit_button(f"Place paper {side.lower()} order", use_container_width=True)
            if submitted:
                try:
                    order = place_order(side, symbol, float(quantity), price)
                    st.success(
                        f"{order['side']} {order['quantity']:.4f} {order['symbol']} filled at "
                        f"{money(order['fill_price'])}. Fee: {money(order['fees'])}."
                    )
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    with right:
        st.markdown('<div class="section-title">Open positions</div>', unsafe_allow_html=True)
        positions = portfolio_positions(account["positions"], market_data)
        if positions:
            st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions. Your virtual balance is available for simulated orders.")
    st.markdown('<div class="section-title">Transaction history</div>', unsafe_allow_html=True)
    trades = get_trades()
    if trades:
        history = pd.DataFrame(trades)
        history["Executed"] = pd.to_datetime(history["created_at"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
        history = history.rename(columns={"side": "Side", "symbol": "Symbol", "quantity": "Quantity", "fill_price": "Fill price", "fees": "Fee"})
        st.dataframe(history[["Executed", "Side", "Symbol", "Quantity", "Fill price", "Fee"]], use_container_width=True, hide_index=True)
    else:
        st.caption("No paper orders have been placed yet.")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_history():
    return fetch_historical_data(list(ASSETS))


def data_notice(message: str = "Historical data are delayed and may be incomplete. Never treat them as executable prices.") -> None:
    if st.session_state.get("atlas_source") == "DEMO":
        demo_banner("DEMO DATA — synthetic prices, not real market history. " + message)
    else:
        st.info("HISTORICAL DATA — Yahoo Finance via yfinance; personal research only. " + message)


initialize_database()
with st.sidebar:
    st.markdown("### Data source")
    source_choice = st.radio("Market data mode", ["Demo (offline)", "Historical (free, internet required)"], key="atlas_mode")
    if st.button("Refresh historical data"):
        cached_history.clear()
if source_choice.startswith("Historical"):
    try:
        market_data, failures = cached_history()
        if not market_data:
            st.error("No historical quotes available. Switch to Demo mode. No fabricated prices substituted.")
            st.stop()
        st.session_state["atlas_source"] = "HISTORICAL"
        if failures:
            st.warning("Unavailable instruments (excluded): " + ", ".join(failures))
    except Exception as exc:
        st.error(f"Historical data could not load: {exc}. Select Demo mode to continue.")
        st.stop()
else:
    market_data = build_market_data()
    st.session_state["atlas_source"] = "DEMO"
# Keep the existing asset menu consistent with the actual downloaded instruments.
active_assets = set(market_data)
account = portfolio_state(market_data)

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark">A</div>
          <div><div class="brand-name">ATLAS AI TRADING</div><div class="brand-sub">PORTFOLIO INTELLIGENCE</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Overview", "Markets", "AI Signals", "Risk Engine", "Strategy Lab", "Paper Trading"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="eyebrow">ENVIRONMENT</div>', unsafe_allow_html=True)
    st.markdown(f"**Paper account** · {st.session_state['atlas_source']} data")
    st.caption(f"Virtual cash · {money(account['cash'])}")
    st.caption(f"Updated · {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    st.divider()
    st.caption("Research prototype only. Not investment advice.")

if page == "Overview":
    render_overview(market_data, account)
elif page == "Markets":
    render_markets(market_data)
elif page == "AI Signals":
    render_signals(market_data)
elif page == "Risk Engine":
    render_risk(market_data, account)
elif page == "Strategy Lab":
    render_strategy(market_data)
else:
    render_paper_trading(market_data, account)
