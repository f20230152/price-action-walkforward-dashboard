from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from price_action_engine import SpikeParams, backtest_spike_strategy, compute_metrics, load_second_prices


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
OUT_DIR = APP_DIR / "outputs" / "clean_walkforward"
REPORT_XLSX = OUT_DIR / "walkforward_report.xlsx"
SINGLE_DEFAULT_TRADES = OUT_DIR / "single_default_trades.csv"
SINGLE_DEFAULT_DAILY = OUT_DIR / "single_default_daily.csv"
SINGLE_DEFAULT_METRICS = OUT_DIR / "single_default_metrics.csv"
DEFAULT_SINGLE_PARAMS = SpikeParams(
    delay_s=120,
    threshold_cents=75,
    lookback_s=1800,
    hold_s=21600,
    volume_window_s=300,
    volume_multiple=0.0,
    direction="momentum",
)
DEFAULT_COST_CENTS = 4.0
DEFAULT_MAX_TRADES_PER_DAY = 1


st.set_page_config(
    page_title="Clean Walk-Forward Dashboard",
    page_icon="WF",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def fmt_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):,.{digits}f}"


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.1%}"


@st.cache_data(show_spinner=False)
def load_outputs() -> dict[str, pd.DataFrame]:
    files = {
        "metrics": "walkforward_metrics.csv",
        "decisions": "walkforward_decisions_all.csv",
        "trades": "walkforward_trades_all.csv",
        "daily": "walkforward_daily_pnl_all.csv",
        "config": "research_config.csv",
        "universe": "parameter_universe.csv",
        "rankings": "walkforward_train_rankings_all.csv",
    }
    out = {}
    for key, name in files.items():
        path = OUT_DIR / name
        out[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return out


@st.cache_data(show_spinner=False)
def load_default_single_backtest() -> tuple[pd.DataFrame, pd.Series, dict, str] | None:
    if not (SINGLE_DEFAULT_TRADES.exists() and SINGLE_DEFAULT_DAILY.exists() and SINGLE_DEFAULT_METRICS.exists()):
        return None
    trades = pd.read_csv(SINGLE_DEFAULT_TRADES)
    for col in ["signal_time", "entry_time", "exit_time"]:
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col])
    daily_df = pd.read_csv(SINGLE_DEFAULT_DAILY, index_col=0, parse_dates=True)
    daily = daily_df.iloc[:, 0].rename("daily_pnl_cents") if not daily_df.empty else pd.Series(dtype=float)
    metrics_row = pd.read_csv(SINGLE_DEFAULT_METRICS).iloc[0]
    label = str(metrics_row["params"])
    metrics = metrics_row.drop(labels=["params"], errors="ignore").to_dict()
    return trades, daily, metrics, label


@st.cache_data(show_spinner=False)
def load_price_bars() -> pd.DataFrame:
    bars, _ = load_second_prices(
        DATA_DIR,
        "2025-10-01",
        "2026-03-26",
        cache_dir=APP_DIR / ".price_cache",
        workers=6,
    )
    return bars


@st.cache_data(show_spinner=False)
def run_single_backtest(
    delay_s: int,
    threshold_cents: float,
    lookback_s: int,
    hold_s: int,
    cost_cents: float,
    max_trades_per_day: int,
    volume_multiple: float,
    volume_window_s: int,
    direction: str,
) -> tuple[pd.DataFrame, pd.Series, dict, str]:
    bars = load_price_bars()
    params = SpikeParams(
        delay_s=int(delay_s),
        threshold_cents=float(threshold_cents),
        lookback_s=int(lookback_s),
        hold_s=int(hold_s),
        volume_window_s=int(volume_window_s),
        volume_multiple=float(volume_multiple),
        direction=str(direction),
    )
    cap = None if int(max_trades_per_day) <= 0 else int(max_trades_per_day)
    trades, daily = backtest_spike_strategy(bars, params, float(cost_cents), cap)
    metrics = compute_metrics(daily, trades)
    return trades, daily, metrics, params.label


def metric_tiles(row: pd.Series) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("WF Net PnL", f"{fmt_num(row['total_pnl_cents'])} c")
    c2.metric("WF Sharpe", fmt_num(row["daily_sharpe"]))
    c3.metric("Max DD", f"{fmt_num(row['max_drawdown_cents'])} c")
    c4.metric("Trades", f"{int(row['trades']):,}")
    c5.metric("Trades/Day", fmt_num(row["trades_per_day"], 2))
    c6.metric("Profit Factor", fmt_num(row["profit_factor"]))


def single_metric_tiles(metrics: dict) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Net PnL", f"{fmt_num(metrics['total_pnl_cents'])} c")
    c2.metric("Sharpe", fmt_num(metrics["daily_sharpe"]))
    c3.metric("Max DD", f"{fmt_num(metrics['max_drawdown_cents'])} c")
    c4.metric("Trades", f"{int(metrics['trades']):,}")
    c5.metric("Trades/Day", fmt_num(metrics["trades_per_day"], 2))
    c6.metric("Profit Factor", fmt_num(metrics["profit_factor"]))


def equity_figure(daily: pd.DataFrame) -> go.Figure:
    df = daily.copy()
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    fig = go.Figure()
    for col in df.columns[1:]:
        fig.add_trace(go.Scatter(x=df[date_col], y=df[col].fillna(0).cumsum(), mode="lines", name=col))
    fig.update_layout(
        title="Stitched Walk-Forward OOS Equity",
        xaxis_title="Date",
        yaxis_title="Cumulative cents",
        height=430,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def single_equity_figure(daily: pd.Series, title: str) -> go.Figure:
    df = daily.fillna(0.0).rename("daily_pnl_cents").to_frame()
    df["equity_cents"] = df["daily_pnl_cents"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["equity_cents"], mode="lines", name="Equity"))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative cents",
        height=420,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def render_walk_forward_tab(data: dict[str, pd.DataFrame]) -> None:
    st.caption(
        "Parameters are selected using the prior 3 months and traded on the next unseen month. "
        "This is the primary walk-forward view."
    )
    metrics = data["metrics"]
    decisions = data["decisions"]
    trades = data["trades"]
    daily = data["daily"]
    config = data["config"]
    universe = data["universe"]
    rankings = data["rankings"]

    if metrics.empty or decisions.empty:
        st.error("Clean walk-forward outputs are missing. Run `python clean_walkforward_research.py` first.")
        st.stop()

    best = metrics.sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).iloc[0]
    st.subheader("Primary Result")
    st.success(
        "Recommended selector: optimize parameters by training Sharpe, then trade the selected parameters next month."
        if best["selector_objective"] == "daily_sharpe"
        else f"Recommended selector: {best['selector_objective']}"
    )
    metric_tiles(best)

    cfg_cols = st.columns(5)
    if not config.empty:
        cfg = config.iloc[0]
        cfg_cols[0].metric("Candidate Universe", f"{int(cfg['candidate_count']):,}")
        cfg_cols[1].metric("Train Window", f"{int(cfg['train_months'])}M")
        cfg_cols[2].metric("Test/Rebalance", f"{int(cfg['test_months'])}M")
        cfg_cols[3].metric("Round Trip Cost", f"{fmt_num(cfg['cost_cents_round_trip'])} c")
        cfg_cols[4].metric("Trade Cap", f"{int(cfg['max_trades_per_day'])}/day")

    if REPORT_XLSX.exists():
        st.download_button(
            "Download Excel Report",
            data=REPORT_XLSX.read_bytes(),
            file_name="walkforward_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()
    st.subheader("Selector Comparison")
    show_metrics = metrics[
        [
            "selector_objective",
            "total_pnl_cents",
            "daily_sharpe",
            "max_drawdown_cents",
            "trades",
            "trades_per_day",
            "win_rate",
            "profit_factor",
        ]
    ].sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False)
    st.dataframe(show_metrics, use_container_width=True, hide_index=True)

    if not daily.empty:
        st.plotly_chart(equity_figure(daily), use_container_width=True)

    st.subheader("Monthly Rebalance Decisions")
    selector = st.selectbox(
        "Selector objective",
        options=show_metrics["selector_objective"].tolist(),
        index=0,
    )
    selected_decisions = decisions[decisions["selector_objective"] == selector].copy()
    st.dataframe(selected_decisions, use_container_width=True, hide_index=True)

    if not selected_decisions.empty:
        month_fig = px.bar(
            selected_decisions,
            x="test_start",
            y="test_total_pnl_cents",
            color="test_total_pnl_cents",
            color_continuous_scale="RdYlGn",
            title=f"Monthly OOS PnL: {selector}",
            labels={"test_start": "Test month", "test_total_pnl_cents": "PnL cents"},
        )
        month_fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(month_fig, use_container_width=True)

    st.subheader("Trades Taken")
    selected_trades = trades[trades["selector_objective"] == selector].copy() if not trades.empty else pd.DataFrame()
    if selected_trades.empty:
        st.info("No trades were taken for this selector.")
    else:
        display_cols = [
            "test_month",
            "signal_time",
            "entry_time",
            "exit_time",
            "side",
            "move_cents",
            "signal_volume",
            "entry_price",
            "exit_price",
            "gross_pnl_cents",
            "cost_cents",
            "net_pnl_cents",
            "selected_params",
        ]
        st.dataframe(selected_trades[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download Trades CSV",
            data=selected_trades.to_csv(index=False).encode("utf-8"),
            file_name=f"walkforward_trades_{selector}.csv",
            mime="text/csv",
        )

    with st.expander("Parameter Universe And Train Rankings"):
        st.markdown("The universe below was defined before the monthly tests. Each test month only uses its own prior 3-month train window.")
        st.dataframe(universe, use_container_width=True, hide_index=True)
        st.markdown("Top train-window rankings saved during each rebalance:")
        st.dataframe(rankings, use_container_width=True, hide_index=True)


def render_single_backtest_tab() -> None:
    st.caption(
        "This tab is a diagnostic single-parameter backtest over the full available data period. "
        "It is not used to select the walk-forward result."
    )

    with st.form("single_backtest_params"):
        st.markdown("Default shown: Shubham set from the latest screenshot.")
        r1 = st.columns(4)
        delay_s = r1[0].number_input("Entry delay a (seconds)", min_value=0, value=DEFAULT_SINGLE_PARAMS.delay_s, step=1)
        threshold_cents = r1[1].number_input(
            "Move threshold (cents)",
            min_value=0.01,
            value=float(DEFAULT_SINGLE_PARAMS.threshold_cents),
            step=1.0,
        )
        lookback_s = r1[2].number_input("Move window (seconds)", min_value=1, value=DEFAULT_SINGLE_PARAMS.lookback_s, step=1)
        hold_s = r1[3].number_input("Hold time (seconds)", min_value=1, value=DEFAULT_SINGLE_PARAMS.hold_s, step=1)

        r2 = st.columns(4)
        cost_cents = r2[0].number_input("Round-trip cost/slippage (cents)", min_value=0.0, value=DEFAULT_COST_CENTS, step=0.5)
        max_trades_per_day = r2[1].number_input(
            "Max trades per day",
            min_value=0,
            value=DEFAULT_MAX_TRADES_PER_DAY,
            step=1,
            help="Set 0 for unlimited trades per day.",
        )
        volume_multiple = r2[2].number_input(
            "Volume multiple filter",
            min_value=0.0,
            value=float(DEFAULT_SINGLE_PARAMS.volume_multiple),
            step=0.5,
            help="0 means volume filter off.",
        )
        direction = r2[3].selectbox("Direction", options=["momentum", "fade"], index=0)

        volume_window_s = st.number_input(
            "Volume baseline window (seconds)",
            min_value=1,
            value=DEFAULT_SINGLE_PARAMS.volume_window_s,
            step=1,
        )
        submitted = st.form_submit_button("Run Single Backtest")

    if not submitted and "single_backtest_ready" not in st.session_state:
        default_result = load_default_single_backtest()
        if default_result is None:
            st.info("Click Run Single Backtest to calculate the whole-period trade list for these parameters.")
            return
        trades, daily, metrics, label = default_result
        st.info("Showing the saved whole-period run for Shubham's default parameters. Click Run Single Backtest after changing inputs.")
    else:
        with st.spinner("Running whole-period single-parameter backtest..."):
            trades, daily, metrics, label = run_single_backtest(
                int(delay_s),
                float(threshold_cents),
                int(lookback_s),
                int(hold_s),
                float(cost_cents),
                int(max_trades_per_day),
                float(volume_multiple),
                int(volume_window_s),
                direction,
            )
        st.session_state["single_backtest_ready"] = True

    st.subheader("Single Backtest Result")
    st.code(label)
    single_metric_tiles(metrics)

    if daily.empty:
        st.warning("No days were available for this backtest.")
        return

    st.plotly_chart(single_equity_figure(daily, "Whole-Period Single Backtest Equity"), use_container_width=True)

    monthly = daily.fillna(0.0).groupby(daily.index.to_period("M")).sum().rename("pnl_cents").reset_index()
    monthly["date"] = monthly["date"].astype(str)
    month_fig = px.bar(
        monthly,
        x="date",
        y="pnl_cents",
        color="pnl_cents",
        color_continuous_scale="RdYlGn",
        title="Monthly PnL For Selected Single Parameters",
        labels={"date": "Month", "pnl_cents": "PnL cents"},
    )
    month_fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(month_fig, use_container_width=True)

    st.subheader("Whole-Period Trade List")
    if trades.empty:
        st.info("No trades were taken for these parameters over the full period.")
        return

    out_trades = trades.copy()
    out_trades["side_label"] = out_trades["side"].map({1: "LONG", -1: "SHORT"})
    out_trades["params"] = label
    display_cols = [
        "signal_time",
        "entry_time",
        "exit_time",
        "side_label",
        "move_cents",
        "signal_volume",
        "entry_price",
        "exit_price",
        "gross_pnl_cents",
        "cost_cents",
        "net_pnl_cents",
        "params",
    ]
    st.dataframe(out_trades[display_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Download Single Backtest Trades CSV",
        data=out_trades[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="single_parameter_whole_period_trades.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Price Action Research Dashboard")
    data = load_outputs()

    walk_forward_tab, single_tab = st.tabs(["Walk-Forward", "Single Parameter Backtest"])
    with walk_forward_tab:
        render_walk_forward_tab(data)
    with single_tab:
        render_single_backtest_tab()


if __name__ == "__main__":
    main()
