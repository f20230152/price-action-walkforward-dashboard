from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from price_action_engine import (
    backtest_strategy,
    compute_metrics,
    daily_close,
    discover_csv_files,
    evaluate_grid,
    load_second_prices,
    parameter_grid,
    parse_number_list,
    run_walk_forward,
    StrategyParams,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = APP_DIR / "data" if (APP_DIR / "data").exists() else APP_DIR.parent


st.set_page_config(
    page_title="Price Action Walk-Forward Dashboard",
    page_icon="PA",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.45rem;}
    div[data-testid="stDataFrame"] {border: 1px solid #e6e8eb; border-radius: 6px;}
    .small-note {color: #5e6773; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "NA"
    if np.isinf(x):
        return "inf"
    return f"{x:,.{digits}f}"


def fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{x:.1%}"


@st.cache_data(show_spinner=False)
def cached_manifest(data_dir: str) -> pd.DataFrame:
    return discover_csv_files(data_dir)


@st.cache_data(show_spinner=True, persist=True)
def cached_prices(data_dir: str, start_date: str, end_date: str) -> tuple[pd.Series, pd.DataFrame]:
    return load_second_prices(
        data_dir,
        pd.to_datetime(start_date),
        pd.to_datetime(end_date),
        cache_dir=APP_DIR / ".price_cache",
    )


def make_grid_from_sidebar(direction: str) -> list[StrategyParams]:
    delays = parse_number_list(st.session_state["delay_values"], int)
    thresholds = parse_number_list(st.session_state["threshold_values"], float)
    lookbacks = parse_number_list(st.session_state["lookback_values"], int)
    holds = parse_number_list(st.session_state["hold_values"], int)
    return parameter_grid(delays, thresholds, lookbacks, holds, direction)


def metric_row(metrics: dict) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Net PnL", f"{fmt_num(metrics['total_pnl_cents'])} c")
    c2.metric("Daily Sharpe", fmt_num(metrics["daily_sharpe"]))
    c3.metric("Max DD", f"{fmt_num(metrics['max_drawdown_cents'])} c")
    c4.metric("Trades", f"{int(metrics['trades']):,}")
    c5.metric("Win Rate", fmt_pct(metrics["win_rate"]))
    c6.metric("Profit Factor", fmt_num(metrics["profit_factor"]))


def equity_chart(daily: pd.Series, title: str) -> go.Figure:
    equity = daily.fillna(0.0).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Net PnL"))
    fig.update_layout(
        title=title,
        yaxis_title="Cumulative cents per 1 contract",
        xaxis_title="Date",
        margin=dict(l=10, r=10, t=45, b=10),
        height=420,
    )
    return fig


def drawdown_chart(daily: pd.Series, title: str) -> go.Figure:
    equity = daily.fillna(0.0).cumsum()
    dd = equity - equity.cummax()
    fig = px.area(x=dd.index, y=dd.values, title=title, labels={"x": "Date", "y": "Drawdown cents"})
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=320)
    return fig


def main() -> None:
    st.title("Price Action Walk-Forward Dashboard")
    st.caption(
        "Rule: after price moves b cents within c seconds, enter after a seconds, hold for d seconds, "
        "and re-optimize a/b/c/d with a rolling walk-forward process."
    )

    with st.sidebar:
        st.header("Data")
        data_dir = st.text_input("CSV folder", value=str(DEFAULT_DATA_DIR))
        manifest = cached_manifest(data_dir)
        if manifest.empty:
            st.error("No daily CSV files named YYYY-MM-DD.csv were found.")
            st.stop()

        min_date = pd.to_datetime(manifest["date"].min()).date()
        max_date = pd.to_datetime(manifest["date"].max()).date()
        default_start = max(pd.Timestamp(max_date) - pd.DateOffset(days=45), pd.Timestamp(min_date)).date()
        start_date = st.date_input("Start date", default_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)
        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()

        st.header("Execution")
        cost_cents = st.number_input(
            "Round-trip cost in cents",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Applied once per completed trade, in price cents.",
        )
        direction = st.selectbox(
            "Trade direction",
            ["momentum", "fade"],
            help="Momentum buys after an up move and sells after a down move. Fade does the opposite.",
        )

        st.header("Grid")
        st.text_input("a delay seconds", value="0,1", key="delay_values")
        st.text_input("b threshold cents", value="2,5", key="threshold_values")
        st.text_input("c move window seconds", value="10,30", key="lookback_values")
        st.text_input("d hold seconds", value="30,60", key="hold_values")
        min_train_trades = st.number_input("Minimum train trades", min_value=0, value=20, step=5)
        objective_label = st.selectbox("Optimization objective", ["Daily Sharpe", "Total PnL", "Profit Factor"])
        objective = {
            "Daily Sharpe": "daily_sharpe",
            "Total PnL": "total_pnl_cents",
            "Profit Factor": "profit_factor",
        }[objective_label]

        st.header("Walk Forward")
        train_months = st.number_input("Train months", min_value=1, max_value=12, value=3, step=1)
        test_months = st.number_input("Test months", min_value=1, max_value=6, value=1, step=1)

    with st.spinner("Loading and normalizing tick files to 1-second prices..."):
        price, file_stats = cached_prices(data_dir, str(start_date), str(end_date))
    if price.empty:
        st.error("No valid price rows were loaded for the selected date range.")
        st.stop()

    params_list = make_grid_from_sidebar(direction)
    if not params_list:
        st.error("Parameter grid is empty. Check the comma-separated values in the sidebar.")
        st.stop()

    combo_count = len(params_list)
    if combo_count > 100:
        st.warning(f"The grid has {combo_count:,} combinations. Reduce it if the run feels slow.")

    close = daily_close(price)
    loaded_days = int(price.index.normalize().nunique())
    selected_files = file_stats[file_stats["raw_rows"] > 0].copy() if not file_stats.empty else file_stats

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Files Loaded", f"{len(selected_files):,}")
    k2.metric("Trading Days", f"{loaded_days:,}")
    k3.metric("1s Rows", f"{len(price):,}")
    k4.metric("Price Range", f"{fmt_num(price.min())} - {fmt_num(price.max())}")
    k5.metric("Grid Combos", f"{combo_count:,}")

    tabs = st.tabs(["Overview", "Single Strategy Lab", "Walk-Forward Optimizer", "Data Diagnostics"])

    with tabs[0]:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.line(
                close.reset_index(),
                x="index",
                y="close",
                title="Daily Close From Tick Files",
                labels={"index": "Date", "close": "Price"},
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            returns = close.pct_change().dropna()
            overview = pd.DataFrame(
                [
                    ["First tick", str(price.index.min())],
                    ["Last tick", str(price.index.max())],
                    ["Daily close return", fmt_pct(close.iloc[-1] / close.iloc[0] - 1)],
                    ["Daily vol", fmt_pct(returns.std() * np.sqrt(252)) if len(returns) else "NA"],
                    ["Skipped/empty files", f"{int((file_stats['raw_rows'] == 0).sum()):,}"],
                ],
                columns=["Item", "Value"],
            )
            st.dataframe(overview, use_container_width=True, hide_index=True)
            st.markdown(
                "<div class='small-note'>PnL is reported in price cents per one notional contract. "
                "No slippage is assumed unless entered as round-trip cost.</div>",
                unsafe_allow_html=True,
            )

    with tabs[1]:
        st.subheader("Single Parameter Backtest")
        p1, p2, p3, p4 = st.columns(4)
        delay_s = p1.number_input("a delay", min_value=0, value=1, step=1)
        threshold_cents = p2.number_input("b cents", min_value=0.1, value=2.0, step=0.1)
        lookback_s = p3.number_input("c seconds", min_value=1, value=30, step=1)
        hold_s = p4.number_input("d seconds", min_value=1, value=60, step=1)
        params = StrategyParams(int(delay_s), float(threshold_cents), int(lookback_s), int(hold_s), direction)

        with st.spinner("Running single strategy backtest..."):
            trades, daily = backtest_strategy(price, params, cost_cents)
            metrics = compute_metrics(daily, trades)
        metric_row(metrics)
        left, right = st.columns(2)
        left.plotly_chart(equity_chart(daily, "Single Strategy Equity"), use_container_width=True)
        right.plotly_chart(drawdown_chart(daily, "Single Strategy Drawdown"), use_container_width=True)

        st.markdown("**Recent Trades**")
        if trades.empty:
            st.info("No trades fired for this parameter set.")
        else:
            show_cols = [
                "signal_time",
                "entry_time",
                "exit_time",
                "side",
                "entry_price",
                "exit_price",
                "net_pnl_cents",
            ]
            st.dataframe(trades[show_cols].tail(250), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Monthly Walk-Forward Optimization")
        run = st.button("Run walk-forward optimization", type="primary")
        if not run:
            st.info("Use the sidebar to set the grid, then run the walk-forward optimizer.")
        else:
            with st.spinner("Optimizing each training window and stitching out-of-sample months..."):
                details, oos_daily, oos_trades, train_grid = run_walk_forward(
                    price=price,
                    params_list=params_list,
                    train_months=int(train_months),
                    test_months=int(test_months),
                    objective=objective,
                    cost_cents=float(cost_cents),
                    min_train_trades=int(min_train_trades),
                )

            if details.empty or oos_daily.empty:
                st.warning("No walk-forward periods produced trades. Try lower thresholds or a wider date range.")
            else:
                oos_metrics = compute_metrics(oos_daily, oos_trades)
                metric_row(oos_metrics)
                l1, l2 = st.columns(2)
                l1.plotly_chart(equity_chart(oos_daily, "Walk-Forward OOS Equity"), use_container_width=True)
                l2.plotly_chart(drawdown_chart(oos_daily, "Walk-Forward OOS Drawdown"), use_container_width=True)

                st.markdown("**Rebalance Decisions**")
                st.dataframe(details, use_container_width=True, hide_index=True)

                selected_counts = details["selected_params"].value_counts().reset_index()
                selected_counts.columns = ["selected_params", "months_selected"]
                bar = px.bar(
                    selected_counts,
                    x="months_selected",
                    y="selected_params",
                    orientation="h",
                    title="Selected Parameter Stability",
                )
                bar.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(bar, use_container_width=True)

                st.markdown("**Training Grid Top Rows**")
                if not train_grid.empty:
                    st.dataframe(
                        train_grid.sort_values("score", ascending=False).head(200),
                        use_container_width=True,
                        hide_index=True,
                    )

                if not oos_trades.empty:
                    st.download_button(
                        "Download OOS trades CSV",
                        data=oos_trades.to_csv(index=False).encode("utf-8"),
                        file_name="price_action_oos_trades.csv",
                        mime="text/csv",
                    )
                    st.download_button(
                        "Download rebalance decisions CSV",
                        data=details.to_csv(index=False).encode("utf-8"),
                        file_name="price_action_walkforward_decisions.csv",
                        mime="text/csv",
                    )

    with tabs[3]:
        st.subheader("Loaded File Diagnostics")
        st.dataframe(file_stats, use_container_width=True, hide_index=True)

        st.subheader("Standalone Grid Scan On Full Selected Sample")
        if st.button("Run full-sample grid scan"):
            with st.spinner("Evaluating every parameter combination on the selected sample..."):
                grid = evaluate_grid(price, params_list, objective, cost_cents, int(min_train_trades))
            st.dataframe(grid.drop(columns=["params"]).head(250), use_container_width=True, hide_index=True)
            st.download_button(
                "Download full-sample grid CSV",
                data=grid.drop(columns=["params"]).to_csv(index=False).encode("utf-8"),
                file_name="price_action_full_sample_grid.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
