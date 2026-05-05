from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from price_action_engine import (
    backtest_strategy,
    backtest_spike_strategy,
    compute_metrics,
    daily_close,
    discover_csv_files,
    evaluate_grid,
    evaluate_spike_grid,
    load_second_prices,
    move_frequency_grid,
    parameter_grid,
    parse_number_list,
    run_spike_walk_forward,
    run_walk_forward,
    spike_parameter_grid,
    SpikeParams,
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
def cached_prices(data_dir: str, start_date: str, end_date: str, cache_version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Net PnL", f"{fmt_num(metrics['total_pnl_cents'])} c")
    c2.metric("Daily Sharpe", fmt_num(metrics["daily_sharpe"]))
    c3.metric("Max DD", f"{fmt_num(metrics['max_drawdown_cents'])} c")
    c4.metric("Trades", f"{int(metrics['trades']):,}")
    c5.metric("Trades/Day", fmt_num(metrics.get("trades_per_day", 0.0), 1))
    c6.metric("Win Rate", fmt_pct(metrics["win_rate"]))
    c7.metric("Profit Factor", fmt_num(metrics["profit_factor"]))


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


def event_audit_chart(bars: pd.DataFrame, trade: pd.Series, minutes_before: int = 60, minutes_after: int = 240) -> go.Figure:
    signal_time = pd.to_datetime(trade["signal_time"])
    entry_time = pd.to_datetime(trade["entry_time"])
    exit_time = pd.to_datetime(trade["exit_time"])
    start = signal_time - pd.Timedelta(minutes=minutes_before)
    end = signal_time + pd.Timedelta(minutes=minutes_after)
    window = bars.loc[start:end].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window.index, y=window["price"], mode="lines", name="Price"))
    fig.add_trace(
        go.Bar(
            x=window.index,
            y=window["volume"],
            name="Volume",
            yaxis="y2",
            marker_color="rgba(80, 120, 180, 0.25)",
        )
    )
    fig.add_vline(x=signal_time, line_dash="dash", line_color="#111827", annotation_text="signal")
    fig.add_vline(x=entry_time, line_dash="dot", line_color="#2563eb", annotation_text="entry")
    fig.add_vline(x=exit_time, line_dash="dot", line_color="#dc2626", annotation_text="exit")
    fig.update_layout(
        title=f"Trade Audit | Net {fmt_num(trade['net_pnl_cents'])}c",
        yaxis_title="Price",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        height=460,
        margin=dict(l=10, r=10, t=45, b=10),
    )
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
        default_start = max(pd.Timestamp(max_date) - pd.DateOffset(days=150), pd.Timestamp(min_date)).date()
        start_date = st.date_input("Start date", default_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)
        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()

        st.header("Execution")
        slippage_per_side_cents = st.number_input(
            "Slippage per side in cents",
            min_value=0.0,
            value=2.0,
            step=0.1,
            help="A completed trade pays this once at entry and once at exit.",
        )
        cost_cents = 2.0 * float(slippage_per_side_cents)
        st.caption(f"Round-trip slippage applied: {fmt_num(cost_cents)} cents per completed trade.")
        max_trades_per_day_value = st.number_input(
            "Max trades per day",
            min_value=1,
            value=1,
            step=1,
            help="Use 1 to model taking only the first valid signal each trading day.",
        )
        max_trades_per_day = int(max_trades_per_day_value)
        direction = st.selectbox(
            "Trade direction",
            ["momentum", "fade"],
            help="Momentum buys after an up move and sells after a down move. Fade does the opposite.",
        )

        st.header("Grid")
        st.text_input("a delay seconds", value="1", key="delay_values")
        st.text_input("b threshold cents", value="15,40,80,90,100", key="threshold_values")
        st.text_input("c move window seconds", value="10,90,180,300", key="lookback_values")
        st.text_input("d hold seconds", value="10800,14400,21600", key="hold_values")
        min_train_trades = st.number_input("Minimum train trades", min_value=0, value=5, step=5)
        objective_label = st.selectbox("Optimization objective", ["Total PnL", "Daily Sharpe", "Profit Factor"])
        objective = {
            "Daily Sharpe": "daily_sharpe",
            "Total PnL": "total_pnl_cents",
            "Profit Factor": "profit_factor",
        }[objective_label]

        st.header("Walk Forward")
        train_months = st.number_input("Train months", min_value=1, max_value=12, value=3, step=1)
        test_months = st.number_input("Test months", min_value=1, max_value=6, value=1, step=1)

    with st.spinner("Loading and normalizing tick files to 1-second prices..."):
        price, file_stats = cached_prices(data_dir, str(start_date), str(end_date), "v3-price-volume")
    if isinstance(price, pd.Series):
        price = price.to_frame("price")
        price["volume"] = 0.0
    if price.empty:
        st.error("No valid price rows were loaded for the selected date range.")
        st.stop()
    price_px = price["price"] if isinstance(price, pd.DataFrame) else price

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
    k4.metric("Price Range", f"{fmt_num(price_px.min())} - {fmt_num(price_px.max())}")
    k5.metric("Grid Combos", f"{combo_count:,}")

    tabs = st.tabs(
        [
            "Overview",
            "Preset Search Results",
            "Rare Move Study",
            "Spike Event Strategy",
            "Single Strategy Lab",
            "Walk-Forward Optimizer",
            "Data Diagnostics",
        ]
    )

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
                "Slippage is applied as entry plus exit cost on every completed trade.</div>",
                unsafe_allow_html=True,
            )

    with tabs[1]:
        st.subheader("Large Preset Search Results")
        st.caption(
            "Precomputed with 3-month train / 1-month OOS windows, momentum direction, a=1s, "
            "2c per-side slippage, and max 1 trade/day."
        )
        fixed_path = APP_DIR / "preset_search_fixed_oos.csv"
        adaptive_path = APP_DIR / "preset_search_adaptive_decisions.csv"
        if fixed_path.exists():
            fixed = pd.read_csv(fixed_path)
            show_cols = [
                "label",
                "total_pnl_cents",
                "daily_sharpe",
                "max_drawdown_cents",
                "trades",
                "trades_per_day",
                "win_rate",
                "profit_factor",
            ]
            best_pnl = fixed.sort_values(["total_pnl_cents", "daily_sharpe"], ascending=False).iloc[0]
            best_sharpe = fixed[fixed["trades"] >= 5].sort_values(
                ["daily_sharpe", "total_pnl_cents"], ascending=False
            ).iloc[0]
            best_balanced = fixed.loc[
                fixed.assign(rank_pnl=fixed["total_pnl_cents"].rank(ascending=False), rank_sharpe=fixed["daily_sharpe"].rank(ascending=False))
                .eval("rank_pnl + rank_sharpe")
                .idxmin()
            ]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best PnL Preset", f"{fmt_num(best_pnl['total_pnl_cents'])} c", best_pnl["label"])
            c2.metric("Best Sharpe Preset", fmt_num(best_sharpe["daily_sharpe"]), best_sharpe["label"])
            c3.metric("Best Balanced Preset", f"{fmt_num(best_balanced['total_pnl_cents'])} c", best_balanced["label"])
            st.markdown("**Top Fixed OOS Presets By Net PnL**")
            st.dataframe(fixed[show_cols].head(100), use_container_width=True, hide_index=True)
            st.markdown("**Top Fixed OOS Presets By Sharpe**")
            st.dataframe(
                fixed[fixed["trades"] >= 5].sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False)[show_cols].head(100),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Preset search CSV is not available in this checkout.")

        if adaptive_path.exists():
            adaptive = pd.read_csv(adaptive_path)
            st.markdown("**Adaptive 3M/1M Walk-Forward Decisions**")
            st.dataframe(adaptive, use_container_width=True, hide_index=True)

        iterative_dir = APP_DIR / "outputs" / "iterative_spike_search"
        refined_path = iterative_dir / "stage2_quick_refine_fixed_oos.csv"
        refined_adaptive_path = iterative_dir / "stage3_quick_refine_adaptive_decisions.csv"
        refined_metrics_path = iterative_dir / "stage3_quick_refine_adaptive_metrics.csv"
        if refined_path.exists():
            refined = pd.read_csv(refined_path)
            st.markdown("**Iterative Spike Search: Refined Time Buckets**")
            st.caption(
                "Coarse scan identified winning buckets, then a finer grid searched inside those ranges. "
                "All rows use 2c per-side slippage and max 1 trade/day."
            )
            best_refined_pnl = refined.sort_values(["total_pnl_cents", "daily_sharpe"], ascending=False).iloc[0]
            best_refined_sharpe = refined[refined["trades"] >= 5].sort_values(
                ["daily_sharpe", "total_pnl_cents"], ascending=False
            ).iloc[0]
            refined["rank_pnl"] = refined["total_pnl_cents"].rank(ascending=False)
            refined["rank_sharpe"] = refined["daily_sharpe"].rank(ascending=False)
            refined["rank_dd"] = refined["max_drawdown_cents"].abs().rank(ascending=True)
            best_refined_balanced = refined.sort_values(
                ["rank_pnl", "rank_sharpe", "rank_dd"], ascending=True
            ).iloc[0]
            r1, r2, r3 = st.columns(3)
            r1.metric("Refined Best PnL", f"{fmt_num(best_refined_pnl['total_pnl_cents'])} c", best_refined_pnl["label"])
            r2.metric("Refined Best Sharpe", fmt_num(best_refined_sharpe["daily_sharpe"]), best_refined_sharpe["label"])
            r3.metric(
                "Refined Balanced",
                f"{fmt_num(best_refined_balanced['total_pnl_cents'])} c",
                best_refined_balanced["label"],
            )
            st.markdown("**Refined Fixed OOS: Top By Net PnL**")
            st.dataframe(refined[show_cols].head(100), use_container_width=True, hide_index=True)
            st.markdown("**Refined Fixed OOS: Top By Sharpe**")
            st.dataframe(
                refined[refined["trades"] >= 5]
                .sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False)[show_cols]
                .head(100),
                use_container_width=True,
                hide_index=True,
            )
        if refined_metrics_path.exists():
            st.markdown("**Refined Adaptive 3M/1M Metrics**")
            st.dataframe(pd.read_csv(refined_metrics_path), use_container_width=True, hide_index=True)
        if refined_adaptive_path.exists():
            st.markdown("**Refined Adaptive 3M/1M Decisions**")
            st.dataframe(pd.read_csv(refined_adaptive_path), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("How Rare Is Each Move?")
        st.caption(
            "This measures every rolling x-second price change before trade filtering. "
            "A threshold near or below 5% event-window frequency is a better starting point for sparse trading."
        )
        freq_lookbacks = parse_number_list(st.text_input("Frequency lookbacks seconds", value="5,10,30,60,120"), int)
        freq_thresholds = parse_number_list(
            st.text_input("Frequency thresholds cents", value="2,5,10,20,30,40,50,60,80,100"), float
        )
        if st.button("Run rare-move frequency study"):
            with st.spinner("Scanning rolling price moves by lookback and threshold..."):
                freq = move_frequency_grid(price, freq_lookbacks, freq_thresholds)
            if freq.empty:
                st.warning("No frequency rows were produced for the selected data.")
            else:
                display = freq.copy()
                display["event_window_pct"] = display["event_window_pct"].map(lambda x: f"{x:.2%}")
                st.dataframe(display, use_container_width=True, hide_index=True)
                rare = freq[freq["event_window_pct"] <= 0.05].sort_values(
                    ["event_window_pct", "raw_event_windows_per_day"], ascending=[False, False]
                )
                st.markdown("**Candidates At Or Below 5% Event-Window Frequency**")
                st.dataframe(rare.head(100), use_container_width=True, hide_index=True)
                fig = px.line(
                    freq,
                    x="threshold_cents",
                    y="event_window_pct",
                    color="lookback_s",
                    markers=True,
                    title="Event-Window Frequency By Threshold",
                    labels={"event_window_pct": "Event window %", "threshold_cents": "Threshold cents"},
                )
                fig.add_hline(y=0.05, line_dash="dash", annotation_text="5% target")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Spike Event Strategy")
        st.caption(
            "Targets large price jumps like the screenshot: price moves X cents in Y seconds/minutes, "
            "optionally confirmed by volume expansion, then holds for a longer window."
        )
        s1, s2, s3, s4 = st.columns(4)
        spike_delay = s1.number_input("Spike a delay seconds", min_value=0, value=7, step=1)
        spike_threshold = s2.number_input("Spike b cents", min_value=1.0, value=77.5, step=2.5)
        spike_lookback = s3.number_input("Spike c seconds", min_value=1, value=75, step=5)
        spike_hold = s4.number_input("Spike d hold seconds", min_value=1, value=16200, step=300)
        v1, v2, v3 = st.columns(3)
        spike_vol_window = v1.number_input("Volume baseline seconds", min_value=10, value=300, step=30)
        spike_vol_mult = v2.number_input("Volume multiple filter", min_value=0.0, value=0.0, step=0.5)
        spike_max_trades = v3.number_input("Spike max trades/day", min_value=1, value=1, step=1)
        spike_params = SpikeParams(
            int(spike_delay),
            float(spike_threshold),
            int(spike_lookback),
            int(spike_hold),
            int(spike_vol_window),
            float(spike_vol_mult),
            direction,
        )
        if st.button("Run spike preset backtest"):
            with st.spinner("Running spike event backtest..."):
                spike_trades, spike_daily = backtest_spike_strategy(
                    price, spike_params, cost_cents, max_trades_per_day=int(spike_max_trades)
                )
                spike_metrics = compute_metrics(spike_daily, spike_trades)
            metric_row(spike_metrics)
            l1, l2 = st.columns(2)
            l1.plotly_chart(equity_chart(spike_daily, "Spike Strategy Equity"), use_container_width=True)
            l2.plotly_chart(drawdown_chart(spike_daily, "Spike Strategy Drawdown"), use_container_width=True)
            if spike_trades.empty:
                st.info("No spike trades fired for this preset.")
            else:
                st.dataframe(spike_trades.tail(250), use_container_width=True, hide_index=True)
                pick_idx = st.slider("Audit trade index", min_value=0, max_value=len(spike_trades) - 1, value=len(spike_trades) - 1)
                st.plotly_chart(event_audit_chart(price, spike_trades.iloc[pick_idx]), use_container_width=True)

        st.markdown("**Spike Preset Optimizer**")
        sg1, sg2 = st.columns(2)
        spike_thresholds = parse_number_list(
            sg1.text_input("Spike grid thresholds cents", value="65,70,72.5,75,77.5,80,85"), float
        )
        spike_lookbacks = parse_number_list(
            sg2.text_input("Spike grid lookbacks seconds", value="45,60,75,90"), int
        )
        sg3, sg4 = st.columns(2)
        spike_holds = parse_number_list(
            sg3.text_input("Spike grid holds seconds", value="12600,14400,16200,18000"), int
        )
        spike_vol_mults = parse_number_list(sg4.text_input("Spike volume multiples", value="0,2"), float)
        if st.button("Run spike 3M/1M optimizer", type="primary"):
            spike_grid = spike_parameter_grid(
                [int(spike_delay)],
                spike_thresholds,
                spike_lookbacks,
                spike_holds,
                [int(spike_vol_window)],
                spike_vol_mults,
                direction,
            )
            with st.spinner(f"Running {len(spike_grid):,} spike presets through 3M/1M walk-forward..."):
                sp_details, sp_oos, sp_trades, sp_train_grid = run_spike_walk_forward(
                    price,
                    spike_grid,
                    train_months=int(train_months),
                    test_months=int(test_months),
                    objective=objective,
                    cost_cents=cost_cents,
                    min_train_trades=int(min_train_trades),
                    max_trades_per_day=int(spike_max_trades),
                )
            if sp_details.empty:
                st.warning("No spike walk-forward periods produced trades.")
            else:
                metric_row(compute_metrics(sp_oos, sp_trades))
                st.dataframe(sp_details, use_container_width=True, hide_index=True)
                st.markdown("**Top Spike Train Grid Rows**")
                st.dataframe(sp_train_grid.sort_values("score", ascending=False).head(200), use_container_width=True, hide_index=True)
                if not sp_trades.empty:
                    st.download_button(
                        "Download spike OOS trades CSV",
                        data=sp_trades.to_csv(index=False).encode("utf-8"),
                        file_name="spike_oos_trades.csv",
                        mime="text/csv",
                    )

    with tabs[4]:
        st.subheader("Single Parameter Backtest")
        preset = st.selectbox(
            "Preset",
            [
                "Best PnL: 15c in 10s, hold 6h",
                "Best Sharpe: 90c in 180s, hold 3h",
                "Best Balanced: 90c in 180s, hold 4h",
                "2c in 30s, hold 60s",
                "40c in 10s, hold 4h",
                "Custom",
            ],
            help="The first preset is the sanity-check case; the second is Shubham's rare-move idea.",
        )
        if preset == "Best PnL: 15c in 10s, hold 6h":
            default_delay, default_threshold, default_lookback, default_hold = 1, 15.0, 10, 21600
        elif preset == "Best Sharpe: 90c in 180s, hold 3h":
            default_delay, default_threshold, default_lookback, default_hold = 1, 90.0, 180, 10800
        elif preset == "Best Balanced: 90c in 180s, hold 4h":
            default_delay, default_threshold, default_lookback, default_hold = 1, 90.0, 180, 14400
        elif preset == "40c in 10s, hold 4h":
            default_delay, default_threshold, default_lookback, default_hold = 1, 40.0, 10, 14400
        else:
            default_delay, default_threshold, default_lookback, default_hold = 1, 2.0, 30, 60
        p1, p2, p3, p4 = st.columns(4)
        delay_s = p1.number_input("a delay", min_value=0, value=default_delay, step=1, key=f"delay_{preset}")
        threshold_cents = p2.number_input("b cents", min_value=0.1, value=default_threshold, step=0.1, key=f"thr_{preset}")
        lookback_s = p3.number_input("c seconds", min_value=1, value=default_lookback, step=1, key=f"look_{preset}")
        hold_s = p4.number_input("d seconds", min_value=1, value=default_hold, step=1, key=f"hold_{preset}")
        params = StrategyParams(int(delay_s), float(threshold_cents), int(lookback_s), int(hold_s), direction)

        with st.spinner("Running single strategy backtest..."):
            trades, daily = backtest_strategy(price, params, cost_cents, max_trades_per_day=max_trades_per_day)
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
                "gross_pnl_cents",
                "cost_cents",
                "net_pnl_cents",
            ]
            st.dataframe(trades[show_cols].tail(250), use_container_width=True, hide_index=True)

    with tabs[5]:
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
                    max_trades_per_day=max_trades_per_day,
                )

            if details.empty or oos_daily.empty:
                st.warning(
                    "No walk-forward periods produced trades. This usually means the selected date range has fewer "
                    "than train months plus test months, or every candidate failed the minimum-trade filter."
                )
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

    with tabs[6]:
        st.subheader("Loaded File Diagnostics")
        st.dataframe(file_stats, use_container_width=True, hide_index=True)

        st.subheader("Standalone Grid Scan On Full Selected Sample")
        if st.button("Run full-sample grid scan"):
            with st.spinner("Evaluating every parameter combination on the selected sample..."):
                grid = evaluate_grid(
                    price,
                    params_list,
                    objective,
                    cost_cents,
                    int(min_train_trades),
                    max_trades_per_day=max_trades_per_day,
                )
            st.dataframe(grid.drop(columns=["params"]).head(250), use_container_width=True, hide_index=True)
            st.download_button(
                "Download full-sample grid CSV",
                data=grid.drop(columns=["params"]).to_csv(index=False).encode("utf-8"),
                file_name="price_action_full_sample_grid.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
