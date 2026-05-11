from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from price_action_engine import (
    DUBAI_UTC_OFFSET_HOURS,
    DYNAMIC_COST_CENTS_PER_PRICE_UNIT,
    SpikeParams,
    backtest_spike_strategy,
    compute_metrics,
    load_second_prices,
)


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
OUT_DIR = APP_DIR / "outputs" / "clean_walkforward"
REPORT_XLSX = OUT_DIR / "walkforward_report.xlsx"
SINGLE_DEFAULT_TRADES = OUT_DIR / "single_default_trades.csv"
SINGLE_DEFAULT_EXCLUDED = OUT_DIR / "single_default_out_of_session_trades.csv"
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
DEFAULT_MAX_TRADES_PER_DAY = 1
SESSION_START_HOUR_DUBAI = 11
SESSION_END_HOUR_DUBAI = 24


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
        "out_of_session": "walkforward_out_of_session_trades_all.csv",
        "config": "research_config.csv",
        "universe": "parameter_universe.csv",
        "rankings": "walkforward_train_rankings_all.csv",
    }
    out = {}
    for key, name in files.items():
        path = OUT_DIR / name
        out[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return out


def add_session_audit_columns(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    signal = pd.to_datetime(out["signal_time"])
    signal_dubai = signal + pd.Timedelta(hours=DUBAI_UTC_OFFSET_HOURS)
    out["signal_time_dubai"] = signal_dubai
    seconds = signal_dubai.dt.hour * 3600 + signal_dubai.dt.minute * 60 + signal_dubai.dt.second
    start_seconds = SESSION_START_HOUR_DUBAI * 3600
    end_seconds = SESSION_END_HOUR_DUBAI * 3600
    inside = (seconds >= start_seconds) & (seconds < end_seconds)
    out["session_bucket"] = inside.map(
        {
            True: f"Inside {SESSION_START_HOUR_DUBAI:02d}:00-24:00 Dubai",
            False: f"Outside {SESSION_START_HOUR_DUBAI:02d}:00-24:00 Dubai",
        }
    )
    out["out_of_frame_category"] = inside.map({True: "Tradable session", False: "Excluded: before 11:00 Dubai"})
    return out


@st.cache_data(show_spinner=False)
def load_default_single_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict, str] | None:
    if not (SINGLE_DEFAULT_TRADES.exists() and SINGLE_DEFAULT_DAILY.exists() and SINGLE_DEFAULT_METRICS.exists()):
        return None
    trades = pd.read_csv(SINGLE_DEFAULT_TRADES)
    excluded = pd.read_csv(SINGLE_DEFAULT_EXCLUDED) if SINGLE_DEFAULT_EXCLUDED.exists() else pd.DataFrame()
    for col in ["signal_time", "entry_time", "exit_time"]:
        if col in trades.columns:
            trades[col] = pd.to_datetime(trades[col])
        if col in excluded.columns:
            excluded[col] = pd.to_datetime(excluded[col])
    daily_df = pd.read_csv(SINGLE_DEFAULT_DAILY, index_col=0, parse_dates=True)
    daily = daily_df.iloc[:, 0].rename("daily_pnl_cents") if not daily_df.empty else pd.Series(dtype=float)
    metrics_row = pd.read_csv(SINGLE_DEFAULT_METRICS).iloc[0]
    label = str(metrics_row["params"])
    metrics = metrics_row.drop(labels=["params"], errors="ignore").to_dict()
    return trades, excluded, daily, metrics, label


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
    max_trades_per_day: int,
    volume_multiple: float,
    volume_window_s: int,
    direction: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict, str]:
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
    trades, daily = backtest_spike_strategy(
        bars,
        params,
        None,
        cap,
        session_start_hour=SESSION_START_HOUR_DUBAI,
        session_end_hour=SESSION_END_HOUR_DUBAI,
        session_tz_offset_hours=DUBAI_UTC_OFFSET_HOURS,
    )
    unrestricted_trades, _ = backtest_spike_strategy(bars, params, None, cap)
    unrestricted_trades = add_session_audit_columns(unrestricted_trades)
    excluded = (
        unrestricted_trades[unrestricted_trades["session_bucket"].str.startswith("Outside")].copy()
        if not unrestricted_trades.empty
        else pd.DataFrame()
    )
    if not excluded.empty:
        excluded["exclusion_reason"] = f"Signal outside {SESSION_START_HOUR_DUBAI:02d}:00-24:00 Dubai tradable window"
    metrics = compute_metrics(daily, trades)
    return trades, excluded, daily, metrics, params.label


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


def _prep_excluded_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["signal_time", "signal_time_dubai", "entry_time", "exit_time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    out["net_pnl_cents"] = pd.to_numeric(out["net_pnl_cents"], errors="coerce").fillna(0.0)
    out["gross_pnl_cents"] = pd.to_numeric(out["gross_pnl_cents"], errors="coerce").fillna(0.0)
    out["cost_cents"] = pd.to_numeric(out["cost_cents"], errors="coerce").fillna(0.0)
    if "signal_time_dubai" in out.columns:
        out["dubai_hour"] = out["signal_time_dubai"].dt.hour
        out["dubai_month"] = out["signal_time_dubai"].dt.to_period("M").astype(str)
    else:
        out["dubai_hour"] = pd.NA
        out["dubai_month"] = ""
    out["side_label"] = out["side"].map({1: "LONG", -1: "SHORT"}) if "side" in out.columns else ""
    return out


def _excluded_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            trades=("net_pnl_cents", "size"),
            net_pnl_cents=("net_pnl_cents", "sum"),
            gross_pnl_cents=("gross_pnl_cents", "sum"),
            total_cost_cents=("cost_cents", "sum"),
            avg_trade_cents=("net_pnl_cents", "mean"),
            win_rate=("net_pnl_cents", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    return grouped.sort_values("net_pnl_cents", ascending=False)


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
        candidate_count = cfg.get("candidate_count", "")
        cfg_cols[0].metric(
            "Candidate Universe",
            f"{int(candidate_count):,}" if pd.notna(candidate_count) and candidate_count != "" else "Selected WF",
        )
        cfg_cols[1].metric("Train Window", f"{int(cfg['train_months'])}M")
        cfg_cols[2].metric("Test/Rebalance", f"{int(cfg['test_months'])}M")
        cost_label = (
            cfg.get("cost_formula", f"abs(entry_price) * {DYNAMIC_COST_CENTS_PER_PRICE_UNIT:g} cents")
            if "cost_formula" in cfg.index
            else f"abs(entry_price) * {DYNAMIC_COST_CENTS_PER_PRICE_UNIT:g} cents"
        )
        cfg_cols[3].metric("Bid/Ask Cost", "Dynamic")
        cfg_cols[4].metric("Trade Cap", f"{int(cfg['max_trades_per_day'])}/day")
        st.caption(f"Trading window: 11:00 to 24:00 Dubai time. Cost model: {cost_label}.")

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
            "signal_time_dubai",
            "entry_time",
            "entry_time_dubai",
            "exit_time",
            "exit_time_dubai",
            "session_bucket",
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

    out_of_session = data["out_of_session"]
    selected_out = out_of_session[out_of_session["selector_objective"] == selector].copy() if not out_of_session.empty else pd.DataFrame()
    st.subheader("Excluded Out-Of-Frame Trades")
    st.caption("These are trades that the same selected parameters would have taken without the Dubai-time restriction.")
    if selected_out.empty:
        st.info("No out-of-frame trades were removed for this selector.")
    else:
        out_cols = [
            "test_month",
            "signal_time",
            "signal_time_dubai",
            "out_of_frame_category",
            "entry_time",
            "exit_time",
            "side",
            "move_cents",
            "entry_price",
            "exit_price",
            "gross_pnl_cents",
            "cost_cents",
            "net_pnl_cents",
            "selected_params",
            "exclusion_reason",
        ]
        st.dataframe(selected_out[[c for c in out_cols if c in selected_out.columns]], use_container_width=True, hide_index=True)

    with st.expander("Parameter Universe"):
        st.markdown("The universe below was defined before the monthly tests. Displayed performance and trades use the Dubai-time window and dynamic bid/ask cost.")
        st.dataframe(universe, use_container_width=True, hide_index=True)


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
        max_trades_per_day = r2[0].number_input(
            "Max trades per day",
            min_value=0,
            value=DEFAULT_MAX_TRADES_PER_DAY,
            step=1,
            help="Set 0 for unlimited trades per day.",
        )
        volume_multiple = r2[1].number_input(
            "Volume multiple filter",
            min_value=0.0,
            value=float(DEFAULT_SINGLE_PARAMS.volume_multiple),
            step=0.5,
            help="0 means volume filter off.",
        )
        direction = r2[2].selectbox("Direction", options=["momentum", "fade"], index=0)
        r2[3].metric("Bid/Ask Cost", f"{DYNAMIC_COST_CENTS_PER_PRICE_UNIT:g}c x price")

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
        trades, excluded, daily, metrics, label = default_result
        st.info("Showing the saved whole-period run for Shubham's default parameters. Click Run Single Backtest after changing inputs.")
    else:
        with st.spinner("Running whole-period single-parameter backtest..."):
            trades, excluded, daily, metrics, label = run_single_backtest(
                int(delay_s),
                float(threshold_cents),
                int(lookback_s),
                int(hold_s),
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
        "signal_time_dubai",
        "entry_time",
        "entry_time_dubai",
        "exit_time",
        "exit_time_dubai",
        "session_bucket",
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

    st.subheader("Excluded Out-Of-Frame Trades")
    st.caption("These trades are from the same parameters without the 11:00-24:00 Dubai signal-time filter.")
    if excluded.empty:
        st.info("No unrestricted trades fell outside the Dubai-time window.")
    else:
        excluded_out = excluded.copy()
        excluded_out["side_label"] = excluded_out["side"].map({1: "LONG", -1: "SHORT"})
        excluded_out["params"] = label
        excluded_cols = [
            "signal_time",
            "signal_time_dubai",
            "out_of_frame_category",
            "entry_time",
            "exit_time",
            "side_label",
            "move_cents",
            "entry_price",
            "exit_price",
            "gross_pnl_cents",
            "cost_cents",
            "net_pnl_cents",
            "params",
            "exclusion_reason",
        ]
        st.dataframe(excluded_out[[c for c in excluded_cols if c in excluded_out.columns]], use_container_width=True, hide_index=True)


def render_out_of_session_tab(data: dict[str, pd.DataFrame]) -> None:
    st.caption(
        "Trades here are signals that did not match the allowed Dubai-time window: 11:00 to 24:00. "
        "They show what was excluded by the time filter."
    )
    wf_excluded = _prep_excluded_for_analysis(data["out_of_session"])
    metrics = data["metrics"]
    single_default = load_default_single_backtest()
    single_excluded = _prep_excluded_for_analysis(single_default[1]) if single_default is not None else pd.DataFrame()

    source_options = ["Walk-forward excluded trades"]
    if not single_excluded.empty:
        source_options.append("Single-parameter default excluded trades")
    source = st.selectbox("Source", source_options)
    if source.startswith("Single"):
        df = single_excluded.copy()
        active_label = single_default[4] if single_default is not None else ""
        in_window_pnl = float(single_default[3].get("total_pnl_cents", 0.0)) if single_default is not None else 0.0
        st.code(active_label)
    else:
        if wf_excluded.empty:
            st.info("No walk-forward trades were excluded by the Dubai-time filter.")
            return
        selectors = sorted(wf_excluded["selector_objective"].dropna().unique().tolist())
        selector = st.selectbox("Selector objective", selectors, index=selectors.index("daily_sharpe") if "daily_sharpe" in selectors else 0)
        df = wf_excluded[wf_excluded["selector_objective"] == selector].copy()
        metric_row = metrics[metrics["selector_objective"] == selector]
        in_window_pnl = float(metric_row["total_pnl_cents"].iloc[0]) if not metric_row.empty else 0.0

    if df.empty:
        st.info("No excluded trades for this selection.")
        return

    excluded_pnl = float(df["net_pnl_cents"].sum())
    excluded_gross = float(df["gross_pnl_cents"].sum())
    excluded_cost = float(df["cost_cents"].sum())
    combined_pnl = in_window_pnl + excluded_pnl
    pnl_word = "profit missed" if excluded_pnl > 0 else "loss avoided"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Excluded Trades", f"{len(df):,}")
    c2.metric("Excluded Net PnL", f"{fmt_num(excluded_pnl)} c")
    c3.metric("Impact", pnl_word)
    c4.metric("In-Window PnL", f"{fmt_num(in_window_pnl)} c")
    c5.metric("If Included", f"{fmt_num(combined_pnl)} c")
    c6.metric("Excluded Costs", f"{fmt_num(excluded_cost)} c")

    st.caption(
        f"Interpretation: excluding these trades changed PnL by {fmt_num(excluded_pnl)}c. "
        f"Positive means profit was left out; negative means the time filter avoided a loss."
    )

    summary_cols = st.columns(2)
    by_month = _excluded_summary(df, ["dubai_month"])
    by_hour = _excluded_summary(df, ["dubai_hour"])
    with summary_cols[0]:
        st.subheader("Excluded PnL By Month")
        st.dataframe(by_month, use_container_width=True, hide_index=True)
    with summary_cols[1]:
        st.subheader("Excluded PnL By Dubai Hour")
        st.dataframe(by_hour, use_container_width=True, hide_index=True)

    if not by_hour.empty:
        hour_fig = px.bar(
            by_hour,
            x="dubai_hour",
            y="net_pnl_cents",
            color="net_pnl_cents",
            color_continuous_scale="RdYlGn",
            title="Excluded Net PnL By Dubai Signal Hour",
            labels={"dubai_hour": "Dubai signal hour", "net_pnl_cents": "Net PnL cents"},
        )
        hour_fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(hour_fig, use_container_width=True)

    side_summary = _excluded_summary(df, ["side_label"])
    param_summary = _excluded_summary(df, ["selected_params"]) if "selected_params" in df.columns else pd.DataFrame()
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Long Vs Short")
        st.dataframe(side_summary, use_container_width=True, hide_index=True)
    with c_right:
        st.subheader("Parameter Set Impact")
        if param_summary.empty:
            st.info("No parameter labels available for this source.")
        else:
            st.dataframe(param_summary, use_container_width=True, hide_index=True)

    st.subheader("Excluded Trade List")
    display_cols = [
        "selector_objective",
        "test_month",
        "signal_time",
        "signal_time_dubai",
        "dubai_hour",
        "out_of_frame_category",
        "entry_time",
        "exit_time",
        "side_label",
        "move_cents",
        "entry_price",
        "exit_price",
        "gross_pnl_cents",
        "cost_cents",
        "net_pnl_cents",
        "selected_params",
        "exclusion_reason",
    ]
    shown = df[[c for c in display_cols if c in df.columns]].sort_values("signal_time_dubai")
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Excluded Trades CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name="excluded_out_of_session_trades.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Price Action Research Dashboard")
    data = load_outputs()

    walk_forward_tab, excluded_tab, single_tab = st.tabs(
        ["Walk-Forward", "Out-Of-Session Analysis", "Single Parameter Backtest"]
    )
    with walk_forward_tab:
        render_walk_forward_tab(data)
    with excluded_tab:
        render_out_of_session_tab(data)
    with single_tab:
        render_single_backtest_tab()


if __name__ == "__main__":
    main()
