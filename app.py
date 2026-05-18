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
VOL_OUT_DIR = APP_DIR / "outputs" / "volatility_walkforward"
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


def objective_label(value: str) -> str:
    labels = {
        "daily_sharpe": "Sharpe objective",
        "total_pnl_cents": "PnL objective",
        "profit_factor": "Profit factor objective",
    }
    return labels.get(str(value), str(value))


def clean_label(value: str) -> str:
    text = str(value)
    replacements = {
        "daily_sharpe": "sharpe_objective",
        "total_pnl_cents": "pnl_objective",
        "percent_to_dollar": "percent_to_dollar_sigma",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(
        columns=[
            "trades_per_day",
            "positive_day_rate",
            "active_day_rate",
            "avg_daily_pnl_cents",
            "daily_sortino",
            "days",
        ],
        errors="ignore",
    )
    for col in ["selector_objective", "objective"]:
        if col in out.columns:
            out[col] = out[col].map(objective_label)
    for col in ["run_key"]:
        if col in out.columns:
            out[col] = out[col].map(clean_label)
    return out.rename(
        columns={
            "selector_objective": "selector",
            "run_key": "run",
            "daily_sharpe": "sharpe",
            "avg_daily_pnl_cents": "avg_pnl_cents",
            "daily_sortino": "sortino",
            "daily_sigma_dollars": "rolling_sigma_dollars",
            "vol_window_days": "vol_window_sessions",
            "max_trades_per_day": "max_trades_per_session",
        }
    )


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


@st.cache_data(show_spinner=False, ttl=30)
def load_vol_outputs() -> dict[str, pd.DataFrame]:
    files = {
        "metrics": "metrics.csv",
        "decisions": "decisions.csv",
        "trades": "trades.csv",
        "daily": "daily_pnl.csv",
        "config": "config.csv",
        "universe": "parameter_universe.csv",
        "rankings": "train_rankings.csv",
        "multiple_backtests": "multiple_backtests.csv",
        "winning_percent_metrics": "winning_percent_vol_backtest_metrics.csv",
        "winning_percent_daily": "winning_percent_vol_backtest_daily.csv",
        "winning_percent_trades": "winning_percent_vol_backtest_trades.csv",
        "robust_metrics": "robust_metrics.csv",
        "robust_decisions": "robust_decisions.csv",
        "robust_trades": "robust_trades.csv",
        "robust_daily": "robust_daily_pnl.csv",
        "robust_rankings": "robust_train_rankings.csv",
        "stability_monthly": "stability_monthly_audit.csv",
        "stability_repeated": "stability_repeated_clocks.csv",
        "stability_top_trades": "stability_top_trades.csv",
        "stable_4var_metrics": "stable_4var_metrics.csv",
        "stable_4var_decisions": "stable_4var_decisions.csv",
        "stable_4var_trades": "stable_4var_trades.csv",
        "stable_4var_daily": "stable_4var_daily_pnl.csv",
        "stable_4var_rankings": "stable_4var_train_rankings.csv",
        "stable_4var_universe": "stable_parameter_universe.csv",
        "rare_move_metrics": "rare_move_metrics.csv",
        "rare_move_decisions": "rare_move_decisions.csv",
        "rare_move_trades": "rare_move_trades.csv",
        "rare_move_daily": "rare_move_daily_pnl.csv",
        "rare_move_rankings": "rare_move_train_rankings.csv",
        "rare_move_universe": "rare_move_parameter_universe.csv",
        "regime_rare_metrics": "regime_rare_metrics.csv",
        "regime_rare_decisions": "regime_rare_decisions.csv",
        "regime_rare_trades": "regime_rare_trades.csv",
        "regime_rare_daily": "regime_rare_daily_pnl.csv",
        "regime_rare_rankings": "regime_rare_train_rankings.csv",
        "regime_rare_universe": "regime_rare_parameter_universe.csv",
    }
    out = {}
    for key, name in files.items():
        path = VOL_OUT_DIR / name
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
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("WF Net PnL", f"{fmt_num(row['total_pnl_cents'])} c")
    c2.metric("WF Sharpe", fmt_num(row["daily_sharpe"]))
    c3.metric("Max DD", f"{fmt_num(row['max_drawdown_cents'])} c")
    c4.metric("Trades", f"{int(row['trades']):,}")
    c5.metric("Profit Factor", fmt_num(row["profit_factor"]))


def single_metric_tiles(metrics: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net PnL", f"{fmt_num(metrics['total_pnl_cents'])} c")
    c2.metric("Sharpe", fmt_num(metrics["daily_sharpe"]))
    c3.metric("Max DD", f"{fmt_num(metrics['max_drawdown_cents'])} c")
    c4.metric("Trades", f"{int(metrics['trades']):,}")
    c5.metric("Profit Factor", fmt_num(metrics["profit_factor"]))


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
        cfg_cols[4].metric("Trade Cap", f"{int(cfg['max_trades_per_day'])} per session")
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
            "win_rate",
            "profit_factor",
        ]
    ].sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False)
    st.dataframe(display_table(show_metrics), use_container_width=True, hide_index=True)

    if not daily.empty:
        st.plotly_chart(equity_figure(daily), use_container_width=True)

    st.subheader("Monthly Rebalance Decisions")
    selector = st.selectbox(
        "Selector objective",
        options=show_metrics["selector_objective"].tolist(),
        index=0,
        format_func=objective_label,
        key="walk_forward_selector_objective",
    )
    selected_decisions = decisions[decisions["selector_objective"] == selector].copy()
    st.dataframe(display_table(selected_decisions), use_container_width=True, hide_index=True)

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
        st.dataframe(display_table(selected_trades[display_cols]), use_container_width=True, hide_index=True)
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
        st.dataframe(display_table(selected_out[[c for c in out_cols if c in selected_out.columns]]), use_container_width=True, hide_index=True)

    with st.expander("Parameter Universe"):
        st.markdown("The universe below was defined before the monthly tests. Displayed performance and trades use the Dubai-time window and dynamic bid/ask cost.")
        st.dataframe(universe, use_container_width=True, hide_index=True)


def render_volatility_tab(vol_data: dict[str, pd.DataFrame], fixed_data: dict[str, pd.DataFrame], vol_method: str) -> None:
    title = "Baseline Percent-to-Dollar Walk-Forward"
    st.caption(
        "True walk-forward: each month chooses parameters using only past data, then trades the next unseen month. "
        "This baseline still optimizes for Sharpe or PnL, so it can be less stable than the robust versions."
    )
    st.info(
        "Percent-to-Dollar means: calculate volatility from daily percent returns, convert that back into dollar terms at the current price level, "
        "then trigger when the intraday move is greater than `sigma multiple x rolling dollar-equivalent volatility`. "
        "The signal now requires the full 30-minute move window to be inside 11:00-24:00 Dubai time, so 11:02 entries caused by pre-session moves are filtered out."
    )
    metrics = vol_data["metrics"]
    decisions = vol_data["decisions"]
    trades = vol_data["trades"]
    daily = vol_data["daily"]
    universe = vol_data["universe"]
    rankings = vol_data["rankings"]
    multiple_backtests = vol_data["multiple_backtests"]

    if metrics.empty:
        st.warning("Volatility walk-forward outputs are missing. Run `python volatility_walkforward_research.py` first.")
        return

    method_metrics = metrics[metrics["vol_method"] == vol_method].copy()
    if method_metrics.empty:
        st.info(f"No results found for {vol_method}.")
        return

    best = method_metrics.sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).iloc[0]
    st.subheader("Best Volatility-Scaled Result")
    metric_tiles(best)

    st.subheader("All Rebalance Results")
    show_metrics = method_metrics[
        [
            "run_key",
            "rebalance",
            "objective",
            "total_pnl_cents",
            "daily_sharpe",
            "max_drawdown_cents",
            "trades",
            "win_rate",
            "profit_factor",
        ]
    ].sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False)
    st.dataframe(display_table(show_metrics), use_container_width=True, hide_index=True)

    method_multiples = multiple_backtests[multiple_backtests["vol_method"] == vol_method].copy() if not multiple_backtests.empty else pd.DataFrame()
    if not method_multiples.empty:
        st.subheader("Sigma Multiple Backtests")
        st.caption(
            "These are fixed-parameter full-period diagnostics for each tested sigma multiple. "
            "They are for checking threshold sensitivity; the selected production-style result still comes from walk-forward only."
        )
        best_by_multiple = (
            method_multiples.sort_values(["sigma_multiple", "daily_sharpe", "total_pnl_cents"], ascending=[True, False, False])
            .groupby("sigma_multiple", as_index=False)
            .head(1)
            .sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False)
        )
        multiple_cols = [
            "sigma_multiple",
            "signal_mode",
            "hold_s",
            "total_pnl_cents",
            "daily_sharpe",
            "max_drawdown_cents",
            "trades",
            "win_rate",
            "profit_factor",
            "params",
        ]
        st.dataframe(
            display_table(best_by_multiple[[c for c in multiple_cols if c in best_by_multiple.columns]]),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("All Sigma Multiple Backtests"):
            all_multiple_cols = [
                "sigma_multiple",
                "signal_mode",
                "hold_s",
                "backtest_start",
                "backtest_end",
                "total_pnl_cents",
                "daily_sharpe",
                "max_drawdown_cents",
                "trades",
                "win_rate",
                "profit_factor",
                "params",
            ]
            st.dataframe(
                display_table(method_multiples[[c for c in all_multiple_cols if c in method_multiples.columns]]),
                use_container_width=True,
                hide_index=True,
            )

    run_key = st.selectbox(
        "Volatility run",
        options=show_metrics["run_key"].tolist(),
        index=0,
        format_func=clean_label,
        key=f"vol_run_{vol_method}",
    )
    selected_decisions = decisions[decisions["run_key"] == run_key].copy()
    st.subheader("Rebalance Decisions")
    st.dataframe(display_table(selected_decisions), use_container_width=True, hide_index=True)

    if not selected_decisions.empty:
        fig = px.bar(
            selected_decisions,
            x="test_start",
            y="test_total_pnl_cents",
            color="test_total_pnl_cents",
            color_continuous_scale="RdYlGn",
            title=f"Out-of-Sample PnL By Monthly Rebalance Period: {clean_label(run_key)}",
            labels={"test_start": "Test start", "test_total_pnl_cents": "PnL cents"},
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if not daily.empty and run_key in daily.columns:
        date_col = daily.columns[0]
        series = daily[[date_col, run_key]].copy()
        series[date_col] = pd.to_datetime(series[date_col])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series[date_col], y=series[run_key].fillna(0).cumsum(), mode="lines", name=clean_label(run_key)))
        fig.update_layout(
            title=f"Equity Curve: {clean_label(run_key)}",
            xaxis_title="Date",
            yaxis_title="Cumulative cents",
            height=390,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    selected_trades = trades[trades["run_key"] == run_key].copy() if not trades.empty else pd.DataFrame()
    st.subheader("Trades Taken")
    if selected_trades.empty:
        st.info("No trades for this run.")
    else:
        display_cols = [
            "test_start",
            "test_end",
            "signal_time",
            "signal_time_dubai",
            "entry_time",
            "entry_time_dubai",
            "exit_time",
            "exit_time_dubai",
            "side",
            "move_cents",
            "threshold_cents",
            "daily_sigma_dollars",
            "sigma_multiple",
            "entry_price",
            "exit_price",
            "gross_pnl_cents",
            "cost_cents",
            "net_pnl_cents",
            "signal_mode",
            "bad_hour_rule",
            "selected_params",
        ]
        st.dataframe(display_table(selected_trades[[c for c in display_cols if c in selected_trades.columns]]), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Volatility Trades CSV",
            data=selected_trades.to_csv(index=False).encode("utf-8"),
            file_name=f"{clean_label(run_key)}_trades.csv",
            mime="text/csv",
            key=f"vol_trade_download_{vol_method}",
        )

    with st.expander("Candidate Universe And Train Rankings"):
        st.dataframe(display_table(universe[universe["vol_method"] == vol_method]), use_container_width=True, hide_index=True)
        st.dataframe(display_table(rankings[rankings["run_key"] == run_key]), use_container_width=True, hide_index=True)


def render_robust_walkforward_tab(vol_data: dict[str, pd.DataFrame]) -> None:
    st.caption(
        "Robust walk-forward uses the same monthly train-then-test process, but changes how parameters are selected. "
        "It rewards positive Sharpe, PnL after removing the best trade, and profitable-month consistency, while penalizing dependence on one trade or one month."
    )
    st.info(
        "Simple language: this tab asks, 'Which parameter is less likely to be a lucky March-only result?' "
        "Technical language: score = Sharpe + top-trade-removed PnL + profitable-month rate - concentration penalties."
    )

    metrics = vol_data["robust_metrics"]
    decisions = vol_data["robust_decisions"]
    trades = vol_data["robust_trades"]
    daily = vol_data["robust_daily"]
    rankings = vol_data["robust_rankings"]

    if metrics.empty:
        st.warning("Robust walk-forward outputs are missing. Run `python volatility_walkforward_research.py` first.")
        return

    show_cols = [
        "run_key",
        "run_label",
        "total_pnl_cents",
        "daily_sharpe",
        "max_drawdown_cents",
        "trades",
        "win_rate",
        "profit_factor",
        "profitable_month_rate",
        "top_trade_pnl_cents",
        "top_trade_share",
        "best_month_pnl_cents",
        "best_month_share",
        "top_trade_removed_pnl_cents",
    ]
    st.subheader("Robust Walk-Forward Results")
    st.dataframe(display_table(metrics[[c for c in show_cols if c in metrics.columns]].sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False)), use_container_width=True, hide_index=True)

    run_key = st.selectbox(
        "Robust run",
        metrics["run_key"].tolist(),
        index=0,
        format_func=lambda x: str(metrics.loc[metrics["run_key"].eq(x), "run_label"].iloc[0]) if x in metrics["run_key"].values else clean_label(x),
        key="robust_run_key",
    )

    selected_metrics = metrics[metrics["run_key"] == run_key].iloc[0]
    metric_tiles(selected_metrics)

    selected_decisions = decisions[decisions["run_key"] == run_key].copy()
    st.subheader("Monthly Rebalance Decisions")
    st.caption(
        "`train_*` columns show what the optimizer saw in the past window. "
        "`test_*` columns show what happened in the next unseen month."
    )
    decision_cols = [
        "test_start",
        "test_end",
        "selected_params",
        "train_score",
        "train_total_pnl_cents",
        "train_sharpe",
        "train_top_trade_removed_pnl_cents",
        "train_top_trade_share",
        "test_total_pnl_cents",
        "test_sharpe",
        "test_trades",
        "test_top_trade_removed_pnl_cents",
        "test_top_trade_share",
    ]
    st.dataframe(display_table(selected_decisions[[c for c in decision_cols if c in selected_decisions.columns]]), use_container_width=True, hide_index=True)

    if not selected_decisions.empty:
        fig = px.bar(
            selected_decisions,
            x="test_start",
            y="test_total_pnl_cents",
            color="test_total_pnl_cents",
            color_continuous_scale="RdYlGn",
            title="OOS PnL By Rebalance Month",
            labels={"test_start": "Out-of-sample month", "test_total_pnl_cents": "PnL cents"},
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if not daily.empty and run_key in daily.columns:
        date_col = daily.columns[0]
        curve = daily[[date_col, run_key]].copy()
        curve[date_col] = pd.to_datetime(curve[date_col])
        curve["equity_cents"] = curve[run_key].fillna(0).cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve[date_col], y=curve["equity_cents"], mode="lines", name=clean_label(run_key)))
        fig.update_layout(
            title="OOS Equity Curve",
            xaxis_title="Date",
            yaxis_title="Cumulative out-of-sample PnL in cents",
            height=390,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    selected_trades = trades[trades["run_key"] == run_key].copy() if not trades.empty else pd.DataFrame()
    st.subheader("Trades Taken")
    if selected_trades.empty:
        st.info("No trades for this robust run.")
    else:
        display_cols = [
            "test_start",
            "signal_time_dubai",
            "entry_time_dubai",
            "exit_time_dubai",
            "side",
            "move_cents",
            "threshold_cents",
            "daily_sigma_dollars",
            "sigma_multiple",
            "entry_price",
            "exit_price",
            "net_pnl_cents",
            "selected_params",
        ]
        st.dataframe(display_table(selected_trades[[c for c in display_cols if c in selected_trades.columns]]), use_container_width=True, hide_index=True)

    with st.expander("Training Rankings For Selected Robust Run"):
        st.dataframe(display_table(rankings[rankings["run_key"] == run_key]), use_container_width=True, hide_index=True)


def render_stable_4var_tab(vol_data: dict[str, pd.DataFrame]) -> None:
    st.caption(
        "Purpose: find a steadier version of the same spike idea by optimizing the four variables a, b, c, and d, then rejecting fragile winners."
    )
    st.info(
        "Simple: the strategy only trades next month if the past training window shows a parameter set that made money without depending on one lucky trade, "
        "one lucky month, or one exact magic number. Technical: each monthly rebalance searches a wider parameter grid and selects by a stability score using "
        "Sharpe, PnL after removing the best trade, profitable-month rate, drawdown, concentration penalties, and nearby-parameter cluster support."
    )

    metrics = vol_data["stable_4var_metrics"]
    decisions = vol_data["stable_4var_decisions"]
    trades = vol_data["stable_4var_trades"]
    daily = vol_data["stable_4var_daily"]
    rankings = vol_data["stable_4var_rankings"]
    universe = vol_data["stable_4var_universe"]

    if metrics.empty:
        st.warning("Stable 4-variable outputs are missing. Run `python volatility_walkforward_research.py` first.")
        return

    st.subheader("What The Four Variables Mean")
    variable_rows = pd.DataFrame(
        [
            {
                "variable": "a = entry delay",
                "simple_explanation": "After a spike is detected, wait this many seconds before entering.",
                "technical_explanation": "Entry index = signal index + a seconds. Tested values: 60, 120, 300 seconds.",
            },
            {
                "variable": "b = volatility multiple",
                "simple_explanation": "How big the move must be before it counts as a real spike.",
                "technical_explanation": "Threshold = b x rolling 63-session percent-return sigma converted to dollars. Tested values: 0.25x, 0.40x, 0.55x, 0.75x, 1.00x.",
            },
            {
                "variable": "c = move window",
                "simple_explanation": "The number of seconds over which the price move is measured.",
                "technical_explanation": "Signal move = price_now - price_c_seconds_ago. Tested values: 900, 1800, 3600 seconds.",
            },
            {
                "variable": "d = holding time",
                "simple_explanation": "How long the position is held after entry, unless midnight Dubai comes first.",
                "technical_explanation": "Exit index = entry index + d seconds, capped by the hard 24:00 Dubai exit. Tested values: 3600, 7200, 14400, 21600 seconds.",
            },
        ]
    )
    st.dataframe(variable_rows, use_container_width=True, hide_index=True)

    row = metrics.iloc[0]
    st.subheader("Stable 4-Variable Walk-Forward Result")
    metric_tiles(row)
    st.caption(
        "Read this as true out-of-sample performance. The selected parameters for each month were chosen only from prior data. "
        "If the optimizer found no stable candidate, it sat out that month."
    )
    if float(row.get("total_pnl_cents", 0.0)) <= 0 or float(row.get("top_trade_removed_pnl_cents", 0.0)) <= 0:
        st.error(
            "Conclusion: the expanded 4-variable stability test did not find a tradeable stable strategy on this dataset. "
            "Simple: even after asking the optimizer to avoid lucky one-trade or one-month winners, the OOS curve still broke. "
            "Technical: the selected training parameters passed in-sample concentration and neighborhood checks, but failed in the next unseen month, "
            "so the current spike logic is regime-sensitive and should not be considered robust without more data or an additional regime/risk filter."
        )

    metric_cols = [
        "run_label",
        "total_pnl_cents",
        "daily_sharpe",
        "max_drawdown_cents",
        "trades",
        "win_rate",
        "profit_factor",
        "profitable_month_rate",
        "top_trade_share",
        "best_month_share",
        "top_trade_removed_pnl_cents",
    ]
    st.dataframe(display_table(metrics[[c for c in metric_cols if c in metrics.columns]]), use_container_width=True, hide_index=True)

    st.subheader("Monthly Parameter Decisions")
    st.caption(
        "Simple: this table explains what the strategy decided before each month. Technical: train columns are in-sample selection diagnostics; "
        "test columns are the next-month OOS result using the selected parameter."
    )
    decision_cols = [
        "test_start",
        "test_end",
        "selected_params",
        "train_stable_score",
        "train_total_pnl_cents",
        "train_sharpe",
        "train_top_trade_removed_pnl_cents",
        "train_top_trade_share",
        "train_best_month_share",
        "train_cluster_positive_neighbors",
        "train_cluster_positive_rate",
        "test_total_pnl_cents",
        "test_sharpe",
        "test_trades",
        "test_top_trade_removed_pnl_cents",
        "test_top_trade_share",
    ]
    st.dataframe(display_table(decisions[[c for c in decision_cols if c in decisions.columns]]), use_container_width=True, hide_index=True)

    if not daily.empty:
        date_col = daily.columns[0]
        pnl_col = "stable_4var_percent" if "stable_4var_percent" in daily.columns else daily.columns[-1]
        curve = daily[[date_col, pnl_col]].copy()
        curve[date_col] = pd.to_datetime(curve[date_col])
        curve["equity_cents"] = curve[pnl_col].fillna(0).cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve[date_col], y=curve["equity_cents"], mode="lines", name="Stable 4-variable OOS equity"))
        fig.update_layout(
            title="OOS Equity Curve: Stable 4-Variable Strategy",
            xaxis_title="Date",
            yaxis_title="Cumulative out-of-sample PnL in cents",
            height=390,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Simple: a smoother upward line would mean the strategy is making money repeatedly. Technical: this is the cumulative sum of daily OOS PnL "
            "from each monthly test period, excluding training PnL."
        )

    st.subheader("Trades Taken")
    st.caption(
        "Each row is one actual OOS trade generated by the stable walk-forward decision for that month. "
        "`move_cents` is the spike that triggered the signal; `threshold_cents` is the volatility-scaled trigger level."
    )
    if trades.empty:
        st.info("No trades were taken because no stable candidate passed the filters.")
    else:
        display_cols = [
            "test_start",
            "signal_time_dubai",
            "entry_time_dubai",
            "exit_time_dubai",
            "side",
            "move_cents",
            "threshold_cents",
            "daily_sigma_dollars",
            "sigma_multiple",
            "entry_price",
            "exit_price",
            "net_pnl_cents",
            "selected_params",
        ]
        st.dataframe(display_table(trades[[c for c in display_cols if c in trades.columns]]), use_container_width=True, hide_index=True)

    with st.expander("Full Stable Parameter Universe And Training Rankings"):
        st.caption(
            "Universe = every a/b/c/d candidate tested before walk-forward selection. Rankings = top training candidates seen before each OOS month."
        )
        st.dataframe(display_table(universe), use_container_width=True, hide_index=True)
        st.dataframe(display_table(rankings), use_container_width=True, hide_index=True)


def render_rare_move_tab(vol_data: dict[str, pd.DataFrame]) -> None:
    st.caption(
        "Purpose: test Shubham's original rare-spike idea directly. This keeps a, b, c, d, but changes b from a volatility multiple into a rarity percentile."
    )
    st.info(
        "Simple: instead of saying 'trade after a 55% daily-vol move', this asks 'is this move larger than 95%, 97.5%, or 99% of recent intraday moves?' "
        "Technical: for each c-second window, the threshold is the rolling percentile of absolute c-second intraday moves from the prior 63 sessions. "
        "The current month is never used to set its own threshold or parameters."
    )

    metrics = vol_data["rare_move_metrics"]
    decisions = vol_data["rare_move_decisions"]
    trades = vol_data["rare_move_trades"]
    daily = vol_data["rare_move_daily"]
    rankings = vol_data["rare_move_rankings"]
    universe = vol_data["rare_move_universe"]

    if metrics.empty:
        st.warning("Rare move walk-forward outputs are missing. Run `python volatility_walkforward_research.py` first.")
        return

    st.subheader("What The Four Variables Mean Here")
    variable_rows = pd.DataFrame(
        [
            {
                "variable": "a = entry delay",
                "simple_explanation": "Wait this many seconds after a rare move is detected before entering.",
                "technical_explanation": "Entry time = signal time + a. Tested values: 60, 120, 300 seconds.",
            },
            {
                "variable": "b = rarity percentile",
                "simple_explanation": "Only trade moves that are unusually large compared with recent intraday moves.",
                "technical_explanation": "Threshold = rolling 95th, 97.5th, or 99th percentile of absolute c-second moves over the prior 63 sessions.",
            },
            {
                "variable": "c = move window",
                "simple_explanation": "The time bucket used to measure the spike.",
                "technical_explanation": "Signal move = price_now - price_c_seconds_ago. Tested values: 300, 900, 1800, 3600 seconds.",
            },
            {
                "variable": "d = holding time",
                "simple_explanation": "How long to hold after entry, unless the Dubai midnight hard stop exits first.",
                "technical_explanation": "Exit = entry + d seconds, capped at 24:00 Dubai. Tested values: 3600, 7200, 14400, 21600 seconds.",
            },
        ]
    )
    st.dataframe(variable_rows, use_container_width=True, hide_index=True)

    row = metrics.iloc[0]
    st.subheader("Rare Move Walk-Forward Result")
    metric_tiles(row)
    if float(row.get("total_pnl_cents", 0.0)) <= 0 or float(row.get("top_trade_removed_pnl_cents", 0.0)) <= 0:
        st.error(
            "Conclusion: the rare-move version did not produce a robust tradeable result on this data. "
            "Simple: filtering for rare moves alone was not enough to create steady growth. "
            "Technical: the walk-forward OOS result failed the stability requirement because total PnL or PnL after removing the best trade is not positive."
        )
    else:
        st.success(
            "Conclusion: this rare-move version passed the first stability check. It still needs inspection for month and trade concentration before live use."
        )

    metric_cols = [
        "run_label",
        "total_pnl_cents",
        "daily_sharpe",
        "max_drawdown_cents",
        "trades",
        "win_rate",
        "profit_factor",
        "profitable_month_rate",
        "top_trade_share",
        "best_month_share",
        "top_trade_removed_pnl_cents",
    ]
    st.dataframe(display_table(metrics[[c for c in metric_cols if c in metrics.columns]]), use_container_width=True, hide_index=True)

    st.subheader("Monthly Walk-Forward Decisions")
    st.caption(
        "Simple: each row shows what parameters were selected before that month and what happened in that unseen month. "
        "Technical: `train_*` columns are computed only on the prior training window; `test_*` columns are out-of-sample."
    )
    decision_cols = [
        "test_start",
        "test_end",
        "selected_params",
        "train_rare_stability_score",
        "train_total_pnl_cents",
        "train_sharpe",
        "train_top_trade_removed_pnl_cents",
        "train_top_trade_share",
        "train_best_month_share",
        "train_cluster_positive_neighbors",
        "train_cluster_positive_rate",
        "test_total_pnl_cents",
        "test_sharpe",
        "test_trades",
        "test_top_trade_removed_pnl_cents",
        "test_top_trade_share",
    ]
    st.dataframe(display_table(decisions[[c for c in decision_cols if c in decisions.columns]]), use_container_width=True, hide_index=True)

    if not daily.empty:
        date_col = daily.columns[0]
        pnl_col = "rare_move_percentile_wf" if "rare_move_percentile_wf" in daily.columns else daily.columns[-1]
        curve = daily[[date_col, pnl_col]].copy()
        curve[date_col] = pd.to_datetime(curve[date_col])
        curve["equity_cents"] = curve[pnl_col].fillna(0).cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve[date_col], y=curve["equity_cents"], mode="lines", name="Rare move OOS equity"))
        fig.update_layout(
            title="OOS Equity Curve: Rare Move Strategy",
            xaxis_title="Date",
            yaxis_title="Cumulative out-of-sample PnL in cents",
            height=390,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Simple: this line should grow steadily if rare moves create a repeatable edge. "
            "Technical: cumulative daily OOS PnL from the monthly walk-forward test periods only."
        )

    st.subheader("Trades Taken")
    st.caption(
        "`rarity_percentile` is the threshold percentile selected from training. `threshold_cents` is the actual rolling percentile threshold for that day and window."
    )
    if trades.empty:
        st.info("No trades were taken because no stable rare-move candidate passed the filters.")
    else:
        display_cols = [
            "test_start",
            "signal_time_dubai",
            "entry_time_dubai",
            "exit_time_dubai",
            "side",
            "move_cents",
            "threshold_cents",
            "rarity_percentile",
            "entry_price",
            "exit_price",
            "net_pnl_cents",
            "selected_params",
        ]
        st.dataframe(display_table(trades[[c for c in display_cols if c in trades.columns]]), use_container_width=True, hide_index=True)

    with st.expander("Rare Move Universe And Training Rankings"):
        st.caption(
            "Universe = every rare-move a/b/c/d candidate tested. Rankings = top training candidates before each OOS month."
        )
        st.dataframe(display_table(universe), use_container_width=True, hide_index=True)
        st.dataframe(display_table(rankings), use_container_width=True, hide_index=True)


def render_regime_rare_tab(vol_data: dict[str, pd.DataFrame]) -> None:
    st.caption(
        "Purpose: test whether the rare-move signal only works in specific market conditions. This is the same rare-spike logic, but trades are allowed only when the selected regime filter is active."
    )
    st.info(
        "Simple: before taking a rare-move trade, the strategy asks: is the recent trend up/down, is volatility high, and what Dubai time bucket is this? "
        "Technical: the walk-forward optimizer selects a, b, c, d, direction, 5-day trend filter, rolling-volatility-rank filter, and signal-time bucket using only prior data."
    )

    metrics = vol_data["regime_rare_metrics"]
    decisions = vol_data["regime_rare_decisions"]
    trades = vol_data["regime_rare_trades"]
    daily = vol_data["regime_rare_daily"]
    rankings = vol_data["regime_rare_rankings"]
    universe = vol_data["regime_rare_universe"]

    if metrics.empty:
        st.warning("Regime-gated rare move outputs are missing. Run `python volatility_walkforward_research.py` first.")
        return

    st.subheader("What The Extra Regime Filters Mean")
    regime_rows = pd.DataFrame(
        [
            {
                "filter": "trend filter",
                "simple_explanation": "Only trade when the last 5 sessions were up, down, or either.",
                "technical_explanation": "`up` means prior 5-session close-to-close return > 0. `down` means < 0. The current test day is not used in this calculation.",
            },
            {
                "filter": "volatility filter",
                "simple_explanation": "Only trade when recent volatility is high, or allow any volatility.",
                "technical_explanation": "`high` means the rolling 63-session percent-to-dollar volatility rank is at least 60%.",
            },
            {
                "filter": "time bucket",
                "simple_explanation": "Only trade rare moves in a selected Dubai-time part of the day.",
                "technical_explanation": "`early` = 11:00-14:00, `mid` = 14:00-18:00, `late` = 18:00-24:00 Dubai signal time.",
            },
        ]
    )
    st.dataframe(regime_rows, use_container_width=True, hide_index=True)

    row = metrics.iloc[0]
    st.subheader("Regime-Gated Rare Move Walk-Forward Result")
    metric_tiles(row)
    if float(row.get("total_pnl_cents", 0.0)) <= 0 or float(row.get("top_trade_removed_pnl_cents", 0.0)) <= 0:
        st.error(
            "Conclusion: the regime gate did not yet create a robust tradeable strategy. "
            "Simple: choosing trend, volatility, and time filters was not enough to make the curve consistently grow. "
            "Technical: OOS PnL or top-trade-removed OOS PnL is non-positive, so the apparent edge is still not stable."
        )
    else:
        st.success(
            "Conclusion: this passed the first OOS stability check. Next inspect month concentration, top-trade dependence, and whether the selected regimes make economic sense."
        )

    metric_cols = [
        "run_label",
        "total_pnl_cents",
        "daily_sharpe",
        "max_drawdown_cents",
        "trades",
        "win_rate",
        "profit_factor",
        "profitable_month_rate",
        "top_trade_share",
        "best_month_share",
        "top_trade_removed_pnl_cents",
    ]
    st.dataframe(display_table(metrics[[c for c in metric_cols if c in metrics.columns]]), use_container_width=True, hide_index=True)

    st.subheader("Monthly Walk-Forward Decisions")
    st.caption(
        "Simple: each row shows the market regime and rare-move parameter selected before that OOS month. "
        "Technical: train columns are in-sample diagnostics; test columns are next-month OOS results."
    )
    decision_cols = [
        "test_start",
        "test_end",
        "selected_params",
        "train_rare_stability_score",
        "train_total_pnl_cents",
        "train_sharpe",
        "train_top_trade_removed_pnl_cents",
        "train_top_trade_share",
        "train_best_month_share",
        "train_cluster_positive_neighbors",
        "train_cluster_positive_rate",
        "test_total_pnl_cents",
        "test_sharpe",
        "test_trades",
        "test_top_trade_removed_pnl_cents",
        "test_top_trade_share",
    ]
    st.dataframe(display_table(decisions[[c for c in decision_cols if c in decisions.columns]]), use_container_width=True, hide_index=True)

    if not daily.empty:
        date_col = daily.columns[0]
        pnl_col = "regime_rare_move_wf" if "regime_rare_move_wf" in daily.columns else daily.columns[-1]
        curve = daily[[date_col, pnl_col]].copy()
        curve[date_col] = pd.to_datetime(curve[date_col])
        curve["equity_cents"] = curve[pnl_col].fillna(0).cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve[date_col], y=curve["equity_cents"], mode="lines", name="Regime-gated rare move OOS equity"))
        fig.update_layout(
            title="OOS Equity Curve: Regime-Gated Rare Move Strategy",
            xaxis_title="Date",
            yaxis_title="Cumulative out-of-sample PnL in cents",
            height=390,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Simple: this should rise smoothly if the regime gate is helping. Technical: cumulative daily OOS PnL from monthly walk-forward test periods only."
        )

    st.subheader("Trades Taken")
    st.caption(
        "`trend_5d_pct`, `vol_rank`, and `time_bucket` explain why the selected regime allowed that trade. `threshold_cents` is the rolling rare-move threshold."
    )
    if trades.empty:
        st.info("No trades were taken because no stable regime-gated rare-move candidate passed the filters.")
    else:
        display_cols = [
            "test_start",
            "signal_time_dubai",
            "entry_time_dubai",
            "exit_time_dubai",
            "side",
            "move_cents",
            "threshold_cents",
            "rarity_percentile",
            "trend_filter",
            "vol_filter",
            "time_bucket",
            "trend_5d_pct",
            "vol_rank",
            "entry_price",
            "exit_price",
            "net_pnl_cents",
            "selected_params",
        ]
        st.dataframe(display_table(trades[[c for c in display_cols if c in trades.columns]]), use_container_width=True, hide_index=True)

    with st.expander("Regime-Gated Rare Move Universe And Training Rankings"):
        st.caption(
            "Universe = every rare-move plus regime-filter candidate. Rankings = top training candidates before each OOS month."
        )
        st.dataframe(display_table(universe), use_container_width=True, hide_index=True)
        st.dataframe(display_table(rankings), use_container_width=True, hide_index=True)


def render_stability_audit_tab(vol_data: dict[str, pd.DataFrame]) -> None:
    st.caption(
        "This tab is for distrust. It shows whether PnL is coming from one month, one trade, or repeated clock-time artifacts."
    )
    st.info(
        "The previous repeated 11:02 trades were caused by the 11:00 session boundary: a 30-minute move from before 11:00 could trigger exactly at 11:00, "
        "then enter at 11:02 after the 120-second delay. The current research run filters those out by requiring the full lookback window inside session."
    )
    monthly = vol_data["stability_monthly"]
    repeated = vol_data["stability_repeated"]
    top_trades = vol_data["stability_top_trades"]

    if monthly.empty and repeated.empty and top_trades.empty:
        st.warning("Stability audit outputs are missing. Run `python volatility_walkforward_research.py` first.")
        return

    run_options = sorted(set(monthly.get("run_key", pd.Series(dtype=str)).dropna().tolist()) | set(top_trades.get("run_key", pd.Series(dtype=str)).dropna().tolist()))
    if not run_options:
        st.info("No audit rows available.")
        return
    run_key = st.selectbox("Audit run", run_options, format_func=clean_label, key="audit_run_key")

    st.subheader("Monthly Contribution")
    selected_monthly = monthly[monthly["run_key"] == run_key].copy() if not monthly.empty else pd.DataFrame()
    if not selected_monthly.empty:
        st.dataframe(display_table(selected_monthly), use_container_width=True, hide_index=True)
        fig = px.bar(
            selected_monthly,
            x="month",
            y="month_pnl_cents",
            color="month_pnl_cents",
            color_continuous_scale="RdYlGn",
            title="Monthly Contribution To OOS PnL",
        )
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Largest Absolute Trades")
    selected_top = top_trades[top_trades["run_key"] == run_key].copy() if not top_trades.empty else pd.DataFrame()
    if selected_top.empty:
        st.info("No top-trade rows for this run.")
    else:
        st.dataframe(display_table(selected_top), use_container_width=True, hide_index=True)

    st.subheader("Repeated Entry Clock Times")
    selected_repeated = repeated[repeated["run_key"] == run_key].copy() if not repeated.empty else pd.DataFrame()
    if selected_repeated.empty:
        st.success("No repeated entry clock times with count >= 2 for this run.")
    else:
        st.dataframe(display_table(selected_repeated.sort_values("count", ascending=False)), use_container_width=True, hide_index=True)


def render_winning_percent_backtest_tab(vol_data: dict[str, pd.DataFrame]) -> None:
    st.caption(
        "This is a fixed-parameter diagnostic, not walk-forward. It uses the best single Percent-to-Dollar parameter after looking at the full period, "
        "so it can be overfit and should not be treated as live-like performance."
    )
    metrics = vol_data["winning_percent_metrics"]
    daily = vol_data["winning_percent_daily"]
    trades = vol_data["winning_percent_trades"]

    if metrics.empty:
        st.warning("Winning percent-vol backtest output is missing. Run `python volatility_walkforward_research.py` first.")
        return

    row = metrics.iloc[0]
    st.subheader("Winning Fixed Percent-Vol Parameters")
    st.code(str(row["selected_params"]), language="text")
    metric_tiles(row)

    detail_cols = [
        "selection_rule",
        "backtest_start",
        "backtest_end",
        "delay_s",
        "sigma_multiple",
        "move_window_s",
        "hold_s",
        "vol_window_days",
        "vol_method",
        "signal_mode",
        "bad_hour_rule",
    ]
    st.dataframe(display_table(metrics[[c for c in detail_cols if c in metrics.columns]]), use_container_width=True, hide_index=True)

    if not daily.empty:
        date_col = daily.columns[0]
        pnl_col = "daily_pnl_cents" if "daily_pnl_cents" in daily.columns else daily.columns[-1]
        curve = daily[[date_col, pnl_col]].copy()
        curve[date_col] = pd.to_datetime(curve[date_col])
        curve["equity_cents"] = curve[pnl_col].fillna(0).cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve[date_col], y=curve["equity_cents"], mode="lines", name="winning percent-vol backtest"))
        fig.update_layout(
            title="Equity Curve: Winning Fixed Percent-Vol Backtest",
            xaxis_title="Date",
            yaxis_title="Cumulative cents",
            height=390,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trade List")
    if trades.empty:
        st.info("No trades in this backtest.")
        return

    display_cols = [
        "signal_time",
        "signal_time_dubai",
        "entry_time",
        "entry_time_dubai",
        "exit_time",
        "exit_time_dubai",
        "side",
        "move_cents",
        "threshold_cents",
        "daily_sigma_dollars",
        "sigma_multiple",
        "entry_price",
        "exit_price",
        "gross_pnl_cents",
        "cost_cents",
        "net_pnl_cents",
        "signal_mode",
        "bad_hour_rule",
        "selected_params",
    ]
    shown = trades[[c for c in display_cols if c in trades.columns]].copy()
    st.dataframe(display_table(shown), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Winning Percent-Vol Trades CSV",
        data=trades.to_csv(index=False).encode("utf-8"),
        file_name="winning_percent_vol_backtest_trades.csv",
        mime="text/csv",
        key="winning_percent_vol_trade_download",
    )


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
            "Max trades per session",
            min_value=0,
            value=DEFAULT_MAX_TRADES_PER_DAY,
            step=1,
            help="Set 0 for unlimited trades.",
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
        st.warning("No data was available for this backtest.")
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
    source = st.selectbox("Source", source_options, key="out_of_session_source")
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
        selector = st.selectbox(
            "Selector objective",
            selectors,
            index=selectors.index("daily_sharpe") if "daily_sharpe" in selectors else 0,
            format_func=objective_label,
            key="out_of_session_selector_objective",
        )
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
    st.dataframe(display_table(shown), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Excluded Trades CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name="excluded_out_of_session_trades.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Price Action Research Dashboard")
    data = load_outputs()
    vol_data = load_vol_outputs()

    baseline_tab, stable_4var_tab, rare_move_tab, regime_rare_tab, robust_tab, fixed_diagnostic_tab, audit_tab = st.tabs(
        [
            "Baseline Percent-Vol WF",
            "Stable 4-Variable WF",
            "Rare Move WF",
            "Regime-Gated Rare WF",
            "Robust Percent-Vol WF",
            "Fixed Winner Diagnostic",
            "Stability Audit",
        ]
    )
    with baseline_tab:
        render_volatility_tab(vol_data, data, "percent_to_dollar")
    with stable_4var_tab:
        render_stable_4var_tab(vol_data)
    with rare_move_tab:
        render_rare_move_tab(vol_data)
    with regime_rare_tab:
        render_regime_rare_tab(vol_data)
    with robust_tab:
        render_robust_walkforward_tab(vol_data)
    with fixed_diagnostic_tab:
        render_winning_percent_backtest_tab(vol_data)
    with audit_tab:
        render_stability_audit_tab(vol_data)


if __name__ == "__main__":
    main()
