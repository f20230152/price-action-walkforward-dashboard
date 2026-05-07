from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "outputs" / "clean_walkforward"
REPORT_XLSX = OUT_DIR / "walkforward_report.xlsx"


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


def metric_tiles(row: pd.Series) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("WF Net PnL", f"{fmt_num(row['total_pnl_cents'])} c")
    c2.metric("WF Sharpe", fmt_num(row["daily_sharpe"]))
    c3.metric("Max DD", f"{fmt_num(row['max_drawdown_cents'])} c")
    c4.metric("Trades", f"{int(row['trades']):,}")
    c5.metric("Trades/Day", fmt_num(row["trades_per_day"], 2))
    c6.metric("Profit Factor", fmt_num(row["profit_factor"]))


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


def main() -> None:
    st.title("Clean Walk-Forward Price Action Dashboard")
    st.caption(
        "Only walk-forward results are shown. Parameters are selected using the prior 3 months and traded on the next "
        "unseen month. Fixed/single backtests are intentionally removed."
    )

    data = load_outputs()
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


if __name__ == "__main__":
    main()
