from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
OUT = APP_DIR / "outputs"


REQUIRED_FILES = [
    "data_coverage.csv",
    "prompt_contract_calendar.csv",
    "parameter_universe_stage1.csv",
    "parameter_universe_stage2.csv",
    "train_rankings.csv",
    "decisions.csv",
    "trades.csv",
    "daily_pnl.csv",
    "monthly_pnl.csv",
    "oos_equity.csv",
    "fold_winners.csv",
    "fill_model_comparison.csv",
    "robustness_report.csv",
    "randomized_entry_distribution.csv",
    "full_period_backtest_trades.csv",
    "full_period_backtest_equity.csv",
    "validation_checks.csv",
    "verdict.json",
]


@st.cache_data(show_spinner=False)
def load_outputs() -> dict:
    missing = [name for name in REQUIRED_FILES if not (OUT / name).exists()]
    if missing:
        return {"missing": missing}
    data = {}
    for name in REQUIRED_FILES:
        if name.endswith(".json"):
            data[name] = json.loads((OUT / name).read_text(encoding="utf-8"))
        elif name.endswith(".md"):
            data[name] = (OUT / name).read_text(encoding="utf-8")
        else:
            data[name] = pd.read_csv(OUT / name)
    optional = [
        "robustness_summary.csv",
        "final_parameter_choices.csv",
        "previous_run_summary.json",
        "fix_comparison.md",
        "fix_comparison_6m.csv",
        "fix_comparison_3m.csv",
        "fix_trade_count_diff.csv",
        "fix_winner_stability_comparison.csv",
    ]
    for name in optional:
        path = OUT / name
        if not path.exists():
            data[name] = {} if name.endswith(".json") else ("" if name.endswith(".md") else pd.DataFrame())
        elif name.endswith(".json"):
            data[name] = json.loads(path.read_text(encoding="utf-8"))
        elif name.endswith(".md"):
            data[name] = path.read_text(encoding="utf-8")
        else:
            data[name] = pd.read_csv(path)
    return data


def download_table(df: pd.DataFrame, label: str, key: str) -> None:
    st.download_button(
        label=f"Download {label} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{label.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=key,
    )


def show_table(df: pd.DataFrame, label: str, key: str, height: int = 360) -> None:
    st.subheader(label)
    st.dataframe(df, use_container_width=True, height=height)
    download_table(df, label, key)


def verdict_color(verdict: str) -> str:
    if verdict == "STABLE EDGE FOUND":
        return "#0b7a3b"
    if verdict == "WEAK / EPISODIC EDGE":
        return "#9a6700"
    return "#b42318"


def render_verdict_banner(verdict: dict) -> None:
    label = verdict.get("verdict", "UNKNOWN")
    color = verdict_color(label)
    reasons = verdict.get("reasons", [])
    st.markdown(
        f"""
        <div style="border-left: 8px solid {color}; padding: 0.75rem 1rem; background: #f7f7f7; margin-bottom: 1rem;">
          <div style="font-size: 1.4rem; font-weight: 700; color: {color};">{label}</div>
          <div style="font-size: 0.95rem; color: #333;">Realistic benchmark is Model B: actual bid/ask on parquet months, CSV fallback to dynamic cost for Dec-Mar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if reasons:
        with st.expander("Failed or blocking robustness conditions", expanded=(label != "STABLE EDGE FOUND")):
            for reason in reasons:
                st.markdown(f"- `{reason}`")


def pass_style(df: pd.DataFrame):
    def row_style(row):
        if "pass" in row.index and str(row["pass"]).lower() == "false":
            return ["background-color: #fde8e8"] * len(row)
        if "pass" in row.index and str(row["pass"]).lower() == "true":
            return ["background-color: #e8f5e9"] * len(row)
        return [""] * len(row)

    return df.style.apply(row_style, axis=1)


def selected_summary(verdict: dict) -> pd.DataFrame:
    rows = []
    for fill_model, item in verdict.get("selected", {}).items():
        rows.append({"fill_model": fill_model, **item})
    return pd.DataFrame(rows)


def previous_selected_summary(previous: dict) -> pd.DataFrame:
    rows = []
    for fill_model, item in previous.get("selected", {}).items():
        rows.append({"fill_model": fill_model, **item})
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Fresh Brent Walk-Forward", layout="wide")
    st.title("Fresh Brent Price-Action Walk-Forward Dashboard")
    outputs = load_outputs()
    if outputs.get("missing"):
        st.error("Research outputs are missing. Run `python -m src.run_research` first.")
        st.write(outputs["missing"])
        return

    verdict = outputs["verdict.json"]
    previous = outputs["previous_run_summary.json"]
    fix_md = outputs["fix_comparison.md"]
    fix_6m = outputs["fix_comparison_6m.csv"]
    fix_3m = outputs["fix_comparison_3m.csv"]
    fix_trade_count = outputs["fix_trade_count_diff.csv"]
    fix_stability = outputs["fix_winner_stability_comparison.csv"]
    coverage = outputs["data_coverage.csv"]
    prompt_calendar = outputs["prompt_contract_calendar.csv"]
    rankings = outputs["train_rankings.csv"]
    decisions = outputs["decisions.csv"]
    trades = outputs["trades.csv"]
    daily = outputs["daily_pnl.csv"]
    monthly = outputs["monthly_pnl.csv"]
    equity = outputs["oos_equity.csv"]
    winners = outputs["fold_winners.csv"]
    robustness = outputs["robustness_report.csv"]
    randomized = outputs["randomized_entry_distribution.csv"]
    full_trades = outputs["full_period_backtest_trades.csv"]
    full_equity = outputs["full_period_backtest_equity.csv"]
    validation = outputs["validation_checks.csv"]
    fill_comparison = outputs["fill_model_comparison.csv"]
    final_choices = outputs["final_parameter_choices.csv"]

    tabs = st.tabs(
        [
            "Verdict & Summary",
            "Walk-Forward Schedule",
            "OOS Performance",
            "Parameter Stability",
            "Robustness Report",
            "All Trades",
            "Monthly Rebalance Decisions",
            "Full-Period Backtest",
            "Data Coverage",
            "Fix Comparison",
        ]
    )

    with tabs[0]:
        view = st.radio("Run view", ["After fixes", "Before fixes"], horizontal=True)
        if previous.get("verdict") and previous.get("verdict") != verdict.get("verdict"):
            st.warning("Verdict changed after fixing roll + sigma bugs - see Fix Comparison tab.")
        if view == "Before fixes":
            render_verdict_banner({"verdict": previous.get("verdict", "UNKNOWN"), "reasons": []})
            summary_df = previous_selected_summary(previous)
        else:
            render_verdict_banner(verdict)
            summary_df = selected_summary(verdict)
        cols = st.columns(4)
        if not summary_df.empty:
            model_b = summary_df[summary_df["fill_model"] == "actual_bid_ask"]
            model_a = summary_df[summary_df["fill_model"] == "mid_dynamic_cost"]
            for idx, (title, frame) in enumerate([("Model A PnL", model_a), ("Model B PnL", model_b)]):
                value = frame["oos_total_pnl_cents"].iloc[0] if not frame.empty else float("nan")
                cols[idx].metric(title, f"{value:,.1f}c")
            for idx, (title, frame) in enumerate([("Model A Sharpe", model_a), ("Model B Sharpe", model_b)], start=2):
                value = frame["oos_daily_sharpe"].iloc[0] if not frame.empty else float("nan")
                cols[idx].metric(title, f"{value:,.2f}")
        show_table(summary_df, "Selected Parameter Sets", "summary_selected")
        if view == "After fixes":
            show_table(fill_comparison, "Fill Model Comparison", "summary_fill_comparison")
            key_tests = robustness[robustness["test_name"].isin(["top_3_trade_removed_pnl", "win_month_rate", "randomized_entry_p_value", "bootstrap_mean_trade_ci_low"])].copy()
            st.subheader("Key Robustness Tests")
            st.dataframe(pass_style(key_tests), use_container_width=True, height=360)
            download_table(key_tests, "Key Robustness Tests", "summary_key_tests")

    with tabs[1]:
        render_verdict_banner(verdict)
        st.write("Both schedules are causal. Each fold optimizes only on its train months, then evaluates the selected set once on the next test month.")
        show_table(winners, "Per-Fold Winners", "wf_winners", height=440)
        top_rankings = rankings.sort_values(["stage", "schedule", "fill_model", "fold", "rank"]).groupby(["stage", "schedule", "fill_model", "fold"]).head(5)
        show_table(top_rankings, "Top 5 Train Rankings Per Fold", "wf_rankings", height=440)

    with tabs[2]:
        render_verdict_banner(verdict)
        if not equity.empty:
            fig = px.line(equity, x="date", y="cum_pnl_cents", color="fill_model", line_dash="schedule", title="OOS Cumulative PnL")
            st.plotly_chart(fig, use_container_width=True)
        if not monthly.empty:
            fig = px.bar(monthly, x="month", y="pnl_cents", color="fill_model", facet_row="schedule", barmode="group", title="OOS Monthly PnL")
            st.plotly_chart(fig, use_container_width=True)
        show_table(equity, "OOS Equity", "oos_equity")
        show_table(monthly, "Monthly PnL", "monthly_pnl")
        show_table(daily, "Daily PnL", "daily_pnl")

    with tabs[3]:
        render_verdict_banner(verdict)
        stage2 = rankings[rankings["stage"] == 2].copy()
        if stage2.empty:
            st.warning("No Stage 2 ranking data available.")
            show_table(rankings, "Train Rankings", "stability_rankings")
        else:
            c1, c2, c3 = st.columns(3)
            schedule = c1.selectbox("Schedule", sorted(stage2["schedule"].unique()), key="stab_schedule")
            fill_model = c2.selectbox("Fill model", sorted(stage2["fill_model"].unique()), key="stab_fill")
            fold = c3.selectbox("Fold", sorted(stage2[(stage2["schedule"] == schedule) & (stage2["fill_model"] == fill_model)]["fold"].unique()), key="stab_fold")
            subset = stage2[(stage2["schedule"] == schedule) & (stage2["fill_model"] == fill_model) & (stage2["fold"] == fold)].copy()
            if not subset.empty:
                ab = subset.pivot_table(index="a", columns="b", values="composite_score", aggfunc="max")
                fig1 = px.imshow(ab, text_auto=".2f", aspect="auto", title="Composite Heatmap: a vs b")
                st.plotly_chart(fig1, use_container_width=True)
                cd = subset.pivot_table(index="c", columns="d", values="composite_score", aggfunc="max")
                fig2 = px.imshow(cd, text_auto=".2f", aspect="auto", title="Composite Heatmap: c vs d")
                st.plotly_chart(fig2, use_container_width=True)
            show_table(subset.sort_values("composite_score", ascending=False), "Selected Fold Ranking Surface", "stability_surface", height=500)

    with tabs[4]:
        render_verdict_banner(verdict)
        st.subheader("Robustness Report")
        st.dataframe(pass_style(robustness), use_container_width=True, height=440)
        download_table(robustness, "Robustness Report", "robustness_report")
        if not randomized.empty:
            c1, c2 = st.columns(2)
            schedule = c1.selectbox("Schedule", sorted(randomized["schedule"].unique()), key="rand_schedule")
            fill_model = c2.selectbox("Fill model", sorted(randomized["fill_model"].unique()), key="rand_fill")
            subset = randomized[(randomized["schedule"] == schedule) & (randomized["fill_model"] == fill_model)]
            fig = px.histogram(subset, x="randomized_pnl_cents", nbins=50, title="Randomized Entry Benchmark")
            real = subset["real_oos_pnl_cents"].iloc[0] if not subset.empty else 0.0
            fig.add_vline(x=real, line_color="red", line_width=3, annotation_text="Real OOS")
            st.plotly_chart(fig, use_container_width=True)
            show_table(subset, "Randomized Entry Distribution", "randomized_distribution", height=360)

    with tabs[5]:
        render_verdict_banner(verdict)
        filtered = trades.copy()
        c1, c2, c3 = st.columns(3)
        fill_options = ["All"] + sorted(filtered["fill_model"].dropna().unique())
        schedule_options = ["All"] + sorted(filtered["schedule"].dropna().unique())
        fold_options = ["All"] + sorted(filtered["fold"].dropna().unique())
        fill_choice = c1.selectbox("Fill model", fill_options, key="trade_fill")
        schedule_choice = c2.selectbox("Schedule", schedule_options, key="trade_schedule")
        fold_choice = c3.selectbox("Fold", fold_options, key="trade_fold")
        if fill_choice != "All":
            filtered = filtered[filtered["fill_model"] == fill_choice]
        if schedule_choice != "All":
            filtered = filtered[filtered["schedule"] == schedule_choice]
        if fold_choice != "All":
            filtered = filtered[filtered["fold"] == fold_choice]
        display_cols = [
            "entry_time_utc",
            "entry_time_dubai",
            "direction",
            "entry_px",
            "exit_time_utc",
            "exit_px",
            "hold_seconds",
            "pnl_cents",
            "fill_model",
            "schedule",
            "fold",
            "parameter_set",
        ]
        show_table(filtered[[c for c in display_cols if c in filtered.columns]], "Visible Trades", "visible_trades", height=540)

    with tabs[6]:
        render_verdict_banner(verdict)
        filtered = decisions.copy()
        c1, c2, c3 = st.columns(3)
        stage = c1.selectbox("Stage", sorted(filtered["stage"].dropna().unique()), key="dec_stage")
        fill_model = c2.selectbox("Fill model", sorted(filtered["fill_model"].dropna().unique()), key="dec_fill")
        schedule = c3.selectbox("Schedule", sorted(filtered["schedule"].dropna().unique()), key="dec_schedule")
        filtered = filtered[(filtered["stage"] == stage) & (filtered["fill_model"] == fill_model) & (filtered["schedule"] == schedule)]
        show_table(filtered, "Monthly Rebalance Decisions", "rebalance_decisions", height=560)

    with tabs[7]:
        render_verdict_banner(verdict)
        st.warning("This chart is a visual full-period backtest. Training months are in-sample and must not be treated as robustness evidence.")
        if not full_equity.empty:
            fig = px.line(full_equity, x="entry_time_utc", y="cum_pnl_cents", color="fill_model", title="Full-Period Visual Equity")
            st.plotly_chart(fig, use_container_width=True)
        show_table(final_choices, "Final Parameter Choices", "full_final_choices")
        show_table(full_equity, "Full-Period Equity", "full_equity")
        show_table(full_trades, "Full-Period Trades", "full_trades", height=520)

    with tabs[8]:
        render_verdict_banner(verdict)
        st.info("Jan-Nov parquet data is now an explicit prompt-contract series: calendar month plus two delivery months using ICE Brent month codes. Example: May -> July contract (N), June -> August contract (Q). Dec-Mar CSV remains the legacy continuous %BRN 1!-ICE series.")
        if not coverage.empty:
            fig = px.bar(coverage, x="month", y="row_count", color="source_type", title="Loaded Row Count by Month")
            st.plotly_chart(fig, use_container_width=True)
        show_table(coverage, "Data Coverage", "data_coverage")
        show_table(prompt_calendar, "Prompt Contract Calendar", "prompt_contract_calendar")
        st.subheader("Validation Checks")
        st.dataframe(pass_style(validation), use_container_width=True, height=360)
        download_table(validation, "Validation Checks", "validation_checks")

    with tabs[9]:
        render_verdict_banner(verdict)
        if not fix_md:
            st.warning("Fix comparison outputs are not generated yet. Run `python -m src.run_research` after the D-drive parquet source is available.")
            return
        st.markdown(fix_md)
        show_table(fix_6m, "Fix Comparison 6m", "fix_comparison_6m")
        show_table(fix_3m, "Fix Comparison 3m", "fix_comparison_3m")
        show_table(fix_trade_count, "Fix Trade Count Diff", "fix_trade_count")
        show_table(fix_stability, "Fix Winner Stability", "fix_stability")
        heatmap_paths = sorted((OUT / "diagnostics").glob("*_postfix.svg"))
        if heatmap_paths:
            st.subheader("Post-Fix Representative Heatmaps")
            for path in heatmap_paths:
                st.image(str(path), caption=path.name, use_container_width=True)


if __name__ == "__main__":
    main()
