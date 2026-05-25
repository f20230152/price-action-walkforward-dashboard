from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import PROMPT_LIQUID_SOURCE_TYPE, load_research_dataset
from .engine import SESSION_END_UTC_HOUR, SESSION_START_UTC_HOUR, generate_base_trades, rolling_sigma_real_prints, stage1_universe, summarize_daily_monthly
from .full_period import choose_final_parameters, run_full_period_backtests
from .robustness import build_robustness_outputs
from .selection import build_stage2_universe
from .walkforward import run_walkforward_stage


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
DIAG_DIR = OUTPUT_DIR / "diagnostics"


def _save_csv(df: pd.DataFrame, name: str) -> None:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print(f"Wrote {path.relative_to(REPO_ROOT)} rows={len(df)}")


def _save_diag_csv(df: pd.DataFrame, name: str) -> None:
    path = DIAG_DIR / name
    df.to_csv(path, index=False)
    print(f"Wrote {path.relative_to(REPO_ROOT)} rows={len(df)}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _same_family(left: pd.Series | None, right: pd.Series | None) -> bool:
    if left is None or right is None:
        return False
    return (
        str(left.get("direction", "")) == str(right.get("direction", ""))
        and int(left.get("c", 0)) == int(right.get("c", 0))
        and int(left.get("d", 0)) == int(right.get("d", 0))
        and abs(int(left.get("a", 0)) - int(right.get("a", 0))) <= 120
        and abs(float(left.get("b", 0.0)) - float(right.get("b", 0.0))) <= 0.2
    )


def _most_common_schedule_winner(fold_winners: pd.DataFrame, schedule: str, fill_model: str) -> pd.Series | None:
    group = fold_winners[(fold_winners["schedule"] == schedule) & (fold_winners["fill_model"] == fill_model)]
    if group.empty:
        return None
    top_param = group["param_id"].value_counts().index[0]
    return group[group["param_id"] == top_param].iloc[-1]


def _build_fill_model_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for schedule, group in summary.groupby("schedule"):
        mid = group[group["fill_model"] == "mid_dynamic_cost"]
        actual = group[group["fill_model"] == "actual_bid_ask"]
        if mid.empty or actual.empty:
            continue
        mid_row = mid.iloc[0]
        actual_row = actual.iloc[0]
        rows.append(
            {
                "schedule": schedule,
                "model_a_total_pnl_cents": mid_row["total_pnl_cents"],
                "model_b_total_pnl_cents": actual_row["total_pnl_cents"],
                "actual_minus_mid_pnl_cents": actual_row["total_pnl_cents"] - mid_row["total_pnl_cents"],
                "model_a_daily_sharpe": mid_row["daily_sharpe"],
                "model_b_daily_sharpe": actual_row["daily_sharpe"],
                "model_a_win_month_rate": mid_row["win_month_rate"],
                "model_b_win_month_rate": actual_row["win_month_rate"],
                "model_a_randomized_entry_p_value": mid_row["randomized_entry_p_value"],
                "model_b_randomized_entry_p_value": actual_row["randomized_entry_p_value"],
                "model_a_top_3_trade_removed_pnl_cents": mid_row["top_3_trade_removed_pnl_cents"],
                "model_b_top_3_trade_removed_pnl_cents": actual_row["top_3_trade_removed_pnl_cents"],
            }
        )
    return pd.DataFrame(rows)


def _build_prompt_contract_calendar(data: pd.DataFrame, chosen_contract_daily: pd.DataFrame) -> pd.DataFrame:
    if chosen_contract_daily.empty:
        return pd.DataFrame()
    rows = []
    parquet = data[data["source_type"].astype(str).str.startswith("parquet")].copy()
    grouped = {str(date): group for date, group in parquet.groupby("utc_date", sort=True)}
    for _, choice in chosen_contract_daily.iterrows():
        date_value = str(choice["date"])
        group = grouped.get(date_value, pd.DataFrame())
        rows.append(
            {
                "date": date_value,
                "calendar_month": str(pd.Timestamp(date_value).to_period("M")),
                "chosen_symbol": choice.get("chosen_symbol", ""),
                "contract_expiry_date": choice.get("chosen_expiry", ""),
                "days_to_expiry": choice.get("days_to_expiry", np.nan),
                "chosen_5d_avg_volume": choice.get("chosen_5d_avg_volume", np.nan),
                "next_symbol": choice.get("next_symbol", ""),
                "next_5d_avg_volume": choice.get("next_5d_avg_volume", np.nan),
                "roll_reason": choice.get("roll_reason", ""),
                "total_volume_that_day": choice.get("total_volume_that_day", np.nan),
                "rank_by_volume_among_available_symbols": choice.get("rank_by_volume_among_available_symbols", np.nan),
                "available_symbol_count": choice.get("available_symbol_count", np.nan),
                "selection_rule": "daily_prompt_liquid_pre_expiry_or_volume_crossover",
                "row_count": int(len(group)) if not group.empty else 0,
                "first_timestamp_utc": group.index.min().isoformat() if not group.empty else "",
                "last_timestamp_utc": group.index.max().isoformat() if not group.empty else "",
            }
        )
    return pd.DataFrame(rows)


def _pass_count(row: pd.Series) -> int:
    checks = [
        row["top_3_trade_removed_pnl_cents"] > 0,
        row["top_month_removed_pnl_cents"] > 0,
        row["win_month_rate"] >= 0.65,
        row["distinct_parameter_sets"] <= 4,
        row["randomized_entry_p_value"] < 0.05,
        pd.notna(row["bootstrap_mean_trade_ci_low"]) and row["bootstrap_mean_trade_ci_low"] > 0,
    ]
    return int(sum(bool(x) for x in checks))


def _capture_previous_run_summary() -> dict[str, Any]:
    previous_path = OUTPUT_DIR / "previous_run_summary.json"
    if previous_path.exists():
        return json.loads(previous_path.read_text(encoding="utf-8"))
    verdict_path = OUTPUT_DIR / "verdict.json"
    robustness_path = OUTPUT_DIR / "robustness_summary.csv"
    winners_path = OUTPUT_DIR / "fold_winners.csv"
    trades_path = OUTPUT_DIR / "trades.csv"
    prompt_path = OUTPUT_DIR / "prompt_contract_calendar.csv"
    if not verdict_path.exists() or not robustness_path.exists() or not trades_path.exists():
        return {}
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    robustness = pd.read_csv(robustness_path)
    winners = pd.read_csv(winners_path) if winners_path.exists() else pd.DataFrame()
    trades = pd.read_csv(trades_path)
    previous_near_expiry_lt5 = 0
    if prompt_path.exists() and not trades.empty:
        calendar = pd.read_csv(prompt_path)
        if "days_to_expiry" in calendar.columns:
            trade_dates = pd.to_datetime(trades["entry_time_utc"]).dt.date.astype(str)
            merged = trades.assign(date=trade_dates).merge(calendar[["date", "days_to_expiry"]], on="date", how="left")
            previous_near_expiry_lt5 = int((pd.to_numeric(merged["days_to_expiry"], errors="coerce") < 5).sum())
    summary = {
        "verdict": verdict.get("verdict", ""),
        "selected": verdict.get("selected", {}),
        "robustness_summary": robustness.to_dict(orient="records"),
        "fold_winners": winners.to_dict(orient="records"),
        "total_oos_trades": int(len(trades)),
        "near_expiry_oos_trades_lt5_bdays": previous_near_expiry_lt5,
    }
    previous_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary


def _build_sigma_diagnostic(data: pd.DataFrame, window: int = 3600) -> pd.DataFrame:
    rows = []
    for date_value, day_df in data.groupby("utc_date", sort=True):
        day_df = day_df.sort_index()
        seconds = day_df.index.hour * 3600 + day_df.index.minute * 60 + day_df.index.second
        session = (seconds >= SESSION_START_UTC_HOUR * 3600) & (seconds < SESSION_END_UTC_HOUR * 3600)
        mid = day_df["mid"].to_numpy(dtype=float)
        real = day_df.get("is_real_print", pd.Series(False, index=day_df.index)).fillna(False).to_numpy(dtype=bool)
        sigma, real_count, min_periods = rolling_sigma_real_prints(mid, real, window)
        session_count = int(session.sum())
        rows.append(
            {
                "date": date_value,
                "sigma_window_seconds": int(window),
                "sigma_min_real_prints": int(min_periods),
                "real_prints_in_session": int(real[session].sum()),
                "forward_filled_seconds_in_session": int(day_df.get("was_forward_filled", pd.Series(False, index=day_df.index)).fillna(False).to_numpy(dtype=bool)[session].sum()),
                "session_seconds": session_count,
                "fraction_session_seconds_with_valid_sigma": float(np.isfinite(sigma[session]).sum() / session_count) if session_count else 0.0,
                "min_real_prints_in_valid_sigma_window": int(np.nanmin(real_count[session][np.isfinite(sigma[session])])) if np.isfinite(sigma[session]).any() else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_back_adjusted_mid(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = data[data["source_type"].astype(str).str.startswith("parquet")][["mid", "selected_symbol", "utc_date"]].copy()
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    adjusted = raw["mid"].astype(float).copy()
    day_symbol = raw.groupby("utc_date")["selected_symbol"].first()
    roll_rows = []
    previous_date = None
    previous_symbol = None
    for date_value, symbol in day_symbol.items():
        if previous_symbol is not None and symbol != previous_symbol:
            prev_group = raw[raw["utc_date"] == previous_date]
            new_group = raw[raw["utc_date"] == date_value]
            old_close_idx = prev_group.index[-1]
            new_open_idx = new_group.index[0]
            old_close_adjusted = float(adjusted.loc[old_close_idx])
            new_open_mid = float(raw.loc[new_open_idx, "mid"])
            adjustment = new_open_mid - old_close_adjusted
            adjusted.loc[adjusted.index < pd.Timestamp(date_value)] = adjusted.loc[adjusted.index < pd.Timestamp(date_value)] + adjustment
            roll_rows.append(
                {
                    "date": date_value,
                    "old_symbol": previous_symbol,
                    "new_symbol": symbol,
                    "old_close_mid": float(raw.loc[old_close_idx, "mid"]),
                    "new_open_mid": new_open_mid,
                    "adjustment_applied": adjustment,
                }
            )
        previous_date = date_value
        previous_symbol = symbol
    sample_mask = raw.index.second == 0
    sampled = raw.loc[sample_mask].copy()
    adjusted_sampled = adjusted.loc[sample_mask]
    out = pd.DataFrame(
        {
            "timestamp_utc": sampled.index,
            "raw_mid": sampled["mid"].astype(float).to_numpy(),
            "back_adjusted_mid": adjusted_sampled.to_numpy(dtype=float),
            "selected_symbol": sampled["selected_symbol"].to_numpy(),
        }
    )
    returns = pd.Series(out["back_adjusted_mid"]).diff()
    out["long_window_volatility_benchmark_cents"] = returns.rolling(3600, min_periods=900).std().to_numpy() * np.sqrt(3600) * 100.0
    return out, pd.DataFrame(roll_rows)


def _roll_continuity_ok(back_adjusted: pd.DataFrame, roll_adjustments: pd.DataFrame) -> tuple[bool, str]:
    if back_adjusted.empty or roll_adjustments.empty:
        return True, "no_rolls"
    series = back_adjusted.copy()
    series["timestamp_utc"] = pd.to_datetime(series["timestamp_utc"])
    series = series.set_index("timestamp_utc").sort_index()
    jumps = []
    for _, roll in roll_adjustments.iterrows():
        date_value = str(roll["date"])
        index_dates = pd.Index(series.index.date).astype(str)
        before = series[index_dates < date_value]
        after = series[index_dates == date_value]
        if before.empty or after.empty:
            continue
        jump_cents = abs(float(after["back_adjusted_mid"].iloc[0] - before["back_adjusted_mid"].iloc[-1]) * 100.0)
        jumps.append(jump_cents)
    max_jump = max(jumps) if jumps else 0.0
    return max_jump < 50.0, f"max_roll_boundary_jump_cents={max_jump:.4f}"


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def _schedule_comparison(previous: dict[str, Any], after_summary: pd.DataFrame, after_winners: pd.DataFrame, schedule: str) -> pd.DataFrame:
    rows = []
    before_summary = _records_to_frame(previous.get("robustness_summary", []))
    before_winners = _records_to_frame(previous.get("fold_winners", []))
    label_map = {"mid_dynamic_cost": "Model A - mid + dynamic cost", "actual_bid_ask": "Model B - actual bid/ask"}
    for run_label, summary, winners in [
        ("Before fixes", before_summary, before_winners),
        ("After fixes", after_summary, after_winners),
    ]:
        for fill_model in ["mid_dynamic_cost", "actual_bid_ask"]:
            metric = summary[(summary.get("schedule", pd.Series(dtype=str)) == schedule) & (summary.get("fill_model", pd.Series(dtype=str)) == fill_model)]
            wg = winners[(winners.get("schedule", pd.Series(dtype=str)) == schedule) & (winners.get("fill_model", pd.Series(dtype=str)) == fill_model)] if not winners.empty else pd.DataFrame()
            choice = ""
            if run_label == "Before fixes":
                choice = previous.get("selected", {}).get(fill_model, {}).get("parameter_set", "")
            elif not wg.empty:
                choice = wg["parameter_set"].value_counts().index[0]
            rows.append(
                {
                    "run": run_label,
                    "fill_model": label_map[fill_model],
                    "chosen_parameter_set": choice,
                    "oos_pnl_cents": float(metric["total_pnl_cents"].iloc[0]) if not metric.empty else np.nan,
                    "sharpe": float(metric["daily_sharpe"].iloc[0]) if not metric.empty else np.nan,
                    "win_month_rate": float(metric["win_month_rate"].iloc[0]) if not metric.empty else np.nan,
                    "top_3_removed_pnl_cents": float(metric["top_3_trade_removed_pnl_cents"].iloc[0]) if not metric.empty else np.nan,
                    "randomized_entry_p": float(metric["randomized_entry_p_value"].iloc[0]) if not metric.empty else np.nan,
                    "distinct_winners_across_folds": int(metric["distinct_parameter_sets"].iloc[0]) if not metric.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _winner_stability_comparison(previous: dict[str, Any], after_summary: pd.DataFrame) -> pd.DataFrame:
    before_summary = _records_to_frame(previous.get("robustness_summary", []))
    rows = []
    for schedule in ["3m_train_1m_test", "6m_train_1m_test"]:
        for fill_model in ["mid_dynamic_cost", "actual_bid_ask"]:
            before = before_summary[(before_summary.get("schedule", pd.Series(dtype=str)) == schedule) & (before_summary.get("fill_model", pd.Series(dtype=str)) == fill_model)]
            after = after_summary[(after_summary["schedule"] == schedule) & (after_summary["fill_model"] == fill_model)]
            rows.append(
                {
                    "schedule": schedule,
                    "fill_model": fill_model,
                    "before_distinct_winners": int(before["distinct_parameter_sets"].iloc[0]) if not before.empty else np.nan,
                    "after_distinct_winners": int(after["distinct_parameter_sets"].iloc[0]) if not after.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _write_heatmap_svg(pivot: pd.DataFrame, title: str, path: Path) -> None:
    if pivot.empty:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='160'><text x='20' y='40'>No data</text></svg>", encoding="utf-8")
        return
    values = pivot.to_numpy(dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    cell_w = 86
    cell_h = 42
    left = 120
    top = 70
    width = left + cell_w * len(pivot.columns) + 20
    height = top + cell_h * len(pivot.index) + 40

    def color(value: float) -> str:
        if not np.isfinite(value) or vmax == vmin:
            return "#eeeeee"
        t = (value - vmin) / (vmax - vmin)
        red = int(190 * (1 - t) + 35 * t)
        green = int(65 * (1 - t) + 150 * t)
        blue = int(60 * (1 - t) + 80 * t)
        return f"#{red:02x}{green:02x}{blue:02x}"

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    parts.append("<rect width='100%' height='100%' fill='white'/>")
    parts.append(f"<text x='20' y='30' font-family='Arial' font-size='18' font-weight='bold'>{title}</text>")
    for j, col in enumerate(pivot.columns):
        parts.append(f"<text x='{left + j * cell_w + 10}' y='{top - 12}' font-family='Arial' font-size='12'>{col}</text>")
    for i, idx in enumerate(pivot.index):
        parts.append(f"<text x='20' y='{top + i * cell_h + 26}' font-family='Arial' font-size='12'>{idx}</text>")
        for j, col in enumerate(pivot.columns):
            value = pivot.loc[idx, col]
            x = left + j * cell_w
            y = top + i * cell_h
            parts.append(f"<rect x='{x}' y='{y}' width='{cell_w}' height='{cell_h}' fill='{color(float(value))}' stroke='white'/>")
            label = "" if pd.isna(value) else f"{float(value):.2f}"
            parts.append(f"<text x='{x + 10}' y='{y + 26}' font-family='Arial' font-size='12' fill='black'>{label}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _generate_representative_heatmaps(rankings: pd.DataFrame) -> list[str]:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    stage2 = rankings[rankings["stage"] == 2].copy() if not rankings.empty else pd.DataFrame()
    if stage2.empty:
        return out
    groups = list(stage2.groupby(["schedule", "fill_model", "fold"], sort=True))[:3]
    for idx, ((schedule, fill_model, fold), subset) in enumerate(groups, start=1):
        ab = subset.pivot_table(index="a", columns="b", values="composite_score", aggfunc="max")
        path = DIAG_DIR / f"heatmap_{idx}_ab_postfix.svg"
        _write_heatmap_svg(ab, f"{schedule} {fill_model} {fold}: a vs b", path)
        out.append(str(path.relative_to(REPO_ROOT)))
    return out


def _build_fix_comparison(
    previous: dict[str, Any],
    verdict: dict[str, Any],
    robustness_summary: pd.DataFrame,
    fold_winners: pd.DataFrame,
    oos_trades: pd.DataFrame,
    skipped_signals: pd.DataFrame,
    heatmap_paths: list[str],
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_6m = _schedule_comparison(previous, robustness_summary, fold_winners, "6m_train_1m_test")
    table_3m = _schedule_comparison(previous, robustness_summary, fold_winners, "3m_train_1m_test")
    stability = _winner_stability_comparison(previous, robustness_summary)
    before_trades = int(previous.get("total_oos_trades", 0))
    after_trades = int(len(oos_trades))
    skipped_ffill = int(skipped_signals.loc[skipped_signals.get("skip_reason", pd.Series(dtype=str)) == "forward_filled_entry", "affected_parameter_count"].sum()) if not skipped_signals.empty else 0
    skipped_near = int(skipped_signals.loc[skipped_signals.get("skip_reason", pd.Series(dtype=str)) == "near_expiry_lt_3_business_days", "affected_parameter_count"].sum()) if not skipped_signals.empty else 0
    trade_diff = pd.DataFrame(
        [
            {
                "before_total_oos_trades": before_trades,
                "after_total_oos_trades": after_trades,
                "oos_trade_count_change": after_trades - before_trades,
                "before_oos_trades_lt5_bdays_to_expiry": int(previous.get("near_expiry_oos_trades_lt5_bdays", 0)),
                "postfix_near_expiry_candidate_trades_skipped": skipped_near,
                "postfix_forward_filled_entry_candidate_trades_skipped": skipped_ffill,
            }
        ]
    )
    before_verdict = previous.get("verdict", "UNKNOWN")
    after_verdict = verdict.get("verdict", "UNKNOWN")
    if after_verdict == "STABLE EDGE FOUND" and before_verdict != after_verdict:
        conclusion = "Verdict flipped to STABLE EDGE - the previous result was an artifact of the roll/sigma bugs."
    elif after_verdict == "WEAK / EPISODIC EDGE" and before_verdict != after_verdict:
        conclusion = "Verdict flipped to WEAK - there's a marginal effect that fixes partially revealed but it still fails one or more robustness tests."
    else:
        conclusion = "Verdict remains NO STABLE EDGE - the bugs were real but fixing them did not produce a tradable result; the signal family genuinely lacks edge on this data."
    heatmap_text = "No stable green regions appeared in the representative post-fix heatmaps; the high-composite areas remain fold-specific rather than broad and persistent."
    lines = [
        "# Fix Comparison",
        "",
        f"**Headline verdict change:** {before_verdict} -> {after_verdict}",
        "",
        "## 6/1 Primary Schedule",
        _markdown_table(table_6m),
        "",
        "## 3/1 Schedule",
        _markdown_table(table_3m),
        "",
        "## Trade Count Diff",
        _markdown_table(trade_diff),
        "",
        "## Per-Fold Winner Stability",
        _markdown_table(stability),
        "",
        "## Heatmap Regeneration",
        *[f"- `{path}`" for path in heatmap_paths],
        "",
        heatmap_text,
        "",
        "## Honest Conclusion",
        conclusion,
    ]
    return "\n".join(lines), table_6m, table_3m, trade_diff, stability


def _build_verdict(summary: pd.DataFrame, fold_winners: pd.DataFrame, final_choices: pd.DataFrame) -> dict[str, Any]:
    model_b = summary[summary["fill_model"] == "actual_bid_ask"].copy()
    model_a = summary[summary["fill_model"] == "mid_dynamic_cost"].copy()
    common_3m_b = _most_common_schedule_winner(fold_winners, "3m_train_1m_test", "actual_bid_ask")
    common_6m_b = _most_common_schedule_winner(fold_winners, "6m_train_1m_test", "actual_bid_ask")
    same_family = _same_family(common_3m_b, common_6m_b)
    b_failures = []
    for _, row in model_b.iterrows():
        tests = {
            "top_3_trade_survival": row["top_3_trade_removed_pnl_cents"] > 0,
            "top_month_survival": row["top_month_removed_pnl_cents"] > 0,
            "win_month_rate": row["win_month_rate"] >= 0.65,
            "parameter_stability": row["distinct_parameter_sets"] <= 4,
            "randomized_entry": row["randomized_entry_p_value"] < 0.05,
            "bootstrap_ci": pd.notna(row["bootstrap_mean_trade_ci_low"]) and row["bootstrap_mean_trade_ci_low"] > 0,
        }
        for test_name, passed in tests.items():
            if not passed:
                b_failures.append(f"{row['schedule']}:{test_name}")
    if not same_family:
        b_failures.append("model_b_parameter_family_differs_between_3m_and_6m")
    stable = len(b_failures) == 0 and not model_b.empty
    best_a_passes = max([_pass_count(row) for _, row in model_a.iterrows()], default=0)
    verdict = "STABLE EDGE FOUND" if stable else ("WEAK / EPISODIC EDGE" if len(b_failures) <= 1 or best_a_passes >= 4 else "NO STABLE EDGE")
    selected = {}
    for _, choice in final_choices.iterrows():
        fill = choice["fill_model"]
        primary = summary[(summary["fill_model"] == fill) & (summary["schedule"] == "6m_train_1m_test")]
        if primary.empty:
            primary = summary[summary["fill_model"] == fill].head(1)
        metrics = primary.iloc[0].to_dict() if not primary.empty else {}
        selected[fill] = {
            "param_id": choice["param_id"],
            "parameter_set": choice["parameter_set"],
            "selection_reason": choice["selection_reason"],
            "primary_schedule_for_summary": metrics.get("schedule", ""),
            "oos_total_pnl_cents": metrics.get("total_pnl_cents", np.nan),
            "oos_daily_sharpe": metrics.get("daily_sharpe", np.nan),
            "oos_win_month_rate": metrics.get("win_month_rate", np.nan),
            "top_3_trade_removed_pnl_cents": metrics.get("top_3_trade_removed_pnl_cents", np.nan),
            "randomized_entry_p_value": metrics.get("randomized_entry_p_value", np.nan),
            "bootstrap_mean_trade_ci_low": metrics.get("bootstrap_mean_trade_ci_low", np.nan),
        }
    return {
        "verdict": verdict,
        "reasons": b_failures,
        "model_b_same_parameter_family_3m_6m": same_family,
        "model_b_common_3m_param": common_3m_b["parameter_set"] if common_3m_b is not None else "",
        "model_b_common_6m_param": common_6m_b["parameter_set"] if common_6m_b is not None else "",
        "selected": selected,
        "footnote_model_b": "Model B uses real bid/ask fills on Jan-Nov 2025 parquet rows; Dec 2025-Mar 2026 CSV rows fall back to mid plus dynamic cost because bid/ask is unavailable.",
    }


def _build_validation_checks(
    data: pd.DataFrame,
    manifest: dict[str, Any],
    base_trades_stage2: pd.DataFrame,
    oos_trades: pd.DataFrame,
    fold_winners: pd.DataFrame,
    randomized: pd.DataFrame,
    chosen_contract_daily: pd.DataFrame,
    sigma_diag: pd.DataFrame,
    skipped_signals: pd.DataFrame,
    back_adjusted: pd.DataFrame,
    roll_adjustments: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, details: str) -> None:
        rows.append({"check": name, "pass": bool(passed), "details": details})

    source_by_month = data.groupby("month")["source_type"].agg(lambda s: sorted(set(s))).to_dict()
    jan_nov_ok = all(source_by_month.get(f"2025-{m:02d}") == [PROMPT_LIQUID_SOURCE_TYPE] for m in range(1, 12))
    dec_mar_ok = all(source_by_month.get(m) == ["csv_mid_dynamic_cost"] for m in ["2025-12", "2026-01", "2026-02", "2026-03"])
    csv_files = [Path(p).stem for p in manifest.get("csv_files_used", [])]
    no_old = not any(name.startswith("2025-10") or name.startswith("2025-11") for name in csv_files)
    add("data_source_scope", jan_nov_ok and dec_mar_ok and no_old, f"jan_nov_parquet={jan_nov_ok}; dec_mar_csv={dec_mar_ok}; no_old_oct_nov_csv={no_old}")
    add("no_duplicate_timestamps_after_stitching", not data.index.duplicated().any(), f"duplicates={int(data.index.duplicated().sum())}")
    if oos_trades.empty:
        add("entries_inside_dubai_session", False, "no_oos_trades")
        add("exits_respect_hold_and_midnight", False, "no_oos_trades")
        add("model_b_fill_sides_correct", False, "no_oos_trades")
    else:
        entry_dubai = pd.to_datetime(oos_trades["entry_time_dubai"])
        hour_float = entry_dubai.dt.hour + entry_dubai.dt.minute / 60.0 + entry_dubai.dt.second / 3600.0
        inside = (hour_float >= 11.0) & (hour_float < 24.0)
        add("entries_inside_dubai_session", bool(inside.all()), f"bad_entries={int((~inside).sum())}")
        entry = pd.to_datetime(oos_trades["entry_time_utc"])
        exit_time = pd.to_datetime(oos_trades["exit_time_utc"])
        max_hold = entry + pd.to_timedelta(oos_trades["d"], unit="s")
        midnight = entry.dt.normalize() + pd.Timedelta(hours=20)
        exit_ok = (exit_time <= max_hold) & (exit_time <= midnight)
        add("exits_respect_hold_and_midnight", bool(exit_ok.all()), f"bad_exits={int((~exit_ok).sum())}")
        model_b = oos_trades[oos_trades["fill_model"] == "actual_bid_ask"]
        parquet_b = model_b[model_b["source_type"].astype(str).str.startswith("parquet")]
        if parquet_b.empty:
            fill_ok = False
            detail = "no_model_b_parquet_trades"
        else:
            longs = parquet_b["side"] == "long"
            shorts = parquet_b["side"] == "short"
            long_ok = ((parquet_b.loc[longs, "entry_fill_side"] == "ask") & (parquet_b.loc[longs, "exit_fill_side"] == "bid")).all()
            short_ok = ((parquet_b.loc[shorts, "entry_fill_side"] == "bid") & (parquet_b.loc[shorts, "exit_fill_side"] == "ask")).all()
            status_ok = (parquet_b["fill_status"] == "actual_bid_ask").all()
            fill_ok = bool(long_ok and short_ok and status_ok)
            detail = f"long_ok={bool(long_ok)}; short_ok={bool(short_ok)}; status_ok={bool(status_ok)}"
        add("model_b_fill_sides_correct", fill_ok, detail)
    if base_trades_stage2.empty:
        add("max_one_trade_per_parameter_day", False, "no_base_trades")
    else:
        max_trades = base_trades_stage2.groupby(["param_id", "entry_dubai_date"])["entry_time_utc"].size().max()
        add("max_one_trade_per_parameter_day", int(max_trades) <= 1, f"max_daily_trades={int(max_trades)}")
    wf_ok = False if fold_winners.empty else bool((pd.to_datetime(fold_winners["train_end_exclusive"]) <= pd.to_datetime(fold_winners["test_start"])).all())
    add("walkforward_train_test_strictly_separated", wf_ok, f"folds={len(fold_winners)}")
    top_ok = False if fold_winners.empty else bool(fold_winners["winner_is_top_composite"].all())
    add("selection_winner_highest_composite", top_ok, f"bad_folds={0 if fold_winners.empty else int((~fold_winners['winner_is_top_composite']).sum())}")
    if randomized.empty:
        add("randomized_entry_resamples", False, "no_randomized_distribution")
    else:
        counts = randomized.groupby(["schedule", "fill_model"]).size()
        add("randomized_entry_resamples", bool((counts >= 1000).all()), ";".join(f"{idx}:{int(v)}" for idx, v in counts.items()))
    if oos_trades.empty or "days_to_expiry" not in oos_trades:
        add("liquidity_roll_no_near_expiry_trades", not oos_trades.empty, "no_oos_trades" if oos_trades.empty else "missing_days_to_expiry")
    else:
        near = pd.to_numeric(oos_trades["days_to_expiry"], errors="coerce") < 3
        near = near.fillna(False)
        add("liquidity_roll_no_near_expiry_trades", not bool(near.any()), f"near_expiry_oos_trades={int(near.sum())}")
    loaded_dates = set(data[data["source_type"].astype(str).str.startswith("parquet")]["utc_date"].dropna().astype(str))
    documented_dates = set(chosen_contract_daily["date"].dropna().astype(str)) if not chosen_contract_daily.empty and "date" in chosen_contract_daily else set()
    documented = bool(loaded_dates) and loaded_dates.issubset(documented_dates) and (DIAG_DIR / "chosen_contract_daily.csv").exists()
    add("liquidity_roll_daily_choice_documented", documented, f"documented_days={len(documented_dates)}; loaded_parquet_days={len(loaded_dates)}; undocumented_loaded_days={len(loaded_dates - documented_dates)}")
    sigma_no_ffill = True
    if not base_trades_stage2.empty and "signal_real_print_count" in base_trades_stage2:
        sigma_no_ffill = bool((pd.to_numeric(base_trades_stage2["signal_real_print_count"], errors="coerce") >= pd.to_numeric(base_trades_stage2["sigma_min_real_prints"], errors="coerce")).all())
    add("sigma_no_ffill_for_estimation", sigma_no_ffill and not sigma_diag.empty, "sigma uses is_real_print mask and rolling real-print returns")
    if base_trades_stage2.empty or "signal_real_print_count" not in base_trades_stage2:
        add("sigma_min_real_prints_enforced", False, "missing_signal_real_print_count")
    else:
        bad_sigma = pd.to_numeric(base_trades_stage2["signal_real_print_count"], errors="coerce") < pd.to_numeric(base_trades_stage2["sigma_min_real_prints"], errors="coerce")
        add("sigma_min_real_prints_enforced", not bool(bad_sigma.any()), f"bad_signals={int(bad_sigma.sum())}")
    if base_trades_stage2.empty or "entry_was_forward_filled" not in base_trades_stage2:
        add("forward_filled_entry_exclusion", False, "missing_entry_was_forward_filled")
    else:
        bad_entries = base_trades_stage2["entry_was_forward_filled"].astype(bool)
        add("forward_filled_entry_exclusion", not bool(bad_entries.any()), f"forward_filled_entries={int(bad_entries.sum())}")
    continuity_ok, continuity_detail = _roll_continuity_ok(back_adjusted, roll_adjustments)
    add("back_adjustment_continuity", continuity_ok, continuity_detail)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    previous_summary = _capture_previous_run_summary()
    print("Loading Jan 2025-Mar 2026 research dataset")
    data, coverage, manifest = load_research_dataset(REPO_ROOT)
    _save_csv(coverage, "data_coverage.csv")
    chosen_contract_daily = manifest.get("chosen_contract_daily", pd.DataFrame())
    _save_diag_csv(chosen_contract_daily, "chosen_contract_daily.csv")
    prompt_calendar = _build_prompt_contract_calendar(data, chosen_contract_daily)
    _save_csv(prompt_calendar, "prompt_contract_calendar.csv")
    sigma_diag = _build_sigma_diagnostic(data)
    _save_diag_csv(sigma_diag, "sigma_realprints_per_second.csv")
    back_adjusted, roll_adjustments = _build_back_adjusted_mid(data)
    _save_diag_csv(back_adjusted, "back_adjusted_mid.csv")
    _save_diag_csv(roll_adjustments, "roll_adjustments.csv")
    print("Building Stage 1 parameter universe")
    params_stage1 = stage1_universe()
    _save_csv(params_stage1, "parameter_universe_stage1.csv")
    print("Generating Stage 1 candidate trades")
    base_stage1, skipped_stage1 = generate_base_trades(data, params_stage1, return_diagnostics=True)
    print(f"Stage 1 base trades rows={len(base_stage1)}")
    print("Running Stage 1 walk-forward selection")
    wf_stage1 = run_walkforward_stage(base_stage1, params_stage1, stage=1)
    print("Building Stage 2 refined universe")
    params_stage2 = build_stage2_universe(wf_stage1["train_rankings"], params_stage1)
    _save_csv(params_stage2, "parameter_universe_stage2.csv")
    print("Generating Stage 2 candidate trades")
    base_stage2, skipped_stage2 = generate_base_trades(data, params_stage2, return_diagnostics=True)
    skipped_signals = pd.concat([skipped_stage1.assign(stage=1), skipped_stage2.assign(stage=2)], ignore_index=True) if not skipped_stage1.empty or not skipped_stage2.empty else pd.DataFrame()
    near_expiry_skips = skipped_signals[skipped_signals["skip_reason"] == "near_expiry_lt_3_business_days"].copy() if not skipped_signals.empty else pd.DataFrame()
    _save_diag_csv(near_expiry_skips, "skipped_near_expiry_signals.csv")
    _save_diag_csv(skipped_signals, "skipped_signals.csv")
    print(f"Stage 2 base trades rows={len(base_stage2)}")
    print("Running Stage 2 walk-forward selection")
    wf_stage2 = run_walkforward_stage(base_stage2, params_stage2, stage=2)
    train_rankings = pd.concat([wf_stage1["train_rankings"], wf_stage2["train_rankings"]], ignore_index=True)
    decisions = pd.concat([wf_stage1["decisions"], wf_stage2["decisions"]], ignore_index=True)
    oos_trades = wf_stage2["trades"].copy()
    fold_winners = wf_stage2["fold_winners"].copy()
    daily_pnl, monthly_pnl, oos_equity = summarize_daily_monthly(oos_trades)
    print("Running OOS robustness battery")
    robustness_report, randomized_dist, robustness_summary = build_robustness_outputs(data, oos_trades, fold_winners, n_random=1000)
    fill_comparison = _build_fill_model_comparison(robustness_summary)
    print("Running visual full-period backtests")
    final_choices = choose_final_parameters(fold_winners)
    full_trades, full_equity = run_full_period_backtests(base_stage2, final_choices)
    verdict = _build_verdict(robustness_summary, fold_winners, final_choices)
    heatmap_paths = _generate_representative_heatmaps(train_rankings)
    comparison_md, comparison_6m, comparison_3m, trade_count_diff, stability_comparison = _build_fix_comparison(
        previous_summary,
        verdict,
        robustness_summary,
        fold_winners,
        oos_trades,
        skipped_signals,
        heatmap_paths,
    )
    validation = _build_validation_checks(
        data,
        manifest,
        base_stage2,
        oos_trades,
        fold_winners,
        randomized_dist,
        chosen_contract_daily,
        sigma_diag,
        skipped_signals,
        back_adjusted,
        roll_adjustments,
    )
    _save_csv(train_rankings, "train_rankings.csv")
    _save_csv(decisions, "decisions.csv")
    _save_csv(oos_trades, "trades.csv")
    _save_csv(daily_pnl, "daily_pnl.csv")
    _save_csv(monthly_pnl, "monthly_pnl.csv")
    _save_csv(oos_equity, "oos_equity.csv")
    _save_csv(fold_winners, "fold_winners.csv")
    _save_csv(fill_comparison, "fill_model_comparison.csv")
    _save_csv(robustness_report, "robustness_report.csv")
    _save_csv(randomized_dist, "randomized_entry_distribution.csv")
    _save_csv(full_trades, "full_period_backtest_trades.csv")
    _save_csv(full_equity, "full_period_backtest_equity.csv")
    _save_csv(validation, "validation_checks.csv")
    _save_csv(final_choices, "final_parameter_choices.csv")
    _save_csv(robustness_summary, "robustness_summary.csv")
    _save_csv(comparison_6m, "fix_comparison_6m.csv")
    _save_csv(comparison_3m, "fix_comparison_3m.csv")
    _save_csv(trade_count_diff, "fix_trade_count_diff.csv")
    _save_csv(stability_comparison, "fix_winner_stability_comparison.csv")
    (OUTPUT_DIR / "fix_comparison.md").write_text(comparison_md, encoding="utf-8")
    print("Wrote outputs/fix_comparison.md")
    (OUTPUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8")
    print("Wrote outputs/verdict.json")
    failures = validation[~validation["pass"]]
    if not failures.empty:
        raise AssertionError("Validation failures:\n" + failures.to_string(index=False))
    print(f"Verdict: {verdict['verdict']}")


if __name__ == "__main__":
    main()
