from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import load_research_dataset, prompt_contract_symbol
from .engine import generate_base_trades, stage1_universe, summarize_daily_monthly
from .full_period import choose_final_parameters, run_full_period_backtests
from .robustness import build_robustness_outputs
from .selection import build_stage2_universe
from .walkforward import run_walkforward_stage


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"


def _save_csv(df: pd.DataFrame, name: str) -> None:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print(f"Wrote {path.relative_to(REPO_ROOT)} rows={len(df)}")


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


def _build_prompt_contract_calendar(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    parquet = data[data["source_type"].astype(str).str.startswith("parquet")].copy()
    if parquet.empty:
        return pd.DataFrame()
    for date_value, group in parquet.groupby("utc_date", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "date": date_value,
                "calendar_month": str(pd.Timestamp(date_value).to_period("M")),
                "chosen_symbol": first.get("selected_symbol", ""),
                "prompt_delivery_month": first.get("prompt_delivery_month", ""),
                "prompt_month_code": first.get("prompt_month_code", ""),
                "contract_expiry_date": first.get("contract_expiry_date", ""),
                "days_to_expiry": (
                    pd.Timestamp(first.get("contract_expiry_date")) - pd.Timestamp(date_value)
                ).days
                if first.get("contract_expiry_date", "")
                else np.nan,
                "total_volume_that_day": first.get("total_volume_that_day", np.nan),
                "rank_by_volume_among_available_symbols": first.get("rank_by_volume_among_available_symbols", np.nan),
                "available_symbol_count": first.get("available_symbol_count", np.nan),
                "selection_rule": first.get("contract_selection_rule", ""),
                "row_count": int(len(group)),
                "first_timestamp_utc": group.index.min().isoformat(),
                "last_timestamp_utc": group.index.max().isoformat(),
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
) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, details: str) -> None:
        rows.append({"check": name, "pass": bool(passed), "details": details})

    source_by_month = data.groupby("month")["source_type"].agg(lambda s: sorted(set(s))).to_dict()
    jan_nov_ok = all(source_by_month.get(f"2025-{m:02d}") == ["parquet_prompt_bid_ask"] for m in range(1, 12))
    dec_mar_ok = all(source_by_month.get(m) == ["csv_mid_dynamic_cost"] for m in ["2025-12", "2026-01", "2026-02", "2026-03"])
    csv_files = [Path(p).stem for p in manifest.get("csv_files_used", [])]
    no_old = not any(name.startswith("2025-10") or name.startswith("2025-11") for name in csv_files)
    add("data_source_scope", jan_nov_ok and dec_mar_ok and no_old, f"jan_nov_parquet={jan_nov_ok}; dec_mar_csv={dec_mar_ok}; no_old_oct_nov_csv={no_old}")
    prompt_ok = True
    prompt_details = []
    for month in range(1, 12):
        month_key = f"2025-{month:02d}"
        expected = prompt_contract_symbol(2025, month)
        seen = sorted(set(data.loc[data["month"] == month_key, "selected_symbol"].dropna().astype(str)))
        ok = seen == [expected]
        prompt_ok = prompt_ok and ok
        prompt_details.append(f"{month_key}:{expected}:{ok}")
    add("prompt_contract_month_plus_2_mapping", prompt_ok, ";".join(prompt_details))
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
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading Jan 2025-Mar 2026 research dataset")
    data, coverage, manifest = load_research_dataset(REPO_ROOT)
    _save_csv(coverage, "data_coverage.csv")
    prompt_calendar = _build_prompt_contract_calendar(data)
    _save_csv(prompt_calendar, "prompt_contract_calendar.csv")
    print("Building Stage 1 parameter universe")
    params_stage1 = stage1_universe()
    _save_csv(params_stage1, "parameter_universe_stage1.csv")
    print("Generating Stage 1 candidate trades")
    base_stage1 = generate_base_trades(data, params_stage1)
    print(f"Stage 1 base trades rows={len(base_stage1)}")
    print("Running Stage 1 walk-forward selection")
    wf_stage1 = run_walkforward_stage(base_stage1, params_stage1, stage=1)
    print("Building Stage 2 refined universe")
    params_stage2 = build_stage2_universe(wf_stage1["train_rankings"], params_stage1)
    _save_csv(params_stage2, "parameter_universe_stage2.csv")
    print("Generating Stage 2 candidate trades")
    base_stage2 = generate_base_trades(data, params_stage2)
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
    validation = _build_validation_checks(data, manifest, base_stage2, oos_trades, fold_winners, randomized_dist)
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
    (OUTPUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print("Wrote outputs/verdict.json")
    failures = validation[~validation["pass"]]
    if not failures.empty:
        raise AssertionError("Validation failures:\n" + failures.to_string(index=False))
    print(f"Verdict: {verdict['verdict']}")


if __name__ == "__main__":
    main()
