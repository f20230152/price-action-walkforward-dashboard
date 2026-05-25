from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .engine import DYNAMIC_COST_CENTS_PER_PRICE_UNIT, SESSION_END_UTC_HOUR, SESSION_START_UTC_HOUR, sharpe_from_daily, trade_metrics


def _random_fill_pnl(data: pd.DataFrame, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, side: int, fill_model: str) -> float:
    if entry_ts not in data.index or exit_ts not in data.index:
        return np.nan
    entry = data.loc[entry_ts]
    exit_row = data.loc[exit_ts]
    if bool(entry.get("was_forward_filled", False)):
        return np.nan
    entry_mid = float(entry["mid"])
    exit_mid = float(exit_row["mid"])
    mid_pnl = side * (exit_mid - entry_mid) * 100.0 - abs(entry_mid) * DYNAMIC_COST_CENTS_PER_PRICE_UNIT
    if fill_model == "mid_dynamic_cost":
        return float(mid_pnl)
    if not str(entry.get("source_type", "")).startswith("parquet"):
        return float(mid_pnl)
    if side == 1:
        ask_entry = entry.get("ask_last", np.nan)
        bid_exit = exit_row.get("bid_last", np.nan)
        if pd.isna(ask_entry) or pd.isna(bid_exit):
            return np.nan
        return float((float(bid_exit) - float(ask_entry)) * 100.0)
    bid_entry = entry.get("bid_last", np.nan)
    ask_exit = exit_row.get("ask_last", np.nan)
    if pd.isna(bid_entry) or pd.isna(ask_exit):
        return np.nan
    return float((float(bid_entry) - float(ask_exit)) * 100.0)


def randomized_entry_distribution(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    fill_model: str,
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real_total = float(trades["pnl_cents"].sum()) if not trades.empty else 0.0
    if trades.empty:
        return pd.DataFrame({"run_id": np.arange(n_resamples), "randomized_pnl_cents": np.zeros(n_resamples), "real_oos_pnl_cents": real_total})
    rows = []
    session_start = SESSION_START_UTC_HOUR * 3600
    session_end = SESSION_END_UTC_HOUR * 3600
    for run_id in range(n_resamples):
        total = 0.0
        for _, trade in trades.iterrows():
            entry_ts = pd.Timestamp(trade["entry_time_utc"])
            day = entry_ts.normalize()
            hold = int(trade["hold_seconds"])
            latest = session_end - max(1, hold)
            if latest <= session_start:
                continue
            sampled = int(rng.integers(session_start, latest + 1))
            rand_entry = day + pd.Timedelta(seconds=sampled)
            rand_exit = min(rand_entry + pd.Timedelta(seconds=hold), day + pd.Timedelta(hours=SESSION_END_UTC_HOUR))
            pnl = _random_fill_pnl(data, rand_entry, rand_exit, int(trade["side_sign"]), fill_model)
            if np.isfinite(pnl):
                total += pnl
        rows.append({"run_id": run_id, "randomized_pnl_cents": total, "real_oos_pnl_cents": real_total})
    return pd.DataFrame(rows)


def bootstrap_mean_trade_ci(pnl: pd.Series, n_resamples: int = 2000, seed: int = 123) -> tuple[float, float]:
    values = pd.to_numeric(pnl, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _top_n_removed_pnl(pnl: pd.Series, n: int) -> float:
    values = pd.to_numeric(pnl, errors="coerce").dropna().sort_values(ascending=False)
    if values.empty:
        return 0.0
    return float(values.iloc[n:].sum()) if len(values) > n else 0.0


def _monthly_pnl_for_group(trades: pd.DataFrame, fold_winners: pd.DataFrame) -> pd.Series:
    months = pd.to_datetime(fold_winners["test_start"]).dt.to_period("M").astype(str).tolist()
    idx = pd.Index(months, name="month")
    if trades.empty:
        return pd.Series(0.0, index=idx)
    pnl = trades.assign(month=pd.to_datetime(trades["entry_time_utc"]).dt.to_period("M").astype(str)).groupby("month")["pnl_cents"].sum()
    return pnl.reindex(idx, fill_value=0.0)


def build_robustness_outputs(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    fold_winners: pd.DataFrame,
    n_random: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report_rows: list[dict[str, Any]] = []
    random_frames = []
    summary_rows: list[dict[str, Any]] = []
    for (schedule, fill_model), winners_group in fold_winners.groupby(["schedule", "fill_model"], dropna=False):
        tg = trades[(trades["schedule"] == schedule) & (trades["fill_model"] == fill_model)].copy()
        metrics = trade_metrics(tg)
        monthly = _monthly_pnl_for_group(tg, winners_group)
        total = metrics["total_pnl_cents"]
        top_1 = _top_n_removed_pnl(tg.get("pnl_cents", pd.Series(dtype=float)), 1)
        top_3 = _top_n_removed_pnl(tg.get("pnl_cents", pd.Series(dtype=float)), 3)
        top_5 = _top_n_removed_pnl(tg.get("pnl_cents", pd.Series(dtype=float)), 5)
        top_month_removed = float(total - monthly.max()) if len(monthly) else total
        win_month_rate = float((monthly > 0).mean()) if len(monthly) else 0.0
        distinct = int(winners_group["param_id"].nunique())
        seed = 9100 + sum(ord(ch) for ch in f"{schedule}-{fill_model}")
        rand = randomized_entry_distribution(data, tg, fill_model, n_resamples=n_random, seed=seed)
        rand["schedule"] = schedule
        rand["fill_model"] = fill_model
        random_frames.append(rand)
        real_total = float(rand["real_oos_pnl_cents"].iloc[0]) if not rand.empty else total
        p_value = float(((rand["randomized_pnl_cents"] >= real_total).sum() + 1) / (len(rand) + 1))
        ci_low, ci_high = bootstrap_mean_trade_ci(tg.get("pnl_cents", pd.Series(dtype=float)))
        summary = {
            "schedule": schedule,
            "fill_model": fill_model,
            "total_pnl_cents": total,
            "daily_sharpe": metrics["daily_sharpe"],
            "trades": metrics["trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "max_drawdown_cents": metrics["max_drawdown_cents"],
            "top_1_trade_removed_pnl_cents": top_1,
            "top_3_trade_removed_pnl_cents": top_3,
            "top_5_trade_removed_pnl_cents": top_5,
            "top_month_removed_pnl_cents": top_month_removed,
            "win_month_rate": win_month_rate,
            "distinct_parameter_sets": distinct,
            "randomized_entry_p_value": p_value,
            "bootstrap_mean_trade_ci_low": ci_low,
            "bootstrap_mean_trade_ci_high": ci_high,
            "randomized_resamples": int(len(rand)),
        }
        summary_rows.append(summary)
        tests = [
            ("top_1_trade_removed_pnl", top_1, "> 0", top_1 > 0),
            ("top_3_trade_removed_pnl", top_3, "> 0", top_3 > 0),
            ("top_5_trade_removed_pnl", top_5, "> 0", top_5 > 0),
            ("top_month_removed_pnl", top_month_removed, "> 0", top_month_removed > 0),
            ("win_month_rate", win_month_rate, ">= 0.65", win_month_rate >= 0.65),
            ("parameter_stability_distinct_sets", distinct, "<= 4", distinct <= 4),
            ("randomized_entry_p_value", p_value, "< 0.05", p_value < 0.05),
            ("bootstrap_mean_trade_ci_low", ci_low, "> 0", pd.notna(ci_low) and ci_low > 0),
        ]
        for name, value, threshold, passed in tests:
            report_rows.append({**summary, "test_name": name, "test_value": value, "threshold": threshold, "pass": bool(passed)})
    return (
        pd.DataFrame(report_rows),
        pd.concat(random_frames, ignore_index=True) if random_frames else pd.DataFrame(),
        pd.DataFrame(summary_rows),
    )


def daily_sharpe_from_trades(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return sharpe_from_daily(trades.groupby("entry_dubai_date")["pnl_cents"].sum())
