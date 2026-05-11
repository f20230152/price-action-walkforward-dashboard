from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from clean_walkforward_research import (
    MAX_TRADES_PER_DAY,
    SESSION_END_HOUR_DUBAI,
    SESSION_START_HOUR_DUBAI,
    add_session_audit_columns,
)
from price_action_engine import (
    DUBAI_UTC_OFFSET_HOURS,
    DYNAMIC_COST_CENTS_PER_PRICE_UNIT,
    SpikeParams,
    backtest_spike_strategy,
    compute_metrics,
    load_second_prices,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "clean_walkforward"
PARAM_RE = re.compile(
    r"a=(?P<delay>\d+)s \| move=(?P<threshold>[\d.]+)c/(?P<lookback>\d+)s \| "
    r"hold=(?P<hold>\d+)s \| (?P<volume>vol off|vol>(?P<vol_multiple>[\d.]+)x/(?P<vol_window>\d+)s) \| "
    r"(?P<direction>\w+)"
)


def parse_params(label: str) -> SpikeParams:
    match = PARAM_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"Cannot parse params: {label}")
    return SpikeParams(
        delay_s=int(match.group("delay")),
        threshold_cents=float(match.group("threshold")),
        lookback_s=int(match.group("lookback")),
        hold_s=int(match.group("hold")),
        volume_window_s=int(match.group("vol_window") or 300),
        volume_multiple=float(match.group("vol_multiple") or 0.0),
        direction=match.group("direction"),
    )


def backtest_window(bars: pd.DataFrame, params: SpikeParams) -> tuple[pd.DataFrame, pd.Series, dict]:
    trades, daily = backtest_spike_strategy(
        bars,
        params,
        None,
        MAX_TRADES_PER_DAY,
        session_start_hour=SESSION_START_HOUR_DUBAI,
        session_end_hour=SESSION_END_HOUR_DUBAI,
        session_tz_offset_hours=DUBAI_UTC_OFFSET_HOURS,
    )
    return trades, daily, compute_metrics(daily, trades)


def month_slice(bars: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    periods = bars.index.to_period("M")
    start_period = pd.Period(start, freq="M")
    end_period = pd.Period(end, freq="M")
    return bars[(periods >= start_period) & (periods <= end_period)]


def main() -> None:
    decisions_path = OUT_DIR / "walkforward_decisions_all.csv"
    old_decisions = pd.read_csv(decisions_path)
    bars, stats = load_second_prices(DATA_DIR, "2025-10-01", "2026-03-26", cache_dir=BASE_DIR / ".price_cache", workers=8)

    decision_rows = []
    daily_by_objective: dict[str, list[pd.Series]] = {}
    trades_all = []
    excluded_all = []

    for _, row in old_decisions.iterrows():
        params = parse_params(str(row["selected_params"]))
        train_bars = month_slice(bars, row["train_start"], row["train_end"])
        test_bars = month_slice(bars, row["test_start"], row["test_end"])

        _, _, train_metrics = backtest_window(train_bars, params)
        test_trades, test_daily, test_metrics = backtest_window(test_bars, params)

        objective = row["selector_objective"]
        daily_by_objective.setdefault(objective, []).append(test_daily)
        if not test_trades.empty:
            out_trades = test_trades.copy()
            out_trades["selector_objective"] = objective
            out_trades["test_month"] = row["test_start"]
            out_trades["selected_params"] = params.label
            trades_all.append(out_trades)

        unrestricted_trades, _ = backtest_spike_strategy(test_bars, params, None, MAX_TRADES_PER_DAY)
        unrestricted_trades = add_session_audit_columns(unrestricted_trades)
        if not unrestricted_trades.empty:
            excluded = unrestricted_trades[unrestricted_trades["session_bucket"].str.startswith("Outside")].copy()
            if not excluded.empty:
                excluded["selector_objective"] = objective
                excluded["test_month"] = row["test_start"]
                excluded["selected_params"] = params.label
                excluded["exclusion_reason"] = f"Signal outside {SESSION_START_HOUR_DUBAI:02d}:00-24:00 Dubai tradable window"
                excluded_all.append(excluded)

        decision_rows.append(
            {
                "selector_objective": objective,
                "train_start": row["train_start"],
                "train_end": row["train_end"],
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "selected_params": params.label,
                "train_total_pnl_cents": train_metrics["total_pnl_cents"],
                "train_sharpe": train_metrics["daily_sharpe"],
                "train_trades": train_metrics["trades"],
                "test_total_pnl_cents": test_metrics["total_pnl_cents"],
                "test_sharpe": test_metrics["daily_sharpe"],
                "test_trades": test_metrics["trades"],
                "test_win_rate": test_metrics["win_rate"],
                "test_max_drawdown_cents": test_metrics["max_drawdown_cents"],
            }
        )

    decisions = pd.DataFrame(decision_rows)
    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    excluded = pd.concat(excluded_all, ignore_index=True) if excluded_all else pd.DataFrame()

    daily_frames = []
    metrics_rows = []
    for objective, series_list in daily_by_objective.items():
        daily = pd.concat(series_list).sort_index().groupby(level=0).sum()
        obj_trades = trades[trades["selector_objective"] == objective].copy() if not trades.empty else pd.DataFrame()
        metrics_rows.append({"selector_objective": objective, **compute_metrics(daily, obj_trades)})
        daily_frames.append(daily.rename(objective))
        daily.to_csv(OUT_DIR / f"daily_pnl_{objective}.csv", header=[objective])
        obj_trades.to_csv(OUT_DIR / f"trades_{objective}.csv", index=False)
        obj_excluded = excluded[excluded["selector_objective"] == objective].copy() if not excluded.empty else pd.DataFrame()
        obj_excluded.to_csv(OUT_DIR / f"out_of_session_trades_{objective}.csv", index=False)
        decisions[decisions["selector_objective"] == objective].to_csv(OUT_DIR / f"decisions_{objective}.csv", index=False)

    pd.DataFrame(metrics_rows).sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).to_csv(
        OUT_DIR / "walkforward_metrics.csv", index=False
    )
    decisions.to_csv(OUT_DIR / "walkforward_decisions_all.csv", index=False)
    trades.to_csv(OUT_DIR / "walkforward_trades_all.csv", index=False)
    excluded.to_csv(OUT_DIR / "walkforward_out_of_session_trades_all.csv", index=False)
    pd.concat(daily_frames, axis=1).to_csv(OUT_DIR / "walkforward_daily_pnl_all.csv")

    config = pd.DataFrame(
        [
            {
                "data_start": str(bars.index.min()),
                "data_end": str(bars.index.max()),
                "days": int(bars.index.normalize().nunique()),
                "raw_files": int(len(stats)),
                "candidate_count": "",
                "train_months": 3,
                "test_months": 1,
                "cost_model": "dynamic_absolute_price_level",
                "cost_formula": f"round-trip cost cents = abs(entry_price) * {DYNAMIC_COST_CENTS_PER_PRICE_UNIT:g}",
                "cost_cents_round_trip": "",
                "bid_ask_cost_cents_per_price_unit": DYNAMIC_COST_CENTS_PER_PRICE_UNIT,
                "session_timezone": "Dubai UTC+4",
                "session_start_hour": SESSION_START_HOUR_DUBAI,
                "session_end_hour": SESSION_END_HOUR_DUBAI,
                "max_trades_per_day": MAX_TRADES_PER_DAY,
                "refresh_method": "existing_selected_walkforward_params_retested_with_session_and_dynamic_cost",
            }
        ]
    )
    config.to_csv(OUT_DIR / "research_config.csv", index=False)
    print(pd.read_csv(OUT_DIR / "walkforward_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
