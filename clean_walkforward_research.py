from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from price_action_engine import (
    DUBAI_UTC_OFFSET_HOURS,
    DYNAMIC_COST_CENTS_PER_PRICE_UNIT,
    SpikeParams,
    backtest_spike_strategy,
    compute_metrics,
    evaluate_spike_grid,
    load_second_prices,
    spike_parameter_grid,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "clean_walkforward"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MONTHS = 3
TEST_MONTHS = 1
COST_CENTS = None
MAX_TRADES_PER_DAY = 1
MIN_TRAIN_TRADES = 5
SESSION_START_HOUR_DUBAI = 11
SESSION_END_HOUR_DUBAI = 24


def build_universe() -> list[SpikeParams]:
    return spike_parameter_grid(
        delays=[1, 5, 10, 30, 60, 120],
        thresholds=[10, 20, 30, 50, 75, 100, 150, 200, 300],
        lookbacks=[30, 60, 120, 180, 300, 600, 900, 1800],
        holds=[900, 1800, 3600, 7200, 10800, 14400, 21600],
        volume_windows=[300],
        volume_multiples=[0, 2],
        direction="momentum",
    )


def select_candidate(grid: pd.DataFrame, objective: str) -> pd.Series | None:
    eligible = grid[
        (grid["trades"] >= MIN_TRAIN_TRADES)
        & (grid["total_pnl_cents"] > 0)
        & (grid["daily_sharpe"] > 0)
        & np.isfinite(grid["profit_factor"])
    ].copy()
    if eligible.empty:
        return None
    if objective == "daily_sharpe":
        return eligible.sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).iloc[0]
    if objective == "profit_factor":
        return eligible.sort_values(["profit_factor", "total_pnl_cents"], ascending=False).iloc[0]
    return eligible.sort_values(["total_pnl_cents", "daily_sharpe"], ascending=False).iloc[0]


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
    out["session_bucket"] = np.where(
        inside,
        f"Inside {SESSION_START_HOUR_DUBAI:02d}:00-24:00 Dubai",
        f"Outside {SESSION_START_HOUR_DUBAI:02d}:00-24:00 Dubai",
    )
    out["out_of_frame_category"] = np.where(
        inside,
        "Tradable session",
        "Excluded: before 11:00 Dubai",
    )
    return out


def run_all_objectives(
    bars: pd.DataFrame,
    params: list[SpikeParams],
    objectives: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    month_key = bars.index.to_period("M")
    months = pd.PeriodIndex(sorted(month_key.unique()))
    decisions = {objective: [] for objective in objectives}
    test_daily = {objective: [] for objective in objectives}
    test_trades = {objective: [] for objective in objectives}
    out_of_session_trades = {objective: [] for objective in objectives}
    ranking_rows = {objective: [] for objective in objectives}

    for start_i in range(TRAIN_MONTHS, len(months), TEST_MONTHS):
        train_periods = months[start_i - TRAIN_MONTHS : start_i]
        test_periods = months[start_i : start_i + TEST_MONTHS]
        if len(test_periods) == 0:
            continue

        print(f"running train {train_periods[0]}-{train_periods[-1]} test {test_periods[0]}", flush=True)
        train_bars = bars[month_key.isin(train_periods)]
        test_bars = bars[month_key.isin(test_periods)]
        train_grid = evaluate_spike_grid(
            train_bars,
            params,
            objective="total_pnl_cents",
            cost_cents=COST_CENTS,
            min_trades=0,
            max_trades_per_day=MAX_TRADES_PER_DAY,
            session_start_hour=SESSION_START_HOUR_DUBAI,
            session_end_hour=SESSION_END_HOUR_DUBAI,
            session_tz_offset_hours=DUBAI_UTC_OFFSET_HOURS,
        )

        for objective in objectives:
            if objective == "daily_sharpe":
                ranked_grid = train_grid.sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False)
            elif objective == "profit_factor":
                ranked_grid = train_grid.sort_values(["profit_factor", "total_pnl_cents"], ascending=False)
            else:
                ranked_grid = train_grid.sort_values(["total_pnl_cents", "daily_sharpe"], ascending=False)
            top = ranked_grid.drop(columns=["params"]).head(50).copy()
            top["selector_objective"] = objective
            top["train_start"] = str(train_periods[0])
            top["train_end"] = str(train_periods[-1])
            top["test_start"] = str(test_periods[0])
            top["test_end"] = str(test_periods[-1])
            ranking_rows[objective].append(top)

            selected = select_candidate(train_grid, objective)
            if selected is None:
                decisions[objective].append(
                    {
                        "selector_objective": objective,
                        "train_start": str(train_periods[0]),
                        "train_end": str(train_periods[-1]),
                        "test_start": str(test_periods[0]),
                        "test_end": str(test_periods[-1]),
                        "selected_params": "NO_ELIGIBLE_CANDIDATE",
                        "train_total_pnl_cents": 0.0,
                        "train_sharpe": 0.0,
                        "train_trades": 0,
                        "test_total_pnl_cents": 0.0,
                        "test_sharpe": 0.0,
                        "test_trades": 0,
                        "test_win_rate": 0.0,
                        "test_max_drawdown_cents": 0.0,
                    }
                )
                continue

            selected_params: SpikeParams = selected["params"]
            trades, daily = backtest_spike_strategy(
                test_bars,
                selected_params,
                COST_CENTS,
                MAX_TRADES_PER_DAY,
                session_start_hour=SESSION_START_HOUR_DUBAI,
                session_end_hour=SESSION_END_HOUR_DUBAI,
                session_tz_offset_hours=DUBAI_UTC_OFFSET_HOURS,
            )
            metrics = compute_metrics(daily, trades)
            test_daily[objective].append(daily)
            if not trades.empty:
                out_trades = trades.copy()
                out_trades["selector_objective"] = objective
                out_trades["test_month"] = ",".join(str(p) for p in test_periods)
                out_trades["selected_params"] = selected_params.label
                test_trades[objective].append(out_trades)

            unrestricted_trades, _ = backtest_spike_strategy(test_bars, selected_params, COST_CENTS, MAX_TRADES_PER_DAY)
            unrestricted_trades = add_session_audit_columns(unrestricted_trades)
            if not unrestricted_trades.empty:
                excluded = unrestricted_trades[unrestricted_trades["session_bucket"].str.startswith("Outside")].copy()
                if not excluded.empty:
                    excluded["selector_objective"] = objective
                    excluded["test_month"] = ",".join(str(p) for p in test_periods)
                    excluded["selected_params"] = selected_params.label
                    excluded["exclusion_reason"] = f"Signal outside {SESSION_START_HOUR_DUBAI:02d}:00-24:00 Dubai tradable window"
                    out_of_session_trades[objective].append(excluded)

            decisions[objective].append(
                {
                    "selector_objective": objective,
                    "train_start": str(train_periods[0]),
                    "train_end": str(train_periods[-1]),
                    "test_start": str(test_periods[0]),
                    "test_end": str(test_periods[-1]),
                    "selected_params": selected_params.label,
                    "train_total_pnl_cents": float(selected["total_pnl_cents"]),
                    "train_sharpe": float(selected["daily_sharpe"]),
                    "train_trades": int(selected["trades"]),
                    "test_total_pnl_cents": metrics["total_pnl_cents"],
                    "test_sharpe": metrics["daily_sharpe"],
                    "test_trades": metrics["trades"],
                    "test_win_rate": metrics["win_rate"],
                    "test_max_drawdown_cents": metrics["max_drawdown_cents"],
                }
            )

    decisions_df = {objective: pd.DataFrame(rows) for objective, rows in decisions.items()}
    daily_df = {
        objective: pd.concat(rows).sort_index().groupby(level=0).sum() if rows else pd.Series(dtype=float, name="daily_pnl_cents")
        for objective, rows in test_daily.items()
    }
    trades_df = {objective: pd.concat(rows, ignore_index=True) if rows else pd.DataFrame() for objective, rows in test_trades.items()}
    rankings_df = {objective: pd.concat(rows, ignore_index=True) if rows else pd.DataFrame() for objective, rows in ranking_rows.items()}
    out_df = {
        objective: pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        for objective, rows in out_of_session_trades.items()
    }
    return decisions_df, daily_df, trades_df, rankings_df, out_df


def main() -> None:
    bars, stats = load_second_prices(DATA_DIR, "2025-10-01", "2026-03-26", cache_dir=BASE_DIR / ".price_cache", workers=8)
    params = build_universe()
    pd.DataFrame(
        [
            {
                "label": p.label,
                "delay_s": p.delay_s,
                "threshold_cents": p.threshold_cents,
                "lookback_s": p.lookback_s,
                "hold_s": p.hold_s,
                "volume_window_s": p.volume_window_s,
                "volume_multiple": p.volume_multiple,
                "direction": p.direction,
            }
            for p in params
        ]
    ).to_csv(OUT_DIR / "parameter_universe.csv", index=False)

    all_decisions = []
    all_trades = []
    all_out_of_session = []
    metrics_rows = []
    daily_frames = []
    rankings = []

    objectives = ["total_pnl_cents", "daily_sharpe", "profit_factor"]
    print("running objectives", ",".join(objectives), "candidates", len(params), flush=True)
    decisions_by_obj, daily_by_obj, trades_by_obj, rankings_by_obj, out_by_obj = run_all_objectives(bars, params, objectives)

    for objective in objectives:
        decisions = decisions_by_obj[objective]
        daily = daily_by_obj[objective]
        trades = trades_by_obj[objective]
        ranking = rankings_by_obj[objective]
        out_of_session = out_by_obj[objective]
        decisions.to_csv(OUT_DIR / f"decisions_{objective}.csv", index=False)
        daily.rename(objective).to_csv(OUT_DIR / f"daily_pnl_{objective}.csv", header=True)
        trades.to_csv(OUT_DIR / f"trades_{objective}.csv", index=False)
        out_of_session.to_csv(OUT_DIR / f"out_of_session_trades_{objective}.csv", index=False)
        ranking.to_csv(OUT_DIR / f"train_rankings_{objective}.csv", index=False)
        metrics = compute_metrics(daily, trades)
        metrics_rows.append({"selector_objective": objective, **metrics})
        all_decisions.append(decisions)
        if not trades.empty:
            all_trades.append(trades)
        if not out_of_session.empty:
            all_out_of_session.append(out_of_session)
        daily_frames.append(daily.rename(objective))
        rankings.append(ranking)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False)
    metrics_df.to_csv(OUT_DIR / "walkforward_metrics.csv", index=False)
    pd.concat(all_decisions, ignore_index=True).to_csv(OUT_DIR / "walkforward_decisions_all.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(OUT_DIR / "walkforward_trades_all.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT_DIR / "walkforward_trades_all.csv", index=False)
    if all_out_of_session:
        pd.concat(all_out_of_session, ignore_index=True).to_csv(OUT_DIR / "walkforward_out_of_session_trades_all.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT_DIR / "walkforward_out_of_session_trades_all.csv", index=False)
    pd.concat(daily_frames, axis=1).to_csv(OUT_DIR / "walkforward_daily_pnl_all.csv")
    pd.concat(rankings, ignore_index=True).to_csv(OUT_DIR / "walkforward_train_rankings_all.csv", index=False)

    summary = {
        "data_start": str(bars.index.min()),
        "data_end": str(bars.index.max()),
        "days": int(bars.index.normalize().nunique()),
        "raw_files": int(len(stats)),
        "candidate_count": int(len(params)),
        "train_months": TRAIN_MONTHS,
        "test_months": TEST_MONTHS,
        "cost_model": "dynamic_absolute_price_level",
        "cost_formula": f"round-trip cost cents = abs(entry_price) * {DYNAMIC_COST_CENTS_PER_PRICE_UNIT:g}",
        "cost_cents_round_trip": "",
        "bid_ask_cost_cents_per_price_unit": DYNAMIC_COST_CENTS_PER_PRICE_UNIT,
        "session_timezone": "Dubai UTC+4",
        "session_start_hour": SESSION_START_HOUR_DUBAI,
        "session_end_hour": SESSION_END_HOUR_DUBAI,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "min_train_trades": MIN_TRAIN_TRADES,
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "research_config.csv", index=False)
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
