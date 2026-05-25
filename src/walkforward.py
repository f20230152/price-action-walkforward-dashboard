from __future__ import annotations

from typing import Iterable

import pandas as pd

from .engine import materialize_fill_trades, trade_metrics
from .selection import FoldSpec, select_fold_winner


FILL_MODELS = ["mid_dynamic_cost", "actual_bid_ask"]


def make_folds(train_months: int) -> list[FoldSpec]:
    if train_months == 3:
        first_test = "2025-04"
        schedule = "3m_train_1m_test"
    elif train_months == 6:
        first_test = "2025-07"
        schedule = "6m_train_1m_test"
    else:
        raise ValueError("Only 3-month and 6-month schedules are supported")
    folds = []
    for test_period in pd.period_range(first_test, "2026-03", freq="M"):
        test_start = test_period.to_timestamp()
        folds.append(
            FoldSpec(
                schedule=schedule,
                train_months=train_months,
                fold=f"{schedule}_{str(test_period)}",
                train_start=(test_period - train_months).to_timestamp(),
                train_end=test_start,
                test_start=test_start,
                test_end=(test_period + 1).to_timestamp(),
            )
        )
    return folds


def _date_filter(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    return trades[(trades["entry_time_utc"] >= start) & (trades["entry_time_utc"] < end)].copy()


def run_walkforward_stage(
    base_trades: pd.DataFrame,
    universe: pd.DataFrame,
    stage: int,
    fill_models: Iterable[str] = FILL_MODELS,
) -> dict[str, pd.DataFrame]:
    ranking_frames = []
    decision_frames = []
    selected_trade_frames = []
    winner_rows = []
    folds = make_folds(3) + make_folds(6)
    for fill_model in fill_models:
        fill_trades = materialize_fill_trades(base_trades, fill_model)
        for fold in folds:
            train_trades = _date_filter(fill_trades, fold.train_start, fold.train_end)
            rankings, decisions, winner = select_fold_winner(train_trades, universe, fold, fill_model, stage)
            ranking_frames.append(rankings)
            decision_frames.append(decisions)
            if winner is None:
                continue
            test_all = _date_filter(fill_trades, fold.test_start, fold.test_end)
            test_trades = test_all[test_all["param_id"] == winner["param_id"]].copy()
            test_trades["stage"] = stage
            test_trades["schedule"] = fold.schedule
            test_trades["fold"] = fold.fold
            test_trades["train_start"] = fold.train_start.date().isoformat()
            test_trades["train_end_exclusive"] = fold.train_end.date().isoformat()
            test_trades["test_start"] = fold.test_start.date().isoformat()
            test_trades["test_end_exclusive"] = fold.test_end.date().isoformat()
            test_trades["selected_composite_score"] = winner.get("composite_score")
            test_trades["selection_mode"] = winner.get("selection_mode")
            selected_trade_frames.append(test_trades)
            test_metrics = trade_metrics(test_trades)
            winner_rows.append(
                {
                    "stage": stage,
                    "schedule": fold.schedule,
                    "fill_model": fill_model,
                    "fold": fold.fold,
                    "train_start": fold.train_start.date().isoformat(),
                    "train_end_exclusive": fold.train_end.date().isoformat(),
                    "test_start": fold.test_start.date().isoformat(),
                    "test_end_exclusive": fold.test_end.date().isoformat(),
                    "param_id": winner["param_id"],
                    "parameter_set": winner["parameter_set"],
                    "a": int(winner["a"]),
                    "b": float(winner["b"]),
                    "c": int(winner["c"]),
                    "d": int(winner["d"]),
                    "direction": winner["direction"],
                    "selection_mode": winner.get("selection_mode"),
                    "composite_score": winner.get("composite_score"),
                    "train_total_pnl_cents": winner.get("total_pnl_cents"),
                    "train_daily_sharpe": winner.get("daily_sharpe"),
                    "train_top_trade_removed_pnl_cents": winner.get("top_trade_removed_pnl_cents"),
                    "train_profit_factor": winner.get("profit_factor"),
                    "train_max_drawdown_cents": winner.get("max_drawdown_cents"),
                    "train_neighborhood_count": winner.get("neighborhood_count"),
                    "train_consistency_gate_pass": bool(winner.get("consistency_gate_pass")),
                    "train_top_trade_gate_pass": bool(winner.get("top_trade_gate_pass")),
                    "train_neighborhood_gate_pass": bool(winner.get("neighborhood_gate_pass")),
                    "winner_is_top_composite": bool(winner.get("winner_is_top_composite")),
                    "test_total_pnl_cents": test_metrics["total_pnl_cents"],
                    "test_daily_sharpe": test_metrics["daily_sharpe"],
                    "test_trades": test_metrics["trades"],
                    "test_win_rate": test_metrics["win_rate"],
                    "test_profit_factor": test_metrics["profit_factor"],
                    "test_max_drawdown_cents": test_metrics["max_drawdown_cents"],
                    "test_top_trade_removed_pnl_cents": test_metrics["top_trade_removed_pnl_cents"],
                }
            )
    return {
        "train_rankings": pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame(),
        "decisions": pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame(),
        "trades": pd.concat(selected_trade_frames, ignore_index=True) if selected_trade_frames else pd.DataFrame(),
        "fold_winners": pd.DataFrame(winner_rows),
    }

