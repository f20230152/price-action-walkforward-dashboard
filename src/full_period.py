from __future__ import annotations

import pandas as pd

from .engine import materialize_fill_trades


def choose_final_parameters(fold_winners: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if fold_winners.empty:
        return pd.DataFrame()
    for fill_model, group in fold_winners.groupby("fill_model", dropna=False):
        counts = group["param_id"].value_counts()
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        clear = top_count >= 2 and (counts == top_count).sum() == 1
        if clear:
            selected = group[group["param_id"] == counts.index[0]].iloc[-1]
            reason = "most_common_oos_fold_winner"
        else:
            six = group[group["schedule"] == "6m_train_1m_test"].sort_values("test_start")
            selected = six.iloc[-1] if not six.empty else group.sort_values("test_start").iloc[-1]
            reason = "fallback_final_6m_fold_winner"
        rows.append(
            {
                "fill_model": fill_model,
                "param_id": selected["param_id"],
                "parameter_set": selected["parameter_set"],
                "a": int(selected["a"]),
                "b": float(selected["b"]),
                "c": int(selected["c"]),
                "d": int(selected["d"]),
                "direction": selected["direction"],
                "selection_reason": reason,
                "fold_win_count": int(counts.get(selected["param_id"], 0)),
            }
        )
    return pd.DataFrame(rows)


def run_full_period_backtests(base_trades: pd.DataFrame, final_choices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for _, choice in final_choices.iterrows():
        fill_trades = materialize_fill_trades(base_trades, str(choice["fill_model"]))
        selected = fill_trades[fill_trades["param_id"] == choice["param_id"]].copy()
        selected["full_period_label"] = "visual_in_sample_full_period"
        selected["selection_reason"] = choice["selection_reason"]
        selected["is_in_sample_visual"] = True
        frames.append(selected)
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if trades.empty:
        return trades, pd.DataFrame()
    equity = trades.sort_values(["fill_model", "entry_time_utc"]).copy()
    equity["cum_pnl_cents"] = equity.groupby("fill_model")["pnl_cents"].cumsum()
    return trades, equity[
        [
            "fill_model",
            "entry_time_utc",
            "entry_time_dubai",
            "pnl_cents",
            "cum_pnl_cents",
            "param_id",
            "parameter_set",
            "is_in_sample_visual",
        ]
    ]

