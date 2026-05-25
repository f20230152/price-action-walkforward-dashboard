from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .engine import make_parameter_universe, trade_metrics


@dataclass(frozen=True)
class FoldSpec:
    schedule: str
    train_months: int
    fold: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - values.mean()) / std


def subfold_ranges(train_start: pd.Timestamp, train_end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    months = pd.period_range(train_start.to_period("M"), (train_end - pd.Timedelta(days=1)).to_period("M"), freq="M")
    chunks = np.array_split(months, 3)
    ranges = []
    for chunk in chunks:
        if len(chunk):
            ranges.append((chunk[0].to_timestamp(), (chunk[-1] + 1).to_timestamp()))
    return ranges


def _param_tuple(row: pd.Series) -> tuple[Any, ...]:
    return (int(row["a"]), float(row["b"]), int(row["c"]), int(row["d"]), str(row["direction"]))


def _neighborhood_counts(candidates: pd.DataFrame, universe: pd.DataFrame) -> dict[str, int]:
    if candidates.empty:
        return {}
    axis_values = {axis: sorted(universe[axis].unique()) for axis in ["a", "b", "c", "d", "direction"]}
    value_to_idx = {axis: {value: i for i, value in enumerate(values)} for axis, values in axis_values.items()}
    surviving = {_param_tuple(row): row["param_id"] for _, row in candidates.iterrows()}
    surviving_tuples = set(surviving)
    counts = {pid: 0 for pid in candidates["param_id"]}
    for tup, param_id in surviving.items():
        values = {"a": tup[0], "b": tup[1], "c": tup[2], "d": tup[3], "direction": tup[4]}
        for axis in ["a", "b", "c", "d", "direction"]:
            idx = value_to_idx[axis][values[axis]]
            for nidx in [idx - 1, idx + 1]:
                if 0 <= nidx < len(axis_values[axis]):
                    test = values.copy()
                    test[axis] = axis_values[axis][nidx]
                    ntup = (int(test["a"]), float(test["b"]), int(test["c"]), int(test["d"]), str(test["direction"]))
                    if ntup in surviving_tuples:
                        counts[param_id] += 1
    return counts


def _metrics_for_param(trades: pd.DataFrame, param_id: str, fold: FoldSpec) -> dict[str, Any]:
    pt = trades[trades["param_id"] == param_id].copy()
    metrics = trade_metrics(pt)
    sub_pnls = []
    for start, end in subfold_ranges(fold.train_start, fold.train_end):
        sub = pt[(pt["entry_time_utc"] >= start) & (pt["entry_time_utc"] < end)]
        sub_pnls.append(float(sub["pnl_cents"].sum()) if not sub.empty else 0.0)
    metrics.update(
        {
            "subfold_1_pnl_cents": sub_pnls[0] if len(sub_pnls) > 0 else 0.0,
            "subfold_2_pnl_cents": sub_pnls[1] if len(sub_pnls) > 1 else 0.0,
            "subfold_3_pnl_cents": sub_pnls[2] if len(sub_pnls) > 2 else 0.0,
            "profitable_train_subfolds": int(sum(p > 0 for p in sub_pnls)),
        }
    )
    metrics["consistency_gate_pass"] = metrics["profitable_train_subfolds"] >= 2
    metrics["top_trade_gate_pass"] = metrics["top_trade_removed_pnl_cents"] >= 0
    return metrics


def select_fold_winner(
    train_trades: pd.DataFrame,
    universe: pd.DataFrame,
    fold: FoldSpec,
    fill_model: str,
    stage: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    rows = []
    for _, param in universe.iterrows():
        rows.append(
            {
                "stage": stage,
                "schedule": fold.schedule,
                "fill_model": fill_model,
                "fold": fold.fold,
                "train_start": fold.train_start.date().isoformat(),
                "train_end_exclusive": fold.train_end.date().isoformat(),
                "test_start": fold.test_start.date().isoformat(),
                "test_end_exclusive": fold.test_end.date().isoformat(),
                **param.to_dict(),
                **_metrics_for_param(train_trades, str(param["param_id"]), fold),
            }
        )
    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored, scored, None
    base_mask = scored["consistency_gate_pass"] & scored["top_trade_gate_pass"]
    counts = _neighborhood_counts(scored[base_mask].copy(), universe)
    scored["neighborhood_count"] = scored["param_id"].map(counts).fillna(0).astype(int)
    scored["neighborhood_gate_pass"] = scored["neighborhood_count"] >= 3
    scored["survives_base_gates"] = base_mask
    scored["survives_all_gates"] = base_mask & scored["neighborhood_gate_pass"]
    if scored["survives_all_gates"].any():
        pop_idx = scored.index[scored["survives_all_gates"]]
        mode = "strict_all_gates"
    elif scored["survives_base_gates"].any():
        pop_idx = scored.index[scored["survives_base_gates"]]
        mode = "fallback_no_neighbor_gate_survivors"
    else:
        pop_idx = scored.index
        mode = "fallback_no_base_gate_survivors"
    scored["eligible_for_selection"] = False
    scored.loc[pop_idx, "eligible_for_selection"] = True
    scored["selection_mode"] = mode
    scored.loc[pop_idx, "z_sharpe"] = zscore(scored.loc[pop_idx, "daily_sharpe"])
    scored.loc[pop_idx, "z_top_trade_removed_pnl"] = zscore(scored.loc[pop_idx, "top_trade_removed_pnl_cents"])
    scored.loc[pop_idx, "z_profit_factor"] = zscore(scored.loc[pop_idx, "profit_factor"])
    scored.loc[pop_idx, "z_neighborhood_count"] = zscore(scored.loc[pop_idx, "neighborhood_count"])
    scored.loc[pop_idx, "z_drawdown"] = zscore(-scored.loc[pop_idx, "max_drawdown_cents"])
    scored.loc[pop_idx, "composite_score"] = (
        0.30 * scored.loc[pop_idx, "z_sharpe"]
        + 0.25 * scored.loc[pop_idx, "z_top_trade_removed_pnl"]
        + 0.15 * scored.loc[pop_idx, "z_profit_factor"]
        + 0.15 * scored.loc[pop_idx, "z_neighborhood_count"]
        + 0.15 * scored.loc[pop_idx, "z_drawdown"]
    )
    winner_idx = scored.loc[pop_idx, "composite_score"].idxmax()
    scored["is_winner"] = False
    scored.loc[winner_idx, "is_winner"] = True
    scored["winner_is_top_composite"] = bool(np.isclose(scored.loc[winner_idx, "composite_score"], scored.loc[pop_idx, "composite_score"].max()))
    scored["rank"] = np.nan
    scored.loc[pop_idx, "rank"] = scored.loc[pop_idx, "composite_score"].rank(ascending=False, method="first")
    reasons = []
    for _, row in scored.iterrows():
        eliminated = []
        if not row["consistency_gate_pass"]:
            eliminated.append("train_subfold_consistency")
        if not row["top_trade_gate_pass"]:
            eliminated.append("top_trade_survival")
        if row["survives_base_gates"] and not row["neighborhood_gate_pass"]:
            eliminated.append("neighborhood_support")
        reasons.append(";".join(eliminated) if eliminated else "survived")
    scored["elimination_reason"] = reasons
    rankings = scored[scored["eligible_for_selection"]].sort_values("composite_score", ascending=False).copy()
    decisions = scored.sort_values(["is_winner", "eligible_for_selection", "composite_score"], ascending=[False, False, False]).copy()
    return rankings, decisions, scored.loc[winner_idx].copy()


def build_stage2_universe(stage1_rankings: pd.DataFrame, stage1_params: pd.DataFrame) -> pd.DataFrame:
    strict = stage1_rankings[stage1_rankings["survives_all_gates"] == True].copy() if "survives_all_gates" in stage1_rankings else pd.DataFrame()
    if strict.empty and "survives_base_gates" in stage1_rankings:
        strict = stage1_rankings[stage1_rankings["survives_base_gates"] == True].copy()
    if strict.empty:
        out = stage1_params.copy()
        out["stage"] = 2
        out["stage2_source"] = "fallback_stage1_no_survivors"
        return out

    strict = strict.copy()
    strict["rank_for_seed"] = pd.to_numeric(strict.get("rank", 9999), errors="coerce").fillna(9999)
    seed_scores = (
        strict.groupby("param_id", as_index=False)
        .agg(
            appearances=("param_id", "size"),
            best_rank=("rank_for_seed", "min"),
            mean_composite=("composite_score", "mean"),
            a=("a", "first"),
            b=("b", "first"),
            c=("c", "first"),
            d=("d", "first"),
            direction=("direction", "first"),
        )
        .sort_values(["appearances", "best_rank", "mean_composite"], ascending=[False, True, False])
        .head(40)
    )

    original_values = {axis: sorted(stage1_params[axis].unique()) for axis in ["a", "b", "c", "d"]}

    def local_axis_values(axis: str, value: Any) -> list[Any]:
        values = set([value])
        original = original_values[axis]
        idx_map = {v: i for i, v in enumerate(original)}
        if value in idx_map:
            idx = idx_map[value]
            for nidx in [idx - 1, idx + 1]:
                if 0 <= nidx < len(original):
                    neighbor = original[nidx]
                    values.add(neighbor)
                    if axis == "b":
                        values.add(round((float(value) + float(neighbor)) / 2.0, 4))
                    else:
                        lo, hi = sorted([int(value), int(neighbor)])
                        gap = hi - lo
                        if gap >= 3:
                            values.add(int(round(lo + gap / 3.0)))
                            values.add(int(round(lo + 2.0 * gap / 3.0)))
        if axis == "b":
            return [float(f"{float(v):.4f}") for v in sorted(values)]
        return [int(v) for v in sorted(values)]

    rows: list[dict[str, Any]] = []
    for _, seed in seed_scores.iterrows():
        base = {
            "a": int(seed["a"]),
            "b": float(seed["b"]),
            "c": int(seed["c"]),
            "d": int(seed["d"]),
            "direction": str(seed["direction"]),
        }
        combos = [base.copy()]
        for axis in ["a", "b", "c", "d"]:
            for local_value in local_axis_values(axis, base[axis]):
                combo = base.copy()
                combo[axis] = local_value
                combos.append(combo)
        for direction in sorted(stage1_params["direction"].unique()):
            combo = base.copy()
            combo["direction"] = direction
            combos.append(combo)
        for combo in combos:
            rows.append(combo)

    out = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    out = make_parameter_universe(
        sorted(out["a"].unique()),
        sorted(out["b"].unique()),
        sorted(out["c"].unique()),
        sorted(out["d"].unique()),
        sorted(out["direction"].unique()),
        stage=2,
    )
    valid = pd.DataFrame(rows).drop_duplicates()
    out = out.merge(valid.assign(_local_refined=True), on=["a", "b", "c", "d", "direction"], how="inner")
    out = out.drop(columns=["_local_refined"])
    out["stage2_source"] = "refined_from_stage1_gate_survivors"
    return out
