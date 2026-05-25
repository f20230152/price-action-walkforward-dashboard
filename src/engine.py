from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd


DYNAMIC_COST_CENTS_PER_PRICE_UNIT = 0.04
DUBAI_UTC_OFFSET_HOURS = 4
PRE_SESSION_START_UTC_HOUR = 6
SESSION_START_UTC_HOUR = 7
SESSION_END_UTC_HOUR = 20


@dataclass(frozen=True)
class ParameterSet:
    a: int
    b: float
    c: int
    d: int
    direction: str

    @property
    def param_id(self) -> str:
        b_txt = f"{self.b:.4f}".rstrip("0").rstrip(".")
        return f"a{self.a}_b{b_txt}_c{self.c}_d{self.d}_{self.direction}"

    @property
    def label(self) -> str:
        b_txt = f"{self.b:.2f}".rstrip("0").rstrip(".")
        return f"a={self.a}s | b={b_txt}x sigma | c={self.c}s | d={self.d}s | {self.direction}"


def make_parameter_universe(
    a_values: Iterable[int],
    b_values: Iterable[float],
    c_values: Iterable[int],
    d_values: Iterable[int],
    directions: Iterable[str],
    stage: int,
) -> pd.DataFrame:
    rows = []
    for a, b, c, d, direction in product(a_values, b_values, c_values, d_values, directions):
        ps = ParameterSet(int(a), float(b), int(c), int(d), str(direction))
        rows.append(
            {
                "stage": int(stage),
                "param_id": ps.param_id,
                "parameter_set": ps.label,
                "a": ps.a,
                "b": ps.b,
                "c": ps.c,
                "d": ps.d,
                "direction": ps.direction,
            }
        )
    return pd.DataFrame(rows).sort_values(["a", "b", "c", "d", "direction"]).reset_index(drop=True)


def stage1_universe() -> pd.DataFrame:
    return make_parameter_universe(
        [60, 120, 300],
        [0.3, 0.5, 0.7, 0.9],
        [900, 1800, 3600],
        [1800, 7200, 14400],
        ["continuation", "reversal"],
        stage=1,
    )


def sharpe_from_daily(daily_pnl: pd.Series) -> float:
    daily_pnl = pd.Series(daily_pnl, dtype=float).dropna()
    if len(daily_pnl) < 2:
        return 0.0
    std = daily_pnl.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float(daily_pnl.mean() / std * np.sqrt(252))


def max_drawdown_cents(pnl: pd.Series) -> float:
    pnl = pd.Series(pnl, dtype=float).dropna()
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    return float((equity - equity.cummax()).min())


def trade_metrics(trades: pd.DataFrame, pnl_col: str = "pnl_cents") -> dict:
    empty = {
        "total_pnl_cents": 0.0,
        "trades": 0,
        "daily_sharpe": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_cents": 0.0,
        "top_trade_pnl_cents": 0.0,
        "top_trade_removed_pnl_cents": 0.0,
    }
    if trades.empty or pnl_col not in trades:
        return empty
    pnl = pd.to_numeric(trades[pnl_col], errors="coerce").dropna()
    if pnl.empty:
        return empty
    total = float(pnl.sum())
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gross_win = float(winners.sum())
    gross_loss = float(losers.sum())
    profit_factor = gross_win / abs(gross_loss) if gross_loss < 0 else (10.0 if gross_win > 0 else 0.0)
    daily_key = "entry_dubai_date" if "entry_dubai_date" in trades else None
    daily = trades.assign(_pnl=trades[pnl_col]).groupby(daily_key)["_pnl"].sum() if daily_key else pnl
    top_trade = float(winners.max()) if not winners.empty else 0.0
    return {
        "total_pnl_cents": total,
        "trades": int(len(pnl)),
        "daily_sharpe": sharpe_from_daily(daily),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(min(profit_factor, 10.0)),
        "max_drawdown_cents": max_drawdown_cents(pnl),
        "top_trade_pnl_cents": top_trade,
        "top_trade_removed_pnl_cents": float(total - top_trade),
    }


def rolling_sigma_real_prints(mid: np.ndarray, is_real_print: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Sigma from real-print mid changes only, scaled by sqrt(window).

    The scaling assumes independent one-second real-print changes. Missing
    seconds are not forward-filled into the estimator. A signal is allowed
    only when at least max(30, window // 4) real prints exist in the window.
    """

    min_periods = max(30, int(window) // 4)
    mid_series = pd.Series(mid, dtype="float64")
    real_mask = pd.Series(is_real_print, dtype=bool)
    real_returns_sparse = pd.Series(np.nan, index=mid_series.index, dtype="float64")
    real_returns_sparse.loc[real_mask] = mid_series.loc[real_mask].diff()
    real_count = real_mask.astype(int).rolling(window=window, min_periods=1).sum()
    sigma = real_returns_sparse.rolling(window=window, min_periods=min_periods).std() * np.sqrt(window)
    sigma = sigma.mask(real_count < min_periods)
    return sigma.to_numpy(dtype=float), real_count.to_numpy(dtype=float), min_periods


def _fill_prices(entry: pd.Series, exit_row: pd.Series, side: int, source_type: str) -> dict:
    entry_mid = float(entry["mid"])
    exit_mid = float(exit_row["mid"])
    mid_gross = side * (exit_mid - entry_mid) * 100.0
    mid_cost = abs(entry_mid) * DYNAMIC_COST_CENTS_PER_PRICE_UNIT
    mid_pnl = mid_gross - mid_cost
    out = {
        "mid_entry_px": entry_mid,
        "mid_exit_px": exit_mid,
        "mid_dynamic_cost_cents": mid_cost,
        "mid_dynamic_pnl_cents": mid_pnl,
        "actual_entry_px": np.nan,
        "actual_exit_px": np.nan,
        "actual_pnl_cents": np.nan,
        "actual_fill_status": "missing",
        "actual_entry_side": "",
        "actual_exit_side": "",
    }
    if str(source_type).startswith("parquet"):
        if side == 1:
            out["actual_entry_px"] = entry.get("ask_last", np.nan)
            out["actual_exit_px"] = exit_row.get("bid_last", np.nan)
            out["actual_entry_side"] = "ask"
            out["actual_exit_side"] = "bid"
            if pd.notna(out["actual_entry_px"]) and pd.notna(out["actual_exit_px"]):
                out["actual_pnl_cents"] = (float(out["actual_exit_px"]) - float(out["actual_entry_px"])) * 100.0
                out["actual_fill_status"] = "actual_bid_ask"
            else:
                out["actual_fill_status"] = "missing_bid_ask_skipped"
        else:
            out["actual_entry_px"] = entry.get("bid_last", np.nan)
            out["actual_exit_px"] = exit_row.get("ask_last", np.nan)
            out["actual_entry_side"] = "bid"
            out["actual_exit_side"] = "ask"
            if pd.notna(out["actual_entry_px"]) and pd.notna(out["actual_exit_px"]):
                out["actual_pnl_cents"] = (float(out["actual_entry_px"]) - float(out["actual_exit_px"])) * 100.0
                out["actual_fill_status"] = "actual_bid_ask"
            else:
                out["actual_fill_status"] = "missing_bid_ask_skipped"
    else:
        out["actual_entry_px"] = entry_mid
        out["actual_exit_px"] = exit_mid
        out["actual_pnl_cents"] = mid_pnl
        out["actual_fill_status"] = "csv_dynamic_cost_fallback"
        out["actual_entry_side"] = "mid"
        out["actual_exit_side"] = "mid"
    return out


def generate_base_trades(data: pd.DataFrame, universe: pd.DataFrame, return_diagnostics: bool = False):
    """Generate candidate trades.

    Sigma is the std of real-print one-second mid changes over c, scaled by
    sqrt(c) to make it comparable to the c-second price move. Forward-filled
    bars remain available for exits, but they are not consumed by sigma.
    """
    if data.empty:
        empty = pd.DataFrame()
        return (empty, empty) if return_diagnostics else empty
    data = data.sort_index()
    param_lookup = universe.set_index("param_id")["parameter_set"].to_dict()
    grouped = universe.groupby(["a", "b", "c"], sort=False)
    rows = []
    skipped_rows = []
    for date_value, day_df in data.groupby(pd.Series(data.index.date, index=data.index), sort=True):
        day_df = day_df.sort_index()
        mid = day_df["mid"].to_numpy(dtype=float)
        is_real_print = day_df.get("is_real_print", pd.Series(True, index=day_df.index)).fillna(False).to_numpy(dtype=bool)
        was_forward_filled = day_df.get("was_forward_filled", pd.Series(False, index=day_df.index)).fillna(False).to_numpy(dtype=bool)
        if len(mid) < 7200 or np.isnan(mid).all():
            continue
        ts = day_df.index
        seconds = ts.hour.to_numpy() * 3600 + ts.minute.to_numpy() * 60 + ts.second.to_numpy()
        session_start = SESSION_START_UTC_HOUR * 3600
        session_end = SESSION_END_UTC_HOUR * 3600
        midnight_positions = np.flatnonzero(seconds >= session_end)
        if len(midnight_positions) == 0:
            continue
        midnight_pos = int(midnight_positions[0])
        signal_session = (seconds >= session_start) & (seconds < session_end)
        by_c = {}
        for c in sorted(universe["c"].unique()):
            c = int(c)
            delta = np.full(len(mid), np.nan)
            if len(mid) > c:
                delta[c:] = mid[c:] - mid[:-c]
            sigma, real_count, min_periods = rolling_sigma_real_prints(mid, is_real_print, c)
            by_c[c] = (delta, sigma, real_count, min_periods)
        positions = np.arange(len(mid))
        for (a, b, c), group in grouped:
            a = int(a)
            b = float(b)
            c = int(c)
            delta, sigma, real_count, min_periods = by_c[c]
            trigger = (
                signal_session
                & (positions + a < midnight_pos)
                & np.isfinite(delta)
                & np.isfinite(sigma)
                & (sigma > 0)
                & (real_count >= min_periods)
                & (np.abs(delta) > b * sigma)
            )
            signal_positions = np.flatnonzero(trigger)
            if len(signal_positions) == 0:
                continue
            signal_pos = int(signal_positions[0])
            move_sign = int(np.sign(delta[signal_pos]))
            if move_sign == 0:
                continue
            entry_pos = signal_pos + a
            entry = day_df.iloc[entry_pos]
            entry_ts = ts[entry_pos]
            source_type = str(entry.get("source_type", ""))
            day_days_to_expiry = entry.get("days_to_expiry", np.nan)
            if pd.notna(day_days_to_expiry) and float(day_days_to_expiry) < 3:
                skipped_rows.append(
                    {
                        "skip_reason": "near_expiry_lt_3_business_days",
                        "date": pd.Timestamp(date_value).date().isoformat(),
                        "signal_time_utc": ts[signal_pos],
                        "entry_time_utc": entry_ts,
                        "selected_symbol": entry.get("selected_symbol", ""),
                        "contract_expiry_date": entry.get("contract_expiry_date", ""),
                        "days_to_expiry": day_days_to_expiry,
                        "a": a,
                        "b": b,
                        "c": c,
                        "affected_parameter_count": int(len(group)),
                    }
                )
                continue
            if bool(was_forward_filled[entry_pos]):
                skipped_rows.append(
                    {
                        "skip_reason": "forward_filled_entry",
                        "date": pd.Timestamp(date_value).date().isoformat(),
                        "signal_time_utc": ts[signal_pos],
                        "entry_time_utc": entry_ts,
                        "selected_symbol": entry.get("selected_symbol", ""),
                        "contract_expiry_date": entry.get("contract_expiry_date", ""),
                        "days_to_expiry": day_days_to_expiry,
                        "a": a,
                        "b": b,
                        "c": c,
                        "affected_parameter_count": int(len(group)),
                    }
                )
                continue
            for _, param in group.iterrows():
                side = move_sign if param["direction"] == "continuation" else -move_sign
                d = int(param["d"])
                exit_pos = min(entry_pos + d, midnight_pos)
                if exit_pos <= entry_pos:
                    continue
                exit_row = day_df.iloc[exit_pos]
                exit_ts = ts[exit_pos]
                fills = _fill_prices(entry, exit_row, side, source_type)
                param_id = str(param["param_id"])
                entry_dubai = entry_ts + pd.Timedelta(hours=DUBAI_UTC_OFFSET_HOURS)
                exit_dubai = exit_ts + pd.Timedelta(hours=DUBAI_UTC_OFFSET_HOURS)
                rows.append(
                    {
                        "stage": int(param["stage"]),
                        "param_id": param_id,
                        "parameter_set": param_lookup[param_id],
                        "a": int(param["a"]),
                        "b": float(param["b"]),
                        "c": c,
                        "d": d,
                        "direction": str(param["direction"]),
                        "side": "long" if side == 1 else "short",
                        "side_sign": side,
                        "signal_time_utc": ts[signal_pos],
                        "signal_move_cents": float(delta[signal_pos] * 100.0),
                        "signal_sigma_cents": float(sigma[signal_pos] * 100.0),
                        "signal_real_print_count": int(real_count[signal_pos]),
                        "sigma_min_real_prints": int(min_periods),
                        "signal_was_forward_filled": bool(was_forward_filled[signal_pos]),
                        "entry_time_utc": entry_ts,
                        "entry_time_dubai": entry_dubai,
                        "entry_dubai_date": entry_dubai.date().isoformat(),
                        "entry_was_forward_filled": bool(was_forward_filled[entry_pos]),
                        "exit_time_utc": exit_ts,
                        "exit_time_dubai": exit_dubai,
                        "exit_was_forward_filled": bool(was_forward_filled[exit_pos]),
                        "hold_seconds": int((exit_ts - entry_ts).total_seconds()),
                        "source_type": source_type,
                        "selected_symbol": entry.get("selected_symbol", ""),
                        "contract_expiry_date": entry.get("contract_expiry_date", ""),
                        "days_to_expiry": entry.get("days_to_expiry", np.nan),
                        "roll_reason": entry.get("roll_reason", ""),
                        **fills,
                    }
                )
    trades = pd.DataFrame(rows)
    if trades.empty:
        skipped = pd.DataFrame(skipped_rows)
        return (trades, skipped) if return_diagnostics else trades
    for col in ["signal_time_utc", "entry_time_utc", "entry_time_dubai", "exit_time_utc", "exit_time_dubai"]:
        trades[col] = pd.to_datetime(trades[col])
    trades = trades.sort_values(["param_id", "entry_time_utc"]).reset_index(drop=True)
    skipped = pd.DataFrame(skipped_rows)
    return (trades, skipped) if return_diagnostics else trades


def materialize_fill_trades(base_trades: pd.DataFrame, fill_model: str) -> pd.DataFrame:
    if base_trades.empty:
        return pd.DataFrame()
    trades = base_trades.copy()
    if fill_model == "mid_dynamic_cost":
        trades["entry_px"] = trades["mid_entry_px"]
        trades["exit_px"] = trades["mid_exit_px"]
        trades["pnl_cents"] = trades["mid_dynamic_pnl_cents"]
        trades["fill_status"] = "mid_dynamic_cost"
        trades["entry_fill_side"] = "mid"
        trades["exit_fill_side"] = "mid"
    elif fill_model == "actual_bid_ask":
        trades["entry_px"] = trades["actual_entry_px"]
        trades["exit_px"] = trades["actual_exit_px"]
        trades["pnl_cents"] = trades["actual_pnl_cents"]
        trades["fill_status"] = trades["actual_fill_status"]
        trades["entry_fill_side"] = trades["actual_entry_side"]
        trades["exit_fill_side"] = trades["actual_exit_side"]
        trades = trades[trades["pnl_cents"].notna()].copy()
    else:
        raise ValueError(f"Unknown fill model: {fill_model}")
    trades["fill_model"] = fill_model
    return trades.reset_index(drop=True)


def summarize_daily_monthly(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    daily = (
        trades.groupby(["schedule", "fill_model", "entry_dubai_date"], dropna=False)
        .agg(pnl_cents=("pnl_cents", "sum"), trades=("pnl_cents", "size"))
        .reset_index()
        .rename(columns={"entry_dubai_date": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = daily.groupby(["schedule", "fill_model", "month"], dropna=False).agg(
        pnl_cents=("pnl_cents", "sum"), trades=("trades", "sum")
    ).reset_index()
    equity = daily.sort_values(["schedule", "fill_model", "date"]).copy()
    equity["cum_pnl_cents"] = equity.groupby(["schedule", "fill_model"])["pnl_cents"].cumsum()
    return daily, monthly, equity
