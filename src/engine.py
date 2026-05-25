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


def _rolling_sigma_scaled(mid: np.ndarray, window: int) -> np.ndarray:
    returns = pd.Series(mid, dtype="float64").diff()
    min_periods = max(30, window // 3)
    return returns.rolling(window=window, min_periods=min_periods).std().to_numpy() * np.sqrt(window)


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


def generate_base_trades(data: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Generate candidate trades.

    Sigma is the close-to-close std of one-second mid changes over c, scaled by
    sqrt(c) to make it comparable to the c-second price move.
    """
    if data.empty:
        return pd.DataFrame()
    data = data.sort_index()
    param_lookup = universe.set_index("param_id")["parameter_set"].to_dict()
    grouped = universe.groupby(["a", "b", "c"], sort=False)
    rows = []
    for date_value, day_df in data.groupby(pd.Series(data.index.date, index=data.index), sort=True):
        day_df = day_df.sort_index()
        mid = day_df["mid"].to_numpy(dtype=float)
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
            by_c[c] = (delta, _rolling_sigma_scaled(mid, c))
        positions = np.arange(len(mid))
        for (a, b, c), group in grouped:
            a = int(a)
            b = float(b)
            c = int(c)
            delta, sigma = by_c[c]
            trigger = (
                signal_session
                & (positions + a < midnight_pos)
                & np.isfinite(delta)
                & np.isfinite(sigma)
                & (sigma > 0)
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
                        "entry_time_utc": entry_ts,
                        "entry_time_dubai": entry_dubai,
                        "entry_dubai_date": entry_dubai.date().isoformat(),
                        "exit_time_utc": exit_ts,
                        "exit_time_dubai": exit_dubai,
                        "hold_seconds": int((exit_ts - entry_ts).total_seconds()),
                        "source_type": source_type,
                        "selected_symbol": entry.get("selected_symbol", ""),
                        **fills,
                    }
                )
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades
    for col in ["signal_time_utc", "entry_time_utc", "entry_time_dubai", "exit_time_utc", "exit_time_dubai"]:
        trades[col] = pd.to_datetime(trades[col])
    return trades.sort_values(["param_id", "entry_time_utc"]).reset_index(drop=True)


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

