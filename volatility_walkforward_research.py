from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from price_action_engine import (
    DUBAI_UTC_OFFSET_HOURS,
    DYNAMIC_COST_CENTS_PER_PRICE_UNIT,
    compute_metrics,
    load_second_prices,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "volatility_walkforward"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MONTHS = 3
WALKFORWARD_START = pd.Timestamp("2025-11-01")
VOL_WINDOWS_DAYS = [63]
VOL_MIN_OBSERVATIONS = 21
SIGMA_MULTIPLES = [0.25, 0.40, 0.55, 0.75, 1.00]
STABLE_DELAYS = [60, 120, 300]
STABLE_SIGMA_MULTIPLES = [0.25, 0.40, 0.55, 0.75, 1.00]
STABLE_MOVE_WINDOWS = [900, 1800, 3600]
STABLE_HOLDS = [3600, 7200, 14400, 21600]
RARE_DELAYS = [60, 120, 300]
RARE_PERCENTILES = [95.0, 97.5, 99.0]
RARE_MOVE_WINDOWS = [300, 900, 1800, 3600]
RARE_HOLDS = [3600, 7200, 14400, 21600]
RARE_LOOKBACK_DAYS = 63
RARE_MIN_OBSERVATIONS = 500
REGIME_RARE_DELAYS = [120]
REGIME_RARE_PERCENTILES = [95.0, 97.5, 99.0]
REGIME_RARE_MOVE_WINDOWS = [900, 1800]
REGIME_RARE_HOLDS = [3600, 7200]
REGIME_TREND_FILTERS = ["any", "up", "down"]
REGIME_VOL_FILTERS = ["any", "high"]
REGIME_TIME_BUCKETS = ["any", "early", "mid", "late"]
SESSION_START_HOUR_DUBAI = 11
SESSION_END_HOUR_DUBAI = 24
MAX_TRADES_PER_DAY = 1
MIN_TRAIN_TRADES = 5
REQUIRE_FULL_SIGNAL_WINDOW_IN_SESSION = True
ROBUST_OBJECTIVE_NAME = "robust_stability_score"


@dataclass(frozen=True)
class VolParams:
    delay_s: int
    sigma_multiple: float
    move_window_s: int
    hold_s: int
    vol_window_days: int
    vol_method: str
    signal_mode: str
    bad_hour_rule: str

    @property
    def label(self) -> str:
        return (
            f"a={self.delay_s}s | move={self.sigma_multiple:g}x {self.vol_method} sigma "
            f"({self.vol_window_days}D)/{self.move_window_s}s | hold={self.hold_s}s | "
            f"{self.signal_mode} | {self.bad_hour_rule}"
        )


@dataclass(frozen=True)
class RareMoveParams:
    delay_s: int
    percentile: float
    move_window_s: int
    hold_s: int
    signal_mode: str
    bad_hour_rule: str

    @property
    def label(self) -> str:
        percentile_label = f"{self.percentile:g}th percentile"
        return (
            f"a={self.delay_s}s | move>{percentile_label} of recent {self.move_window_s}s moves | "
            f"hold={self.hold_s}s | {self.signal_mode} | {self.bad_hour_rule}"
        )


@dataclass(frozen=True)
class RegimeRareMoveParams:
    delay_s: int
    percentile: float
    move_window_s: int
    hold_s: int
    signal_mode: str
    bad_hour_rule: str
    trend_filter: str
    vol_filter: str
    time_bucket: str

    @property
    def label(self) -> str:
        percentile_label = f"{self.percentile:g}th percentile"
        return (
            f"a={self.delay_s}s | move>{percentile_label} of recent {self.move_window_s}s moves | "
            f"hold={self.hold_s}s | {self.signal_mode} | trend={self.trend_filter} | "
            f"vol={self.vol_filter} | time={self.time_bucket} | {self.bad_hour_rule}"
        )


def build_universe() -> list[VolParams]:
    return [
        VolParams(
            delay_s=delay,
            sigma_multiple=multiple,
            move_window_s=lookback,
            hold_s=hold,
            vol_window_days=vol_window,
            vol_method=vol_method,
            signal_mode=mode,
            bad_hour_rule=rule,
        )
        for delay, multiple, lookback, hold, vol_window, vol_method, mode, rule in itertools.product(
            [120],
            SIGMA_MULTIPLES,
            [1800],
            [3600, 21600],
            VOL_WINDOWS_DAYS,
            ["percent_to_dollar"],
            ["continuation", "reversal"],
            ["exit_by_midnight"],
        )
    ]


def build_rare_universe() -> list[RareMoveParams]:
    return [
        RareMoveParams(
            delay_s=delay,
            percentile=percentile,
            move_window_s=lookback,
            hold_s=hold,
            signal_mode=mode,
            bad_hour_rule="exit_by_midnight",
        )
        for delay, percentile, lookback, hold, mode in itertools.product(
            RARE_DELAYS,
            RARE_PERCENTILES,
            RARE_MOVE_WINDOWS,
            RARE_HOLDS,
            ["continuation", "reversal"],
        )
    ]


def build_regime_rare_universe() -> list[RegimeRareMoveParams]:
    return [
        RegimeRareMoveParams(
            delay_s=delay,
            percentile=percentile,
            move_window_s=lookback,
            hold_s=hold,
            signal_mode=mode,
            bad_hour_rule="exit_by_midnight",
            trend_filter=trend_filter,
            vol_filter=vol_filter,
            time_bucket=time_bucket,
        )
        for delay, percentile, lookback, hold, mode, trend_filter, vol_filter, time_bucket in itertools.product(
            REGIME_RARE_DELAYS,
            REGIME_RARE_PERCENTILES,
            REGIME_RARE_MOVE_WINDOWS,
            REGIME_RARE_HOLDS,
            ["continuation", "reversal"],
            REGIME_TREND_FILTERS,
            REGIME_VOL_FILTERS,
            REGIME_TIME_BUCKETS,
        )
    ]


def build_stable_universe() -> list[VolParams]:
    return [
        VolParams(
            delay_s=delay,
            sigma_multiple=multiple,
            move_window_s=lookback,
            hold_s=hold,
            vol_window_days=vol_window,
            vol_method="percent_to_dollar",
            signal_mode=mode,
            bad_hour_rule="exit_by_midnight",
        )
        for delay, multiple, lookback, hold, vol_window, mode in itertools.product(
            STABLE_DELAYS,
            STABLE_SIGMA_MULTIPLES,
            STABLE_MOVE_WINDOWS,
            STABLE_HOLDS,
            VOL_WINDOWS_DAYS,
            ["continuation", "reversal"],
        )
    ]


def objective_slug(objective: str) -> str:
    if objective == "daily_sharpe":
        return "sharpe_objective"
    if objective == "total_pnl_cents":
        return "pnl_objective"
    return objective


def add_volatility_columns(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    daily_close = out["price"].groupby(out.index.normalize()).last().sort_index()
    dollar_change = daily_close.diff()
    pct_change = daily_close.pct_change()
    vol_cols: dict[str, pd.Series] = {}
    for window in VOL_WINDOWS_DAYS:
        dollar_sigma = dollar_change.rolling(window, min_periods=VOL_MIN_OBSERVATIONS).std().shift(1)
        pct_sigma_to_dollar = (pct_change.rolling(window, min_periods=VOL_MIN_OBSERVATIONS).std().shift(1) * daily_close.shift(1))
        vol_cols[f"sigma_dollar_{window}"] = dollar_sigma
        vol_cols[f"sigma_percent_to_dollar_{window}"] = pct_sigma_to_dollar
        vol_cols[f"trend_5d_pct_{window}"] = daily_close.pct_change(5).shift(1)
        vol_cols[f"vol_rank_{window}"] = pct_sigma_to_dollar.rolling(window, min_periods=VOL_MIN_OBSERVATIONS).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        )

    date_index = out.index.normalize()
    for name, series in vol_cols.items():
        out[name] = date_index.map(series).astype(float)
    return out


def prepare_days(bars: pd.DataFrame) -> list[dict]:
    days = []
    for date, day_bars in bars.dropna(subset=["price"]).groupby(bars.index.normalize(), sort=True):
        dubai_index = day_bars.index + pd.Timedelta(hours=DUBAI_UTC_OFFSET_HOURS)
        days.append(
            {
                "date": pd.Timestamp(date),
                "times": day_bars.index.to_numpy(),
                "prices": day_bars["price"].to_numpy(dtype=float),
                "dubai_times": dubai_index,
                "dubai_seconds": (
                    dubai_index.hour.to_numpy() * 3600
                    + dubai_index.minute.to_numpy() * 60
                    + dubai_index.second.to_numpy()
                ),
                "sigma_dollar_63": _day_scalar(day_bars, "sigma_dollar_63"),
                "sigma_percent_to_dollar_63": _day_scalar(day_bars, "sigma_percent_to_dollar_63"),
                "trend_5d_pct_63": _day_scalar(day_bars, "trend_5d_pct_63"),
                "vol_rank_63": _day_scalar(day_bars, "vol_rank_63"),
            }
        )
    return days


def _day_scalar(day_bars: pd.DataFrame, col: str) -> float:
    if col not in day_bars.columns:
        return np.nan
    values = day_bars[col].dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def add_rare_move_thresholds(days: list[dict]) -> list[dict]:
    enriched = []
    history: dict[int, list[np.ndarray]] = {window: [] for window in RARE_MOVE_WINDOWS}
    start_seconds = SESSION_START_HOUR_DUBAI * 3600
    for day in days:
        day = dict(day)
        prices = day["prices"]
        dubai_seconds = day["dubai_seconds"]
        n = len(prices)

        for window in RARE_MOVE_WINDOWS:
            prior = history[window][-RARE_LOOKBACK_DAYS:]
            sample = np.concatenate(prior) if prior else np.asarray([], dtype=float)
            for percentile in RARE_PERCENTILES:
                key = f"rare_threshold_{window}_{percentile:g}"
                day[key] = float(np.percentile(sample, percentile)) if sample.size >= RARE_MIN_OBSERVATIONS else np.nan

            if n > window + 2:
                idx = np.arange(window, n)
                keep = (dubai_seconds[idx] >= start_seconds) & (dubai_seconds[idx] < 24 * 3600)
                if REQUIRE_FULL_SIGNAL_WINDOW_IN_SESSION:
                    keep = keep & (dubai_seconds[idx - window] >= start_seconds)
                if np.any(keep):
                    deltas = np.abs(prices[idx[keep]] - prices[idx[keep] - window])
                    history[window].append(deltas.astype(float))
                else:
                    history[window].append(np.asarray([], dtype=float))
            else:
                history[window].append(np.asarray([], dtype=float))

        enriched.append(day)
    return enriched


def backtest_days(days: list[dict], params: VolParams, collect_trades: bool = False) -> tuple[pd.DataFrame, pd.Series, dict]:
    trades: list[dict] = []
    pnls: list[float] = []
    sides: list[int] = []
    daily_values = []
    daily_index = []

    sigma_col = f"sigma_{params.vol_method}_{params.vol_window_days}"
    for day in days:
        day_trades, day_pnls, day_sides = scan_day(day, params, sigma_col, collect_trades)
        if collect_trades and day_trades:
            trades.extend(day_trades)
        pnls.extend(day_pnls)
        sides.extend(day_sides)
        daily_index.append(day["date"])
        daily_values.append(float(np.sum(day_pnls)) if day_pnls else 0.0)

    daily = pd.Series(daily_values, index=pd.Index(daily_index, name="date"), name="daily_pnl_cents")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, daily, _trade_summary(pnls, sides)


def scan_day(day: dict, params: VolParams, sigma_col: str, collect_trades: bool) -> tuple[list[dict], list[float], list[int]]:
    sigma = day.get(sigma_col, np.nan)
    if not np.isfinite(sigma) or sigma <= 0:
        return [], [], []

    prices = day["prices"]
    times = day["times"]
    dubai_times = day["dubai_times"]
    dubai_seconds = day["dubai_seconds"]
    n = len(prices)
    min_required = params.move_window_s + params.delay_s + 2
    if n <= min_required:
        return [], [], []

    threshold_cents = sigma * 100.0 * params.sigma_multiple
    delta = prices[params.move_window_s :] - prices[: -params.move_window_s]
    raw_idx = np.flatnonzero(np.abs(delta) >= threshold_cents / 100.0) + params.move_window_s
    if raw_idx.size == 0:
        return [], [], []

    start_seconds = SESSION_START_HOUR_DUBAI * 3600
    keep = (dubai_seconds[raw_idx] >= start_seconds) & (dubai_seconds[raw_idx] < 24 * 3600)
    if REQUIRE_FULL_SIGNAL_WINDOW_IN_SESSION:
        lookback_start_idx = raw_idx - params.move_window_s
        keep = keep & (dubai_seconds[lookback_start_idx] >= start_seconds)
    raw_idx = raw_idx[keep]
    if raw_idx.size == 0:
        return [], [], []

    trades = []
    pnls = []
    sides = []
    last_exit = -1
    i = 0
    while i < raw_idx.size:
        if len(pnls) >= MAX_TRADES_PER_DAY:
            break
        event_idx = int(raw_idx[i])
        entry_idx = event_idx + params.delay_s
        exit_idx = entry_idx + params.hold_s
        if entry_idx >= n:
            break
        if entry_idx <= last_exit:
            i = int(np.searchsorted(raw_idx, last_exit - params.delay_s + 1, side="left"))
            continue

        if exit_idx >= n:
            break
        adjusted_exit_idx = apply_bad_hour_rule(times, dubai_times, entry_idx, exit_idx, params.bad_hour_rule)
        if adjusted_exit_idx is None:
            i += 1
            continue
        exit_idx = adjusted_exit_idx
        if exit_idx <= entry_idx:
            i += 1
            continue

        move = prices[event_idx] - prices[event_idx - params.move_window_s]
        side = 1 if move > 0 else -1
        if params.signal_mode == "reversal":
            side *= -1

        entry_price = float(prices[entry_idx])
        exit_price = float(prices[exit_idx])
        gross_cents = side * (exit_price - entry_price) * 100.0
        cost_cents = abs(entry_price) * DYNAMIC_COST_CENTS_PER_PRICE_UNIT
        net_cents = gross_cents - cost_cents
        pnls.append(float(net_cents))
        sides.append(side)
        if collect_trades:
            trades.append(
                {
                    "signal_time": pd.Timestamp(times[event_idx]),
                    "signal_time_dubai": pd.Timestamp(dubai_times[event_idx]),
                    "entry_time": pd.Timestamp(times[entry_idx]),
                    "entry_time_dubai": pd.Timestamp(dubai_times[entry_idx]),
                    "exit_time": pd.Timestamp(times[exit_idx]),
                    "exit_time_dubai": pd.Timestamp(dubai_times[exit_idx]),
                    "side": side,
                    "move_cents": move * 100.0,
                    "threshold_cents": threshold_cents,
                    "daily_sigma_dollars": sigma,
                    "sigma_multiple": params.sigma_multiple,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl_cents": gross_cents,
                    "cost_cents": cost_cents,
                    "net_pnl_cents": net_cents,
                    "bad_hour_rule": params.bad_hour_rule,
                    "signal_mode": params.signal_mode,
                    "params": params.label,
                }
            )
        last_exit = exit_idx
        i = int(np.searchsorted(raw_idx, last_exit - params.delay_s + 1, side="left"))

    return trades, pnls, sides


def backtest_rare_days(days: list[dict], params: RareMoveParams, collect_trades: bool = False) -> tuple[pd.DataFrame, pd.Series, dict]:
    trades: list[dict] = []
    pnls: list[float] = []
    sides: list[int] = []
    daily_values = []
    daily_index = []

    for day in days:
        day_trades, day_pnls, day_sides = scan_rare_day(day, params, collect_trades)
        if collect_trades and day_trades:
            trades.extend(day_trades)
        pnls.extend(day_pnls)
        sides.extend(day_sides)
        daily_index.append(day["date"])
        daily_values.append(float(np.sum(day_pnls)) if day_pnls else 0.0)

    daily = pd.Series(daily_values, index=pd.Index(daily_index, name="date"), name="daily_pnl_cents")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, daily, _trade_summary(pnls, sides)


def scan_rare_day(day: dict, params: RareMoveParams, collect_trades: bool) -> tuple[list[dict], list[float], list[int]]:
    threshold_dollars = day.get(f"rare_threshold_{params.move_window_s}_{params.percentile:g}", np.nan)
    if not np.isfinite(threshold_dollars) or threshold_dollars <= 0:
        return [], [], []

    prices = day["prices"]
    times = day["times"]
    dubai_times = day["dubai_times"]
    dubai_seconds = day["dubai_seconds"]
    n = len(prices)
    min_required = params.move_window_s + params.delay_s + 2
    if n <= min_required:
        return [], [], []

    delta = prices[params.move_window_s :] - prices[: -params.move_window_s]
    raw_idx = np.flatnonzero(np.abs(delta) >= threshold_dollars) + params.move_window_s
    if raw_idx.size == 0:
        return [], [], []

    start_seconds = SESSION_START_HOUR_DUBAI * 3600
    keep = (dubai_seconds[raw_idx] >= start_seconds) & (dubai_seconds[raw_idx] < 24 * 3600)
    if REQUIRE_FULL_SIGNAL_WINDOW_IN_SESSION:
        lookback_start_idx = raw_idx - params.move_window_s
        keep = keep & (dubai_seconds[lookback_start_idx] >= start_seconds)
    keep = keep & regime_keep_mask(day, params, raw_idx)
    raw_idx = raw_idx[keep]
    if raw_idx.size == 0:
        return [], [], []

    trades = []
    pnls = []
    sides = []
    last_exit = -1
    i = 0
    threshold_cents = threshold_dollars * 100.0
    while i < raw_idx.size:
        if len(pnls) >= MAX_TRADES_PER_DAY:
            break
        event_idx = int(raw_idx[i])
        entry_idx = event_idx + params.delay_s
        exit_idx = entry_idx + params.hold_s
        if entry_idx >= n:
            break
        if entry_idx <= last_exit:
            i = int(np.searchsorted(raw_idx, last_exit - params.delay_s + 1, side="left"))
            continue

        if exit_idx >= n:
            break
        adjusted_exit_idx = apply_bad_hour_rule(times, dubai_times, entry_idx, exit_idx, params.bad_hour_rule)
        if adjusted_exit_idx is None:
            i += 1
            continue
        exit_idx = adjusted_exit_idx
        if exit_idx <= entry_idx:
            i += 1
            continue

        move = prices[event_idx] - prices[event_idx - params.move_window_s]
        side = 1 if move > 0 else -1
        if params.signal_mode == "reversal":
            side *= -1

        entry_price = float(prices[entry_idx])
        exit_price = float(prices[exit_idx])
        gross_cents = side * (exit_price - entry_price) * 100.0
        cost_cents = abs(entry_price) * DYNAMIC_COST_CENTS_PER_PRICE_UNIT
        net_cents = gross_cents - cost_cents
        pnls.append(float(net_cents))
        sides.append(side)
        if collect_trades:
            trades.append(
                {
                    "signal_time": pd.Timestamp(times[event_idx]),
                    "signal_time_dubai": pd.Timestamp(dubai_times[event_idx]),
                    "entry_time": pd.Timestamp(times[entry_idx]),
                    "entry_time_dubai": pd.Timestamp(dubai_times[entry_idx]),
                    "exit_time": pd.Timestamp(times[exit_idx]),
                    "exit_time_dubai": pd.Timestamp(dubai_times[exit_idx]),
                    "side": side,
                    "move_cents": move * 100.0,
                    "threshold_cents": threshold_cents,
                    "rarity_percentile": params.percentile,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl_cents": gross_cents,
                    "cost_cents": cost_cents,
                    "net_pnl_cents": net_cents,
                    "bad_hour_rule": params.bad_hour_rule,
                    "signal_mode": params.signal_mode,
                    "trend_filter": getattr(params, "trend_filter", "any"),
                    "vol_filter": getattr(params, "vol_filter", "any"),
                    "time_bucket": getattr(params, "time_bucket", "any"),
                    "trend_5d_pct": day.get("trend_5d_pct_63", np.nan),
                    "vol_rank": day.get("vol_rank_63", np.nan),
                    "params": params.label,
                }
            )
        last_exit = exit_idx
        i = int(np.searchsorted(raw_idx, last_exit - params.delay_s + 1, side="left"))

    return trades, pnls, sides


def regime_keep_mask(day: dict, params, raw_idx: np.ndarray) -> np.ndarray:
    keep = np.ones(raw_idx.size, dtype=bool)

    trend_filter = getattr(params, "trend_filter", "any")
    trend = day.get("trend_5d_pct_63", np.nan)
    if trend_filter == "up":
        keep &= np.isfinite(trend) and trend > 0
    elif trend_filter == "down":
        keep &= np.isfinite(trend) and trend < 0

    vol_filter = getattr(params, "vol_filter", "any")
    vol_rank = day.get("vol_rank_63", np.nan)
    if vol_filter == "high":
        keep &= np.isfinite(vol_rank) and vol_rank >= 0.60
    elif vol_filter == "low":
        keep &= np.isfinite(vol_rank) and vol_rank <= 0.40

    time_bucket = getattr(params, "time_bucket", "any")
    if time_bucket != "any":
        seconds = day["dubai_seconds"][raw_idx]
        if time_bucket == "early":
            keep &= (seconds >= 11 * 3600) & (seconds < 14 * 3600)
        elif time_bucket == "mid":
            keep &= (seconds >= 14 * 3600) & (seconds < 18 * 3600)
        elif time_bucket == "late":
            keep &= (seconds >= 18 * 3600) & (seconds < 24 * 3600)
    return keep


def apply_bad_hour_rule(times: np.ndarray, dubai_times: pd.DatetimeIndex, entry_idx: int, exit_idx: int, rule: str) -> int | None:
    if rule == "allow_hold":
        return exit_idx

    entry_dubai = pd.Timestamp(dubai_times[entry_idx])
    exit_dubai = pd.Timestamp(dubai_times[exit_idx])

    if rule == "exit_by_midnight":
        midnight_dubai = pd.Timestamp(entry_dubai.date()) + pd.Timedelta(days=1)
        midnight_utc = midnight_dubai - pd.Timedelta(hours=DUBAI_UTC_OFFSET_HOURS)
        forced = int(np.searchsorted(times, np.datetime64(midnight_utc.to_datetime64()), side="left") - 1)
        return forced if forced > entry_idx else None

    if rule == "avoid_exit_1_3":
        return None if 1 <= exit_dubai.hour < 3 else exit_idx

    if rule == "avoid_hold_1_3":
        return None if overlaps_bad_window(entry_dubai, exit_dubai) else exit_idx

    raise ValueError(f"Unknown bad-hour rule: {rule}")


def overlaps_bad_window(entry_dubai: pd.Timestamp, exit_dubai: pd.Timestamp) -> bool:
    start_day = entry_dubai.normalize()
    end_day = exit_dubai.normalize()
    day = start_day
    while day <= end_day:
        bad_start = day + pd.Timedelta(hours=1)
        bad_end = day + pd.Timedelta(hours=3)
        if entry_dubai < bad_end and exit_dubai > bad_start:
            return True
        day += pd.Timedelta(days=1)
    return False


def _trade_summary(pnls: list[float], sides: list[int]) -> dict:
    if not pnls:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_cents": 0.0,
            "profit_factor": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }
    pnl = np.asarray(pnls, dtype=float)
    sides_arr = np.asarray(sides, dtype=int)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return {
        "trades": int(pnl.size),
        "win_rate": float((pnl > 0).mean()),
        "avg_trade_cents": float(pnl.mean()),
        "profit_factor": gross_profit / (gross_loss + 1e-12) if gross_loss > 0 else np.inf,
        "long_trades": int((sides_arr > 0).sum()),
        "short_trades": int((sides_arr < 0).sum()),
    }


def period_schedule(bars: pd.DataFrame, rebalance: str) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    end = bars.index.max().normalize()
    if rebalance != "monthly":
        raise ValueError(rebalance)
    starts = pd.date_range(WALKFORWARD_START, end, freq="MS")
    test_ends = [min(s + pd.DateOffset(months=1) - pd.Timedelta(days=1), end) for s in starts]

    schedule = []
    for test_start, test_end in zip(starts, test_ends):
        if test_start == pd.Timestamp("2025-11-01"):
            train_start = pd.Timestamp("2025-10-01")
        elif test_start == pd.Timestamp("2025-12-01"):
            train_start = pd.Timestamp("2025-10-01")
        else:
            train_start = test_start - pd.DateOffset(months=TRAIN_MONTHS)
        train_end = test_start - pd.Timedelta(seconds=1)
        if test_start <= end:
            schedule.append((train_start, train_end, test_start, test_end))
    return schedule


def slice_days(days: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    start_date = start.normalize()
    end_date = end.normalize()
    return [day for day in days if start_date <= day["date"] <= end_date]


def evaluate_grid(days: list[dict], params: list[VolParams], objective: str) -> pd.DataFrame:
    rows = []
    for p in params:
        _, daily, summary = backtest_days(days, p, collect_trades=False)
        metrics = compute_metrics(daily, trade_summary=summary)
        score = -np.inf if metrics["trades"] < MIN_TRAIN_TRADES else float(metrics[objective])
        rows.append(
            {
                "params_obj": p,
                "params": p.label,
                "delay_s": p.delay_s,
                "sigma_multiple": p.sigma_multiple,
                "move_window_s": p.move_window_s,
                "hold_s": p.hold_s,
                "vol_window_days": p.vol_window_days,
                "vol_method": p.vol_method,
                "signal_mode": p.signal_mode,
                "bad_hour_rule": p.bad_hour_rule,
                "score": score,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "total_pnl_cents"], ascending=False).reset_index(drop=True)


def precompute_results(days: list[dict], params: list[VolParams]) -> dict[VolParams, tuple[pd.Series, pd.DataFrame]]:
    out = {}
    for idx, p in enumerate(params, 1):
        if idx % 25 == 0:
            print("precomputed", idx, "of", len(params), flush=True)
        trades, daily, _ = backtest_days(days, p, collect_trades=True)
        out[p] = (daily, trades)
    return out


def precompute_rare_results(days: list[dict], params: list[RareMoveParams]) -> dict[RareMoveParams, tuple[pd.Series, pd.DataFrame]]:
    out = {}
    for idx, p in enumerate(params, 1):
        if idx % 25 == 0:
            print("precomputed rare", idx, "of", len(params), flush=True)
        trades, daily, _ = backtest_rare_days(days, p, collect_trades=True)
        out[p] = (daily, trades)
    return out


def slice_daily(daily: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    start_date = start.normalize()
    end_date = end.normalize()
    return daily[(daily.index >= start_date) & (daily.index <= end_date)]


def slice_trades(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return trades
    signal_date = pd.to_datetime(trades["signal_time"]).dt.normalize()
    return trades[(signal_date >= start.normalize()) & (signal_date <= end.normalize())].copy()


def evaluate_grid_cached(
    cache: dict[VolParams, tuple[pd.Series, pd.DataFrame]],
    params: list[VolParams],
    start: pd.Timestamp,
    end: pd.Timestamp,
    objective: str,
) -> pd.DataFrame:
    rows = []
    for p in params:
        daily, trades = cache[p]
        period_daily = slice_daily(daily, start, end)
        period_trades = slice_trades(trades, start, end)
        metrics = compute_metrics(period_daily, period_trades)
        score = -np.inf if metrics["trades"] < MIN_TRAIN_TRADES else float(metrics[objective])
        rows.append(
            {
                "params_obj": p,
                "params": p.label,
                "delay_s": p.delay_s,
                "sigma_multiple": p.sigma_multiple,
                "move_window_s": p.move_window_s,
                "hold_s": p.hold_s,
                "vol_window_days": p.vol_window_days,
                "vol_method": p.vol_method,
                "signal_mode": p.signal_mode,
                "bad_hour_rule": p.bad_hour_rule,
                "score": score,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "total_pnl_cents"], ascending=False).reset_index(drop=True)


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
    return eligible.sort_values(["total_pnl_cents", "daily_sharpe"], ascending=False).iloc[0]


def evaluate_multiple_backtests_cached(
    cache: dict[VolParams, tuple[pd.Series, pd.DataFrame]],
    params: list[VolParams],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for p in params:
        daily_full, trades_full = cache[p]
        period_daily = slice_daily(daily_full, start, end)
        period_trades = slice_trades(trades_full, start, end)
        metrics = compute_metrics(period_daily, period_trades)
        rows.append(
            {
                "params": p.label,
                "delay_s": p.delay_s,
                "sigma_multiple": p.sigma_multiple,
                "move_window_s": p.move_window_s,
                "hold_s": p.hold_s,
                "vol_window_days": p.vol_window_days,
                "vol_method": p.vol_method,
                "signal_mode": p.signal_mode,
                "bad_hour_rule": p.bad_hour_rule,
                "backtest_start": start.date().isoformat(),
                "backtest_end": end.date().isoformat(),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).reset_index(drop=True)


def concentration_metrics(daily: pd.Series, trades: pd.DataFrame) -> dict:
    total_pnl = float(daily.sum()) if not daily.empty else 0.0
    monthly = daily.groupby(pd.Grouper(freq="MS")).sum() if not daily.empty else pd.Series(dtype=float)
    positive_months = monthly[monthly > 0]
    best_month_pnl = float(positive_months.max()) if not positive_months.empty else 0.0
    profitable_months = int((monthly > 0).sum()) if not monthly.empty else 0
    month_count = int(monthly.shape[0])

    top_trade_pnl = 0.0
    top_trade_removed_pnl = total_pnl
    if trades is not None and not trades.empty and "net_pnl_cents" in trades.columns:
        positive_trades = trades[trades["net_pnl_cents"] > 0]["net_pnl_cents"]
        if not positive_trades.empty:
            top_trade_pnl = float(positive_trades.max())
            top_trade_removed_pnl = total_pnl - top_trade_pnl

    denominator = abs(total_pnl) if abs(total_pnl) > 1e-12 else 1.0
    return {
        "profitable_months": profitable_months,
        "month_count": month_count,
        "profitable_month_rate": float(profitable_months / month_count) if month_count else 0.0,
        "best_month_pnl_cents": best_month_pnl,
        "best_month_share": float(best_month_pnl / denominator),
        "top_trade_pnl_cents": top_trade_pnl,
        "top_trade_share": float(top_trade_pnl / denominator),
        "top_trade_removed_pnl_cents": top_trade_removed_pnl,
    }


def robust_score(metrics: dict, concentration: dict) -> float:
    if (
        metrics["trades"] < MIN_TRAIN_TRADES
        or metrics["total_pnl_cents"] <= 0
        or metrics["daily_sharpe"] <= 0
        or not np.isfinite(metrics["profit_factor"])
    ):
        return -np.inf

    return float(
        metrics["daily_sharpe"]
        + 0.002 * concentration["top_trade_removed_pnl_cents"]
        + 0.75 * concentration["profitable_month_rate"]
        - 0.50 * max(concentration["top_trade_share"] - 0.35, 0.0)
        - 0.50 * max(concentration["best_month_share"] - 0.60, 0.0)
        + 0.001 * metrics["max_drawdown_cents"]
    )


def evaluate_robust_grid_cached(
    cache: dict[VolParams, tuple[pd.Series, pd.DataFrame]],
    params: list[VolParams],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for p in params:
        daily, trades = cache[p]
        period_daily = slice_daily(daily, start, end)
        period_trades = slice_trades(trades, start, end)
        metrics = compute_metrics(period_daily, period_trades)
        concentration = concentration_metrics(period_daily, period_trades)
        rows.append(
            {
                "params_obj": p,
                "params": p.label,
                "delay_s": p.delay_s,
                "sigma_multiple": p.sigma_multiple,
                "move_window_s": p.move_window_s,
                "hold_s": p.hold_s,
                "vol_window_days": p.vol_window_days,
                "vol_method": p.vol_method,
                "signal_mode": p.signal_mode,
                "bad_hour_rule": p.bad_hour_rule,
                "score": robust_score(metrics, concentration),
                **metrics,
                **concentration,
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "daily_sharpe", "total_pnl_cents"], ascending=False).reset_index(drop=True)


def _index_distance(value, reference: list) -> int:
    if value not in reference:
        return 99
    return int(reference.index(value))


def add_stable_cluster_scores(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return grid
    out = grid.copy()
    positive = (
        (out["total_pnl_cents"] > 0)
        & (out["daily_sharpe"] > 0)
        & (out["top_trade_removed_pnl_cents"] > 0)
    )
    delay_idx = out["delay_s"].map(lambda x: _index_distance(int(x), STABLE_DELAYS))
    sigma_idx = out["sigma_multiple"].map(lambda x: _index_distance(float(x), STABLE_SIGMA_MULTIPLES))
    lookback_idx = out["move_window_s"].map(lambda x: _index_distance(int(x), STABLE_MOVE_WINDOWS))
    hold_idx = out["hold_s"].map(lambda x: _index_distance(int(x), STABLE_HOLDS))

    cluster_counts = []
    cluster_positive_counts = []
    for idx, row in out.iterrows():
        same_family = out["signal_mode"].eq(row["signal_mode"])
        nearby = (
            same_family
            & ((delay_idx - delay_idx.loc[idx]).abs() <= 1)
            & ((sigma_idx - sigma_idx.loc[idx]).abs() <= 1)
            & ((lookback_idx - lookback_idx.loc[idx]).abs() <= 1)
            & ((hold_idx - hold_idx.loc[idx]).abs() <= 1)
        )
        nearby.loc[idx] = False
        cluster_count = int(nearby.sum())
        cluster_positive_count = int((nearby & positive).sum())
        cluster_counts.append(cluster_count)
        cluster_positive_counts.append(cluster_positive_count)

    out["cluster_neighbors"] = cluster_counts
    out["cluster_positive_neighbors"] = cluster_positive_counts
    out["cluster_positive_rate"] = np.where(
        out["cluster_neighbors"] > 0,
        out["cluster_positive_neighbors"] / out["cluster_neighbors"],
        0.0,
    )
    out["stable_score"] = out["score"] + 0.40 * out["cluster_positive_neighbors"] + 1.50 * out["cluster_positive_rate"]
    return out.sort_values(["stable_score", "cluster_positive_neighbors", "daily_sharpe"], ascending=False).reset_index(drop=True)


def evaluate_stable_grid_cached(
    cache: dict[VolParams, tuple[pd.Series, pd.DataFrame]],
    params: list[VolParams],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return add_stable_cluster_scores(evaluate_robust_grid_cached(cache, params, start, end))


def evaluate_rare_grid_cached(
    cache: dict[RareMoveParams, tuple[pd.Series, pd.DataFrame]],
    params: list[RareMoveParams],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for p in params:
        daily, trades = cache[p]
        period_daily = slice_daily(daily, start, end)
        period_trades = slice_trades(trades, start, end)
        metrics = compute_metrics(period_daily, period_trades)
        concentration = concentration_metrics(period_daily, period_trades)
        rows.append(
            {
                "params_obj": p,
                "params": p.label,
                "delay_s": p.delay_s,
                "rarity_percentile": p.percentile,
                "move_window_s": p.move_window_s,
                "hold_s": p.hold_s,
                "signal_mode": p.signal_mode,
                "bad_hour_rule": p.bad_hour_rule,
                "trend_filter": getattr(p, "trend_filter", "any"),
                "vol_filter": getattr(p, "vol_filter", "any"),
                "time_bucket": getattr(p, "time_bucket", "any"),
                "score": robust_score(metrics, concentration),
                **metrics,
                **concentration,
            }
        )
    return add_rare_cluster_scores(pd.DataFrame(rows))


def add_rare_cluster_scores(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return grid
    out = grid.copy()
    positive = (
        (out["total_pnl_cents"] > 0)
        & (out["daily_sharpe"] > 0)
        & (out["top_trade_removed_pnl_cents"] > 0)
    )
    delay_idx = out["delay_s"].map(lambda x: _index_distance(int(x), RARE_DELAYS))
    percentile_idx = out["rarity_percentile"].map(lambda x: _index_distance(float(x), RARE_PERCENTILES))
    lookback_idx = out["move_window_s"].map(lambda x: _index_distance(int(x), RARE_MOVE_WINDOWS))
    hold_idx = out["hold_s"].map(lambda x: _index_distance(int(x), RARE_HOLDS))

    cluster_counts = []
    cluster_positive_counts = []
    for idx, row in out.iterrows():
        same_family = (
            out["signal_mode"].eq(row["signal_mode"])
            & out["trend_filter"].eq(row["trend_filter"])
            & out["vol_filter"].eq(row["vol_filter"])
            & out["time_bucket"].eq(row["time_bucket"])
        )
        nearby = (
            same_family
            & ((delay_idx - delay_idx.loc[idx]).abs() <= 1)
            & ((percentile_idx - percentile_idx.loc[idx]).abs() <= 1)
            & ((lookback_idx - lookback_idx.loc[idx]).abs() <= 1)
            & ((hold_idx - hold_idx.loc[idx]).abs() <= 1)
        )
        nearby.loc[idx] = False
        cluster_count = int(nearby.sum())
        cluster_positive_count = int((nearby & positive).sum())
        cluster_counts.append(cluster_count)
        cluster_positive_counts.append(cluster_positive_count)

    out["cluster_neighbors"] = cluster_counts
    out["cluster_positive_neighbors"] = cluster_positive_counts
    out["cluster_positive_rate"] = np.where(
        out["cluster_neighbors"] > 0,
        out["cluster_positive_neighbors"] / out["cluster_neighbors"],
        0.0,
    )
    out["rare_stability_score"] = out["score"] + 0.40 * out["cluster_positive_neighbors"] + 1.50 * out["cluster_positive_rate"]
    return out.sort_values(["rare_stability_score", "cluster_positive_neighbors", "daily_sharpe"], ascending=False).reset_index(drop=True)


def select_robust_candidate(grid: pd.DataFrame, strict: bool = False) -> pd.Series | None:
    eligible = grid[np.isfinite(grid["score"])].copy()
    if strict and not eligible.empty:
        enough_history = eligible["month_count"] >= 2
        eligible = eligible[
            (~enough_history)
            | (
                (eligible["profitable_month_rate"] >= 0.67)
                & (eligible["top_trade_removed_pnl_cents"] > 0)
                & (eligible["top_trade_share"] <= 0.50)
                & (eligible["best_month_share"] <= 0.70)
            )
        ].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(["score", "daily_sharpe", "top_trade_removed_pnl_cents"], ascending=False).iloc[0]


def select_stable_candidate(grid: pd.DataFrame) -> pd.Series | None:
    if grid.empty:
        return None
    eligible = grid[
        np.isfinite(grid["stable_score"])
        & (grid["month_count"] >= 2)
        & (grid["trades"] >= MIN_TRAIN_TRADES)
        & (grid["total_pnl_cents"] > 0)
        & (grid["daily_sharpe"] > 0)
        & (grid["top_trade_removed_pnl_cents"] > 0)
        & (grid["cluster_positive_neighbors"] >= 2)
    ].copy()
    if eligible.empty:
        return None

    eligible = eligible[
        (eligible["profitable_month_rate"] >= 0.50)
        & (eligible["top_trade_share"] <= 0.45)
        & (eligible["best_month_share"] <= 0.50)
    ].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["stable_score", "cluster_positive_neighbors", "top_trade_removed_pnl_cents", "daily_sharpe"],
        ascending=False,
    ).iloc[0]


def select_rare_candidate(grid: pd.DataFrame) -> pd.Series | None:
    if grid.empty:
        return None
    eligible = grid[
        np.isfinite(grid["rare_stability_score"])
        & (grid["month_count"] >= 2)
        & (grid["trades"] >= MIN_TRAIN_TRADES)
        & (grid["total_pnl_cents"] > 0)
        & (grid["daily_sharpe"] > 0)
        & (grid["top_trade_removed_pnl_cents"] > 0)
        & (grid["cluster_positive_neighbors"] >= 2)
        & (grid["profitable_month_rate"] >= 0.50)
        & (grid["top_trade_share"] <= 0.45)
        & (grid["best_month_share"] <= 0.55)
    ].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["rare_stability_score", "cluster_positive_neighbors", "top_trade_removed_pnl_cents", "daily_sharpe"],
        ascending=False,
    ).iloc[0]


def select_regime_rare_candidate(grid: pd.DataFrame) -> pd.Series | None:
    if grid.empty:
        return None
    active_regime = (
        grid["trend_filter"].ne("any")
        | grid["vol_filter"].ne("any")
        | grid["time_bucket"].ne("any")
    )
    eligible = grid[
        active_regime
        & np.isfinite(grid["rare_stability_score"])
        & (grid["month_count"] >= 2)
        & (grid["trades"] >= 10)
        & (grid["total_pnl_cents"] > 0)
        & (grid["daily_sharpe"] > 0)
        & (grid["top_trade_removed_pnl_cents"] > 0)
        & (grid["cluster_positive_neighbors"] >= 2)
        & (grid["profitable_month_rate"] >= 0.50)
        & (grid["top_trade_share"] <= 0.35)
        & (grid["best_month_share"] <= 0.50)
    ].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["rare_stability_score", "cluster_positive_neighbors", "top_trade_removed_pnl_cents", "daily_sharpe"],
        ascending=False,
    ).iloc[0]


def export_winning_percent_vol_backtest(
    cache: dict[VolParams, tuple[pd.Series, pd.DataFrame]],
    multiple_backtests: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    if multiple_backtests.empty:
        return
    winner = multiple_backtests.sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).iloc[0]
    winner_label = str(winner["params"])
    match = next((p for p in cache if p.label == winner_label), None)
    if match is None:
        return

    daily_full, trades_full = cache[match]
    daily = slice_daily(daily_full, start, end)
    trades = slice_trades(trades_full, start, end)
    metrics = compute_metrics(daily, trades)

    metrics_row = {
        "selected_params": match.label,
        "backtest_start": start.date().isoformat(),
        "backtest_end": end.date().isoformat(),
        "selection_rule": "Best fixed percent-to-dollar candidate by Sharpe, then PnL",
        "delay_s": match.delay_s,
        "sigma_multiple": match.sigma_multiple,
        "move_window_s": match.move_window_s,
        "hold_s": match.hold_s,
        "vol_window_days": match.vol_window_days,
        "vol_method": match.vol_method,
        "signal_mode": match.signal_mode,
        "bad_hour_rule": match.bad_hour_rule,
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(OUT_DIR / "winning_percent_vol_backtest_metrics.csv", index=False)
    daily.rename("daily_pnl_cents").to_csv(OUT_DIR / "winning_percent_vol_backtest_daily.csv")
    if not trades.empty:
        trades = trades.copy()
        trades["selected_params"] = match.label
        trades["backtest_start"] = start.date().isoformat()
        trades["backtest_end"] = end.date().isoformat()
    trades.to_csv(OUT_DIR / "winning_percent_vol_backtest_trades.csv", index=False)


def run_walkforward(days: list[dict], bars: pd.DataFrame, params: list[VolParams], rebalance: str, objective: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    decisions = []
    trades_list = []
    daily_list = []
    rankings = []

    for train_start, train_end, test_start, test_end in period_schedule(bars, rebalance):
        train_days = slice_days(days, train_start, train_end)
        test_days = slice_days(days, test_start, test_end)
        if not train_days or not test_days:
            continue

        grid = evaluate_grid(train_days, params, objective)
        selected = select_candidate(grid, objective)
        top = grid.drop(columns=["params_obj"]).head(25).copy()
        top["rebalance"] = rebalance
        top["objective"] = objective
        top["train_start"] = train_start.date().isoformat()
        top["train_end"] = train_end.date().isoformat()
        top["test_start"] = test_start.date().isoformat()
        top["test_end"] = test_end.date().isoformat()
        rankings.append(top)

        if selected is None:
            decisions.append(
                {
                    "rebalance": rebalance,
                    "objective": objective,
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "selected_params": "NO_ELIGIBLE_CANDIDATE",
                    "test_total_pnl_cents": 0.0,
                    "test_sharpe": 0.0,
                    "test_trades": 0,
                }
            )
            continue

        p: VolParams = selected["params_obj"]
        trades, daily, _ = backtest_days(test_days, p, collect_trades=True)
        metrics = compute_metrics(daily, trades)
        daily_list.append(daily)
        if not trades.empty:
            out_trades = trades.copy()
            out_trades["rebalance"] = rebalance
            out_trades["objective"] = objective
            out_trades["test_start"] = test_start.date().isoformat()
            out_trades["test_end"] = test_end.date().isoformat()
            out_trades["selected_params"] = p.label
            trades_list.append(out_trades)

        decisions.append(
            {
                "rebalance": rebalance,
                "objective": objective,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_params": p.label,
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

    daily_oos = pd.concat(daily_list).sort_index().groupby(level=0).sum() if daily_list else pd.Series(dtype=float, name="daily_pnl_cents")
    trades_df = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    decisions_df = pd.DataFrame(decisions)
    rankings_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    return decisions_df, trades_df, daily_oos, rankings_df


def run_walkforward_cached(
    cache: dict[VolParams, tuple[pd.Series, pd.DataFrame]],
    bars: pd.DataFrame,
    params: list[VolParams],
    rebalance: str,
    objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    decisions = []
    trades_list = []
    daily_list = []
    rankings = []

    for train_start, train_end, test_start, test_end in period_schedule(bars, rebalance):
        grid = evaluate_grid_cached(cache, params, train_start, train_end, objective)
        selected = select_candidate(grid, objective)
        top = grid.drop(columns=["params_obj"]).head(25).copy()
        top["rebalance"] = rebalance
        top["objective"] = objective
        top["train_start"] = train_start.date().isoformat()
        top["train_end"] = train_end.date().isoformat()
        top["test_start"] = test_start.date().isoformat()
        top["test_end"] = test_end.date().isoformat()
        rankings.append(top)

        if selected is None:
            decisions.append(
                {
                    "rebalance": rebalance,
                    "objective": objective,
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "selected_params": "NO_ELIGIBLE_CANDIDATE",
                    "test_total_pnl_cents": 0.0,
                    "test_sharpe": 0.0,
                    "test_trades": 0,
                }
            )
            continue

        p: VolParams = selected["params_obj"]
        daily_full, trades_full = cache[p]
        daily = slice_daily(daily_full, test_start, test_end)
        trades = slice_trades(trades_full, test_start, test_end)
        metrics = compute_metrics(daily, trades)
        daily_list.append(daily)
        if not trades.empty:
            out_trades = trades.copy()
            out_trades["rebalance"] = rebalance
            out_trades["objective"] = objective
            out_trades["test_start"] = test_start.date().isoformat()
            out_trades["test_end"] = test_end.date().isoformat()
            out_trades["selected_params"] = p.label
            trades_list.append(out_trades)

        decisions.append(
            {
                "rebalance": rebalance,
                "objective": objective,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_params": p.label,
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

    daily_oos = pd.concat(daily_list).sort_index().groupby(level=0).sum() if daily_list else pd.Series(dtype=float, name="daily_pnl_cents")
    trades_df = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    decisions_df = pd.DataFrame(decisions)
    rankings_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    return decisions_df, trades_df, daily_oos, rankings_df


def run_robust_walkforward_cached(
    cache: dict[VolParams, tuple[pd.Series, pd.DataFrame]],
    bars: pd.DataFrame,
    params: list[VolParams],
    run_key: str,
    run_label: str,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    decisions = []
    trades_list = []
    daily_list = []
    rankings = []

    for train_start, train_end, test_start, test_end in period_schedule(bars, "monthly"):
        grid = evaluate_robust_grid_cached(cache, params, train_start, train_end)
        selected = select_robust_candidate(grid, strict=strict)
        top = grid.drop(columns=["params_obj"]).head(25).copy()
        top["run_key"] = run_key
        top["run_label"] = run_label
        top["objective"] = ROBUST_OBJECTIVE_NAME
        top["train_start"] = train_start.date().isoformat()
        top["train_end"] = train_end.date().isoformat()
        top["test_start"] = test_start.date().isoformat()
        top["test_end"] = test_end.date().isoformat()
        rankings.append(top)

        if selected is None:
            decisions.append(
                {
                    "run_key": run_key,
                    "run_label": run_label,
                    "objective": ROBUST_OBJECTIVE_NAME,
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "selected_params": "NO_ELIGIBLE_CANDIDATE",
                    "test_total_pnl_cents": 0.0,
                    "test_sharpe": 0.0,
                    "test_trades": 0,
                }
            )
            continue

        p: VolParams = selected["params_obj"]
        daily_full, trades_full = cache[p]
        daily = slice_daily(daily_full, test_start, test_end)
        trades = slice_trades(trades_full, test_start, test_end)
        metrics = compute_metrics(daily, trades)
        concentration = concentration_metrics(daily, trades)
        daily_list.append(daily)
        if not trades.empty:
            out_trades = trades.copy()
            out_trades["run_key"] = run_key
            out_trades["run_label"] = run_label
            out_trades["objective"] = ROBUST_OBJECTIVE_NAME
            out_trades["test_start"] = test_start.date().isoformat()
            out_trades["test_end"] = test_end.date().isoformat()
            out_trades["selected_params"] = p.label
            trades_list.append(out_trades)

        decisions.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "objective": ROBUST_OBJECTIVE_NAME,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_params": p.label,
                "train_score": float(selected["score"]),
                "train_total_pnl_cents": float(selected["total_pnl_cents"]),
                "train_sharpe": float(selected["daily_sharpe"]),
                "train_trades": int(selected["trades"]),
                "train_profitable_month_rate": float(selected["profitable_month_rate"]),
                "train_top_trade_share": float(selected["top_trade_share"]),
                "train_best_month_share": float(selected["best_month_share"]),
                "train_top_trade_removed_pnl_cents": float(selected["top_trade_removed_pnl_cents"]),
                "test_total_pnl_cents": metrics["total_pnl_cents"],
                "test_sharpe": metrics["daily_sharpe"],
                "test_trades": metrics["trades"],
                "test_win_rate": metrics["win_rate"],
                "test_max_drawdown_cents": metrics["max_drawdown_cents"],
                "test_profitable_month_rate": concentration["profitable_month_rate"],
                "test_top_trade_share": concentration["top_trade_share"],
                "test_best_month_share": concentration["best_month_share"],
                "test_top_trade_removed_pnl_cents": concentration["top_trade_removed_pnl_cents"],
            }
        )

    daily_oos = pd.concat(daily_list).sort_index().groupby(level=0).sum() if daily_list else pd.Series(dtype=float, name="daily_pnl_cents")
    trades_df = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    decisions_df = pd.DataFrame(decisions)
    rankings_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    return decisions_df, trades_df, daily_oos, rankings_df


def run_stable_walkforward_cached(
    cache: dict[VolParams, tuple[pd.Series, pd.DataFrame]],
    bars: pd.DataFrame,
    params: list[VolParams],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    run_key = "stable_4var_percent"
    run_label = "Stable 4-variable percent-vol walk-forward"
    decisions = []
    trades_list = []
    daily_list = []
    rankings = []

    for train_start, train_end, test_start, test_end in period_schedule(bars, "monthly"):
        grid = evaluate_stable_grid_cached(cache, params, train_start, train_end)
        selected = select_stable_candidate(grid)
        top = grid.drop(columns=["params_obj"]).head(50).copy()
        top["run_key"] = run_key
        top["run_label"] = run_label
        top["objective"] = "stable_4var_score"
        top["train_start"] = train_start.date().isoformat()
        top["train_end"] = train_end.date().isoformat()
        top["test_start"] = test_start.date().isoformat()
        top["test_end"] = test_end.date().isoformat()
        rankings.append(top)

        if selected is None:
            decisions.append(
                {
                    "run_key": run_key,
                    "run_label": run_label,
                    "objective": "stable_4var_score",
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "selected_params": "NO_STABLE_CANDIDATE",
                    "test_total_pnl_cents": 0.0,
                    "test_sharpe": 0.0,
                    "test_trades": 0,
                }
            )
            continue

        p: VolParams = selected["params_obj"]
        daily_full, trades_full = cache[p]
        daily = slice_daily(daily_full, test_start, test_end)
        trades = slice_trades(trades_full, test_start, test_end)
        metrics = compute_metrics(daily, trades)
        concentration = concentration_metrics(daily, trades)
        daily_list.append(daily)
        if not trades.empty:
            out_trades = trades.copy()
            out_trades["run_key"] = run_key
            out_trades["run_label"] = run_label
            out_trades["objective"] = "stable_4var_score"
            out_trades["test_start"] = test_start.date().isoformat()
            out_trades["test_end"] = test_end.date().isoformat()
            out_trades["selected_params"] = p.label
            trades_list.append(out_trades)

        decisions.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "objective": "stable_4var_score",
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_params": p.label,
                "train_stable_score": float(selected["stable_score"]),
                "train_total_pnl_cents": float(selected["total_pnl_cents"]),
                "train_sharpe": float(selected["daily_sharpe"]),
                "train_trades": int(selected["trades"]),
                "train_profitable_month_rate": float(selected["profitable_month_rate"]),
                "train_top_trade_share": float(selected["top_trade_share"]),
                "train_best_month_share": float(selected["best_month_share"]),
                "train_top_trade_removed_pnl_cents": float(selected["top_trade_removed_pnl_cents"]),
                "train_cluster_positive_neighbors": int(selected["cluster_positive_neighbors"]),
                "train_cluster_positive_rate": float(selected["cluster_positive_rate"]),
                "test_total_pnl_cents": metrics["total_pnl_cents"],
                "test_sharpe": metrics["daily_sharpe"],
                "test_trades": metrics["trades"],
                "test_win_rate": metrics["win_rate"],
                "test_max_drawdown_cents": metrics["max_drawdown_cents"],
                "test_profitable_month_rate": concentration["profitable_month_rate"],
                "test_top_trade_share": concentration["top_trade_share"],
                "test_best_month_share": concentration["best_month_share"],
                "test_top_trade_removed_pnl_cents": concentration["top_trade_removed_pnl_cents"],
            }
        )

    daily_oos = pd.concat(daily_list).sort_index().groupby(level=0).sum() if daily_list else pd.Series(dtype=float, name="daily_pnl_cents")
    trades_df = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    decisions_df = pd.DataFrame(decisions)
    rankings_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    return decisions_df, trades_df, daily_oos, rankings_df


def run_rare_walkforward_cached(
    cache: dict[RareMoveParams, tuple[pd.Series, pd.DataFrame]],
    bars: pd.DataFrame,
    params: list[RareMoveParams],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    run_key = "rare_move_percentile_wf"
    run_label = "Rare move percentile walk-forward"
    decisions = []
    trades_list = []
    daily_list = []
    rankings = []

    for train_start, train_end, test_start, test_end in period_schedule(bars, "monthly"):
        grid = evaluate_rare_grid_cached(cache, params, train_start, train_end)
        selected = select_rare_candidate(grid)
        top = grid.drop(columns=["params_obj"]).head(50).copy()
        top["run_key"] = run_key
        top["run_label"] = run_label
        top["objective"] = "rare_move_stability_score"
        top["train_start"] = train_start.date().isoformat()
        top["train_end"] = train_end.date().isoformat()
        top["test_start"] = test_start.date().isoformat()
        top["test_end"] = test_end.date().isoformat()
        rankings.append(top)

        if selected is None:
            decisions.append(
                {
                    "run_key": run_key,
                    "run_label": run_label,
                    "objective": "rare_move_stability_score",
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "selected_params": "NO_STABLE_RARE_MOVE_CANDIDATE",
                    "test_total_pnl_cents": 0.0,
                    "test_sharpe": 0.0,
                    "test_trades": 0,
                }
            )
            continue

        p: RareMoveParams = selected["params_obj"]
        daily_full, trades_full = cache[p]
        daily = slice_daily(daily_full, test_start, test_end)
        trades = slice_trades(trades_full, test_start, test_end)
        metrics = compute_metrics(daily, trades)
        concentration = concentration_metrics(daily, trades)
        daily_list.append(daily)
        if not trades.empty:
            out_trades = trades.copy()
            out_trades["run_key"] = run_key
            out_trades["run_label"] = run_label
            out_trades["objective"] = "rare_move_stability_score"
            out_trades["test_start"] = test_start.date().isoformat()
            out_trades["test_end"] = test_end.date().isoformat()
            out_trades["selected_params"] = p.label
            trades_list.append(out_trades)

        decisions.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "objective": "rare_move_stability_score",
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_params": p.label,
                "train_rare_stability_score": float(selected["rare_stability_score"]),
                "train_total_pnl_cents": float(selected["total_pnl_cents"]),
                "train_sharpe": float(selected["daily_sharpe"]),
                "train_trades": int(selected["trades"]),
                "train_profitable_month_rate": float(selected["profitable_month_rate"]),
                "train_top_trade_share": float(selected["top_trade_share"]),
                "train_best_month_share": float(selected["best_month_share"]),
                "train_top_trade_removed_pnl_cents": float(selected["top_trade_removed_pnl_cents"]),
                "train_cluster_positive_neighbors": int(selected["cluster_positive_neighbors"]),
                "train_cluster_positive_rate": float(selected["cluster_positive_rate"]),
                "test_total_pnl_cents": metrics["total_pnl_cents"],
                "test_sharpe": metrics["daily_sharpe"],
                "test_trades": metrics["trades"],
                "test_win_rate": metrics["win_rate"],
                "test_max_drawdown_cents": metrics["max_drawdown_cents"],
                "test_profitable_month_rate": concentration["profitable_month_rate"],
                "test_top_trade_share": concentration["top_trade_share"],
                "test_best_month_share": concentration["best_month_share"],
                "test_top_trade_removed_pnl_cents": concentration["top_trade_removed_pnl_cents"],
            }
        )

    daily_oos = pd.concat(daily_list).sort_index().groupby(level=0).sum() if daily_list else pd.Series(dtype=float, name="daily_pnl_cents")
    trades_df = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    decisions_df = pd.DataFrame(decisions)
    rankings_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    return decisions_df, trades_df, daily_oos, rankings_df


def run_regime_rare_walkforward_cached(
    cache: dict[RegimeRareMoveParams, tuple[pd.Series, pd.DataFrame]],
    bars: pd.DataFrame,
    params: list[RegimeRareMoveParams],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    run_key = "regime_rare_move_wf"
    run_label = "Regime-gated rare move walk-forward"
    decisions = []
    trades_list = []
    daily_list = []
    rankings = []

    for train_start, train_end, test_start, test_end in period_schedule(bars, "monthly"):
        grid = evaluate_rare_grid_cached(cache, params, train_start, train_end)
        selected = select_regime_rare_candidate(grid)
        top = grid.drop(columns=["params_obj"]).head(50).copy()
        top["run_key"] = run_key
        top["run_label"] = run_label
        top["objective"] = "regime_rare_move_stability_score"
        top["train_start"] = train_start.date().isoformat()
        top["train_end"] = train_end.date().isoformat()
        top["test_start"] = test_start.date().isoformat()
        top["test_end"] = test_end.date().isoformat()
        rankings.append(top)

        if selected is None:
            decisions.append(
                {
                    "run_key": run_key,
                    "run_label": run_label,
                    "objective": "regime_rare_move_stability_score",
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "selected_params": "NO_STABLE_REGIME_RARE_MOVE_CANDIDATE",
                    "test_total_pnl_cents": 0.0,
                    "test_sharpe": 0.0,
                    "test_trades": 0,
                }
            )
            continue

        p: RegimeRareMoveParams = selected["params_obj"]
        daily_full, trades_full = cache[p]
        daily = slice_daily(daily_full, test_start, test_end)
        trades = slice_trades(trades_full, test_start, test_end)
        metrics = compute_metrics(daily, trades)
        concentration = concentration_metrics(daily, trades)
        daily_list.append(daily)
        if not trades.empty:
            out_trades = trades.copy()
            out_trades["run_key"] = run_key
            out_trades["run_label"] = run_label
            out_trades["objective"] = "regime_rare_move_stability_score"
            out_trades["test_start"] = test_start.date().isoformat()
            out_trades["test_end"] = test_end.date().isoformat()
            out_trades["selected_params"] = p.label
            trades_list.append(out_trades)

        decisions.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "objective": "regime_rare_move_stability_score",
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_params": p.label,
                "train_rare_stability_score": float(selected["rare_stability_score"]),
                "train_total_pnl_cents": float(selected["total_pnl_cents"]),
                "train_sharpe": float(selected["daily_sharpe"]),
                "train_trades": int(selected["trades"]),
                "train_profitable_month_rate": float(selected["profitable_month_rate"]),
                "train_top_trade_share": float(selected["top_trade_share"]),
                "train_best_month_share": float(selected["best_month_share"]),
                "train_top_trade_removed_pnl_cents": float(selected["top_trade_removed_pnl_cents"]),
                "train_cluster_positive_neighbors": int(selected["cluster_positive_neighbors"]),
                "train_cluster_positive_rate": float(selected["cluster_positive_rate"]),
                "test_total_pnl_cents": metrics["total_pnl_cents"],
                "test_sharpe": metrics["daily_sharpe"],
                "test_trades": metrics["trades"],
                "test_win_rate": metrics["win_rate"],
                "test_max_drawdown_cents": metrics["max_drawdown_cents"],
                "test_profitable_month_rate": concentration["profitable_month_rate"],
                "test_top_trade_share": concentration["top_trade_share"],
                "test_best_month_share": concentration["best_month_share"],
                "test_top_trade_removed_pnl_cents": concentration["top_trade_removed_pnl_cents"],
            }
        )

    daily_oos = pd.concat(daily_list).sort_index().groupby(level=0).sum() if daily_list else pd.Series(dtype=float, name="daily_pnl_cents")
    trades_df = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    decisions_df = pd.DataFrame(decisions)
    rankings_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    return decisions_df, trades_df, daily_oos.rename(run_key), rankings_df


def monthly_audit_rows(daily: pd.Series, trades: pd.DataFrame, run_key: str, run_label: str) -> list[dict]:
    if daily.empty:
        return []
    trade_months = pd.Series(dtype=int)
    if trades is not None and not trades.empty and "entry_time_dubai" in trades.columns:
        entry = pd.to_datetime(trades["entry_time_dubai"])
        trade_months = entry.dt.to_period("M").dt.to_timestamp().value_counts()
    total = float(daily.sum())
    rows = []
    for month, pnl in daily.groupby(pd.Grouper(freq="MS")).sum().items():
        rows.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "month": month.date().isoformat(),
                "month_pnl_cents": float(pnl),
                "month_share_of_total": float(pnl / total) if abs(total) > 1e-12 else 0.0,
                "trades": int(trade_months.get(month, 0)) if not trade_months.empty else 0,
            }
        )
    return rows


def repeated_clock_rows(trades: pd.DataFrame, run_key: str, run_label: str) -> list[dict]:
    if trades is None or trades.empty or "entry_time_dubai" not in trades.columns:
        return []
    out = trades.copy()
    entry = pd.to_datetime(out["entry_time_dubai"])
    out["entry_clock"] = entry.dt.strftime("%H:%M:%S")
    out["entry_date"] = entry.dt.date.astype(str)
    grouped = out.groupby("entry_clock")
    rows = []
    for clock, group in grouped:
        if len(group) < 2:
            continue
        rows.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "entry_clock": clock,
                "count": int(len(group)),
                "dates": ", ".join(group["entry_date"].tolist()),
                "total_pnl_cents": float(group["net_pnl_cents"].sum()),
                "avg_pnl_cents": float(group["net_pnl_cents"].mean()),
            }
        )
    return rows


def top_trade_rows(trades: pd.DataFrame, run_key: str, run_label: str) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["run_key"] = run_key
    out["run_label"] = run_label
    out["abs_pnl_cents"] = out["net_pnl_cents"].abs()
    cols = [
        "run_key",
        "run_label",
        "entry_time_dubai",
        "exit_time_dubai",
        "side",
        "move_cents",
        "threshold_cents",
        "net_pnl_cents",
        "abs_pnl_cents",
        "selected_params",
    ]
    return out.sort_values("abs_pnl_cents", ascending=False)[[c for c in cols if c in out.columns]].head(20)


def main() -> None:
    bars, stats = load_second_prices(DATA_DIR, "2025-10-01", "2026-03-26", cache_dir=BASE_DIR / ".price_cache", workers=8)
    bars = add_volatility_columns(bars)
    days = prepare_days(bars)
    days = add_rare_move_thresholds(days)
    params = build_universe()

    pd.DataFrame([p.__dict__ | {"label": p.label} for p in params]).to_csv(OUT_DIR / "parameter_universe.csv", index=False)

    all_metrics = []
    all_decisions = []
    all_trades = []
    all_daily = []
    all_rankings = []
    all_multiple_backtests = []
    all_robust_metrics = []
    all_robust_decisions = []
    all_robust_trades = []
    all_robust_daily = []
    all_robust_rankings = []
    all_monthly_audit = []
    all_repeated_clocks = []
    all_top_trades = []

    caches: dict[str, dict[VolParams, tuple[pd.Series, pd.DataFrame]]] = {}
    for vol_method in ["percent_to_dollar"]:
        method_params = [p for p in params if p.vol_method == vol_method]
        print("precomputing", vol_method, "candidates", len(method_params), flush=True)
        cache = precompute_results(days, method_params)
        caches[vol_method] = cache
        multiple_backtests = evaluate_multiple_backtests_cached(cache, method_params, WALKFORWARD_START, bars.index.max().normalize())
        if not multiple_backtests.empty:
            all_multiple_backtests.append(multiple_backtests)
        if vol_method == "percent_to_dollar":
            export_winning_percent_vol_backtest(cache, multiple_backtests, WALKFORWARD_START, bars.index.max().normalize())
        for rebalance in ["monthly"]:
            for objective in ["daily_sharpe", "total_pnl_cents"]:
                print("running", vol_method, rebalance, objective, "candidates", len(method_params), flush=True)
                decisions, trades, daily, rankings = run_walkforward_cached(cache, bars, method_params, rebalance, objective)
                metrics = compute_metrics(daily, trades)
                run_key = f"{vol_method}_{rebalance}_{objective_slug(objective)}"
                all_metrics.append({"run_key": run_key, "vol_method": vol_method, "rebalance": rebalance, "objective": objective, **metrics})
                if not decisions.empty:
                    decisions["run_key"] = run_key
                    decisions["vol_method"] = vol_method
                    all_decisions.append(decisions)
                if not trades.empty:
                    trades["run_key"] = run_key
                    trades["vol_method"] = vol_method
                    all_trades.append(trades)
                if not daily.empty:
                    all_daily.append(daily.rename(run_key))
                if not rankings.empty:
                    rankings["run_key"] = run_key
                    rankings["vol_method"] = vol_method
                    all_rankings.append(rankings)

                run_label = f"Baseline {objective_slug(objective).replace('_', ' ')}"
                all_monthly_audit.extend(monthly_audit_rows(daily, trades, run_key, run_label))
                all_repeated_clocks.extend(repeated_clock_rows(trades, run_key, run_label))
                top_trades = top_trade_rows(trades, run_key, run_label)
                if not top_trades.empty:
                    all_top_trades.append(top_trades)

    percent_params = [p for p in params if p.vol_method == "percent_to_dollar"]
    robust_specs: list[tuple[str, str, Iterable[VolParams], bool]] = [
        ("robust_percent_all", "Robust percent-vol: all directions", percent_params, False),
        ("robust_percent_strict", "Strict robust percent-vol: sit out if unstable", percent_params, True),
        ("robust_percent_continuation", "Robust percent-vol: continuation only", [p for p in percent_params if p.signal_mode == "continuation"], False),
        ("robust_percent_reversal", "Robust percent-vol: reversal only", [p for p in percent_params if p.signal_mode == "reversal"], False),
    ]
    percent_cache = caches["percent_to_dollar"]
    for run_key, run_label, run_params, strict in robust_specs:
        run_params = list(run_params)
        print("running", run_key, "candidates", len(run_params), flush=True)
        decisions, trades, daily, rankings = run_robust_walkforward_cached(percent_cache, bars, run_params, run_key, run_label, strict=strict)
        metrics = compute_metrics(daily, trades)
        concentration = concentration_metrics(daily, trades)
        all_robust_metrics.append({"run_key": run_key, "run_label": run_label, "objective": ROBUST_OBJECTIVE_NAME, **metrics, **concentration})
        if not decisions.empty:
            all_robust_decisions.append(decisions)
        if not trades.empty:
            all_robust_trades.append(trades)
        if not daily.empty:
            all_robust_daily.append(daily.rename(run_key))
        if not rankings.empty:
            all_robust_rankings.append(rankings)
        all_monthly_audit.extend(monthly_audit_rows(daily, trades, run_key, run_label))
        all_repeated_clocks.extend(repeated_clock_rows(trades, run_key, run_label))
        top_trades = top_trade_rows(trades, run_key, run_label)
        if not top_trades.empty:
            all_top_trades.append(top_trades)

    stable_params = build_stable_universe()
    pd.DataFrame([p.__dict__ | {"label": p.label} for p in stable_params]).to_csv(OUT_DIR / "stable_parameter_universe.csv", index=False)
    print("precomputing stable 4-variable candidates", len(stable_params), flush=True)
    stable_cache = precompute_results(days, stable_params)
    stable_decisions, stable_trades, stable_daily, stable_rankings = run_stable_walkforward_cached(stable_cache, bars, stable_params)
    stable_metrics = compute_metrics(stable_daily, stable_trades)
    stable_concentration = concentration_metrics(stable_daily, stable_trades)
    stable_run_key = "stable_4var_percent"
    stable_run_label = "Stable 4-variable percent-vol walk-forward"
    stable_metrics_df = pd.DataFrame(
        [
            {
                "run_key": stable_run_key,
                "run_label": stable_run_label,
                "objective": "stable_4var_score",
                **stable_metrics,
                **stable_concentration,
            }
        ]
    )
    all_monthly_audit.extend(monthly_audit_rows(stable_daily, stable_trades, stable_run_key, stable_run_label))
    all_repeated_clocks.extend(repeated_clock_rows(stable_trades, stable_run_key, stable_run_label))
    top_trades = top_trade_rows(stable_trades, stable_run_key, stable_run_label)
    if not top_trades.empty:
        all_top_trades.append(top_trades)

    rare_params = build_rare_universe()
    pd.DataFrame([p.__dict__ | {"label": p.label} for p in rare_params]).to_csv(OUT_DIR / "rare_move_parameter_universe.csv", index=False)
    print("precomputing rare move candidates", len(rare_params), flush=True)
    rare_cache = precompute_rare_results(days, rare_params)
    rare_decisions, rare_trades, rare_daily, rare_rankings = run_rare_walkforward_cached(rare_cache, bars, rare_params)
    rare_metrics = compute_metrics(rare_daily, rare_trades)
    rare_concentration = concentration_metrics(rare_daily, rare_trades)
    rare_run_key = "rare_move_percentile_wf"
    rare_run_label = "Rare move percentile walk-forward"
    rare_metrics_df = pd.DataFrame(
        [
            {
                "run_key": rare_run_key,
                "run_label": rare_run_label,
                "objective": "rare_move_stability_score",
                **rare_metrics,
                **rare_concentration,
            }
        ]
    )
    all_monthly_audit.extend(monthly_audit_rows(rare_daily, rare_trades, rare_run_key, rare_run_label))
    all_repeated_clocks.extend(repeated_clock_rows(rare_trades, rare_run_key, rare_run_label))
    top_trades = top_trade_rows(rare_trades, rare_run_key, rare_run_label)
    if not top_trades.empty:
        all_top_trades.append(top_trades)

    regime_rare_params = build_regime_rare_universe()
    pd.DataFrame([p.__dict__ | {"label": p.label} for p in regime_rare_params]).to_csv(OUT_DIR / "regime_rare_parameter_universe.csv", index=False)
    print("precomputing regime-gated rare move candidates", len(regime_rare_params), flush=True)
    regime_rare_cache = precompute_rare_results(days, regime_rare_params)
    regime_rare_decisions, regime_rare_trades, regime_rare_daily, regime_rare_rankings = run_regime_rare_walkforward_cached(
        regime_rare_cache,
        bars,
        regime_rare_params,
    )
    regime_rare_metrics = compute_metrics(regime_rare_daily, regime_rare_trades)
    regime_rare_concentration = concentration_metrics(regime_rare_daily, regime_rare_trades)
    regime_rare_run_key = "regime_rare_move_wf"
    regime_rare_run_label = "Regime-gated rare move walk-forward"
    regime_rare_metrics_df = pd.DataFrame(
        [
            {
                "run_key": regime_rare_run_key,
                "run_label": regime_rare_run_label,
                "objective": "regime_rare_move_stability_score",
                **regime_rare_metrics,
                **regime_rare_concentration,
            }
        ]
    )
    all_monthly_audit.extend(monthly_audit_rows(regime_rare_daily, regime_rare_trades, regime_rare_run_key, regime_rare_run_label))
    all_repeated_clocks.extend(repeated_clock_rows(regime_rare_trades, regime_rare_run_key, regime_rare_run_label))
    top_trades = top_trade_rows(regime_rare_trades, regime_rare_run_key, regime_rare_run_label)
    if not top_trades.empty:
        all_top_trades.append(top_trades)

    pd.DataFrame(all_metrics).sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).to_csv(OUT_DIR / "metrics.csv", index=False)
    (pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()).to_csv(OUT_DIR / "decisions.csv", index=False)
    (pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()).to_csv(OUT_DIR / "trades.csv", index=False)
    pd.concat(all_daily, axis=1).to_csv(OUT_DIR / "daily_pnl.csv")
    (pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()).to_csv(OUT_DIR / "train_rankings.csv", index=False)
    (pd.concat(all_multiple_backtests, ignore_index=True) if all_multiple_backtests else pd.DataFrame()).to_csv(
        OUT_DIR / "multiple_backtests.csv",
        index=False,
    )
    pd.DataFrame(all_robust_metrics).sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).to_csv(OUT_DIR / "robust_metrics.csv", index=False)
    (pd.concat(all_robust_decisions, ignore_index=True) if all_robust_decisions else pd.DataFrame()).to_csv(OUT_DIR / "robust_decisions.csv", index=False)
    (pd.concat(all_robust_trades, ignore_index=True) if all_robust_trades else pd.DataFrame()).to_csv(OUT_DIR / "robust_trades.csv", index=False)
    pd.concat(all_robust_daily, axis=1).to_csv(OUT_DIR / "robust_daily_pnl.csv")
    (pd.concat(all_robust_rankings, ignore_index=True) if all_robust_rankings else pd.DataFrame()).to_csv(OUT_DIR / "robust_train_rankings.csv", index=False)
    stable_metrics_df.to_csv(OUT_DIR / "stable_4var_metrics.csv", index=False)
    stable_decisions.to_csv(OUT_DIR / "stable_4var_decisions.csv", index=False)
    stable_trades.to_csv(OUT_DIR / "stable_4var_trades.csv", index=False)
    stable_daily.rename(stable_run_key).to_csv(OUT_DIR / "stable_4var_daily_pnl.csv")
    stable_rankings.to_csv(OUT_DIR / "stable_4var_train_rankings.csv", index=False)
    rare_metrics_df.to_csv(OUT_DIR / "rare_move_metrics.csv", index=False)
    rare_decisions.to_csv(OUT_DIR / "rare_move_decisions.csv", index=False)
    rare_trades.to_csv(OUT_DIR / "rare_move_trades.csv", index=False)
    rare_daily.rename(rare_run_key).to_csv(OUT_DIR / "rare_move_daily_pnl.csv")
    rare_rankings.to_csv(OUT_DIR / "rare_move_train_rankings.csv", index=False)
    regime_rare_metrics_df.to_csv(OUT_DIR / "regime_rare_metrics.csv", index=False)
    regime_rare_decisions.to_csv(OUT_DIR / "regime_rare_decisions.csv", index=False)
    regime_rare_trades.to_csv(OUT_DIR / "regime_rare_trades.csv", index=False)
    regime_rare_daily.to_csv(OUT_DIR / "regime_rare_daily_pnl.csv")
    regime_rare_rankings.to_csv(OUT_DIR / "regime_rare_train_rankings.csv", index=False)
    pd.DataFrame(all_monthly_audit).to_csv(OUT_DIR / "stability_monthly_audit.csv", index=False)
    pd.DataFrame(all_repeated_clocks).to_csv(OUT_DIR / "stability_repeated_clocks.csv", index=False)
    (pd.concat(all_top_trades, ignore_index=True) if all_top_trades else pd.DataFrame()).to_csv(OUT_DIR / "stability_top_trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "data_start": str(bars.index.min()),
                "data_end": str(bars.index.max()),
                "raw_files": int(len(stats)),
                "candidate_count": int(len(params)),
                "stable_4var_candidate_count": int(len(stable_params)),
                "rare_move_candidate_count": int(len(rare_params)),
                "regime_rare_move_candidate_count": int(len(regime_rare_params)),
                "train_months": TRAIN_MONTHS,
                "rebalance_options": "monthly",
                "vol_methods": "percent_to_dollar",
                "vol_windows_days": "63",
                "sigma_multiples": ",".join(f"{x:g}" for x in SIGMA_MULTIPLES),
                "stable_4var_a_delay_seconds": ",".join(str(x) for x in STABLE_DELAYS),
                "stable_4var_b_sigma_multiples": ",".join(f"{x:g}" for x in STABLE_SIGMA_MULTIPLES),
                "stable_4var_c_move_windows_seconds": ",".join(str(x) for x in STABLE_MOVE_WINDOWS),
                "stable_4var_d_hold_seconds": ",".join(str(x) for x in STABLE_HOLDS),
                "rare_move_a_delay_seconds": ",".join(str(x) for x in RARE_DELAYS),
                "rare_move_b_percentiles": ",".join(f"{x:g}" for x in RARE_PERCENTILES),
                "rare_move_c_move_windows_seconds": ",".join(str(x) for x in RARE_MOVE_WINDOWS),
                "rare_move_d_hold_seconds": ",".join(str(x) for x in RARE_HOLDS),
                "rare_move_threshold_lookback_days": RARE_LOOKBACK_DAYS,
                "rare_move_threshold_min_observations": RARE_MIN_OBSERVATIONS,
                "regime_rare_trend_filters": ",".join(REGIME_TREND_FILTERS),
                "regime_rare_vol_filters": ",".join(REGIME_VOL_FILTERS),
                "regime_rare_time_buckets": ",".join(REGIME_TIME_BUCKETS),
                "walkforward_test_start": WALKFORWARD_START.date().isoformat(),
                "early_rebalance_training": "Nov uses Oct only; Dec uses Oct-Nov; Jan onward uses rolling 3 months",
                "vol_min_observations": VOL_MIN_OBSERVATIONS,
                "full_signal_window_in_session": REQUIRE_FULL_SIGNAL_WINDOW_IN_SESSION,
                "robust_objective": "Sharpe plus top-trade-removed PnL and profitable-month rate, penalized for top-trade and best-month concentration",
                "bad_hour_rules": "exit_by_midnight",
                "session": "11:00-24:00 Dubai signal time",
                "cost_formula": f"round-trip cost cents = abs(entry_price) * {DYNAMIC_COST_CENTS_PER_PRICE_UNIT:g}",
            }
        ]
    ).to_csv(OUT_DIR / "config.csv", index=False)
    print(pd.read_csv(OUT_DIR / "metrics.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
