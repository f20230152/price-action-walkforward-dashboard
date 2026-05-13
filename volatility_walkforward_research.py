from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

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
VOL_WINDOWS_DAYS = [63]
VOL_MIN_OBSERVATIONS = 21
SESSION_START_HOUR_DUBAI = 11
SESSION_END_HOUR_DUBAI = 24
MAX_TRADES_PER_DAY = 1
MIN_TRAIN_TRADES = 5


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
            [0.40, 0.55],
            [1800],
            [3600, 21600],
            VOL_WINDOWS_DAYS,
            ["dollar", "percent_to_dollar"],
            ["continuation", "reversal"],
            ["exit_by_midnight"],
        )
    ]


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
            }
        )
    return days


def _day_scalar(day_bars: pd.DataFrame, col: str) -> float:
    if col not in day_bars.columns:
        return np.nan
    values = day_bars[col].dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


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
    start = pd.Timestamp("2026-01-01")
    end = bars.index.max().normalize()
    if rebalance != "monthly":
        raise ValueError(rebalance)
    starts = pd.date_range(start, end, freq="MS")
    test_ends = [min(s + pd.DateOffset(months=1) - pd.Timedelta(days=1), end) for s in starts]

    schedule = []
    for test_start, test_end in zip(starts, test_ends):
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


def main() -> None:
    bars, stats = load_second_prices(DATA_DIR, "2025-10-01", "2026-03-26", cache_dir=BASE_DIR / ".price_cache", workers=8)
    bars = add_volatility_columns(bars)
    days = prepare_days(bars)
    params = build_universe()

    pd.DataFrame([p.__dict__ | {"label": p.label} for p in params]).to_csv(OUT_DIR / "parameter_universe.csv", index=False)

    all_metrics = []
    all_decisions = []
    all_trades = []
    all_daily = []
    all_rankings = []
    for vol_method in ["dollar", "percent_to_dollar"]:
        method_params = [p for p in params if p.vol_method == vol_method]
        print("precomputing", vol_method, "candidates", len(method_params), flush=True)
        cache = precompute_results(days, method_params)
        for rebalance in ["monthly"]:
            for objective in ["daily_sharpe", "total_pnl_cents"]:
                print("running", vol_method, rebalance, objective, "candidates", len(method_params), flush=True)
                decisions, trades, daily, rankings = run_walkforward_cached(cache, bars, method_params, rebalance, objective)
                metrics = compute_metrics(daily, trades)
                run_key = f"{vol_method}_{rebalance}_{objective}"
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

    pd.DataFrame(all_metrics).sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).to_csv(OUT_DIR / "metrics.csv", index=False)
    pd.concat(all_decisions, ignore_index=True).to_csv(OUT_DIR / "decisions.csv", index=False)
    (pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()).to_csv(OUT_DIR / "trades.csv", index=False)
    pd.concat(all_daily, axis=1).to_csv(OUT_DIR / "daily_pnl.csv")
    (pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()).to_csv(OUT_DIR / "train_rankings.csv", index=False)
    pd.DataFrame(
        [
            {
                "data_start": str(bars.index.min()),
                "data_end": str(bars.index.max()),
                "raw_files": int(len(stats)),
                "candidate_count": int(len(params)),
                "train_months": TRAIN_MONTHS,
                "rebalance_options": "monthly",
                "vol_methods": "dollar,percent_to_dollar",
                "vol_windows_days": "63",
                "vol_min_observations": VOL_MIN_OBSERVATIONS,
                "bad_hour_rules": "exit_by_midnight",
                "session": "11:00-24:00 Dubai signal time",
                "cost_formula": f"round-trip cost cents = abs(entry_price) * {DYNAMIC_COST_CENTS_PER_PRICE_UNIT:g}",
            }
        ]
    ).to_csv(OUT_DIR / "config.csv", index=False)
    print(pd.read_csv(OUT_DIR / "metrics.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
