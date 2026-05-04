from __future__ import annotations

import itertools
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATE_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.csv$")
EPS = 1e-12
CACHE_VERSION = "v2"


@dataclass(frozen=True)
class StrategyParams:
    delay_s: int
    threshold_cents: float
    lookback_s: int
    hold_s: int
    direction: str = "momentum"

    @property
    def label(self) -> str:
        return (
            f"a={self.delay_s}s | b={self.threshold_cents:g}c | "
            f"c={self.lookback_s}s | d={self.hold_s}s | {self.direction}"
        )


def parse_number_list(raw: str, cast=float) -> list:
    values = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(cast(token))
    return sorted(set(values))


def discover_csv_files(data_dir: str | Path) -> pd.DataFrame:
    data_path = Path(data_dir)
    rows = []
    for file in sorted(data_path.glob("*.csv")):
        if not DATE_FILE_RE.match(file.name):
            continue
        rows.append(
            {
                "date": pd.to_datetime(file.stem).date(),
                "file": str(file),
                "name": file.name,
                "size_mb": file.stat().st_size / 1_000_000,
            }
        )
    return pd.DataFrame(rows)


def _pick_columns(columns: Iterable[str]) -> tuple[str, str]:
    cols = list(columns)
    price_cols = [c for c in cols if "price" in c.lower()]
    if not price_cols:
        raise ValueError(f"Could not find a price column in {cols}")

    time_cols = [c for c in cols if c.lower().startswith("time")]
    if not time_cols:
        raise ValueError(f"Could not find a time column in {cols}")

    # Older files contain Time plus Time.1; Time.1 is ISO-like and less ambiguous.
    time_col = "Time.1" if "Time.1" in time_cols else time_cols[0]
    return time_col, price_cols[0]


def _cache_file_for(file: Path, cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{file.stem}.{CACHE_VERSION}.pkl"


def _load_cached_day(file: Path, cache_file: Path | None) -> tuple[pd.Series, dict] | None:
    if cache_file is None or not cache_file.exists():
        return None
    if cache_file.stat().st_mtime < file.stat().st_mtime:
        return None
    try:
        with cache_file.open("rb") as fh:
            payload = pickle.load(fh)
        return payload["price"], payload["stats"]
    except Exception:
        return None


def _write_cached_day(cache_file: Path | None, price: pd.Series, stats: dict) -> None:
    if cache_file is None:
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as fh:
        pickle.dump({"price": price, "stats": stats}, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_file_to_seconds(file: str | Path, cache_dir: str | Path | None = None) -> tuple[pd.Series, dict]:
    file = Path(file)
    cache_file = _cache_file_for(file, Path(cache_dir) if cache_dir else None)
    cached = _load_cached_day(file, cache_file)
    if cached is not None:
        return cached

    header = pd.read_csv(file, nrows=0)
    time_col, price_col = _pick_columns(header.columns)
    raw = pd.read_csv(file, usecols=[time_col, price_col], dtype={time_col: str})
    raw = raw.rename(columns={time_col: "time", price_col: "price"})
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw = raw.dropna(subset=["time", "price"])

    stats = {
        "date": pd.to_datetime(file.stem).date(),
        "file": file.name,
        "raw_rows": int(len(raw)),
        "seconds": 0,
        "first_time": pd.NaT,
        "last_time": pd.NaT,
        "first_price": np.nan,
        "last_price": np.nan,
    }
    if raw.empty:
        _write_cached_day(cache_file, pd.Series(dtype=float, name="price"), stats)
        return pd.Series(dtype=float, name="price"), stats

    # Timestamp strings are second-granular in these files and heavily duplicated.
    # Grouping before datetime parsing avoids parsing millions of duplicate strings.
    time_key = raw["time"].str.slice(0, 19).str.replace("T", " ", regex=False)
    by_second = raw.groupby(time_key, sort=True)["price"].last().astype(float)
    parsed_index = pd.to_datetime(by_second.index, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    by_second.index = parsed_index
    by_second = by_second[by_second.index.notna()].sort_index()
    if by_second.empty:
        _write_cached_day(cache_file, pd.Series(dtype=float, name="price"), stats)
        return pd.Series(dtype=float, name="price"), stats

    idx = pd.date_range(by_second.index.min(), by_second.index.max(), freq="1s")
    second_price = by_second.reindex(idx).ffill()
    second_price.name = "price"

    stats.update(
        {
            "seconds": int(second_price.shape[0]),
            "first_time": second_price.index.min(),
            "last_time": second_price.index.max(),
            "first_price": float(second_price.iloc[0]),
            "last_price": float(second_price.iloc[-1]),
        }
    )
    _write_cached_day(cache_file, second_price, stats)
    return second_price, stats


def load_second_prices(
    data_dir: str | Path,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    cache_dir: str | Path | None = None,
    workers: int = 6,
) -> tuple[pd.Series, pd.DataFrame]:
    manifest = discover_csv_files(data_dir)
    if manifest.empty:
        return pd.Series(dtype=float, name="price"), pd.DataFrame()

    if start_date is not None:
        start_d = pd.to_datetime(start_date).date()
        manifest = manifest[manifest["date"] >= start_d]
    if end_date is not None:
        end_d = pd.to_datetime(end_date).date()
        manifest = manifest[manifest["date"] <= end_d]

    series = []
    stats_rows = []
    if cache_dir is None:
        cache_dir = Path(data_dir) / "price_action_dashboard" / ".price_cache"

    files = manifest["file"].tolist()
    if workers and workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            results = list(pool.map(lambda f: load_file_to_seconds(f, cache_dir=cache_dir), files))
    else:
        results = [load_file_to_seconds(file, cache_dir=cache_dir) for file in files]

    for daily, stats in results:
        stats_rows.append(stats)
        if not daily.empty:
            series.append(daily)

    if not series:
        return pd.Series(dtype=float, name="price"), pd.DataFrame(stats_rows)

    price = pd.concat(series).sort_index()
    price = price[~price.index.duplicated(keep="last")]
    price.name = "price"
    return price, pd.DataFrame(stats_rows)


def parameter_grid(
    delays: Iterable[int],
    thresholds: Iterable[float],
    lookbacks: Iterable[int],
    holds: Iterable[int],
    direction: str,
) -> list[StrategyParams]:
    return [
        StrategyParams(int(a), float(b), int(c), int(d), direction)
        for a, b, c, d in itertools.product(delays, thresholds, lookbacks, holds)
        if int(c) > 0 and int(d) > 0 and float(b) > 0 and int(a) >= 0
    ]


def prepare_day_arrays(price: pd.Series) -> list[dict]:
    price = price.dropna().sort_index()
    days = []
    for date, day_price in price.groupby(price.index.normalize(), sort=True):
        days.append(
            {
                "date": pd.Timestamp(date),
                "times": day_price.index.to_numpy(),
                "prices": day_price.to_numpy(dtype=float),
            }
        )
    return days


def _scan_day(day: dict, params: StrategyParams, cost_cents: float, collect_trades: bool) -> tuple[list[dict], list[float], list[int]]:
    prices = day["prices"]
    times = day["times"]
    n = len(prices)
    min_required = params.lookback_s + params.delay_s + params.hold_s + 1
    if n <= min_required:
        return [], [], []

    delta = prices[params.lookback_s :] - prices[: -params.lookback_s]
    raw_idx = np.flatnonzero(np.abs(delta) >= params.threshold_cents / 100.0) + params.lookback_s
    if raw_idx.size == 0:
        return [], [], []

    trades = []
    pnls = []
    sides = []
    last_exit = -1
    i = 0
    while i < raw_idx.size:
        event_idx = int(raw_idx[i])
        entry_idx = event_idx + params.delay_s
        exit_idx = entry_idx + params.hold_s
        if exit_idx >= n:
            break
        if entry_idx <= last_exit:
            i = int(np.searchsorted(raw_idx, last_exit - params.delay_s + 1, side="left"))
            continue

        move = prices[event_idx] - prices[event_idx - params.lookback_s]
        side = 1 if move > 0 else -1
        if params.direction == "fade":
            side *= -1

        entry_price = prices[entry_idx]
        exit_price = prices[exit_idx]
        gross_cents = side * (exit_price - entry_price) * 100.0
        net_cents = gross_cents - cost_cents
        pnls.append(float(net_cents))
        sides.append(side)
        if collect_trades:
            trades.append(
                {
                    "signal_time": pd.Timestamp(times[event_idx]),
                    "entry_time": pd.Timestamp(times[entry_idx]),
                    "exit_time": pd.Timestamp(times[exit_idx]),
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl_cents": gross_cents,
                    "cost_cents": cost_cents,
                    "net_pnl_cents": net_cents,
                }
            )
        last_exit = exit_idx
        i = int(np.searchsorted(raw_idx, last_exit - params.delay_s + 1, side="left"))

    return trades, pnls, sides


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
        "profit_factor": gross_profit / (gross_loss + EPS) if gross_loss > 0 else np.inf,
        "long_trades": int((sides_arr > 0).sum()),
        "short_trades": int((sides_arr < 0).sum()),
    }


def _backtest_prepared(
    days: list[dict],
    params: StrategyParams,
    cost_cents: float,
    collect_trades: bool,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    if not days:
        return pd.DataFrame(), pd.Series(dtype=float, name="daily_pnl_cents"), _trade_summary([], [])

    all_trades: list[dict] = []
    all_pnls: list[float] = []
    all_sides: list[int] = []
    daily_values = []
    daily_index = []

    for day in days:
        trades, pnls, sides = _scan_day(day, params, cost_cents, collect_trades)
        if collect_trades and trades:
            all_trades.extend(trades)
        all_pnls.extend(pnls)
        all_sides.extend(sides)
        daily_index.append(day["date"])
        daily_values.append(float(np.sum(pnls)) if pnls else 0.0)

    daily = pd.Series(daily_values, index=pd.Index(daily_index, name="date"), name="daily_pnl_cents")
    trades_df = pd.DataFrame(all_trades) if collect_trades and all_trades else pd.DataFrame()
    return trades_df, daily, _trade_summary(all_pnls, all_sides)


def backtest_strategy(price: pd.Series, params: StrategyParams, cost_cents: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    days = prepare_day_arrays(price)
    trades, daily, _ = _backtest_prepared(days, params, cost_cents, collect_trades=True)
    return trades, daily


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity - peak).min())


def compute_metrics(daily_pnl: pd.Series, trades: pd.DataFrame | None = None, trade_summary: dict | None = None) -> dict:
    daily = daily_pnl.fillna(0.0)
    equity = daily.cumsum()
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float(np.sqrt(252) * daily.mean() / (std + EPS)) if std > 0 else 0.0
    downside = daily[daily < 0].std(ddof=1)
    sortino = float(np.sqrt(252) * daily.mean() / (downside + EPS)) if pd.notna(downside) and downside > 0 else 0.0

    row = {
        "total_pnl_cents": float(daily.sum()),
        "avg_daily_pnl_cents": float(daily.mean()) if len(daily) else 0.0,
        "daily_sharpe": sharpe,
        "daily_sortino": sortino,
        "max_drawdown_cents": max_drawdown(equity),
        "positive_day_rate": float((daily > 0).mean()) if len(daily) else 0.0,
        "active_day_rate": float((daily != 0).mean()) if len(daily) else 0.0,
    }

    if trade_summary is not None:
        row.update(trade_summary)
        return row

    if trades is None or trades.empty:
        row.update(_trade_summary([], []))
        return row

    pnl = trades["net_pnl_cents"].astype(float)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    row.update(
        {
            "trades": int(len(trades)),
            "win_rate": float((pnl > 0).mean()),
            "avg_trade_cents": float(pnl.mean()),
            "profit_factor": gross_profit / (gross_loss + EPS) if gross_loss > 0 else np.inf,
            "long_trades": int((trades["side"] > 0).sum()),
            "short_trades": int((trades["side"] < 0).sum()),
        }
    )
    return row


def evaluate_grid(
    price: pd.Series,
    params_list: list[StrategyParams],
    objective: str,
    cost_cents: float,
    min_trades: int,
) -> pd.DataFrame:
    rows = []
    days = prepare_day_arrays(price)
    for params in params_list:
        _, daily, summary = _backtest_prepared(days, params, cost_cents, collect_trades=False)
        metrics = compute_metrics(daily, trade_summary=summary)
        if metrics["trades"] < min_trades:
            score = -np.inf
        else:
            score = float(metrics.get(objective, metrics["daily_sharpe"]))
        rows.append(
            {
                "params": params,
                "label": params.label,
                "delay_s": params.delay_s,
                "threshold_cents": params.threshold_cents,
                "lookback_s": params.lookback_s,
                "hold_s": params.hold_s,
                "direction": params.direction,
                "score": score,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def run_walk_forward(
    price: pd.Series,
    params_list: list[StrategyParams],
    train_months: int = 3,
    test_months: int = 1,
    objective: str = "daily_sharpe",
    cost_cents: float = 0.0,
    min_train_trades: int = 20,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    price = price.dropna().sort_index()
    if price.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    month_key = price.index.to_period("M")
    months = pd.PeriodIndex(sorted(month_key.unique()))
    detail_rows = []
    all_test_daily = []
    all_test_trades = []
    train_grids = []

    for start_i in range(train_months, len(months), test_months):
        train_periods = months[start_i - train_months : start_i]
        test_periods = months[start_i : start_i + test_months]
        if len(test_periods) == 0:
            continue

        train_price = price[month_key.isin(train_periods)]
        test_price = price[month_key.isin(test_periods)]
        if train_price.empty or test_price.empty:
            continue

        grid = evaluate_grid(train_price, params_list, objective, cost_cents, min_train_trades)
        grid = grid.copy()
        grid["train_start"] = str(train_periods[0])
        grid["train_end"] = str(train_periods[-1])
        grid["test_start"] = str(test_periods[0])
        grid["test_end"] = str(test_periods[-1])
        train_grids.append(grid.drop(columns=["params"]))

        finite = grid[np.isfinite(grid["score"])]
        if finite.empty:
            continue

        best_row = finite.iloc[0]
        best_params = best_row["params"]
        test_trades, test_daily = backtest_strategy(test_price, best_params, cost_cents)
        test_metrics = compute_metrics(test_daily, test_trades)
        all_test_daily.append(test_daily)
        if not test_trades.empty:
            t = test_trades.copy()
            t["params_label"] = best_params.label
            t["test_month"] = ",".join(str(p) for p in test_periods)
            all_test_trades.append(t)

        detail_rows.append(
            {
                "train_start": str(train_periods[0]),
                "train_end": str(train_periods[-1]),
                "test_start": str(test_periods[0]),
                "test_end": str(test_periods[-1]),
                "selected_params": best_params.label,
                "train_score": float(best_row["score"]),
                "train_sharpe": float(best_row["daily_sharpe"]),
                "train_total_pnl_cents": float(best_row["total_pnl_cents"]),
                "train_trades": int(best_row["trades"]),
                "test_sharpe": test_metrics["daily_sharpe"],
                "test_total_pnl_cents": test_metrics["total_pnl_cents"],
                "test_trades": test_metrics["trades"],
                "test_max_drawdown_cents": test_metrics["max_drawdown_cents"],
                "test_win_rate": test_metrics["win_rate"],
            }
        )

    if all_test_daily:
        oos_daily = pd.concat(all_test_daily).sort_index()
        oos_daily = oos_daily.groupby(oos_daily.index).sum()
    else:
        oos_daily = pd.Series(dtype=float, name="daily_pnl_cents")

    trades_df = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    details_df = pd.DataFrame(detail_rows)
    grid_df = pd.concat(train_grids, ignore_index=True) if train_grids else pd.DataFrame()
    return details_df, oos_daily, trades_df, grid_df


def daily_close(price: pd.Series) -> pd.Series:
    if price.empty:
        return pd.Series(dtype=float, name="close")
    close = price.groupby(price.index.normalize()).last()
    close.name = "close"
    return close
