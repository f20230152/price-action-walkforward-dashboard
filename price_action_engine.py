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
CACHE_VERSION = "v3"
DUBAI_UTC_OFFSET_HOURS = 4
DYNAMIC_COST_CENTS_PER_PRICE_UNIT = 0.04


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


@dataclass(frozen=True)
class SpikeParams:
    delay_s: int
    threshold_cents: float
    lookback_s: int
    hold_s: int
    volume_window_s: int = 300
    volume_multiple: float = 0.0
    direction: str = "momentum"

    @property
    def label(self) -> str:
        vol = "vol off" if self.volume_multiple <= 0 else f"vol>{self.volume_multiple:g}x/{self.volume_window_s}s"
        return (
            f"a={self.delay_s}s | move={self.threshold_cents:g}c/{self.lookback_s}s | "
            f"hold={self.hold_s}s | {vol} | {self.direction}"
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


def _pick_columns(columns: Iterable[str]) -> tuple[str, str, str | None]:
    cols = list(columns)
    price_cols = [c for c in cols if "price" in c.lower()]
    if not price_cols:
        raise ValueError(f"Could not find a price column in {cols}")

    time_cols = [c for c in cols if c.lower().startswith("time")]
    if not time_cols:
        raise ValueError(f"Could not find a time column in {cols}")

    # Older files contain Time plus Time.1; Time.1 is ISO-like and less ambiguous.
    time_col = "Time.1" if "Time.1" in time_cols else time_cols[0]
    size_cols = [c for c in cols if "size" in c.lower() or "volume" in c.lower()]
    size_col = size_cols[0] if size_cols else None
    return time_col, price_cols[0], size_col


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
    time_col, price_col, size_col = _pick_columns(header.columns)
    usecols = [time_col, price_col] + ([size_col] if size_col else [])
    raw = pd.read_csv(file, usecols=usecols, dtype={time_col: str})
    rename_cols = {time_col: "time", price_col: "price"}
    if size_col:
        rename_cols[size_col] = "volume"
    raw = raw.rename(columns=rename_cols)
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    if "volume" in raw.columns:
        raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce").fillna(0.0)
    else:
        raw["volume"] = 0.0
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
        empty = pd.DataFrame(columns=["price", "volume"])
        _write_cached_day(cache_file, empty, stats)
        return empty, stats

    # Timestamp strings are second-granular in these files and heavily duplicated.
    # Grouping before datetime parsing avoids parsing millions of duplicate strings.
    time_key = raw["time"].str.slice(0, 19).str.replace("T", " ", regex=False)
    by_second = raw.groupby(time_key, sort=True).agg(price=("price", "last"), volume=("volume", "sum")).astype(float)
    parsed_index = pd.to_datetime(by_second.index, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    by_second.index = parsed_index
    by_second = by_second[by_second.index.notna()].sort_index()
    if by_second.empty:
        empty = pd.DataFrame(columns=["price", "volume"])
        _write_cached_day(cache_file, empty, stats)
        return empty, stats

    idx = pd.date_range(by_second.index.min(), by_second.index.max(), freq="1s")
    second_bars = by_second.reindex(idx)
    second_bars["price"] = second_bars["price"].ffill()
    second_bars["volume"] = second_bars["volume"].fillna(0.0)

    stats.update(
        {
            "seconds": int(second_bars.shape[0]),
            "first_time": second_bars.index.min(),
            "last_time": second_bars.index.max(),
            "first_price": float(second_bars["price"].iloc[0]),
            "last_price": float(second_bars["price"].iloc[-1]),
            "total_volume": float(second_bars["volume"].sum()),
        }
    )
    _write_cached_day(cache_file, second_bars, stats)
    return second_bars, stats


def load_second_prices(
    data_dir: str | Path,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    cache_dir: str | Path | None = None,
    workers: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = discover_csv_files(data_dir)
    if manifest.empty:
        return pd.DataFrame(columns=["price", "volume"]), pd.DataFrame()

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
        return pd.DataFrame(columns=["price", "volume"]), pd.DataFrame(stats_rows)

    bars = pd.concat(series).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    return bars[["price", "volume"]], pd.DataFrame(stats_rows)


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


def spike_parameter_grid(
    delays: Iterable[int],
    thresholds: Iterable[float],
    lookbacks: Iterable[int],
    holds: Iterable[int],
    volume_windows: Iterable[int],
    volume_multiples: Iterable[float],
    direction: str = "momentum",
) -> list[SpikeParams]:
    return [
        SpikeParams(int(a), float(b), int(c), int(d), int(vw), float(vm), direction)
        for a, b, c, d, vw, vm in itertools.product(
            delays, thresholds, lookbacks, holds, volume_windows, volume_multiples
        )
        if int(a) >= 0 and float(b) > 0 and int(c) > 0 and int(d) > 0 and int(vw) > 0 and float(vm) >= 0
    ]


def _price_series(data: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        return data["price"]
    return data


def prepare_day_arrays(price: pd.Series | pd.DataFrame) -> list[dict]:
    price = _price_series(price).dropna().sort_index()
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


def prepare_spike_day_arrays(bars: pd.DataFrame) -> list[dict]:
    if isinstance(bars, pd.Series):
        bars = bars.to_frame("price")
        bars["volume"] = 0.0
    bars = bars.dropna(subset=["price"]).sort_index()
    if "volume" not in bars.columns:
        bars = bars.copy()
        bars["volume"] = 0.0
    days = []
    for date, day_bars in bars.groupby(bars.index.normalize(), sort=True):
        dubai_index = day_bars.index + pd.Timedelta(hours=DUBAI_UTC_OFFSET_HOURS)
        days.append(
            {
                "date": pd.Timestamp(date),
                "times": day_bars.index.to_numpy(),
                "dubai_seconds": (
                    dubai_index.hour.to_numpy() * 3600
                    + dubai_index.minute.to_numpy() * 60
                    + dubai_index.second.to_numpy()
                ),
                "prices": day_bars["price"].to_numpy(dtype=float),
                "volume": day_bars["volume"].fillna(0.0).to_numpy(dtype=float),
            }
        )
    return days


def _scan_day(
    day: dict,
    params: StrategyParams,
    cost_cents: float,
    collect_trades: bool,
    max_trades_per_day: int | None = None,
) -> tuple[list[dict], list[float], list[int]]:
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
        if max_trades_per_day is not None and len(pnls) >= max_trades_per_day:
            break
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


def _scan_spike_day(
    day: dict,
    params: SpikeParams,
    cost_cents: float | None,
    collect_trades: bool,
    max_trades_per_day: int | None = 1,
    session_start_hour: int | None = None,
    session_end_hour: int | None = None,
    session_tz_offset_hours: int = DUBAI_UTC_OFFSET_HOURS,
) -> tuple[list[dict], list[float], list[int]]:
    prices = day["prices"]
    volume = day["volume"]
    times = day["times"]
    dubai_seconds = day.get("dubai_seconds")
    n = len(prices)
    min_required = params.lookback_s + params.delay_s + params.hold_s + 1
    if n <= min_required:
        return [], [], []

    delta = prices[params.lookback_s :] - prices[: -params.lookback_s]
    raw_idx = np.flatnonzero(np.abs(delta) >= params.threshold_cents / 100.0) + params.lookback_s
    if raw_idx.size == 0:
        return [], [], []

    if params.volume_multiple > 0:
        recent_vol = _rolling_sum_min1(volume, params.lookback_s)
        baseline_vol = _rolling_mean(volume, params.volume_window_s, max(1, min(params.volume_window_s, 30)))
        required = params.volume_multiple * baseline_vol * max(params.lookback_s, 1)
        raw_idx = raw_idx[recent_vol[raw_idx] >= required[raw_idx]]
        if raw_idx.size == 0:
            return [], [], []
    else:
        recent_vol = _rolling_sum_min1(volume, params.lookback_s) if collect_trades else None

    if session_start_hour is not None and session_end_hour is not None:
        if dubai_seconds is not None and int(session_tz_offset_hours) == DUBAI_UTC_OFFSET_HOURS:
            seconds = dubai_seconds[raw_idx]
        else:
            session_times = pd.DatetimeIndex(times[raw_idx]) + pd.Timedelta(hours=int(session_tz_offset_hours))
            seconds = session_times.hour * 3600 + session_times.minute * 60 + session_times.second
        start_seconds = int(session_start_hour) * 3600
        end_hour = int(session_end_hour)
        end_seconds = 24 * 3600 if end_hour >= 24 else end_hour * 3600
        if start_seconds < end_seconds:
            keep = (seconds >= start_seconds) & (seconds < end_seconds)
        else:
            keep = (seconds >= start_seconds) | (seconds < end_seconds)
        raw_idx = raw_idx[keep]
        if raw_idx.size == 0:
            return [], [], []

    trades = []
    pnls = []
    sides = []
    last_exit = -1
    i = 0
    while i < raw_idx.size:
        if max_trades_per_day is not None and len(pnls) >= max_trades_per_day:
            break
        event_idx = int(raw_idx[i])
        entry_idx = event_idx + params.delay_s
        exit_idx = entry_idx + params.hold_s
        if exit_idx >= n:
            break
        signal_time = pd.Timestamp(times[event_idx])
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
        trade_cost_cents = _resolve_trade_cost_cents(cost_cents, entry_price)
        net_cents = gross_cents - trade_cost_cents
        pnls.append(float(net_cents))
        sides.append(side)
        if collect_trades:
            entry_time = pd.Timestamp(times[entry_idx])
            exit_time = pd.Timestamp(times[exit_idx])
            trades.append(
                {
                    "signal_time": signal_time,
                    "signal_time_dubai": _to_session_time(signal_time, session_tz_offset_hours),
                    "entry_time": entry_time,
                    "entry_time_dubai": _to_session_time(entry_time, session_tz_offset_hours),
                    "exit_time": exit_time,
                    "exit_time_dubai": _to_session_time(exit_time, session_tz_offset_hours),
                    "session_bucket": _session_bucket(signal_time, session_start_hour, session_end_hour, session_tz_offset_hours),
                    "side": side,
                    "move_cents": move * 100.0,
                    "signal_volume": float(recent_vol[event_idx]) if recent_vol is not None else 0.0,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl_cents": gross_cents,
                    "cost_cents": trade_cost_cents,
                    "net_pnl_cents": net_cents,
                }
            )
        last_exit = exit_idx
        i = int(np.searchsorted(raw_idx, last_exit - params.delay_s + 1, side="left"))

    return trades, pnls, sides


def _to_session_time(ts: pd.Timestamp, tz_offset_hours: int) -> pd.Timestamp:
    return pd.Timestamp(ts) + pd.Timedelta(hours=int(tz_offset_hours))


def _rolling_sum_min1(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = max(int(window), 1)
    cumsum = np.cumsum(arr)
    out = cumsum.copy()
    if window < arr.size:
        out[window:] = cumsum[window:] - cumsum[:-window]
    return out


def _rolling_mean(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = max(int(window), 1)
    min_periods = max(int(min_periods), 1)
    sums = _rolling_sum_min1(arr, window)
    counts = np.minimum(np.arange(1, arr.size + 1), window).astype(float)
    out = sums / counts
    out[counts < min_periods] = np.nan
    return out


def _is_in_session(
    ts: pd.Timestamp,
    session_start_hour: int | None,
    session_end_hour: int | None,
    session_tz_offset_hours: int,
) -> bool:
    if session_start_hour is None or session_end_hour is None:
        return True
    session_ts = _to_session_time(pd.Timestamp(ts), session_tz_offset_hours)
    seconds = session_ts.hour * 3600 + session_ts.minute * 60 + session_ts.second
    start_seconds = int(session_start_hour) * 3600
    end_hour = int(session_end_hour)
    end_seconds = 24 * 3600 if end_hour >= 24 else end_hour * 3600
    if start_seconds < end_seconds:
        return start_seconds <= seconds < end_seconds
    return seconds >= start_seconds or seconds < end_seconds


def _session_bucket(
    ts: pd.Timestamp,
    session_start_hour: int | None,
    session_end_hour: int | None,
    session_tz_offset_hours: int,
) -> str:
    if session_start_hour is None or session_end_hour is None:
        return "Unrestricted"
    return (
        f"Inside {session_start_hour:02d}:00-24:00 Dubai"
        if _is_in_session(ts, session_start_hour, session_end_hour, session_tz_offset_hours)
        else f"Outside {session_start_hour:02d}:00-24:00 Dubai"
    )


def _resolve_trade_cost_cents(cost_cents: float | None, entry_price: float) -> float:
    if cost_cents is None:
        return float(abs(entry_price) * DYNAMIC_COST_CENTS_PER_PRICE_UNIT)
    return float(cost_cents)


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
    max_trades_per_day: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    if not days:
        return pd.DataFrame(), pd.Series(dtype=float, name="daily_pnl_cents"), _trade_summary([], [])

    all_trades: list[dict] = []
    all_pnls: list[float] = []
    all_sides: list[int] = []
    daily_values = []
    daily_index = []

    for day in days:
        trades, pnls, sides = _scan_day(day, params, cost_cents, collect_trades, max_trades_per_day)
        if collect_trades and trades:
            all_trades.extend(trades)
        all_pnls.extend(pnls)
        all_sides.extend(sides)
        daily_index.append(day["date"])
        daily_values.append(float(np.sum(pnls)) if pnls else 0.0)

    daily = pd.Series(daily_values, index=pd.Index(daily_index, name="date"), name="daily_pnl_cents")
    trades_df = pd.DataFrame(all_trades) if collect_trades and all_trades else pd.DataFrame()
    return trades_df, daily, _trade_summary(all_pnls, all_sides)


def _backtest_spike_prepared(
    days: list[dict],
    params: SpikeParams,
    cost_cents: float | None,
    collect_trades: bool,
    max_trades_per_day: int | None = 1,
    session_start_hour: int | None = None,
    session_end_hour: int | None = None,
    session_tz_offset_hours: int = DUBAI_UTC_OFFSET_HOURS,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    if not days:
        return pd.DataFrame(), pd.Series(dtype=float, name="daily_pnl_cents"), _trade_summary([], [])

    all_trades: list[dict] = []
    all_pnls: list[float] = []
    all_sides: list[int] = []
    daily_values = []
    daily_index = []

    for day in days:
        trades, pnls, sides = _scan_spike_day(
            day,
            params,
            cost_cents,
            collect_trades,
            max_trades_per_day,
            session_start_hour,
            session_end_hour,
            session_tz_offset_hours,
        )
        if collect_trades and trades:
            all_trades.extend(trades)
        all_pnls.extend(pnls)
        all_sides.extend(sides)
        daily_index.append(day["date"])
        daily_values.append(float(np.sum(pnls)) if pnls else 0.0)

    daily = pd.Series(daily_values, index=pd.Index(daily_index, name="date"), name="daily_pnl_cents")
    trades_df = pd.DataFrame(all_trades) if collect_trades and all_trades else pd.DataFrame()
    return trades_df, daily, _trade_summary(all_pnls, all_sides)


def backtest_spike_strategy(
    bars: pd.DataFrame | pd.Series,
    params: SpikeParams,
    cost_cents: float | None = 0.0,
    max_trades_per_day: int | None = 1,
    session_start_hour: int | None = None,
    session_end_hour: int | None = None,
    session_tz_offset_hours: int = DUBAI_UTC_OFFSET_HOURS,
) -> tuple[pd.DataFrame, pd.Series]:
    days = prepare_spike_day_arrays(bars)
    trades, daily, _ = _backtest_spike_prepared(
        days,
        params,
        cost_cents,
        True,
        max_trades_per_day,
        session_start_hour,
        session_end_hour,
        session_tz_offset_hours,
    )
    return trades, daily


def evaluate_spike_grid(
    bars: pd.DataFrame | pd.Series,
    params_list: list[SpikeParams],
    objective: str,
    cost_cents: float | None,
    min_trades: int,
    max_trades_per_day: int | None = 1,
    session_start_hour: int | None = None,
    session_end_hour: int | None = None,
    session_tz_offset_hours: int = DUBAI_UTC_OFFSET_HOURS,
) -> pd.DataFrame:
    rows = []
    days = prepare_spike_day_arrays(bars)
    for params in params_list:
        _, daily, summary = _backtest_spike_prepared(
            days,
            params,
            cost_cents,
            False,
            max_trades_per_day,
            session_start_hour,
            session_end_hour,
            session_tz_offset_hours,
        )
        metrics = compute_metrics(daily, trade_summary=summary)
        score = -np.inf if metrics["trades"] < min_trades else float(metrics.get(objective, metrics["total_pnl_cents"]))
        rows.append(
            {
                "params": params,
                "label": params.label,
                "delay_s": params.delay_s,
                "threshold_cents": params.threshold_cents,
                "lookback_s": params.lookback_s,
                "hold_s": params.hold_s,
                "volume_window_s": params.volume_window_s,
                "volume_multiple": params.volume_multiple,
                "direction": params.direction,
                "score": score,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def backtest_strategy(
    price: pd.Series | pd.DataFrame,
    params: StrategyParams,
    cost_cents: float = 0.0,
    max_trades_per_day: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    days = prepare_day_arrays(price)
    trades, daily, _ = _backtest_prepared(days, params, cost_cents, collect_trades=True, max_trades_per_day=max_trades_per_day)
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
        "days": int(len(daily)),
    }

    if trade_summary is not None:
        row.update(trade_summary)
        row["trades_per_day"] = row["trades"] / max(row["days"], 1)
        return row

    if trades is None or trades.empty:
        row.update(_trade_summary([], []))
        row["trades_per_day"] = 0.0
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
    row["trades_per_day"] = row["trades"] / max(row["days"], 1)
    return row


def evaluate_grid(
    price: pd.Series | pd.DataFrame,
    params_list: list[StrategyParams],
    objective: str,
    cost_cents: float,
    min_trades: int,
    max_trades_per_day: int | None = None,
) -> pd.DataFrame:
    rows = []
    days = prepare_day_arrays(price)
    for params in params_list:
        _, daily, summary = _backtest_prepared(
            days,
            params,
            cost_cents,
            collect_trades=False,
            max_trades_per_day=max_trades_per_day,
        )
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


def move_frequency_grid(
    price: pd.Series | pd.DataFrame,
    lookbacks: Iterable[int],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    days = prepare_day_arrays(price)
    rows = []
    for lookback_s in sorted(set(int(x) for x in lookbacks if int(x) > 0)):
        moves = []
        for day in days:
            prices = day["prices"]
            if len(prices) <= lookback_s:
                continue
            delta_cents = np.abs(prices[lookback_s:] - prices[:-lookback_s]) * 100.0
            if delta_cents.size:
                moves.append(delta_cents)
        if not moves:
            continue

        all_moves = np.concatenate(moves)
        window_count = int(all_moves.size)
        trading_days = max(len(days), 1)
        quantiles = {
            "p50_cents": float(np.quantile(all_moves, 0.50)),
            "p90_cents": float(np.quantile(all_moves, 0.90)),
            "p95_cents": float(np.quantile(all_moves, 0.95)),
            "p98_cents": float(np.quantile(all_moves, 0.98)),
            "p99_cents": float(np.quantile(all_moves, 0.99)),
            "p995_cents": float(np.quantile(all_moves, 0.995)),
            "max_cents": float(np.max(all_moves)),
            "avg_abs_move_cents": float(np.mean(all_moves)),
        }
        for threshold_cents in sorted(set(float(x) for x in thresholds if float(x) > 0)):
            event_count = int((all_moves >= threshold_cents).sum())
            rows.append(
                {
                    "lookback_s": lookback_s,
                    "threshold_cents": threshold_cents,
                    "windows": window_count,
                    "event_windows": event_count,
                    "event_window_pct": event_count / window_count if window_count else 0.0,
                    "raw_event_windows_per_day": event_count / trading_days,
                    **quantiles,
                }
            )
    return pd.DataFrame(rows)


def run_walk_forward(
    price: pd.Series | pd.DataFrame,
    params_list: list[StrategyParams],
    train_months: int = 3,
    test_months: int = 1,
    objective: str = "daily_sharpe",
    cost_cents: float = 0.0,
    min_train_trades: int = 20,
    max_trades_per_day: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    price = price.dropna(subset=["price"]).sort_index() if isinstance(price, pd.DataFrame) else price.dropna().sort_index()
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

        grid = evaluate_grid(train_price, params_list, objective, cost_cents, min_train_trades, max_trades_per_day)
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
        test_trades, test_daily = backtest_strategy(test_price, best_params, cost_cents, max_trades_per_day)
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


def run_spike_walk_forward(
    bars: pd.DataFrame | pd.Series,
    params_list: list[SpikeParams],
    train_months: int = 3,
    test_months: int = 1,
    objective: str = "total_pnl_cents",
    cost_cents: float = 0.0,
    min_train_trades: int = 1,
    max_trades_per_day: int | None = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    if isinstance(bars, pd.Series):
        bars = bars.to_frame("price")
        bars["volume"] = 0.0
    bars = bars.dropna(subset=["price"]).sort_index()
    if bars.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    month_key = bars.index.to_period("M")
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

        train_bars = bars[month_key.isin(train_periods)]
        test_bars = bars[month_key.isin(test_periods)]
        if train_bars.empty or test_bars.empty:
            continue

        grid = evaluate_spike_grid(train_bars, params_list, objective, cost_cents, min_train_trades, max_trades_per_day)
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
        test_trades, test_daily = backtest_spike_strategy(test_bars, best_params, cost_cents, max_trades_per_day)
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
                "test_trades_per_day": test_metrics["trades_per_day"],
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
    price = _price_series(price)
    if price.empty:
        return pd.Series(dtype=float, name="close")
    close = price.groupby(price.index.normalize()).last()
    close.name = "close"
    return close
