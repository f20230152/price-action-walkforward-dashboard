from __future__ import annotations

import itertools
import re
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
RAW_ROOT = Path(r"D:\Energin Raw Data")
OUT_DIR = BASE_DIR / "outputs" / "expanded_walkforward"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_MONTHS = list(range(1, 12))
CSV_START = pd.Timestamp("2025-12-01")
CSV_END = pd.Timestamp("2026-03-31")
WALKFORWARD_TEST_START = pd.Timestamp("2025-04-01")
WALKFORWARD_TEST_END = pd.Timestamp("2026-03-31")
TRAIN_MONTHS = 3
VOL_WINDOW_DAYS = 63
VOL_MIN_OBSERVATIONS = 21
SESSION_START_HOUR_DUBAI = 11
SESSION_END_HOUR_DUBAI = 24
MAX_TRADES_PER_DAY = 1
MIN_TRAIN_TRADES = 5
REQUIRE_FULL_SIGNAL_WINDOW_IN_SESSION = True

DELAYS = [60, 90, 120, 180, 300]
SIGMA_MULTIPLES = [round(float(x), 2) for x in np.arange(0.25, 1.051, 0.05)]
MOVE_WINDOWS = [900, 1200, 1800, 2400, 3600]
HOLDS = [1800, 3600, 7200, 14400, 21600]
SIGNAL_MODES = ["continuation", "reversal"]
FILL_MODELS = ["mid_dynamic_cost", "actual_bid_ask"]

PARQUET_COLUMNS = [
    "archive_date",
    "symbol_ticker",
    "second_utc",
    "mid",
    "bid_last",
    "ask_last",
    "spread",
    "trade_volume",
]


@dataclass(frozen=True)
class ExpandedParams:
    delay_s: int
    sigma_multiple: float
    move_window_s: int
    hold_s: int
    signal_mode: str
    vol_window_days: int = VOL_WINDOW_DAYS
    bad_hour_rule: str = "exit_by_midnight"

    @property
    def label(self) -> str:
        return (
            f"a={self.delay_s}s | b={self.sigma_multiple:g}x rolling sigma | "
            f"c={self.move_window_s}s | d={self.hold_s}s | {self.signal_mode} | exit_by_midnight"
        )


def build_universe() -> list[ExpandedParams]:
    return [
        ExpandedParams(int(delay), float(multiple), int(window), int(hold), str(mode))
        for delay, multiple, window, hold, mode in itertools.product(
            DELAYS,
            SIGMA_MULTIPLES,
            MOVE_WINDOWS,
            HOLDS,
            SIGNAL_MODES,
        )
    ]


def parquet_month_dir(month: int) -> Path:
    return RAW_ROOT / f"brent_local_2025_{month:02d}"


def read_symbol_master(month_dir: Path) -> pd.DataFrame:
    path = month_dir / "reference" / "brent_symbol_master.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["symbol_ticker", "expiration_date"])
    master = pd.read_parquet(path)
    master["expiration_ts"] = pd.to_datetime(master.get("expiration_date"), errors="coerce")
    return master


def date_from_parquet_name(path: Path) -> pd.Timestamp | None:
    match = re.search(r"(\d{8})", path.stem)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def select_front_symbol(
    archive_date: pd.Timestamp,
    available_symbols: pd.Series,
    volume_by_symbol: pd.Series,
    symbol_master: pd.DataFrame,
) -> tuple[str | None, str]:
    available = sorted(set(available_symbols.dropna().astype(str)))
    if not available:
        return None, "no_symbols"

    if not symbol_master.empty:
        master = symbol_master[symbol_master["symbol_ticker"].isin(available)].copy()
        master = master[master["expiration_ts"].notna()]
        master = master[master["expiration_ts"] >= archive_date.normalize()]
        if not master.empty:
            master["daily_volume"] = master["symbol_ticker"].map(volume_by_symbol).fillna(0.0).astype(float)
            master = master.sort_values(["expiration_ts", "daily_volume", "symbol_ticker"], ascending=[True, False, True])
            return str(master.iloc[0]["symbol_ticker"]), "nearest_unexpired"

    if not volume_by_symbol.empty:
        volume = volume_by_symbol.reindex(available).fillna(0.0).astype(float)
        if volume.max() > 0:
            return str(volume.sort_values(ascending=False).index[0]), "highest_volume_fallback"

    return available[0], "alphabetical_fallback"


def load_parquet_day(path: Path, symbol_master: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {
        "date": pd.NaT,
        "source_type": "parquet_front_bid_ask",
        "source_file": str(path),
        "selected_symbol": "",
        "selection_rule": "",
        "raw_rows": 0,
        "selected_rows": 0,
        "seconds": 0,
        "first_time": pd.NaT,
        "last_time": pd.NaT,
        "total_volume": 0.0,
    }
    raw = pd.read_parquet(path, columns=PARQUET_COLUMNS)
    stats["raw_rows"] = int(len(raw))
    if raw.empty:
        guessed_date = date_from_parquet_name(path)
        stats["date"] = guessed_date.date() if guessed_date is not None and pd.notna(guessed_date) else pd.NaT
        return pd.DataFrame(), stats

    archive_date = pd.to_datetime(raw["archive_date"].dropna().iloc[0], errors="coerce")
    if pd.isna(archive_date):
        archive_date = date_from_parquet_name(path)
    stats["date"] = archive_date.date() if archive_date is not None and pd.notna(archive_date) else pd.NaT

    raw["trade_volume"] = pd.to_numeric(raw["trade_volume"], errors="coerce").fillna(0.0)
    volume_by_symbol = raw.groupby("symbol_ticker", dropna=True)["trade_volume"].sum()
    selected_symbol, selection_rule = select_front_symbol(
        pd.Timestamp(archive_date),
        raw["symbol_ticker"],
        volume_by_symbol,
        symbol_master,
    )
    stats["selected_symbol"] = selected_symbol or ""
    stats["selection_rule"] = selection_rule
    if selected_symbol is None:
        return pd.DataFrame(), stats

    day = raw[raw["symbol_ticker"].astype(str).eq(selected_symbol)].copy()
    stats["selected_rows"] = int(len(day))
    if day.empty:
        return pd.DataFrame(), stats

    day["second_utc"] = pd.to_datetime(day["second_utc"], utc=True, errors="coerce").dt.floor("s").dt.tz_localize(None)
    for col in ["mid", "bid_last", "ask_last", "spread", "trade_volume"]:
        day[col] = pd.to_numeric(day[col], errors="coerce")
    day = day.dropna(subset=["second_utc"]).sort_values("second_utc")
    if day.empty:
        return pd.DataFrame(), stats

    grouped = (
        day.groupby("second_utc", sort=True)
        .agg(
            price=("mid", "last"),
            bid=("bid_last", "last"),
            ask=("ask_last", "last"),
            spread=("spread", "last"),
            volume=("trade_volume", "sum"),
        )
        .astype(float)
    )
    valid_price = grouped["price"].notna()
    if not valid_price.any():
        return pd.DataFrame(), stats

    first_valid = grouped.index[valid_price][0]
    last_valid = grouped.index[valid_price][-1]
    grouped = grouped.loc[first_valid:last_valid]
    full_index = pd.date_range(first_valid, last_valid, freq="1s")
    bars = grouped.reindex(full_index)
    bars[["price", "bid", "ask", "spread"]] = bars[["price", "bid", "ask", "spread"]].ffill()
    bars["volume"] = bars["volume"].fillna(0.0)
    bars = bars.dropna(subset=["price"])
    bars["source_type"] = "parquet_front_bid_ask"
    bars["has_actual_bid_ask"] = bars["bid"].notna() & bars["ask"].notna()
    bars["selected_symbol"] = selected_symbol

    stats.update(
        {
            "seconds": int(len(bars)),
            "first_time": bars.index.min(),
            "last_time": bars.index.max(),
            "total_volume": float(bars["volume"].sum()),
        }
    )
    return bars, stats


def load_parquet_months() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_bars: list[pd.DataFrame] = []
    stats_rows: list[dict] = []
    for month in PARQUET_MONTHS:
        month_dir = parquet_month_dir(month)
        bars_root = month_dir / "parquet" / "brent_1s_bars"
        symbol_master = read_symbol_master(month_dir)
        files = sorted(bars_root.rglob("*.parquet")) if bars_root.exists() else []
        print(f"loading parquet month 2025-{month:02d}: {len(files)} files", flush=True)
        for idx, path in enumerate(files, 1):
            if idx % 10 == 0:
                print(f"  parquet 2025-{month:02d}: {idx}/{len(files)}", flush=True)
            bars, stats = load_parquet_day(path, symbol_master)
            stats_rows.append(stats)
            if not bars.empty:
                all_bars.append(bars)

    if not all_bars:
        return pd.DataFrame(), pd.DataFrame(stats_rows)
    bars = pd.concat(all_bars).sort_index()
    return bars, pd.DataFrame(stats_rows)


def load_csv_months() -> tuple[pd.DataFrame, pd.DataFrame]:
    bars, stats = load_second_prices(
        DATA_DIR,
        start_date=CSV_START,
        end_date=CSV_END,
        cache_dir=BASE_DIR / ".price_cache",
        workers=8,
    )
    if bars.empty:
        return bars, stats
    bars = bars.copy()
    bars["bid"] = np.nan
    bars["ask"] = np.nan
    bars["spread"] = np.nan
    bars["source_type"] = "csv_mid_dynamic_cost"
    bars["has_actual_bid_ask"] = False
    bars["selected_symbol"] = "%BRN 1!-ICE"
    stats = stats.copy()
    stats["source_type"] = "csv_mid_dynamic_cost"
    stats["source_file"] = stats.get("file", "")
    stats["selected_symbol"] = "%BRN 1!-ICE"
    stats["selection_rule"] = "legacy_csv_continuous"
    stats["selected_rows"] = stats.get("raw_rows", 0)
    return bars, stats


def stitch_research_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    parquet_bars, parquet_stats = load_parquet_months()
    csv_bars, csv_stats = load_csv_months()

    frames = []
    if not parquet_bars.empty:
        frames.append(parquet_bars)
    if not csv_bars.empty:
        frames.append(csv_bars)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    combined = pd.concat(frames).sort_index()
    duplicate_count = int(combined.index.duplicated().sum())
    combined = combined[~combined.index.duplicated(keep="last")]
    stats = pd.concat([parquet_stats, csv_stats], ignore_index=True)
    coverage = build_monthly_coverage(combined, stats, duplicate_count)
    return combined, stats, coverage, duplicate_count


def build_monthly_coverage(bars: pd.DataFrame, day_stats: pd.DataFrame, duplicate_count: int) -> pd.DataFrame:
    if day_stats.empty:
        return pd.DataFrame()
    stats = day_stats.copy()
    stats["date"] = pd.to_datetime(stats["date"], errors="coerce")
    stats["month"] = stats["date"].dt.to_period("M").astype(str)
    stats["seconds"] = pd.to_numeric(stats["seconds"], errors="coerce").fillna(0).astype(int)
    stats["loaded"] = stats["seconds"] > 0

    rows = []
    for month, group in stats.groupby("month", dropna=True):
        month_start = pd.Period(month).to_timestamp()
        month_end = month_start + pd.offsets.MonthEnd(0)
        expected_business = pd.bdate_range(month_start, month_end)
        loaded_dates = set(group.loc[group["loaded"], "date"].dt.normalize())
        missing_business = [d.date().isoformat() for d in expected_business if d.normalize() not in loaded_dates]
        month_bars = bars[bars.index.to_period("M").astype(str) == month] if not bars.empty else pd.DataFrame()
        source_types = sorted(group["source_type"].dropna().astype(str).unique().tolist())
        rows.append(
            {
                "month": month,
                "source_type": ", ".join(source_types),
                "source_files": int(len(group)),
                "loaded_days": int(group["loaded"].sum()),
                "empty_source_files": int((~group["loaded"]).sum()),
                "expected_business_days": int(len(expected_business)),
                "missing_business_days": int(len(missing_business)),
                "missing_business_dates": ", ".join(missing_business),
                "first_timestamp": month_bars.index.min() if not month_bars.empty else pd.NaT,
                "last_timestamp": month_bars.index.max() if not month_bars.empty else pd.NaT,
                "row_count": int(len(month_bars)),
                "duplicate_timestamps_removed_total": duplicate_count,
            }
        )
    return pd.DataFrame(rows)


def add_volatility_columns(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    daily_close = out["price"].groupby(out.index.normalize()).last().sort_index()
    pct_change = daily_close.pct_change()
    sigma = pct_change.rolling(VOL_WINDOW_DAYS, min_periods=VOL_MIN_OBSERVATIONS).std().shift(1) * daily_close.shift(1)
    trend = daily_close.pct_change(5).shift(1)
    vol_rank = sigma.rolling(VOL_WINDOW_DAYS, min_periods=VOL_MIN_OBSERVATIONS).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )
    date_index = out.index.normalize()
    out["sigma_percent_to_dollar_63"] = date_index.map(sigma).astype(float)
    out["trend_5d_pct_63"] = date_index.map(trend).astype(float)
    out["vol_rank_63"] = date_index.map(vol_rank).astype(float)
    return out


def day_scalar(day_bars: pd.DataFrame, col: str) -> float:
    values = day_bars[col].dropna() if col in day_bars.columns else pd.Series(dtype=float)
    return float(values.iloc[0]) if not values.empty else np.nan


def prepare_days(bars: pd.DataFrame) -> list[dict]:
    days = []
    clean = bars.dropna(subset=["price"]).sort_index()
    for date, day_bars in clean.groupby(clean.index.normalize(), sort=True):
        dubai_index = day_bars.index + pd.Timedelta(hours=DUBAI_UTC_OFFSET_HOURS)
        source_types = sorted(day_bars["source_type"].dropna().astype(str).unique().tolist())
        selected_symbols = sorted(day_bars["selected_symbol"].dropna().astype(str).unique().tolist())
        days.append(
            {
                "date": pd.Timestamp(date),
                "times": day_bars.index.to_numpy(),
                "prices": day_bars["price"].to_numpy(dtype=float),
                "bid": day_bars["bid"].to_numpy(dtype=float),
                "ask": day_bars["ask"].to_numpy(dtype=float),
                "spread": day_bars["spread"].to_numpy(dtype=float),
                "has_actual_bid_ask": day_bars["has_actual_bid_ask"].fillna(False).to_numpy(dtype=bool),
                "volume": day_bars["volume"].fillna(0.0).to_numpy(dtype=float),
                "dubai_times": dubai_index,
                "dubai_seconds": (
                    dubai_index.hour.to_numpy() * 3600
                    + dubai_index.minute.to_numpy() * 60
                    + dubai_index.second.to_numpy()
                ),
                "sigma_percent_to_dollar_63": day_scalar(day_bars, "sigma_percent_to_dollar_63"),
                "trend_5d_pct_63": day_scalar(day_bars, "trend_5d_pct_63"),
                "vol_rank_63": day_scalar(day_bars, "vol_rank_63"),
                "source_type": ", ".join(source_types),
                "selected_symbol": ", ".join(selected_symbols),
            }
        )
    return days


def forced_midnight_exit_idx(times: np.ndarray, dubai_times: pd.DatetimeIndex, entry_idx: int) -> int:
    entry_dubai = pd.Timestamp(dubai_times[entry_idx])
    midnight_dubai = pd.Timestamp(entry_dubai.date()) + pd.Timedelta(days=1)
    midnight_utc = midnight_dubai - pd.Timedelta(hours=DUBAI_UTC_OFFSET_HOURS)
    return int(np.searchsorted(times, np.datetime64(midnight_utc.to_datetime64()), side="right") - 1)


def resolve_fill(day: dict, entry_idx: int, exit_idx: int, side: int, fill_model: str) -> dict:
    entry_mid = float(day["prices"][entry_idx])
    exit_mid = float(day["prices"][exit_idx])
    mid_gross = float(side * (exit_mid - entry_mid) * 100.0)
    dynamic_cost = float(abs(entry_mid) * DYNAMIC_COST_CENTS_PER_PRICE_UNIT)

    entry_bid = float(day["bid"][entry_idx]) if np.isfinite(day["bid"][entry_idx]) else np.nan
    entry_ask = float(day["ask"][entry_idx]) if np.isfinite(day["ask"][entry_idx]) else np.nan
    exit_bid = float(day["bid"][exit_idx]) if np.isfinite(day["bid"][exit_idx]) else np.nan
    exit_ask = float(day["ask"][exit_idx]) if np.isfinite(day["ask"][exit_idx]) else np.nan
    has_actual = bool(day["has_actual_bid_ask"][entry_idx] and day["has_actual_bid_ask"][exit_idx])

    if fill_model == "actual_bid_ask" and has_actual:
        entry_fill = entry_ask if side > 0 else entry_bid
        exit_fill = exit_bid if side > 0 else exit_ask
        if np.isfinite(entry_fill) and np.isfinite(exit_fill) and entry_fill > 0 and exit_fill > 0:
            actual_net = float(side * (exit_fill - entry_fill) * 100.0)
            return {
                "entry_price": float(entry_fill),
                "exit_price": float(exit_fill),
                "entry_mid": entry_mid,
                "exit_mid": exit_mid,
                "entry_bid": entry_bid,
                "entry_ask": entry_ask,
                "exit_bid": exit_bid,
                "exit_ask": exit_ask,
                "gross_pnl_cents": mid_gross,
                "cost_cents": float(mid_gross - actual_net),
                "net_pnl_cents": actual_net,
                "fill_model": fill_model,
                "fill_source": "actual_bid_ask",
            }

    return {
        "entry_price": entry_mid,
        "exit_price": exit_mid,
        "entry_mid": entry_mid,
        "exit_mid": exit_mid,
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "exit_bid": exit_bid,
        "exit_ask": exit_ask,
        "gross_pnl_cents": mid_gross,
        "cost_cents": dynamic_cost,
        "net_pnl_cents": float(mid_gross - dynamic_cost),
        "fill_model": fill_model,
        "fill_source": "synthetic_dynamic_cost",
    }


def scan_day(day: dict, params: ExpandedParams, fill_model: str, collect_trades: bool) -> tuple[list[dict], list[float], list[int]]:
    sigma = day.get("sigma_percent_to_dollar_63", np.nan)
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

    threshold_cents = float(sigma * 100.0 * params.sigma_multiple)
    delta = prices[params.move_window_s :] - prices[: -params.move_window_s]
    raw_idx = np.flatnonzero(np.abs(delta) >= threshold_cents / 100.0) + params.move_window_s
    if raw_idx.size == 0:
        return [], [], []

    start_seconds = SESSION_START_HOUR_DUBAI * 3600
    end_seconds = SESSION_END_HOUR_DUBAI * 3600
    keep = (dubai_seconds[raw_idx] >= start_seconds) & (dubai_seconds[raw_idx] < end_seconds)
    if REQUIRE_FULL_SIGNAL_WINDOW_IN_SESSION:
        keep &= dubai_seconds[raw_idx - params.move_window_s] >= start_seconds
    raw_idx = raw_idx[keep]
    if raw_idx.size == 0:
        return [], [], []

    trades: list[dict] = []
    pnls: list[float] = []
    sides: list[int] = []
    last_exit = -1
    i = 0

    while i < raw_idx.size:
        if len(pnls) >= MAX_TRADES_PER_DAY:
            break
        event_idx = int(raw_idx[i])
        entry_idx = event_idx + params.delay_s
        if entry_idx >= n:
            break
        if entry_idx <= last_exit:
            i = int(np.searchsorted(raw_idx, last_exit - params.delay_s + 1, side="left"))
            continue

        midnight_idx = forced_midnight_exit_idx(times, dubai_times, entry_idx)
        natural_exit_idx = entry_idx + params.hold_s
        exit_idx = min(natural_exit_idx, midnight_idx, n - 1)
        if exit_idx <= entry_idx:
            i += 1
            continue

        move = float(prices[event_idx] - prices[event_idx - params.move_window_s])
        side = 1 if move > 0 else -1
        if params.signal_mode == "reversal":
            side *= -1

        fill = resolve_fill(day, entry_idx, exit_idx, side, fill_model)
        pnls.append(float(fill["net_pnl_cents"]))
        sides.append(side)

        if collect_trades:
            entry_time = pd.Timestamp(times[entry_idx])
            exit_time = pd.Timestamp(times[exit_idx])
            actual_hold_s = (exit_time - entry_time).total_seconds()
            exit_reason = "midnight_stop" if midnight_idx <= natural_exit_idx else "holding_time"
            trades.append(
                {
                    "signal_time": pd.Timestamp(times[event_idx]),
                    "signal_time_dubai": pd.Timestamp(dubai_times[event_idx]),
                    "entry_time": entry_time,
                    "entry_time_dubai": pd.Timestamp(dubai_times[entry_idx]),
                    "exit_time": exit_time,
                    "exit_time_dubai": pd.Timestamp(dubai_times[exit_idx]),
                    "side": side,
                    "move_cents": move * 100.0,
                    "threshold_cents": threshold_cents,
                    "daily_sigma_dollars": float(sigma),
                    "sigma_multiple": params.sigma_multiple,
                    "delay_s": params.delay_s,
                    "move_window_s": params.move_window_s,
                    "hold_s": params.hold_s,
                    "actual_hold_s": actual_hold_s,
                    "exit_reason": exit_reason,
                    "signal_mode": params.signal_mode,
                    "source_type": day["source_type"],
                    "selected_symbol": day["selected_symbol"],
                    "trend_5d_pct": day.get("trend_5d_pct_63", np.nan),
                    "vol_rank": day.get("vol_rank_63", np.nan),
                    "params": params.label,
                    **fill,
                }
            )

        last_exit = exit_idx
        i = int(np.searchsorted(raw_idx, last_exit - params.delay_s + 1, side="left"))

    return trades, pnls, sides


def trade_summary(pnls: list[float], sides: list[int]) -> dict:
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
    side_arr = np.asarray(sides, dtype=int)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return {
        "trades": int(pnl.size),
        "win_rate": float((pnl > 0).mean()),
        "avg_trade_cents": float(pnl.mean()),
        "profit_factor": gross_profit / (gross_loss + 1e-12) if gross_loss > 0 else np.inf,
        "long_trades": int((side_arr > 0).sum()),
        "short_trades": int((side_arr < 0).sum()),
    }


def backtest_days(
    days: list[dict],
    params: ExpandedParams,
    fill_model: str,
    collect_trades: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    all_trades: list[dict] = []
    all_pnls: list[float] = []
    all_sides: list[int] = []
    daily_index = []
    daily_values = []

    for day in days:
        trades, pnls, sides = scan_day(day, params, fill_model, collect_trades)
        if collect_trades and trades:
            all_trades.extend(trades)
        all_pnls.extend(pnls)
        all_sides.extend(sides)
        daily_index.append(day["date"])
        daily_values.append(float(np.sum(pnls)) if pnls else 0.0)

    daily = pd.Series(daily_values, index=pd.Index(daily_index, name="date"), name="daily_pnl_cents")
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    return trades_df, daily, trade_summary(all_pnls, all_sides)


def precompute_results(
    days: list[dict],
    params: list[ExpandedParams],
    fill_model: str,
) -> dict[ExpandedParams, tuple[pd.Series, pd.DataFrame]]:
    out: dict[ExpandedParams, tuple[pd.Series, pd.DataFrame]] = {}
    for idx, p in enumerate(params, 1):
        if idx % 100 == 0:
            print(f"precomputed {fill_model}: {idx}/{len(params)}", flush=True)
        trades, daily, _ = backtest_days(days, p, fill_model, collect_trades=True)
        out[p] = (daily, trades)
    return out


def walkforward_schedule(bars: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    max_date = min(bars.index.max().normalize(), WALKFORWARD_TEST_END)
    starts = pd.date_range(WALKFORWARD_TEST_START, max_date, freq="MS")
    schedule = []
    for test_start in starts:
        train_start = test_start - pd.DateOffset(months=TRAIN_MONTHS)
        train_end = test_start - pd.Timedelta(seconds=1)
        test_end = min(test_start + pd.DateOffset(months=1) - pd.Timedelta(days=1), max_date)
        schedule.append((train_start, train_end, test_start, test_end))
    return schedule


def slice_daily(daily: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return daily[(daily.index >= start.normalize()) & (daily.index <= end.normalize())]


def slice_trades(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    signal_date = pd.to_datetime(trades["signal_time"]).dt.normalize()
    return trades[(signal_date >= start.normalize()) & (signal_date <= end.normalize())].copy()


def zero_daily_for_period(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    period = bars[(bars.index >= start) & (bars.index <= end + pd.Timedelta(days=1))]
    dates = pd.Index(sorted(period.index.normalize().unique()), name="date")
    return pd.Series(0.0, index=dates, name="daily_pnl_cents")


def concentration_metrics(daily: pd.Series, trades: pd.DataFrame) -> dict:
    total_pnl = float(daily.sum()) if not daily.empty else 0.0
    monthly = daily.groupby(pd.Grouper(freq="MS")).sum() if not daily.empty else pd.Series(dtype=float)
    positive_months = monthly[monthly > 0]
    best_month_pnl = float(positive_months.max()) if not positive_months.empty else 0.0
    month_count = int(monthly.shape[0])

    top_trade_pnl = 0.0
    top_trade_removed_pnl = total_pnl
    if trades is not None and not trades.empty and "net_pnl_cents" in trades.columns:
        positive_trades = trades[trades["net_pnl_cents"].astype(float) > 0]["net_pnl_cents"].astype(float)
        if not positive_trades.empty:
            top_trade_pnl = float(positive_trades.max())
            top_trade_removed_pnl = total_pnl - top_trade_pnl

    denominator = abs(total_pnl) if abs(total_pnl) > 1e-12 else 1.0
    return {
        "profitable_months": int((monthly > 0).sum()) if not monthly.empty else 0,
        "month_count": month_count,
        "profitable_month_rate": float((monthly > 0).sum() / month_count) if month_count else 0.0,
        "best_month_pnl_cents": best_month_pnl,
        "best_month_share": float(best_month_pnl / denominator),
        "top_trade_pnl_cents": top_trade_pnl,
        "top_trade_share": float(top_trade_pnl / denominator),
        "top_trade_removed_pnl_cents": top_trade_removed_pnl,
    }


def top_trade_neutral_metrics(daily: pd.Series, trades: pd.DataFrame) -> dict:
    deflated_daily = daily.copy()
    deflated_trades = trades.copy() if trades is not None else pd.DataFrame()
    removed_top_win = 0.0

    if trades is not None and not trades.empty and "net_pnl_cents" in trades.columns and "signal_time" in trades.columns:
        positive = trades[trades["net_pnl_cents"].astype(float) > 0]
        if not positive.empty:
            top_idx = positive["net_pnl_cents"].astype(float).idxmax()
            removed_top_win = float(trades.loc[top_idx, "net_pnl_cents"])
            trade_date = pd.to_datetime(trades.loc[top_idx, "signal_time"]).normalize()
            if trade_date in deflated_daily.index:
                deflated_daily.loc[trade_date] = float(deflated_daily.loc[trade_date]) - removed_top_win
            deflated_trades = trades.drop(index=top_idx)

    metrics = compute_metrics(deflated_daily, deflated_trades)
    return {
        "removed_top_win_cents": removed_top_win,
        "deflated_total_pnl_cents": metrics["total_pnl_cents"],
        "deflated_daily_sharpe": metrics["daily_sharpe"],
        "deflated_max_drawdown_cents": metrics["max_drawdown_cents"],
        "deflated_trades": metrics["trades"],
        "deflated_win_rate": metrics["win_rate"],
        "deflated_profit_factor": metrics["profit_factor"],
    }


def standard_score(metrics: dict, objective: str) -> float:
    if (
        metrics["trades"] < MIN_TRAIN_TRADES
        or metrics["total_pnl_cents"] <= 0
        or metrics["daily_sharpe"] <= 0
        or not np.isfinite(metrics["profit_factor"])
    ):
        return -np.inf
    return float(metrics[objective])


def robust_score(metrics: dict, concentration: dict, deflated: dict) -> float:
    if (
        metrics["trades"] < MIN_TRAIN_TRADES
        or metrics["total_pnl_cents"] <= 0
        or metrics["daily_sharpe"] <= 0
        or deflated["deflated_total_pnl_cents"] <= 0
        or deflated["deflated_daily_sharpe"] <= 0
        or concentration["profitable_month_rate"] < 0.50
        or concentration["top_trade_share"] > 0.55
        or concentration["best_month_share"] > 0.70
    ):
        return -np.inf

    return float(
        deflated["deflated_daily_sharpe"]
        + 0.002 * deflated["deflated_total_pnl_cents"]
        + 0.75 * concentration["profitable_month_rate"]
        - 0.75 * max(concentration["top_trade_share"] - 0.35, 0.0)
        - 0.75 * max(concentration["best_month_share"] - 0.55, 0.0)
        + 0.001 * metrics["max_drawdown_cents"]
    )


def index_distance(value, reference: list) -> int:
    if value not in reference:
        return 99
    return int(reference.index(value))


def add_cluster_scores(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return grid
    out = grid.copy()
    positive = (
        (out["total_pnl_cents"] > 0)
        & (out["daily_sharpe"] > 0)
        & (out["deflated_total_pnl_cents"] > 0)
        & (out["deflated_daily_sharpe"] > 0)
    )
    delay_idx = out["delay_s"].map(lambda x: index_distance(int(x), DELAYS))
    sigma_idx = out["sigma_multiple"].map(lambda x: index_distance(float(x), SIGMA_MULTIPLES))
    lookback_idx = out["move_window_s"].map(lambda x: index_distance(int(x), MOVE_WINDOWS))
    hold_idx = out["hold_s"].map(lambda x: index_distance(int(x), HOLDS))

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
    out["robust_top_trade_neutral_score"] = out["score"] + 0.35 * out["cluster_positive_neighbors"] + 1.25 * out["cluster_positive_rate"]
    return out.sort_values(
        ["robust_top_trade_neutral_score", "cluster_positive_neighbors", "deflated_total_pnl_cents", "deflated_daily_sharpe"],
        ascending=False,
    ).reset_index(drop=True)


def evaluate_grid_cached(
    cache: dict[ExpandedParams, tuple[pd.Series, pd.DataFrame]],
    params: list[ExpandedParams],
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
        concentration = concentration_metrics(period_daily, period_trades)
        deflated = top_trade_neutral_metrics(period_daily, period_trades)
        rows.append(
            {
                "params_obj": p,
                "params": p.label,
                "delay_s": p.delay_s,
                "sigma_multiple": p.sigma_multiple,
                "move_window_s": p.move_window_s,
                "hold_s": p.hold_s,
                "vol_window_days": p.vol_window_days,
                "signal_mode": p.signal_mode,
                "score": standard_score(metrics, objective),
                **metrics,
                **concentration,
                **deflated,
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "daily_sharpe", "total_pnl_cents"], ascending=False).reset_index(drop=True)


def evaluate_robust_grid_cached(
    cache: dict[ExpandedParams, tuple[pd.Series, pd.DataFrame]],
    params: list[ExpandedParams],
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
        deflated = top_trade_neutral_metrics(period_daily, period_trades)
        rows.append(
            {
                "params_obj": p,
                "params": p.label,
                "delay_s": p.delay_s,
                "sigma_multiple": p.sigma_multiple,
                "move_window_s": p.move_window_s,
                "hold_s": p.hold_s,
                "vol_window_days": p.vol_window_days,
                "signal_mode": p.signal_mode,
                "score": robust_score(metrics, concentration, deflated),
                **metrics,
                **concentration,
                **deflated,
            }
        )
    return add_cluster_scores(pd.DataFrame(rows))


def select_standard_candidate(grid: pd.DataFrame, objective: str) -> pd.Series | None:
    eligible = grid[np.isfinite(grid["score"])].copy()
    if eligible.empty:
        return None
    return eligible.sort_values([objective, "daily_sharpe", "total_pnl_cents"], ascending=False).iloc[0]


def select_robust_candidate(grid: pd.DataFrame) -> pd.Series | None:
    if grid.empty:
        return None
    eligible = grid[
        np.isfinite(grid["robust_top_trade_neutral_score"])
        & (grid["month_count"] >= 2)
        & (grid["trades"] >= MIN_TRAIN_TRADES)
        & (grid["total_pnl_cents"] > 0)
        & (grid["daily_sharpe"] > 0)
        & (grid["deflated_total_pnl_cents"] > 0)
        & (grid["deflated_daily_sharpe"] > 0)
        & (grid["profitable_month_rate"] >= 0.50)
        & (grid["top_trade_share"] <= 0.55)
        & (grid["best_month_share"] <= 0.70)
        & (grid["cluster_positive_neighbors"] >= 2)
    ].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["robust_top_trade_neutral_score", "cluster_positive_neighbors", "deflated_total_pnl_cents", "deflated_daily_sharpe"],
        ascending=False,
    ).iloc[0]


def selector_label(selector_kind: str) -> str:
    return {
        "train_total_pnl": "Train PnL selector",
        "train_daily_sharpe": "Train Sharpe selector",
        "robust_top_trade_neutral": "Robust top-trade-neutral selector",
    }[selector_kind]


def run_walkforward_cached(
    cache: dict[ExpandedParams, tuple[pd.Series, pd.DataFrame]],
    bars: pd.DataFrame,
    params: list[ExpandedParams],
    fill_model: str,
    selector_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    decisions = []
    trades_list = []
    daily_list = []
    rankings = []
    run_key = f"{fill_model}_{selector_kind}"
    run_label = selector_label(selector_kind)
    objective = "total_pnl_cents" if selector_kind == "train_total_pnl" else "daily_sharpe"

    for train_start, train_end, test_start, test_end in walkforward_schedule(bars):
        if selector_kind == "robust_top_trade_neutral":
            grid = evaluate_robust_grid_cached(cache, params, train_start, train_end)
            selected = select_robust_candidate(grid)
            score_col = "robust_top_trade_neutral_score"
        else:
            grid = evaluate_grid_cached(cache, params, train_start, train_end, objective)
            selected = select_standard_candidate(grid, objective)
            score_col = "score"

        top = grid.drop(columns=["params_obj"], errors="ignore").head(75).copy()
        top["run_key"] = run_key
        top["run_label"] = run_label
        top["fill_model"] = fill_model
        top["selector_kind"] = selector_kind
        top["train_start"] = train_start.date().isoformat()
        top["train_end"] = train_end.date().isoformat()
        top["test_start"] = test_start.date().isoformat()
        top["test_end"] = test_end.date().isoformat()
        rankings.append(top)

        if selected is None:
            zero_daily = zero_daily_for_period(bars, test_start, test_end)
            daily_list.append(zero_daily)
            decisions.append(
                {
                    "run_key": run_key,
                    "run_label": run_label,
                    "fill_model": fill_model,
                    "selector_kind": selector_kind,
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "selected_params": "NO_ELIGIBLE_CANDIDATE",
                    "test_total_pnl_cents": 0.0,
                    "test_sharpe": 0.0,
                    "test_trades": 0,
                    "test_top_trade_removed_pnl_cents": 0.0,
                    "test_deflated_total_pnl_cents": 0.0,
                    "test_deflated_sharpe": 0.0,
                }
            )
            continue

        p: ExpandedParams = selected["params_obj"]
        daily_full, trades_full = cache[p]
        daily = slice_daily(daily_full, test_start, test_end)
        trades = slice_trades(trades_full, test_start, test_end)
        metrics = compute_metrics(daily, trades)
        concentration = concentration_metrics(daily, trades)
        deflated = top_trade_neutral_metrics(daily, trades)
        daily_list.append(daily)
        if not trades.empty:
            out_trades = trades.copy()
            out_trades["run_key"] = run_key
            out_trades["run_label"] = run_label
            out_trades["fill_model"] = fill_model
            out_trades["selector_kind"] = selector_kind
            out_trades["test_start"] = test_start.date().isoformat()
            out_trades["test_end"] = test_end.date().isoformat()
            out_trades["selected_params"] = p.label
            trades_list.append(out_trades)

        decisions.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "fill_model": fill_model,
                "selector_kind": selector_kind,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_params": p.label,
                "train_score": float(selected[score_col]) if score_col in selected.index else float(selected["score"]),
                "train_total_pnl_cents": float(selected["total_pnl_cents"]),
                "train_sharpe": float(selected["daily_sharpe"]),
                "train_trades": int(selected["trades"]),
                "train_top_trade_removed_pnl_cents": float(selected.get("top_trade_removed_pnl_cents", np.nan)),
                "train_deflated_total_pnl_cents": float(selected.get("deflated_total_pnl_cents", np.nan)),
                "train_deflated_sharpe": float(selected.get("deflated_daily_sharpe", np.nan)),
                "train_cluster_positive_neighbors": int(selected.get("cluster_positive_neighbors", 0)),
                "train_cluster_positive_rate": float(selected.get("cluster_positive_rate", 0.0)),
                "test_total_pnl_cents": metrics["total_pnl_cents"],
                "test_sharpe": metrics["daily_sharpe"],
                "test_trades": metrics["trades"],
                "test_win_rate": metrics["win_rate"],
                "test_max_drawdown_cents": metrics["max_drawdown_cents"],
                "test_profitable_month_rate": concentration["profitable_month_rate"],
                "test_top_trade_share": concentration["top_trade_share"],
                "test_best_month_share": concentration["best_month_share"],
                "test_top_trade_removed_pnl_cents": concentration["top_trade_removed_pnl_cents"],
                "test_removed_top_win_cents": deflated["removed_top_win_cents"],
                "test_deflated_total_pnl_cents": deflated["deflated_total_pnl_cents"],
                "test_deflated_sharpe": deflated["deflated_daily_sharpe"],
            }
        )

    daily_oos = pd.concat(daily_list).sort_index().groupby(level=0).sum() if daily_list else pd.Series(dtype=float, name="daily_pnl_cents")
    trades_df = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    decisions_df = pd.DataFrame(decisions)
    rankings_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    metrics = compute_metrics(daily_oos, trades_df)
    concentration = concentration_metrics(daily_oos, trades_df)
    deflated = top_trade_neutral_metrics(daily_oos, trades_df)
    metrics_row = {
        "run_key": run_key,
        "run_label": run_label,
        "fill_model": fill_model,
        "selector_kind": selector_kind,
        "objective": selector_kind,
        **metrics,
        **concentration,
        **deflated,
    }
    return decisions_df, trades_df, daily_oos, rankings_df, metrics_row


def top_trade_rows(trades: pd.DataFrame, run_key: str, run_label: str) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    out = trades.copy()
    out["abs_pnl_cents"] = out["net_pnl_cents"].astype(float).abs()
    cols = [
        "run_key",
        "run_label",
        "fill_model",
        "selector_kind",
        "test_start",
        "signal_time_dubai",
        "entry_time_dubai",
        "exit_time_dubai",
        "side",
        "fill_source",
        "selected_symbol",
        "move_cents",
        "threshold_cents",
        "entry_price",
        "exit_price",
        "net_pnl_cents",
        "abs_pnl_cents",
        "selected_params",
    ]
    return out.sort_values("abs_pnl_cents", ascending=False)[[c for c in cols if c in out.columns]].head(25)


def monthly_audit_rows(daily: pd.Series, trades: pd.DataFrame, run_key: str, run_label: str, fill_model: str, selector_kind: str) -> list[dict]:
    if daily.empty:
        return []
    monthly = daily.groupby(pd.Grouper(freq="MS")).sum()
    trade_months = pd.Series(dtype=int)
    if trades is not None and not trades.empty and "signal_time" in trades.columns:
        trade_months = pd.to_datetime(trades["signal_time"]).dt.to_period("M").astype(str).value_counts()
    total = float(daily.sum())
    rows = []
    for date, pnl in monthly.items():
        month = pd.Period(date, freq="M").strftime("%Y-%m")
        rows.append(
            {
                "run_key": run_key,
                "run_label": run_label,
                "fill_model": fill_model,
                "selector_kind": selector_kind,
                "month": month,
                "month_pnl_cents": float(pnl),
                "month_share_of_total": float(pnl / total) if abs(total) > 1e-12 else 0.0,
                "trades": int(trade_months.get(month, 0)) if not trade_months.empty else 0,
            }
        )
    return rows


def build_winner_comparison(metrics: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fill_model, group in metrics.groupby("fill_model"):
        specs = [
            ("max_oos_pnl", group.sort_values(["total_pnl_cents", "daily_sharpe"], ascending=False).iloc[0]),
            ("max_oos_sharpe", group.sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).iloc[0]),
        ]
        robust = group[group["selector_kind"] == "robust_top_trade_neutral"]
        if not robust.empty:
            specs.append(("robust_top_trade_neutral", robust.iloc[0]))
        for winner_type, row in specs:
            run_key = str(row["run_key"])
            run_decisions = decisions[decisions["run_key"] == run_key].copy()
            latest = run_decisions[~run_decisions["selected_params"].astype(str).str.startswith("NO_")]
            latest_params = latest.iloc[-1]["selected_params"] if not latest.empty else ""
            unique_params = int(latest["selected_params"].nunique()) if not latest.empty else 0
            rows.append(
                {
                    "winner_type": winner_type,
                    "fill_model": fill_model,
                    "run_key": run_key,
                    "run_label": row["run_label"],
                    "selector_kind": row["selector_kind"],
                    "total_pnl_cents": row["total_pnl_cents"],
                    "daily_sharpe": row["daily_sharpe"],
                    "trades": int(row["trades"]),
                    "max_drawdown_cents": row["max_drawdown_cents"],
                    "win_rate": row["win_rate"],
                    "profit_factor": row["profit_factor"],
                    "top_trade_pnl_cents": row["top_trade_pnl_cents"],
                    "top_trade_removed_pnl_cents": row["top_trade_removed_pnl_cents"],
                    "top_trade_share": row["top_trade_share"],
                    "best_month_share": row["best_month_share"],
                    "deflated_total_pnl_cents": row["deflated_total_pnl_cents"],
                    "deflated_daily_sharpe": row["deflated_daily_sharpe"],
                    "unique_monthly_parameter_sets": unique_params,
                    "latest_selected_params": latest_params,
                    "is_robust_by_tests": bool(
                        row["total_pnl_cents"] > 0
                        and row["daily_sharpe"] > 0
                        and row["top_trade_removed_pnl_cents"] > 0
                        and row["top_trade_share"] <= 0.55
                        and row["best_month_share"] <= 0.70
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_fill_model_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for selector_kind, group in metrics.groupby("selector_kind"):
        if set(group["fill_model"]) >= set(FILL_MODELS):
            mid = group[group["fill_model"] == "mid_dynamic_cost"].iloc[0]
            actual = group[group["fill_model"] == "actual_bid_ask"].iloc[0]
            rows.append(
                {
                    "selector_kind": selector_kind,
                    "mid_total_pnl_cents": mid["total_pnl_cents"],
                    "actual_total_pnl_cents": actual["total_pnl_cents"],
                    "actual_minus_mid_pnl_cents": actual["total_pnl_cents"] - mid["total_pnl_cents"],
                    "mid_daily_sharpe": mid["daily_sharpe"],
                    "actual_daily_sharpe": actual["daily_sharpe"],
                    "actual_minus_mid_sharpe": actual["daily_sharpe"] - mid["daily_sharpe"],
                    "mid_trades": int(mid["trades"]),
                    "actual_trades": int(actual["trades"]),
                    "mid_top_trade_removed_pnl_cents": mid["top_trade_removed_pnl_cents"],
                    "actual_top_trade_removed_pnl_cents": actual["top_trade_removed_pnl_cents"],
                }
            )
    return pd.DataFrame(rows)


def validate_outputs(
    bars: pd.DataFrame,
    day_stats: pd.DataFrame,
    trades: pd.DataFrame,
    duplicate_count: int,
) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, details: str) -> None:
        rows.append({"check": name, "passed": bool(passed), "details": details})

    add("no_duplicate_timestamps_after_stitching", not bars.index.duplicated().any(), f"removed_before_final={duplicate_count}")

    stats = day_stats.copy()
    stats["date"] = pd.to_datetime(stats["date"], errors="coerce")
    parquet_stats = stats[stats["source_type"].eq("parquet_front_bid_ask")]
    csv_stats = stats[stats["source_type"].eq("csv_mid_dynamic_cost")]
    parquet_months = sorted(parquet_stats["date"].dt.to_period("M").dropna().astype(str).unique().tolist())
    csv_months = sorted(csv_stats["date"].dt.to_period("M").dropna().astype(str).unique().tolist())
    add("parquet_jan_to_nov_only", parquet_months == [f"2025-{m:02d}" for m in range(1, 12)], ",".join(parquet_months))
    add("csv_dec_to_mar_only", csv_months == ["2025-12", "2026-01", "2026-02", "2026-03"], ",".join(csv_months))

    if trades.empty:
        add("trades_exist", False, "no trades generated")
        return pd.DataFrame(rows)

    signal_dubai = pd.to_datetime(trades["signal_time_dubai"], errors="coerce")
    seconds = signal_dubai.dt.hour * 3600 + signal_dubai.dt.minute * 60 + signal_dubai.dt.second
    in_session = (seconds >= SESSION_START_HOUR_DUBAI * 3600) & (seconds < SESSION_END_HOUR_DUBAI * 3600)
    add("signals_inside_11_to_midnight_dubai", bool(in_session.all()), f"violations={int((~in_session).sum())}")

    grouped = trades.assign(signal_date=pd.to_datetime(trades["signal_time"]).dt.normalize()).groupby(["run_key", "signal_date"]).size()
    max_daily_trades = int(grouped.max()) if not grouped.empty else 0
    add("max_one_trade_per_day_per_run", max_daily_trades <= 1, f"max_daily_trades={max_daily_trades}")

    hold_ok = trades["actual_hold_s"].astype(float) <= trades["hold_s"].astype(float)
    exit_dubai = pd.to_datetime(trades["exit_time_dubai"], errors="coerce")
    entry_dubai = pd.to_datetime(trades["entry_time_dubai"], errors="coerce")
    midnight = entry_dubai.dt.normalize() + pd.Timedelta(days=1)
    midnight_ok = exit_dubai <= midnight
    add("exits_respect_hold_and_midnight_stop", bool((hold_ok & midnight_ok).all()), f"violations={int((~(hold_ok & midnight_ok)).sum())}")

    actual = trades[trades["fill_source"].eq("actual_bid_ask")].copy()
    if actual.empty:
        add("actual_bid_ask_fill_direction", True, "no actual bid/ask rows")
    else:
        long_rows = actual["side"].astype(int) > 0
        short_rows = actual["side"].astype(int) < 0
        long_ok = (
            np.isclose(actual.loc[long_rows, "entry_price"], actual.loc[long_rows, "entry_ask"])
            & np.isclose(actual.loc[long_rows, "exit_price"], actual.loc[long_rows, "exit_bid"])
        )
        short_ok = (
            np.isclose(actual.loc[short_rows, "entry_price"], actual.loc[short_rows, "entry_bid"])
            & np.isclose(actual.loc[short_rows, "exit_price"], actual.loc[short_rows, "exit_ask"])
        )
        violations = int((~pd.Series(long_ok, index=actual.loc[long_rows].index)).sum()) + int(
            (~pd.Series(short_ok, index=actual.loc[short_rows].index)).sum()
        )
        add("actual_bid_ask_fill_direction", violations == 0, f"violations={violations}")

    return pd.DataFrame(rows)


def main() -> None:
    bars, day_stats, coverage, duplicate_count = stitch_research_dataset()
    if bars.empty:
        raise RuntimeError("No research data loaded.")

    day_stats.to_csv(OUT_DIR / "data_coverage_days.csv", index=False)
    coverage.to_csv(OUT_DIR / "data_coverage.csv", index=False)

    bars = add_volatility_columns(bars)
    days = prepare_days(bars)
    params = build_universe()
    pd.DataFrame([p.__dict__ | {"label": p.label} for p in params]).to_csv(OUT_DIR / "parameter_universe.csv", index=False)

    all_metrics = []
    all_decisions = []
    all_trades = []
    all_daily = []
    all_rankings = []
    all_top_trades = []
    all_monthly_audit = []

    for fill_model in FILL_MODELS:
        print(f"precomputing fill model: {fill_model}", flush=True)
        cache = precompute_results(days, params, fill_model)
        for selector_kind in ["train_total_pnl", "train_daily_sharpe", "robust_top_trade_neutral"]:
            print(f"running {fill_model} {selector_kind}", flush=True)
            decisions, trades, daily, rankings, metrics_row = run_walkforward_cached(cache, bars, params, fill_model, selector_kind)
            all_metrics.append(metrics_row)
            if not decisions.empty:
                all_decisions.append(decisions)
            if not trades.empty:
                all_trades.append(trades)
                top = top_trade_rows(trades, metrics_row["run_key"], metrics_row["run_label"])
                if not top.empty:
                    all_top_trades.append(top)
            if not daily.empty:
                all_daily.append(daily.rename(metrics_row["run_key"]))
                all_monthly_audit.extend(
                    monthly_audit_rows(
                        daily,
                        trades,
                        metrics_row["run_key"],
                        metrics_row["run_label"],
                        fill_model,
                        selector_kind,
                    )
                )
            if not rankings.empty:
                all_rankings.append(rankings)
        del cache

    metrics_df = pd.DataFrame(all_metrics).sort_values(["fill_model", "daily_sharpe", "total_pnl_cents"], ascending=[True, False, False])
    decisions_df = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    rankings_df = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    daily_df = pd.concat(all_daily, axis=1).fillna(0.0) if all_daily else pd.DataFrame()
    top_trades_df = pd.concat(all_top_trades, ignore_index=True) if all_top_trades else pd.DataFrame()
    monthly_audit_df = pd.DataFrame(all_monthly_audit)
    winners_df = build_winner_comparison(metrics_df, decisions_df)
    fill_comparison_df = build_fill_model_comparison(metrics_df)
    validation_df = validate_outputs(bars, day_stats, trades_df, duplicate_count)

    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
    decisions_df.to_csv(OUT_DIR / "decisions.csv", index=False)
    trades_df.to_csv(OUT_DIR / "trades.csv", index=False)
    rankings_df.to_csv(OUT_DIR / "train_rankings.csv", index=False)
    daily_df.to_csv(OUT_DIR / "daily_pnl.csv")
    top_trades_df.to_csv(OUT_DIR / "top_trades.csv", index=False)
    monthly_audit_df.to_csv(OUT_DIR / "monthly_audit.csv", index=False)
    winners_df.to_csv(OUT_DIR / "winner_comparison.csv", index=False)
    fill_comparison_df.to_csv(OUT_DIR / "fill_model_comparison.csv", index=False)
    validation_df.to_csv(OUT_DIR / "validation_checks.csv", index=False)
    pd.DataFrame(
        [
            {
                "data_start": str(bars.index.min()),
                "data_end": str(bars.index.max()),
                "parquet_raw_root": str(RAW_ROOT),
                "output_dir": str(OUT_DIR),
                "candidate_count": len(params),
                "fill_models": ",".join(FILL_MODELS),
                "train_months": TRAIN_MONTHS,
                "test_months": 1,
                "walkforward_first_test": WALKFORWARD_TEST_START.date().isoformat(),
                "walkforward_last_test": min(bars.index.max().normalize(), WALKFORWARD_TEST_END).date().isoformat(),
                "delays_seconds": ",".join(str(x) for x in DELAYS),
                "sigma_multiples": ",".join(f"{x:g}" for x in SIGMA_MULTIPLES),
                "move_windows_seconds": ",".join(str(x) for x in MOVE_WINDOWS),
                "holds_seconds": ",".join(str(x) for x in HOLDS),
                "directions": ",".join(SIGNAL_MODES),
                "session": "11:00-24:00 Dubai signal time; full move window must be in session",
                "exit_rule": "min(entry + hold_s, midnight Dubai)",
                "max_trades_per_day": MAX_TRADES_PER_DAY,
                "cost_formula": f"mid model round-trip cents = abs(entry_price) * {DYNAMIC_COST_CENTS_PER_PRICE_UNIT:g}",
                "actual_fill_rule": "long entry ask/exit bid; short entry bid/exit ask where bid/ask exists; CSV months fall back to dynamic cost",
            }
        ]
    ).to_csv(OUT_DIR / "config.csv", index=False)

    print("Winner comparison", flush=True)
    print(winners_df.to_string(index=False), flush=True)
    print("Validation", flush=True)
    print(validation_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
