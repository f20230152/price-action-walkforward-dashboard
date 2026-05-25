from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .engine import PRE_SESSION_START_UTC_HOUR, SESSION_END_UTC_HOUR


RAW_ROOT = Path(r"D:\Energin Raw Data")
PARQUET_MONTH_TEMPLATE = "brent_local_2025_{month:02d}"
CSV_START = pd.Timestamp("2025-12-01")
CSV_END = pd.Timestamp("2026-03-31 23:59:59")
PARQUET_START = pd.Timestamp("2025-01-01")
REQUIRED_PARQUET_COLUMNS = ["second_utc", "symbol_ticker", "mid", "bid_last", "ask_last", "spread", "trade_volume"]
VOLUME_RANK_COLUMNS = ["symbol_ticker", "trade_volume"]
BRENT_MONTH_CODES = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}


def _date_range_for_day(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(
        day.normalize() + pd.Timedelta(hours=PRE_SESSION_START_UTC_HOUR),
        day.normalize() + pd.Timedelta(hours=SESSION_END_UTC_HOUR),
        freq="s",
    )


def _normalize_second_index(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df[df[timestamp_col].notna()].copy()
    df[timestamp_col] = df[timestamp_col].dt.tz_convert(None).dt.floor("s")
    df = df.sort_values(timestamp_col).drop_duplicates(timestamp_col, keep="last")
    return df.set_index(timestamp_col).sort_index()


def _session_reindex_one_day(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    target = _date_range_for_day(day)
    expanded = df.reindex(df.index.union(target)).sort_index().ffill().reindex(target)
    return expanded[expanded["mid"].notna()].copy()


def _parse_csv_file_date(path: Path) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(path.stem)
    except ValueError:
        return None


def _read_symbol_master(month_dir: Path) -> pd.DataFrame:
    ref_path = month_dir / "reference" / "brent_symbol_master.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing symbol master: {ref_path}")
    master = pd.read_parquet(ref_path)
    master["expiration_dt"] = pd.to_datetime(master["expiration_date"], errors="coerce")
    return master


def prompt_delivery_month(year: int, month: int) -> pd.Timestamp:
    """The prompt Brent series uses the contract month two calendar months ahead.

    Examples: May 2025 -> July 2025 (N25), June 2025 -> August 2025 (Q25).
    """

    return pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(months=2)


def prompt_contract_symbol(year: int, month: int) -> str:
    delivery = prompt_delivery_month(year, month)
    code = BRENT_MONTH_CODES[int(delivery.month)]
    return f"F:BRN\\{code}{str(delivery.year)[-2:]}"


def choose_prompt_contract(month_dir: Path, year: int, month: int) -> tuple[str, dict[str, Any]]:
    master = _read_symbol_master(month_dir)
    expected_symbol = prompt_contract_symbol(year, month)
    match = master[master["symbol_ticker"] == expected_symbol].copy()
    if match.empty:
        raise ValueError(f"Prompt contract {expected_symbol} not found in symbol master for {year}-{month:02d}")
    row = match.iloc[0]
    delivery = prompt_delivery_month(year, month)
    metadata = {
        "selection_rule": "prompt_month_calendar_plus_2_using_ice_month_codes",
        "calendar_month": f"{year:04d}-{month:02d}",
        "prompt_delivery_month": delivery.strftime("%Y-%m"),
        "prompt_month_code": BRENT_MONTH_CODES[int(delivery.month)],
        "expected_prompt_symbol": expected_symbol,
        "contract_expiry_date": pd.Timestamp(row["expiration_dt"]).date().isoformat() if pd.notna(row["expiration_dt"]) else "",
    }
    return expected_symbol, metadata


def _read_filtered_parquet(path: Path, symbol_ticker: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path, columns=REQUIRED_PARQUET_COLUMNS, filters=[("symbol_ticker", "==", symbol_ticker)])
    except Exception:
        df = pd.read_parquet(path, columns=REQUIRED_PARQUET_COLUMNS)
    if not df.empty:
        df = df[df["symbol_ticker"] == symbol_ticker].copy()
    return df


def _daily_symbol_volume_rank(file_paths: list[Path], selected_symbol: str) -> dict[str, Any]:
    parts = []
    for path in file_paths:
        try:
            raw = pd.read_parquet(path, columns=VOLUME_RANK_COLUMNS)
        except Exception:
            continue
        if raw.empty:
            continue
        raw = raw.copy()
        raw["trade_volume"] = pd.to_numeric(raw["trade_volume"], errors="coerce").fillna(0.0)
        parts.append(raw[VOLUME_RANK_COLUMNS])
    if not parts:
        return {
            "total_volume_that_day": np.nan,
            "rank_by_volume_among_available_symbols": np.nan,
            "available_symbol_count": 0,
        }
    totals = pd.concat(parts, ignore_index=True).groupby("symbol_ticker")["trade_volume"].sum()
    ranks = totals.rank(method="min", ascending=False).astype(int)
    selected_total = float(totals[selected_symbol]) if selected_symbol in totals.index else np.nan
    selected_rank = int(ranks[selected_symbol]) if selected_symbol in ranks.index else np.nan
    return {
        "total_volume_that_day": selected_total,
        "rank_by_volume_among_available_symbols": selected_rank,
        "available_symbol_count": int(len(totals)),
    }


def _coverage_row(
    month_df: pd.DataFrame,
    year: int,
    month: int,
    source_type: str,
    selected_symbol: str,
    file_count: int,
) -> dict[str, Any]:
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    expected_bdays = pd.bdate_range(month_start, month_end)
    if month_df.empty:
        loaded_dates = set()
        first_ts = ""
        last_ts = ""
        row_count = 0
    else:
        loaded_dates = {pd.Timestamp(d) for d in pd.Series(month_df.index.date).unique()}
        first_ts = month_df.index.min().isoformat()
        last_ts = month_df.index.max().isoformat()
        row_count = int(len(month_df))
    missing = [d.date().isoformat() for d in expected_bdays if pd.Timestamp(d.date()) not in loaded_dates]
    return {
        "month": f"{year:04d}-{month:02d}",
        "source_type": source_type,
        "selected_symbol": selected_symbol,
        "days_loaded": int(len(loaded_dates)),
        "expected_business_days": int(len(expected_bdays)),
        "missing_business_days": ";".join(missing),
        "missing_business_day_count": int(len(missing)),
        "first_timestamp_utc": first_ts,
        "last_timestamp_utc": last_ts,
        "row_count": row_count,
        "file_count": int(file_count),
    }


def load_parquet_month(raw_root: Path, year: int, month: int) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    month_dir = raw_root / PARQUET_MONTH_TEMPLATE.format(month=month)
    if not month_dir.exists():
        raise FileNotFoundError(f"Missing Energin month folder: {month_dir}")
    front_symbol, prompt_meta = choose_prompt_contract(month_dir, year, month)
    base = month_dir / "parquet" / "brent_1s_bars" / f"year={year}" / f"month={month:02d}"
    if not base.exists():
        raise FileNotFoundError(f"Missing parquet path: {base}")
    frames = []
    files_used = []
    for day_dir in sorted(base.glob("day=*")):
        if not day_dir.is_dir():
            continue
        day = pd.Timestamp(f"{year:04d}-{month:02d}-{day_dir.name.split('=')[-1]}")
        day_files = sorted(day_dir.glob("*.parquet"))
        volume_rank = _daily_symbol_volume_rank(day_files, front_symbol)
        day_frames = []
        for file_path in day_files:
            raw = _read_filtered_parquet(file_path, front_symbol)
            if not raw.empty:
                files_used.append(str(file_path))
                day_frames.append(raw)
        if not day_frames:
            continue
        raw = pd.concat(day_frames, ignore_index=True)
        raw = _normalize_second_index(raw, "second_utc").rename(columns={"symbol_ticker": "selected_symbol"})
        raw["source_type"] = "parquet_prompt_bid_ask"
        raw["has_actual_bid_ask"] = raw["bid_last"].notna() & raw["ask_last"].notna()
        session = _session_reindex_one_day(raw, day)
        if not session.empty:
            session["selected_symbol"] = front_symbol
            session["source_type"] = "parquet_prompt_bid_ask"
            session["has_actual_bid_ask"] = session["bid_last"].notna() & session["ask_last"].notna()
            session["contract_expiry_date"] = prompt_meta["contract_expiry_date"]
            session["prompt_delivery_month"] = prompt_meta["prompt_delivery_month"]
            session["prompt_month_code"] = prompt_meta["prompt_month_code"]
            session["contract_selection_rule"] = prompt_meta["selection_rule"]
            session["total_volume_that_day"] = volume_rank["total_volume_that_day"]
            session["rank_by_volume_among_available_symbols"] = volume_rank["rank_by_volume_among_available_symbols"]
            session["available_symbol_count"] = volume_rank["available_symbol_count"]
            frames.append(session)
    month_df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    coverage = _coverage_row(month_df, year, month, "parquet_prompt_bid_ask", front_symbol, len(files_used))
    coverage.update(prompt_meta)
    return month_df, coverage, files_used


def _load_csv_one_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame()
    price_cols = [c for c in raw.columns if c.endswith(".Price")]
    if not price_cols:
        raise ValueError(f"No price column found in {path}")
    timestamp_col = "Time.1" if "Time.1" in raw.columns else raw.columns[1]
    df = pd.DataFrame({"second_utc": raw[timestamp_col], "mid": pd.to_numeric(raw[price_cols[0]], errors="coerce")})
    df = df.dropna(subset=["mid"])
    if df.empty:
        return pd.DataFrame()
    df = _normalize_second_index(df, "second_utc")
    df["bid_last"] = np.nan
    df["ask_last"] = np.nan
    df["spread"] = np.nan
    df["trade_volume"] = np.nan
    df["selected_symbol"] = "%BRN 1!-ICE"
    df["source_type"] = "csv_mid_dynamic_cost"
    df["has_actual_bid_ask"] = False
    df["contract_expiry_date"] = ""
    df["prompt_delivery_month"] = ""
    df["prompt_month_code"] = ""
    df["contract_selection_rule"] = "csv_continuous_front_symbol_from_legacy_data"
    return df


def load_csv_months(data_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    frames = []
    files_used = []
    for path in sorted(data_dir.glob("*.csv")):
        file_date = _parse_csv_file_date(path)
        if file_date is None or file_date < CSV_START or file_date > CSV_END:
            continue
        raw = _load_csv_one_file(path)
        if raw.empty:
            continue
        session = _session_reindex_one_day(raw, file_date)
        if not session.empty:
            session["source_type"] = "csv_mid_dynamic_cost"
            session["selected_symbol"] = "%BRN 1!-ICE"
            session["has_actual_bid_ask"] = False
            frames.append(session)
            files_used.append(str(path))
    csv_df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    coverage = []
    for period in pd.period_range("2025-12", "2026-03", freq="M"):
        month_df = csv_df[csv_df.index.to_period("M") == period] if not csv_df.empty else pd.DataFrame()
        coverage.append(
            _coverage_row(
                month_df,
                period.year,
                period.month,
                "csv_mid_dynamic_cost",
                "%BRN 1!-ICE",
                sum(1 for f in files_used if Path(f).stem.startswith(str(period))),
            )
        )
        coverage[-1].update(
            {
                "selection_rule": "csv_continuous_front_symbol_from_legacy_data",
                "calendar_month": str(period),
                "prompt_delivery_month": "",
                "prompt_month_code": "",
                "expected_prompt_symbol": "%BRN 1!-ICE",
                "contract_expiry_date": "",
            }
        )
    return csv_df, coverage, files_used


def load_research_dataset(repo_root: Path, raw_root: Path = RAW_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frames = []
    coverage_rows = []
    parquet_files = []
    for month in range(1, 12):
        month_df, coverage, files = load_parquet_month(raw_root, 2025, month)
        frames.append(month_df)
        coverage_rows.append(coverage)
        parquet_files.extend(files)
    csv_df, csv_coverage, csv_files = load_csv_months(repo_root / "data")
    frames.append(csv_df)
    coverage_rows.extend(csv_coverage)
    combined = pd.concat([f for f in frames if not f.empty]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")].copy()
    combined = combined[(combined.index >= PARQUET_START) & (combined.index <= CSV_END)].copy()
    combined.index.name = "timestamp_utc"
    combined["month"] = combined.index.to_period("M").astype(str)
    combined["utc_date"] = [d.isoformat() for d in combined.index.date]
    combined["dubai_time"] = combined.index + pd.Timedelta(hours=4)
    combined["dubai_date"] = combined["dubai_time"].dt.date.astype(str)
    if combined.index.duplicated().any():
        raise ValueError(f"Duplicate timestamps after stitching: {int(combined.index.duplicated().sum())}")
    manifest = {
        "csv_files_used": csv_files,
        "parquet_files_used": parquet_files,
        "rows": int(len(combined)),
        "first_timestamp_utc": combined.index.min().isoformat() if not combined.empty else "",
        "last_timestamp_utc": combined.index.max().isoformat() if not combined.empty else "",
        "contract_selection_rule": "Jan-Nov parquet uses calendar month + 2 delivery months via ICE Brent month codes; Dec-Mar CSV uses legacy continuous %BRN 1!-ICE.",
    }
    return combined, pd.DataFrame(coverage_rows), manifest
