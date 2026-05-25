from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .engine import PRE_SESSION_START_UTC_HOUR, SESSION_END_UTC_HOUR


warnings.filterwarnings("ignore", category=FutureWarning)
RAW_ROOT = Path(r"C:\Users\taran\OneDrive\Desktop\Downloads\Backtesting data 2025")
LEGACY_RAW_ROOT = Path(r"D:\Energin Raw Data")
PARQUET_MONTH_TEMPLATE = "brent_local_2025_{month:02d}"
CSV_START = pd.Timestamp("2025-12-01")
CSV_END = pd.Timestamp("2026-03-31 23:59:59")
PARQUET_START = pd.Timestamp("2025-01-01")
REQUIRED_PARQUET_COLUMNS = ["second_utc", "symbol_ticker", "mid", "bid_last", "ask_last", "spread", "trade_volume"]
VOLUME_RANK_COLUMNS = ["symbol_ticker", "trade_volume"]
PROMPT_LIQUID_SOURCE_TYPE = "parquet_prompt_liquid_bid_ask"
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
MONTH_NAME_TO_NUMBER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
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
    real_print_index = df.index.drop_duplicates()
    expanded = df.reindex(df.index.union(target)).sort_index().ffill().reindex(target)
    expanded["is_real_print"] = expanded.index.isin(real_print_index)
    expanded["was_forward_filled"] = expanded["mid"].notna() & ~expanded["is_real_print"]
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
    return pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(months=2)


def prompt_contract_symbol(year: int, month: int) -> str:
    delivery = prompt_delivery_month(year, month)
    code = BRENT_MONTH_CODES[int(delivery.month)]
    return f"F:BRN\\{code}{str(delivery.year)[-2:]}"


def _business_days_to_expiry(day: Any, expiry: Any) -> int:
    if pd.isna(expiry):
        return 9999
    return int(np.busday_count(pd.Timestamp(day).date(), pd.Timestamp(expiry).date()))


def _read_filtered_parquet(path: Path, symbol_ticker: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path, columns=REQUIRED_PARQUET_COLUMNS, filters=[("symbol_ticker", "==", symbol_ticker)])
    except Exception:
        df = pd.read_parquet(path, columns=REQUIRED_PARQUET_COLUMNS)
    if not df.empty:
        df = df[df["symbol_ticker"] == symbol_ticker].copy()
    return df


def _read_filtered_raw(path: Path, symbol_ticker: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=REQUIRED_PARQUET_COLUMNS)
        return df[df["symbol_ticker"] == symbol_ticker].copy()
    return _read_filtered_parquet(path, symbol_ticker)


def _read_volume_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=VOLUME_RANK_COLUMNS)
    return pd.read_parquet(path, columns=VOLUME_RANK_COLUMNS)


def _daily_symbol_volumes(file_paths: list[Path]) -> pd.Series:
    parts = []
    for path in file_paths:
        try:
            raw = _read_volume_raw(path)
        except Exception:
            continue
        if raw.empty:
            continue
        raw = raw.copy()
        raw["trade_volume"] = pd.to_numeric(raw["trade_volume"], errors="coerce").fillna(0.0)
        parts.append(raw[VOLUME_RANK_COLUMNS])
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, ignore_index=True).groupby("symbol_ticker")["trade_volume"].sum().sort_values(ascending=False)


def _month_master_by_month(raw_root: Path) -> dict[str, pd.DataFrame]:
    manifest_path = raw_root / "export_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        rows = []
        for _, row in manifest.drop_duplicates("symbol_ticker").iterrows():
            data_month = str(row["data_month"])
            month_num = MONTH_NAME_TO_NUMBER[data_month]
            expiry = pd.bdate_range(pd.Timestamp(year=2025, month=month_num, day=1), pd.Timestamp(year=2025, month=month_num, day=1) + pd.offsets.MonthEnd(0))[-1]
            rows.append({"symbol_ticker": row["symbol_ticker"], "expiration_dt": expiry})
        master = pd.DataFrame(rows).sort_values(["expiration_dt", "symbol_ticker"])
        return {f"2025-{month:02d}": master for month in range(1, 12)}
    out = {}
    for month in range(1, 12):
        month_dir = raw_root / PARQUET_MONTH_TEMPLATE.format(month=month)
        if not month_dir.exists():
            raise FileNotFoundError(f"Missing Energin month folder: {month_dir}")
        master = _read_symbol_master(month_dir)
        master = master[master["expiration_dt"].notna()].sort_values(["expiration_dt", "symbol_ticker"]).copy()
        out[f"2025-{month:02d}"] = master
    return out


def _raw_parquet_day_inventory(raw_root: Path) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    masters = _month_master_by_month(raw_root)
    manifest_path = raw_root / "export_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        rows = []
        grouped = manifest[manifest["status"].astype(str).str.lower().eq("exported")].groupby("archive_date", sort=True)
        for archive_date, group in grouped:
            day = pd.Timestamp(archive_date)
            symbol_files: dict[str, list[Path]] = {}
            file_paths = []
            for _, item in group.iterrows():
                original = Path(str(item["csv_path"]))
                local_matches = list(raw_root.rglob(original.name))
                if not local_matches:
                    continue
                local_path = local_matches[0]
                symbol = str(item["symbol_ticker"])
                symbol_files.setdefault(symbol, []).append(local_path)
                file_paths.append(local_path)
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "month": f"{day.year:04d}-{day.month:02d}",
                    "day": day,
                    "day_files": sorted(file_paths),
                    "symbol_files": symbol_files,
                    "symbol_volumes": _daily_symbol_volumes(sorted(file_paths)),
                }
            )
        return rows, masters
    rows = []
    for month in range(1, 12):
        month_key = f"2025-{month:02d}"
        month_dir = raw_root / PARQUET_MONTH_TEMPLATE.format(month=month)
        base = month_dir / "parquet" / "brent_1s_bars" / "year=2025" / f"month={month:02d}"
        if not base.exists():
            raise FileNotFoundError(f"Missing parquet path: {base}")
        for day_dir in sorted(base.glob("day=*")):
            if not day_dir.is_dir():
                continue
            day = pd.Timestamp(f"2025-{month:02d}-{day_dir.name.split('=')[-1]}")
            day_files = sorted(day_dir.glob("*.parquet"))
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "month": month_key,
                    "day": day,
                    "day_files": day_files,
                    "symbol_files": {},
                    "symbol_volumes": _daily_symbol_volumes(day_files),
                }
            )
    return rows, masters


def _choose_liquid_contracts(raw_root: Path) -> pd.DataFrame:
    inventory, masters = _raw_parquet_day_inventory(raw_root)
    all_symbols = sorted({sym for row in inventory for sym in row["symbol_volumes"].index})
    volume_rows = []
    for row in inventory:
        volume_rows.append({"date": row["date"], **{sym: float(row["symbol_volumes"].get(sym, 0.0)) for sym in all_symbols}})
    volume_df = pd.DataFrame(volume_rows).set_index("date").sort_index() if volume_rows else pd.DataFrame()
    rolling_5d = volume_df.shift(1).rolling(window=5, min_periods=1).mean()
    chosen_rows = []
    for row in sorted(inventory, key=lambda item: item["date"]):
        day = pd.Timestamp(row["date"])
        master = masters[row["month"]]
        available = master[master["expiration_dt"] >= day].sort_values(["expiration_dt", "symbol_ticker"])
        if available.empty:
            continue
        prompt = available.iloc[0]
        next_contract = available.iloc[1] if len(available) > 1 else None
        prompt_symbol = str(prompt["symbol_ticker"])
        prompt_expiry = pd.Timestamp(prompt["expiration_dt"])
        next_symbol = str(next_contract["symbol_ticker"]) if next_contract is not None else ""
        prompt_days = _business_days_to_expiry(day, prompt_expiry)
        history = rolling_5d.loc[row["date"]] if row["date"] in rolling_5d.index else pd.Series(dtype=float)
        prompt_avg = float(history.get(prompt_symbol, np.nan))
        next_avg = float(history.get(next_symbol, np.nan)) if next_symbol else np.nan
        if not np.isfinite(prompt_avg):
            prompt_avg = float(row["symbol_volumes"].get(prompt_symbol, 0.0))
        if next_symbol and not np.isfinite(next_avg):
            next_avg = float(row["symbol_volumes"].get(next_symbol, 0.0))
        chosen = prompt
        chosen_symbol = prompt_symbol
        chosen_avg = prompt_avg
        roll_reason = "default_most_prompt"
        if next_contract is not None and prompt_days < 5:
            chosen = next_contract
            chosen_symbol = next_symbol
            chosen_avg = next_avg
            roll_reason = "pre_expiry_buffer"
        elif next_contract is not None and np.isfinite(next_avg) and np.isfinite(prompt_avg) and next_avg > prompt_avg:
            chosen = next_contract
            chosen_symbol = next_symbol
            chosen_avg = next_avg
            roll_reason = "volume_crossover"
        chosen_expiry = pd.Timestamp(chosen["expiration_dt"])
        ranks = row["symbol_volumes"].rank(method="min", ascending=False).astype(int)
        chosen_rows.append(
            {
                "date": row["date"],
                "month": row["month"],
                "day": row["day"],
                "day_files": row["day_files"],
                "symbol_files": row.get("symbol_files", {}),
                "chosen_symbol": chosen_symbol,
                "chosen_expiry": chosen_expiry.date().isoformat(),
                "days_to_expiry": _business_days_to_expiry(day, chosen_expiry),
                "chosen_5d_avg_volume": chosen_avg,
                "next_symbol": next_symbol,
                "next_5d_avg_volume": next_avg,
                "roll_reason": roll_reason,
                "total_volume_that_day": float(row["symbol_volumes"].get(chosen_symbol, 0.0)),
                "rank_by_volume_among_available_symbols": int(ranks.get(chosen_symbol, 0)) if chosen_symbol in ranks.index else np.nan,
                "available_symbol_count": int(len(row["symbol_volumes"])),
            }
        )
    return pd.DataFrame(chosen_rows).sort_values("date").reset_index(drop=True)


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


def load_parquet_month(
    raw_root: Path,
    year: int,
    month: int,
    choices: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    frames = []
    files_used = []
    month_key = f"{year:04d}-{month:02d}"
    month_choices = choices[choices["month"] == month_key].copy()
    for _, choice in month_choices.iterrows():
        day = pd.Timestamp(choice["day"])
        selected_symbol = str(choice["chosen_symbol"])
        symbol_files = choice.get("symbol_files", {})
        if isinstance(symbol_files, dict) and selected_symbol in symbol_files:
            selected_files = list(symbol_files[selected_symbol])
        else:
            selected_files = list(choice["day_files"])
        day_frames = []
        for file_path in selected_files:
            raw = _read_filtered_raw(Path(file_path), selected_symbol)
            if not raw.empty:
                files_used.append(str(file_path))
                day_frames.append(raw)
        if not day_frames:
            continue
        raw = pd.concat(day_frames, ignore_index=True)
        raw = _normalize_second_index(raw, "second_utc").rename(columns={"symbol_ticker": "selected_symbol"})
        raw["is_real_print"] = True
        raw["was_forward_filled"] = False
        raw["source_type"] = PROMPT_LIQUID_SOURCE_TYPE
        raw["has_actual_bid_ask"] = raw["bid_last"].notna() & raw["ask_last"].notna()
        session = _session_reindex_one_day(raw, day)
        if not session.empty:
            session["selected_symbol"] = selected_symbol
            session["source_type"] = PROMPT_LIQUID_SOURCE_TYPE
            session["has_actual_bid_ask"] = session["bid_last"].notna() & session["ask_last"].notna()
            session["contract_expiry_date"] = choice["chosen_expiry"]
            session["prompt_delivery_month"] = ""
            session["prompt_month_code"] = ""
            session["contract_selection_rule"] = "daily_prompt_liquid_pre_expiry_or_volume_crossover"
            session["roll_reason"] = choice["roll_reason"]
            session["days_to_expiry"] = int(choice["days_to_expiry"])
            frames.append(session)
    month_df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    selected_symbols = ";".join(sorted(month_choices["chosen_symbol"].dropna().astype(str).unique()))
    coverage = _coverage_row(month_df, year, month, PROMPT_LIQUID_SOURCE_TYPE, selected_symbols, len(files_used))
    coverage.update(
        {
            "selection_rule": "daily_prompt_liquid_pre_expiry_or_volume_crossover",
            "calendar_month": month_key,
            "prompt_delivery_month": "",
            "prompt_month_code": "",
            "expected_prompt_symbol": "",
            "contract_expiry_date": "",
        }
    )
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
    df["is_real_print"] = True
    df["was_forward_filled"] = False
    df["bid_last"] = np.nan
    df["ask_last"] = np.nan
    df["spread"] = np.nan
    df["trade_volume"] = np.nan
    df["selected_symbol"] = "%BRN 1!-ICE"
    df["source_type"] = "csv_mid_dynamic_cost"
    df["has_actual_bid_ask"] = False
    df["contract_expiry_date"] = ""
    df["days_to_expiry"] = np.nan
    df["roll_reason"] = "csv_legacy"
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
            session["contract_selection_rule"] = "csv_continuous_front_symbol_from_legacy_data"
            session["roll_reason"] = "csv_legacy"
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
    daily_choices = _choose_liquid_contracts(raw_root)
    for month in range(1, 12):
        month_df, coverage, files = load_parquet_month(raw_root, 2025, month, daily_choices)
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
    for col in ["mid", "bid_last", "ask_last", "spread", "trade_volume", "days_to_expiry"]:
        if col in combined:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("float32")
    for col in ["is_real_print", "was_forward_filled", "has_actual_bid_ask"]:
        if col in combined:
            combined[col] = combined[col].fillna(False).astype(bool)
    for col in ["selected_symbol", "source_type", "roll_reason", "contract_selection_rule", "contract_expiry_date", "prompt_delivery_month", "prompt_month_code"]:
        if col in combined:
            combined[col] = combined[col].astype("category")
    if combined.index.duplicated().any():
        raise ValueError(f"Duplicate timestamps after stitching: {int(combined.index.duplicated().sum())}")
    manifest = {
        "csv_files_used": csv_files,
        "parquet_files_used": parquet_files,
        "rows": int(len(combined)),
        "first_timestamp_utc": combined.index.min().isoformat() if not combined.empty else "",
        "last_timestamp_utc": combined.index.max().isoformat() if not combined.empty else "",
        "contract_selection_rule": "Jan-Nov parquet uses daily prompt-but-liquid roll: nearest unexpired contract unless <5 business days to expiry or next contract has higher prior 5-day average volume; Dec-Mar CSV uses legacy continuous %BRN 1!-ICE.",
        "chosen_contract_daily": daily_choices.drop(columns=["day", "day_files"], errors="ignore"),
    }
    return combined, pd.DataFrame(coverage_rows), manifest
