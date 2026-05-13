"""
IBKR paper-trading script for the BRN walk-forward spike strategy.

Default deployed parameters come from the latest volatility-scaled walk-forward winner:
percent-vol converted to dollar sigma, monthly rebalance, daily_sharpe objective.

Strategy:
    - Use 1-second price bars from live market data.
    - Only allow signals from 11:00:00 to 23:59:59 Dubai time.
    - Compute rolling daily percent volatility from prior daily closes, converted to dollars.
    - If price moves >= 0.55 x 63D daily dollar-equivalent sigma over 1800 seconds, trade the configured signal mode.
    - Enter 120 seconds after the signal.
    - Hold for 3600 seconds, but always exit by midnight Dubai.
    - Max 1 entry per Dubai calendar day.
    - Quantity: 1 contract.

Install before use:
    pip install ib_insync

Typical paper TWS settings:
    host=127.0.0.1, port=7497, client_id=17

Important:
    This file defaults to PLACE_ORDERS=False. It will calculate signals and log
    intended orders without sending them. After testing the feed/contract in
    paper TWS, set PLACE_ORDERS=True to send paper orders.
"""

from __future__ import annotations

import asyncio
import csv
import math
import signal
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


try:
    from ib_insync import IB, Future, MarketOrder, util
except ImportError:  # Allows `python -m py_compile` and config review without ib_insync installed.
    IB = None
    Future = None
    MarketOrder = None
    util = None


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
KILL_SWITCH_FILE = SCRIPT_DIR / "STOP_TRADING.txt"
DUBAI_TZ = ZoneInfo("Asia/Dubai")
UTC = timezone.utc


@dataclass(frozen=True)
class StrategyConfig:
    delay_s: int = 120
    sigma_multiple: float = 0.55
    vol_window_days: int = 63
    vol_min_observations: int = 21
    vol_method: str = "percent_to_dollar"
    lookback_s: int = 1800
    hold_s: int = 3600
    signal_mode: str = "reversal"
    bad_hour_rule: str = "exit_by_midnight"
    max_trades_per_dubai_day: int = 1
    session_start_dubai: time = time(11, 0, 0)
    session_end_dubai: time = time(0, 0, 0)  # Midnight, exclusive.


@dataclass(frozen=True)
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497  # Paper TWS. Live TWS is usually 7496.
    client_id: int = 17
    readonly: bool = False
    place_orders: bool = False
    symbol: str = "BRN"
    exchange: str = "ICEEU"
    currency: str = "USD"
    contract_month: str = ""  # Example: "202607". Blank lets IBKR try symbol/exchange only.
    quantity: int = 1
    order_type: str = "MKT"
    price_source: str = "last"  # "last" matches the backtest. Use "mid_or_last" only if last trade feed is unavailable.


STRATEGY = StrategyConfig()
IBKR = IBKRConfig()


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._has_header = self.path.exists() and self.path.stat().st_size > 0

    def write(self, row: dict) -> None:
        payload = {key: row.get(key, "") for key in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
            if not self._has_header:
                writer.writeheader()
                self._has_header = True
            writer.writerow(payload)


class WalkForwardSpikeTrader:
    def __init__(self, ib, contract, strategy: StrategyConfig, ibkr: IBKRConfig) -> None:
        self.ib = ib
        self.contract = contract
        self.strategy = strategy
        self.ibkr = ibkr
        self.best_bid: float | None = None
        self.best_ask: float | None = None
        self.last_trade_price: float | None = None
        self.last_bar_second: datetime | None = None
        self.second_bars: deque[tuple[datetime, float]] = deque(maxlen=strategy.lookback_s + strategy.delay_s + strategy.hold_s + 600)
        self.pending_entry: asyncio.Task | None = None
        self.open_position_side: int | None = None
        self.open_entry_price: float | None = None
        self.open_entry_time: datetime | None = None
        self.trades_today: dict[str, int] = {}
        self.daily_sigma_dollars: float | None = None
        self.last_sigma_refresh_dubai_day: str | None = None

        today = datetime.now(DUBAI_TZ).strftime("%Y%m%d")
        self.event_log = CsvLogger(
            LOG_DIR / f"events_{today}.csv",
            [
                "event_utc",
                "event_dubai",
                "event_type",
                "message",
                "signal_time_utc",
                "signal_time_dubai",
                "side",
                "signal_price",
                "lookback_price",
                "move_cents",
                "daily_sigma_dollars",
                "threshold_cents",
                "entry_time_utc",
                "exit_time_utc",
                "order_id",
                "order_status",
                "fill_price",
                "estimated_cost_cents",
                "estimated_pnl_cents",
                "estimated_net_pnl_cents",
            ],
        )

    def log(self, event_type: str, message: str, **kwargs) -> None:
        now_utc = datetime.now(UTC)
        row = {
            "event_utc": now_utc.isoformat(),
            "event_dubai": now_utc.astimezone(DUBAI_TZ).isoformat(),
            "event_type": event_type,
            "message": message,
            **kwargs,
        }
        self.event_log.write(row)
        print(f"{row['event_dubai']} | {event_type} | {message}", flush=True)

    def in_trading_session(self, ts_utc: datetime) -> bool:
        dubai_t = ts_utc.astimezone(DUBAI_TZ).time()
        start = self.strategy.session_start_dubai
        end = self.strategy.session_end_dubai
        if end == time(0, 0, 0):
            return dubai_t >= start
        if start < end:
            return start <= dubai_t < end
        return dubai_t >= start or dubai_t < end

    def current_dubai_day(self) -> str:
        return datetime.now(DUBAI_TZ).strftime("%Y-%m-%d")

    def trades_used_today(self) -> int:
        return self.trades_today.get(self.current_dubai_day(), 0)

    def mark_trade_today(self) -> None:
        day = self.current_dubai_day()
        self.trades_today[day] = self.trades_today.get(day, 0) + 1

    def usable_price(self) -> float | None:
        if self.ibkr.price_source == "last":
            if self.last_trade_price is not None and self.last_trade_price > 0:
                return self.last_trade_price
            return None
        if self.best_bid is not None and self.best_ask is not None and self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2.0
        if self.last_trade_price is not None and self.last_trade_price > 0:
            return self.last_trade_price
        return None

    def on_pending_tickers(self, tickers) -> None:
        for ticker in tickers:
            if ticker.contract.conId != self.contract.conId:
                continue
            if is_valid_price(ticker.bid):
                self.best_bid = float(ticker.bid)
            if is_valid_price(ticker.ask):
                self.best_ask = float(ticker.ask)
            if is_valid_price(ticker.last):
                self.last_trade_price = float(ticker.last)

        price = self.usable_price()
        if price is None:
            return
        now_utc = datetime.now(UTC).replace(microsecond=0)
        self.add_second_bar(now_utc, price)

    def add_second_bar(self, ts_utc: datetime, price: float) -> None:
        if self.last_bar_second is None:
            self.second_bars.append((ts_utc, price))
            self.last_bar_second = ts_utc
            return

        if ts_utc <= self.last_bar_second:
            self.second_bars[-1] = (self.last_bar_second, price)
            return

        fill_ts = self.last_bar_second + timedelta(seconds=1)
        while fill_ts < ts_utc:
            self.second_bars.append((fill_ts, self.second_bars[-1][1]))
            fill_ts += timedelta(seconds=1)

        self.second_bars.append((ts_utc, price))
        self.last_bar_second = ts_utc
        self.evaluate_signal(ts_utc, price)

    def evaluate_signal(self, signal_time_utc: datetime, signal_price: float) -> None:
        if KILL_SWITCH_FILE.exists():
            self.log("KILL_SWITCH", f"{KILL_SWITCH_FILE.name} exists. No new entries.")
            return
        if self.pending_entry is not None and not self.pending_entry.done():
            return
        if self.open_position_side is not None:
            return
        if self.trades_used_today() >= self.strategy.max_trades_per_dubai_day:
            return
        if not self.in_trading_session(signal_time_utc):
            return
        self.refresh_sigma_if_needed()
        if self.daily_sigma_dollars is None or self.daily_sigma_dollars <= 0:
            self.log("NO_SIGMA", "Rolling daily sigma is not available yet.")
            return

        lookback_time = signal_time_utc - timedelta(seconds=self.strategy.lookback_s)
        lookback_price = self.price_at_or_before(lookback_time)
        if lookback_price is None:
            return

        move_cents = (signal_price - lookback_price) * 100.0
        threshold_cents = self.daily_sigma_dollars * 100.0 * self.strategy.sigma_multiple
        if abs(move_cents) < threshold_cents:
            return

        side = 1 if move_cents > 0 else -1
        if self.strategy.signal_mode == "reversal":
            side *= -1

        side_label = "BUY" if side > 0 else "SELL"
        entry_time_utc = signal_time_utc + timedelta(seconds=self.strategy.delay_s)
        scheduled_exit_utc = self.planned_exit_time(entry_time_utc)
        if scheduled_exit_utc is None:
            self.log(
                "SIGNAL_SKIPPED",
                "Signal skipped because entry would occur at/after the Dubai midnight hard stop",
                signal_time_utc=signal_time_utc.isoformat(),
                signal_time_dubai=signal_time_utc.astimezone(DUBAI_TZ).isoformat(),
                side=side_label,
                signal_price=signal_price,
                lookback_price=lookback_price,
                move_cents=move_cents,
                daily_sigma_dollars=self.daily_sigma_dollars,
                threshold_cents=threshold_cents,
                entry_time_utc=entry_time_utc.isoformat(),
                exit_time_utc=scheduled_exit_utc.isoformat(),
            )
            return
        self.log(
            "SIGNAL",
            f"{side_label} signal: move {move_cents:.2f}c vs threshold {threshold_cents:.2f}c; entry scheduled in {self.strategy.delay_s}s",
            signal_time_utc=signal_time_utc.isoformat(),
            signal_time_dubai=signal_time_utc.astimezone(DUBAI_TZ).isoformat(),
            side=side_label,
            signal_price=signal_price,
            lookback_price=lookback_price,
            move_cents=move_cents,
            daily_sigma_dollars=self.daily_sigma_dollars,
            threshold_cents=threshold_cents,
            entry_time_utc=entry_time_utc.isoformat(),
            exit_time_utc=scheduled_exit_utc.isoformat(),
        )
        self.pending_entry = asyncio.create_task(
            self.enter_after_delay(side=side, signal_time_utc=signal_time_utc, scheduled_entry_utc=entry_time_utc)
        )

    def price_at_or_before(self, target_utc: datetime) -> float | None:
        for ts, price in reversed(self.second_bars):
            if ts <= target_utc:
                return price
        return None

    def refresh_sigma_if_needed(self) -> None:
        day = self.current_dubai_day()
        if self.last_sigma_refresh_dubai_day == day and self.daily_sigma_dollars is not None:
            return
        bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr="120 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )
        closes = [float(bar.close) for bar in bars if is_valid_price(bar.close)]
        required = min(self.strategy.vol_window_days, self.strategy.vol_min_observations) + 1
        if len(closes) < required:
            self.log("SIGMA_REFRESH_FAILED", f"Need at least {required} daily closes, got {len(closes)}.")
            return
        if self.strategy.vol_method == "dollar":
            values = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
            window = values[-self.strategy.vol_window_days :] if len(values) >= self.strategy.vol_window_days else values
            self.daily_sigma_dollars = sample_std(window)
        elif self.strategy.vol_method == "percent_to_dollar":
            returns = [(closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0]
            window = returns[-self.strategy.vol_window_days :] if len(returns) >= self.strategy.vol_window_days else returns
            self.daily_sigma_dollars = sample_std(window) * closes[-1]
        else:
            raise ValueError(f"Unknown vol method: {self.strategy.vol_method}")
        self.last_sigma_refresh_dubai_day = day
        self.log(
            "SIGMA_REFRESHED",
            f"Daily dollar sigma refreshed: {self.daily_sigma_dollars:.4f}; threshold={self.daily_sigma_dollars * self.strategy.sigma_multiple * 100.0:.2f}c",
            daily_sigma_dollars=self.daily_sigma_dollars,
            threshold_cents=self.daily_sigma_dollars * self.strategy.sigma_multiple * 100.0,
        )

    def planned_exit_time(self, entry_utc: datetime) -> datetime | None:
        hold_exit = entry_utc + timedelta(seconds=self.strategy.hold_s)
        if self.strategy.bad_hour_rule != "exit_by_midnight":
            return hold_exit
        entry_dubai = entry_utc.astimezone(DUBAI_TZ)
        midnight_dubai = entry_dubai.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        midnight_utc = midnight_dubai.astimezone(UTC)
        planned = min(hold_exit, midnight_utc)
        return planned if planned > entry_utc else None

    def trade_passes_bad_hour_rule(self, entry_utc: datetime, exit_utc: datetime) -> bool:
        rule = self.strategy.bad_hour_rule
        entry_dubai = entry_utc.astimezone(DUBAI_TZ)
        exit_dubai = exit_utc.astimezone(DUBAI_TZ)
        if rule == "avoid_exit_1_3":
            return not (1 <= exit_dubai.hour < 3)
        if rule == "avoid_hold_1_3":
            return not overlaps_bad_window(entry_dubai, exit_dubai)
        if rule == "exit_by_midnight":
            return exit_dubai.date() == entry_dubai.date() or exit_dubai.time() == time(0, 0, 0)
        raise ValueError(f"Unknown bad-hour rule: {rule}")

    async def enter_after_delay(self, side: int, signal_time_utc: datetime, scheduled_entry_utc: datetime) -> None:
        delay = max(0.0, (scheduled_entry_utc - datetime.now(UTC)).total_seconds())
        await asyncio.sleep(delay)

        if KILL_SWITCH_FILE.exists():
            self.log("ENTRY_CANCELLED", "Kill switch present before entry.")
            return
        if self.open_position_side is not None:
            self.log("ENTRY_CANCELLED", "Position already open.")
            return
        if self.trades_used_today() >= self.strategy.max_trades_per_dubai_day:
            self.log("ENTRY_CANCELLED", "Daily trade cap reached before entry.")
            return

        action = "BUY" if side > 0 else "SELL"
        entry_price_snapshot = self.usable_price()
        self.mark_trade_today()

        if not self.ibkr.place_orders:
            self.open_position_side = side
            self.open_entry_price = entry_price_snapshot
            self.open_entry_time = datetime.now(UTC)
            self.log(
                "PAPER_DRY_ENTRY",
                f"Would send {action} {self.ibkr.quantity} {self.contract.localSymbol or self.contract.symbol}",
                signal_time_utc=signal_time_utc.isoformat(),
                entry_time_utc=self.open_entry_time.isoformat(),
                side=action,
                fill_price=entry_price_snapshot,
            )
            planned_exit = self.planned_exit_time(self.open_entry_time)
            if planned_exit is None:
                self.log("EXIT_SCHEDULE_FAILED", "Could not schedule exit before Dubai midnight.")
                return
            asyncio.create_task(self.exit_at_time(side=side, dry_run=True, planned_exit_utc=planned_exit))
            return

        order = MarketOrder(action, self.ibkr.quantity)
        trade = self.ib.placeOrder(self.contract, order)
        self.log("ENTRY_ORDER_SENT", f"Sent {action} {self.ibkr.quantity}", order_id=order.orderId, side=action)
        await wait_for_trade_done(self.ib, trade)
        fill_price = average_fill_price(trade) or entry_price_snapshot
        self.open_position_side = side
        self.open_entry_price = fill_price
        self.open_entry_time = datetime.now(UTC)
        self.log(
            "ENTRY_FILLED",
            f"{action} filled",
            order_id=order.orderId,
            order_status=trade.orderStatus.status,
            fill_price=fill_price,
            entry_time_utc=self.open_entry_time.isoformat(),
            side=action,
        )
        planned_exit = self.planned_exit_time(self.open_entry_time)
        if planned_exit is None:
            self.log("EXIT_SCHEDULE_FAILED", "Could not schedule exit before Dubai midnight.")
            return
        asyncio.create_task(self.exit_at_time(side=side, dry_run=False, planned_exit_utc=planned_exit))

    async def exit_at_time(self, side: int, dry_run: bool, planned_exit_utc: datetime) -> None:
        await asyncio.sleep(max(0.0, (planned_exit_utc - datetime.now(UTC)).total_seconds()))
        exit_action = "SELL" if side > 0 else "BUY"
        exit_price_snapshot = self.usable_price()
        entry_price = self.open_entry_price

        if dry_run:
            estimated_pnl = estimate_pnl_cents(side, entry_price, exit_price_snapshot)
            estimated_cost = dynamic_cost_cents(entry_price)
            estimated_net_pnl = estimated_pnl - estimated_cost if estimated_pnl is not None and estimated_cost is not None else None
            self.log(
                "PAPER_DRY_EXIT",
                f"Would send {exit_action} {self.ibkr.quantity}",
                side=exit_action,
                fill_price=exit_price_snapshot,
                estimated_cost_cents=estimated_cost,
                estimated_pnl_cents=estimated_pnl,
                estimated_net_pnl_cents=estimated_net_pnl,
                exit_time_utc=datetime.now(UTC).isoformat(),
            )
            self.clear_position()
            return

        order = MarketOrder(exit_action, self.ibkr.quantity)
        trade = self.ib.placeOrder(self.contract, order)
        self.log("EXIT_ORDER_SENT", f"Sent {exit_action} {self.ibkr.quantity}", order_id=order.orderId, side=exit_action)
        await wait_for_trade_done(self.ib, trade)
        fill_price = average_fill_price(trade) or exit_price_snapshot
        estimated_pnl = estimate_pnl_cents(side, entry_price, fill_price)
        estimated_cost = dynamic_cost_cents(entry_price)
        estimated_net_pnl = estimated_pnl - estimated_cost if estimated_pnl is not None and estimated_cost is not None else None
        self.log(
            "EXIT_FILLED",
            f"{exit_action} filled",
            order_id=order.orderId,
            order_status=trade.orderStatus.status,
            fill_price=fill_price,
            estimated_cost_cents=estimated_cost,
            estimated_pnl_cents=estimated_pnl,
            estimated_net_pnl_cents=estimated_net_pnl,
            exit_time_utc=datetime.now(UTC).isoformat(),
        )
        self.clear_position()

    def clear_position(self) -> None:
        self.open_position_side = None
        self.open_entry_price = None
        self.open_entry_time = None
        self.pending_entry = None


def is_valid_price(value) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and x > 0


def estimate_pnl_cents(side: int, entry_price: float | None, exit_price: float | None) -> float | None:
    if entry_price is None or exit_price is None:
        return None
    return side * (exit_price - entry_price) * 100.0


def dynamic_cost_cents(entry_price: float | None) -> float | None:
    if entry_price is None:
        return None
    return abs(entry_price) * 0.04


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def overlaps_bad_window(entry_dubai: datetime, exit_dubai: datetime) -> bool:
    day = entry_dubai.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = exit_dubai.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end_day:
        bad_start = day.replace(hour=1)
        bad_end = day.replace(hour=3)
        if entry_dubai < bad_end and exit_dubai > bad_start:
            return True
        day += timedelta(days=1)
    return False


def average_fill_price(trade) -> float | None:
    fills = getattr(trade, "fills", None) or []
    if not fills:
        return None
    qty_price = [(float(fill.execution.shares), float(fill.execution.price)) for fill in fills]
    total_qty = sum(qty for qty, _ in qty_price)
    if total_qty <= 0:
        return None
    return sum(qty * price for qty, price in qty_price) / total_qty


async def wait_for_trade_done(ib, trade, timeout_s: int = 60) -> None:
    start = datetime.now(UTC)
    while not trade.isDone():
        await asyncio.sleep(0.25)
        if (datetime.now(UTC) - start).total_seconds() > timeout_s:
            raise TimeoutError(f"Order did not complete within {timeout_s}s: {trade}")


def build_contract(ibkr: IBKRConfig):
    if Future is None:
        raise RuntimeError("ib_insync is not installed. Run: pip install ib_insync")
    if ibkr.contract_month:
        return Future(
            symbol=ibkr.symbol,
            lastTradeDateOrContractMonth=ibkr.contract_month,
            exchange=ibkr.exchange,
            currency=ibkr.currency,
        )
    return Future(symbol=ibkr.symbol, exchange=ibkr.exchange, currency=ibkr.currency)


async def main() -> None:
    if IB is None:
        raise RuntimeError("ib_insync is not installed. Run: pip install ib_insync")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting BRN walk-forward spike strategy")
    print(f"Parameters: {STRATEGY}")
    print(f"IBKR config: {IBKR}")
    if not IBKR.place_orders:
        print("PLACE_ORDERS=False: script will log intended paper orders but will not send them.")
    elif IBKR.port != 7497:
        raise RuntimeError("Refusing to place orders unless IBKR.port is 7497 paper TWS. Change this safety check manually if needed.")

    ib = IB()
    await ib.connectAsync(IBKR.host, IBKR.port, clientId=IBKR.client_id, readonly=IBKR.readonly)
    contract = build_contract(IBKR)
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        raise RuntimeError(f"IBKR could not qualify contract: {contract}")
    contract = qualified[0]
    print(f"Qualified contract: {contract}")

    trader = WalkForwardSpikeTrader(ib, contract, STRATEGY, IBKR)
    ticker = ib.reqMktData(contract, "", False, False)
    ib.pendingTickersEvent += trader.on_pending_tickers
    trader.log("START", f"Subscribed to market data: {ticker}")

    stop_event = asyncio.Event()

    def request_stop(*_args) -> None:
        trader.log("STOP_REQUESTED", "Shutdown requested.")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    await stop_event.wait()

    ib.pendingTickersEvent -= trader.on_pending_tickers
    ib.cancelMktData(contract)
    ib.disconnect()
    trader.log("STOPPED", "Disconnected from IBKR.")


if __name__ == "__main__":
    if "--check-config" in sys.argv:
        print("Strategy config:", STRATEGY)
        print("IBKR config:", IBKR)
        print(f"Kill switch path: {KILL_SWITCH_FILE}")
        print("Config check passed. Remove --check-config to connect to IBKR.")
        raise SystemExit(0)
    if util is not None:
        util.patchAsyncio()
    asyncio.run(main())
