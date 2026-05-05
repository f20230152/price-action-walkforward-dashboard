from __future__ import annotations

from pathlib import Path

import pandas as pd

from price_action_engine import compute_metrics, evaluate_spike_grid, load_second_prices, run_spike_walk_forward, spike_parameter_grid


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "iterative_spike_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def show(title: str, df: pd.DataFrame, n: int = 15) -> None:
    cols = ["label", "total_pnl_cents", "daily_sharpe", "max_drawdown_cents", "trades", "trades_per_day", "win_rate", "profit_factor"]
    print(f"\n{title}")
    print(df[cols].head(n).to_string(index=False))


def main() -> None:
    bars, _ = load_second_prices(BASE_DIR / "data", "2025-10-01", "2026-03-26", cache_dir=BASE_DIR / ".price_cache", workers=8)
    oos = bars[bars.index.to_period("M").isin(pd.PeriodIndex(["2026-01", "2026-02", "2026-03"], freq="M"))]

    params = []
    # Deep dive: highest PnL pocket around 75c/60s/4h.
    params += spike_parameter_grid(
        [1, 3, 5, 7, 10, 15, 30],
        [65, 70, 72.5, 75, 77.5, 80, 85],
        [45, 60, 75, 90],
        [12600, 14400, 16200, 18000],
        [300],
        [0],
        "momentum",
    )
    # Deep dive: high Sharpe pocket around 75c/60s/2h with volume.
    params += spike_parameter_grid(
        [1, 5, 10, 30, 60],
        [70, 75, 80],
        [45, 60, 75],
        [5400, 7200, 9000, 10800],
        [300],
        [2, 3],
        "momentum",
    )
    # Deep dive: long hold / smaller threshold pocket.
    params += spike_parameter_grid(
        [1, 5, 10, 30, 60, 120],
        [8, 10, 12, 15, 20],
        [120, 180, 240],
        [25200, 28800, 32400],
        [300],
        [0],
        "momentum",
    )
    params = list(dict.fromkeys(params))
    print("quick refine presets", len(params), flush=True)

    fixed = evaluate_spike_grid(oos, params, "total_pnl_cents", 4.0, 1, max_trades_per_day=1).drop(columns=["params"])
    fixed.to_csv(OUT_DIR / "stage2_quick_refine_fixed_oos.csv", index=False)
    show("QUICK REFINE BEST PNL", fixed)
    show("QUICK REFINE BEST SHARPE", fixed[fixed["trades"] >= 5].sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False))

    fixed["rank_pnl"] = fixed["total_pnl_cents"].rank(ascending=False, method="min")
    fixed["rank_sharpe"] = fixed["daily_sharpe"].rank(ascending=False, method="min")
    fixed["rank_dd"] = fixed["max_drawdown_cents"].abs().rank(ascending=True, method="min")
    fixed["combined_rank"] = fixed["rank_pnl"] + fixed["rank_sharpe"] + 0.25 * fixed["rank_dd"]
    combined = fixed.sort_values(["combined_rank", "total_pnl_cents"], ascending=[True, False])
    show("QUICK REFINE BEST COMBINED", combined)

    coarse = pd.read_csv(OUT_DIR / "stage1_coarse_fixed_oos.csv")
    finalists = pd.concat(
        [
            fixed.sort_values(["total_pnl_cents", "daily_sharpe"], ascending=False).head(50),
            fixed.sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).head(50),
            combined.head(50),
            coarse.sort_values(["total_pnl_cents", "daily_sharpe"], ascending=False).head(50),
            coarse.sort_values(["daily_sharpe", "total_pnl_cents"], ascending=False).head(50),
        ],
        ignore_index=True,
    ).drop_duplicates("label")
    wf_params = [
        spike_parameter_grid([int(r.delay_s)], [float(r.threshold_cents)], [int(r.lookback_s)], [int(r.hold_s)], [300], [float(r.volume_multiple)], "momentum")[0]
        for r in finalists.itertuples()
    ]
    print("adaptive finalists", len(wf_params), flush=True)
    details, oos_daily, trades, train_grid = run_spike_walk_forward(
        bars,
        wf_params,
        train_months=3,
        test_months=1,
        objective="total_pnl_cents",
        cost_cents=4.0,
        min_train_trades=1,
        max_trades_per_day=1,
    )
    metrics = compute_metrics(oos_daily, trades)
    details.to_csv(OUT_DIR / "stage3_quick_refine_adaptive_decisions.csv", index=False)
    train_grid.drop(columns=["params"], errors="ignore").to_csv(OUT_DIR / "stage3_quick_refine_train_grid.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "stage3_quick_refine_adaptive_metrics.csv", index=False)
    print("\nADAPTIVE DECISIONS")
    print(details.to_string(index=False))
    print("ADAPTIVE METRICS", metrics)


if __name__ == "__main__":
    main()
