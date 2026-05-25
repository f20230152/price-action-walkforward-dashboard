# Codex Prompt — Fresh Brent Price-Action Walk-Forward Dashboard

You are taking over the repository at `C:\Users\taran\OneDrive\Desktop\Downloads\price_action_dashboard`. The previous walk-forward research produced a result that was statistically fragile (one trade was 73% of OOS PnL; the result inverted under realistic bid/ask fills). The owner wants a complete clean-slate rebuild that is designed from the ground up to surface **stable, robust, walk-forward-validated parameter sets** — not just the highest-PnL set on paper. If no stable edge exists in the data, the dashboard must say so honestly rather than ship a fragile winner.

Read this prompt end-to-end before writing any code. Treat it as the source of truth.

---

## 1. Wipe Scope — Clean Slate

Delete everything inside `price_action_dashboard/` **except**:

- `data/` — the existing old CSV files (Dec 2025 – Mar 2026 source)
- `chat_export_day1.md` — historical record of the previous attempt; do not delete, do not modify
- `.git/`, `.gitignore`, `README.md` if present
- This file (`codex_prompt.md`) — keep it for reference

Specifically remove: `price_action_engine.py`, `expanded_walkforward_research.py`, any `app.py`/Streamlit files, any old research scripts, `__pycache__/`, the entire `outputs/` directory, and any prior tab/dashboard code. Do not preserve any artifact of the previous run. The new dashboard will be the only dashboard.

The raw D-drive parquet data at `D:\Energin Raw Data\brent_local_2025_01` through `brent_local_2025_11` is external and stays untouched.

Commit this wipe as its own commit before writing new code, so the history is clear.

---

## 2. Data Scope

Build a single combined research dataset:

- **Jan 2025 – Nov 2025**: Energin parquet files only, from `D:\Energin Raw Data\brent_local_2025_<MM>/`. Columns to read: `second_utc`, `mid`, `bid_last`, `ask_last`, `spread`, `trade_volume`. Pick the front contract per month from the available parquet contracts.
- **Dec 2025 – Mar 2026**: old CSVs from `price_action_dashboard/data/` only.
- Do **not** use old Oct/Nov 2025 CSVs (they would overlap with the parquet and double-count).
- Normalize all timestamps to UTC second bars.
- Stitch into a single time-indexed DataFrame; assert no duplicate timestamps after stitching.
- Use `mid` as the signal price throughout.
- Preserve `bid_last` and `ask_last` for the actual-fills model.

Write a `data_coverage.csv` showing for each month: source type (parquet/csv), days loaded, missing days, first timestamp, last timestamp, and row count.

---

## 3. Strategy Signal (do not change without reason)

At each second `t` inside the Dubai session:

1. Compute the absolute price change in the window `(t − c, t)` using mid.
2. Compute the rolling volatility `σ` over the same window `c` (use absolute returns or close-to-close standard deviation — pick one and document it).
3. If `|Δmid| > b × σ`, fire a signal.
4. Enter `a` seconds later. Direction is **continuation** (same sign as the move) or **reversal** (opposite sign) — both are candidates in the grid.
5. Exit at `min(entry_time + d, Dubai midnight)`.

Hard constraints (do not weaken):

- Session: 11:00 AM – midnight Dubai time (UTC+4).
- Maximum 1 trade per calendar Dubai day per parameter set.
- Hard midnight stop overrides hold time.

---

## 4. Walk-Forward Design

Run **two walk-forward schedules side-by-side** so we can see whether short training windows are causing winner instability:

- **3-month train / 1-month test**: train Jan–Mar → test Apr; roll monthly through train Dec–Feb → test Mar 2026. Yields 12 OOS folds.
- **6-month train / 1-month test**: train Jan–Jun → test Jul; roll monthly through train Sep–Feb → test Mar 2026. Yields 9 OOS folds.

For every fold, evaluate the **full parameter grid under both fill models**, then apply the selection rule from §6 to pick a winner.

---

## 5. Parameter Universe — Coarsen, Then Refine

**Stage 1 — coarse grid** (run on every fold, both schedules, both fill models):

- `a` (entry delay, s): 60, 120, 300
- `b` (volatility multiple): 0.3, 0.5, 0.7, 0.9
- `c` (lookback, s): 900, 1800, 3600
- `d` (hold, s): 1800, 7200, 14400
- direction: continuation, reversal

That's 3 × 4 × 3 × 3 × 2 = 216 combos per fold. Compute is not a constraint, so this is comfortable.

**Stage 2 — refine around survivors**: after Stage 1 across all folds, identify the "robust region" (defined in §6). Build a finer grid by interpolating one step between the surviving grid points (e.g., if `a=120` and `a=300` both survive, add `a=180, 240`). Run Stage 2 with the same walk-forward schedules and fill models. The Stage-2 winner is the final reported winner.

Save the parameter universe for each stage to `parameter_universe_stage1.csv` and `parameter_universe_stage2.csv`.

---

## 6. Composite Robustness Score — How a Winner Is Picked

For every (parameter set × fold × fill model), compute the following on the **training** data (never the test data — selection must be in-sample):

1. **OOS-style consistency proxy on train**: split the training window into 3 sub-folds and require that the parameter set is profitable in at least 2 of the 3 sub-folds. Sets that fail this gate are eliminated before scoring.
2. **Sharpe on training**.
3. **Top-trade survival**: train PnL after removing the single largest winning trade. Sets where this goes negative are eliminated.
4. **Profit factor** (gross winners / gross losers).
5. **Max drawdown** (penalty).
6. **Parameter-neighborhood support**: for each surviving set, count the number of immediate grid neighbors (one step away in any single axis) that also survived gates 1 and 3. The winner must have at least 3 surviving neighbors.

Compose a normalized score per set:

```
composite = 0.30 * z(sharpe)
          + 0.25 * z(top_trade_removed_pnl)
          + 0.15 * z(profit_factor)
          + 0.15 * z(neighborhood_count)
          + 0.15 * z(-max_drawdown)
```

(`z` = z-score against the surviving population in that fold.) The winner for that fold is the highest-composite set among those that passed all gates. Save composite component values for every surviving set per fold to `train_rankings.csv`.

The chosen set is then evaluated **once on the held-out test month** to produce the OOS row.

---

## 7. Robustness Tests on the OOS Series

After the walk-forward run for each (schedule × fill model), apply this battery to the concatenated OOS trade series:

- **Top-N trade removal**: report total OOS PnL after removing top 1, top 3, top 5 trades. Strategy passes if PnL stays positive at top-3 removal.
- **Top-month removal**: remove the single best OOS month, check remainder is still positive.
- **Win-month rate**: fraction of OOS months with positive PnL. Target ≥ 0.65.
- **Parameter stability**: how many distinct parameter sets won across the folds. Lower is better (target ≤ 4 distinct sets across 12 folds).
- **Randomized-entry benchmark**: for the chosen rule, generate 1,000 bootstrap runs where entry timestamps are shuffled within each session (preserving 1-trade-per-day and session window). Compare the real OOS PnL distribution against the random distribution; report a one-sided p-value. Strategy passes if p < 0.05.
- **Bootstrap CI**: 95% bootstrap CI for the mean per-trade PnL. Strategy passes if the lower bound is > 0.

Write all of this to `robustness_report.csv`.

---

## 8. Fill Models — Both, Side-by-Side

Run the entire pipeline twice:

- **Model A — mid + dynamic cost**: enter and exit at `mid`, apply the existing dynamic round-trip cost (carry over the `DYNAMIC_COST_CENTS_PER_PRICE_UNIT = 0.04` constant from the previous engine). Works for all months.
- **Model B — actual bid/ask where available**: Jan–Nov 2025 parquet months use real fills (long entry = ask, long exit = bid, short entry = bid, short exit = ask). Dec 2025 – Mar 2026 CSV months fall back to mid + dynamic cost because bid/ask is unavailable. Document this in the dashboard as a footnote everywhere Model B numbers appear.

Show both models everywhere. The reported winner under Model B is the realistic-trading benchmark; the gap between A and B is itself a diagnostic.

---

## 9. Verdict Logic

After all of the above, the dashboard renders a single **Verdict** at the top of every tab:

- **STABLE EDGE FOUND** — only if the Model B (actual fills) winner passes: top-3-trade survival positive, top-month survival positive, win-month rate ≥ 0.65, randomized-entry p < 0.05, bootstrap mean-trade lower CI > 0, **and** the same parameter family wins in both the 3/1 and 6/1 schedules.
- **WEAK / EPISODIC EDGE** — Model A passes most tests but Model B does not, or one of the robustness gates fails by a small margin.
- **NO STABLE EDGE** — the Model B winner fails 2+ of the robustness gates, or the winning parameter set changes wildly between schedules.

In the WEAK and NO STABLE EDGE cases, the dashboard still shows a "least-bad" parameter set with **every failed test highlighted in red**. Never present a fragile result as a recommendation.

---

## 10. Final Full-Period Backtest

For the **final winning parameter set under each fill model** (the most-common winner across the OOS folds, or if there's no clear winner, the Model B Stage-2 winner of the final fold), run a single backtest across the entire Jan 1 2025 – Mar 31 2026 window. This is **in-sample** for the training months and must be labeled as such — its purpose is purely visual continuity for the user, not a robustness claim. Save to `full_period_backtest_trades.csv` and `full_period_backtest_equity.csv`.

---

## 11. Dashboard (Streamlit)

Single app, single dashboard. No legacy tabs. Tabs (in order):

1. **Verdict & Summary** — the verdict from §9, the chosen parameter set under each fill model, top-level OOS Sharpe / PnL / win-month rate / robustness pass-fail badges.
2. **Walk-Forward Schedule** — explanation of 3/1 vs 6/1; per-fold winner table for both schedules and both fill models, showing composite score components.
3. **OOS Performance** — cumulative equity curve (line chart) and per-month PnL (bar chart) for the winning rule under each fill model.
4. **Parameter Stability** — heatmaps of composite score across (a, b) and (c, d) for each fold; visual check that winners sit in broad green regions, not on isolated peaks.
5. **Robustness Report** — table of every test from §7 with pass/fail badges; randomized-entry histogram with the real PnL marked.
6. **All Trades** — full trade list filterable by fill model, schedule, fold; columns: entry_time_utc, entry_time_dubai, direction, entry_px, exit_time_utc, exit_px, hold_seconds, pnl_cents, fill_model, fold, parameter_set. Streamlit download button to export the visible table as CSV.
7. **Monthly Rebalance Decisions** — for each fold, the winning set + the runners-up + which gates eliminated which candidates. Downloadable CSV.
8. **Full-Period Backtest** — the §10 result, with an explicit "training months are in-sample" disclaimer banner.
9. **Data Coverage** — the `data_coverage.csv` table plus a row-count timeline.

Every table on every tab must have a "Download CSV" button.

---

## 12. File Layout

```
price_action_dashboard/
  data/                              # untouched old CSVs
  src/
    __init__.py
    data_loader.py                   # parquet + CSV ingestion, stitching
    engine.py                        # signal, fills, trade construction
    walkforward.py                   # fold scheduling, candidate eval
    selection.py                     # composite score, gates, winner pick
    robustness.py                    # all §7 tests
    full_period.py                   # §10 backtest
    run_research.py                  # orchestrator: writes everything to outputs/
  app.py                             # Streamlit dashboard
  outputs/
    data_coverage.csv
    parameter_universe_stage1.csv
    parameter_universe_stage2.csv
    train_rankings.csv               # per-fold composite components for survivors
    decisions.csv                    # per-fold winner + runners-up + elimination reasons
    trades.csv                       # all OOS trades, every schedule × fill model
    daily_pnl.csv
    monthly_pnl.csv
    oos_equity.csv
    fold_winners.csv                 # per-fold winning parameter set under each (schedule, fill)
    fill_model_comparison.csv        # A vs B summary
    robustness_report.csv
    randomized_entry_distribution.csv
    full_period_backtest_trades.csv
    full_period_backtest_equity.csv
    verdict.json                     # the §9 verdict + reasons
    validation_checks.csv
  codex_prompt.md                    # this file
  chat_export_day1.md                # untouched
  README.md
  requirements.txt
```

---

## 13. Validation — Must Pass Before Committing

Write a `validation_checks.csv` capturing pass/fail for each of:

- Data: Jan–Nov parquet only used for those months; Dec–Mar CSV only used for those months; no old Oct/Nov CSV touched.
- Data: zero duplicate timestamps after stitching.
- Engine: every trade entry timestamp inside 11:00–24:00 Dubai.
- Engine: max one trade per (parameter_set, day).
- Engine: every exit timestamp ≤ entry + d, and ≤ Dubai midnight of entry day.
- Engine: Model B fill sides correct — long entry = ask, long exit = bid, short entry = bid, short exit = ask, only on parquet months.
- Walk-forward: every test month's data is strictly outside the training window used to pick its winner.
- Selection: the picked winner per fold is the highest composite among gate-survivors (assert against a recomputed top-3 list).
- Robustness: randomized-entry baseline used at least 1,000 resamples.

Run `python -m py_compile` on every `.py` file. Run `python -m src.run_research` end-to-end. Boot Streamlit at `127.0.0.1:8502` and curl `http://127.0.0.1:8502` — expect HTTP 200.

---

## 14. Acceptance Criteria

The job is done when:

1. The clean-slate wipe is committed as its own commit.
2. The full pipeline runs end-to-end without error.
3. All `outputs/*.csv` files exist and load cleanly into the Streamlit app.
4. Streamlit serves on `127.0.0.1:8502` and the Verdict tab shows one of the three verdicts with all supporting numbers.
5. `validation_checks.csv` shows zero failures.
6. Every tab has working CSV downloads.
7. The work is pushed to `https://github.com/f20230152/price-action-walkforward-dashboard.git` on `main` in a single coherent commit (or a small clean series).
8. The final message to the user contains: the verdict, the chosen parameter set under Model A and under Model B, OOS PnL/Sharpe/win-month rate under each model, top-3-trade-removed PnL under each model, randomized-entry p-value under each model, and a one-line plain-English summary of whether the strategy is trustworthy.

---

## 15. Non-Negotiables (please re-read)

- Do not invent a winner if the robustness tests fail. The owner specifically wants intellectual honesty over a pretty number.
- Do not use future data inside any training window. The walk-forward must be strictly causal.
- Do not optimize against the OOS test data — the OOS month is scored once with the train-picked winner.
- Do not skip the randomized-entry benchmark, even if Streamlit looks good without it. It is the single most important sanity check.
- Do not silently fall back from actual bid/ask to mid on the parquet months. That choice destroyed the previous run's interpretability.

If anything in this prompt seems wrong or under-specified, stop and ask before guessing.
