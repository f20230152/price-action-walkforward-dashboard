# Fresh Brent Price-Action Walk-Forward Dashboard

Clean-slate Streamlit dashboard for the Jan 2025-Mar 2026 Brent price-action research run.

The pipeline:

- Loads Jan-Nov 2025 from external Energin parquet files on `D:\Energin Raw Data`.
- Loads Dec 2025-Mar 2026 from committed CSV files in `data/`.
- Runs 3-month and 6-month rolling walk-forward schedules.
- Selects parameters with train-only robustness gates, not OOS PnL.
- Compares mid plus dynamic cost against actual bid/ask fills where available.
- Publishes an explicit verdict: stable edge, weak/episodic edge, or no stable edge.

## Run Research

```powershell
cd "C:\Users\taran\OneDrive\Desktop\Downloads\price_action_dashboard"
python -m src.run_research
```

## Run Dashboard

```powershell
streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Streamlit Cloud reads committed `outputs/` CSV and JSON files. It does not need the D-drive raw data at runtime.

