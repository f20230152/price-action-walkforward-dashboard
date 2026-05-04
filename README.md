# Price Action Walk-Forward Dashboard

Streamlit dashboard for the rule:

> Enter `a` seconds after price moves `b` cents within `c` seconds, then hold for `d` seconds.

The app reads daily `YYYY-MM-DD.csv` tick files from `data/` when present, otherwise from the parent Downloads folder. It normalizes duplicate `Time` columns, resamples to 1-second last price, and evaluates the rule in cents per one contract.

For Streamlit Community Cloud, keep the app entrypoint as `app.py`.

## Run

```powershell
cd "C:\Users\taran\OneDrive\Desktop\Downloads\price_action_dashboard"
streamlit run app.py
```

## Notes

- Momentum mode buys after an up move and sells after a down move.
- Fade mode does the opposite.
- The walk-forward tab optimizes on the prior N months, then runs the selected a/b/c/d settings out of sample for the next month.
- Round-trip costs are optional and entered in cents per completed trade.
