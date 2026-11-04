# Deep Learning Climate-Scenario Stress Testing of Sector Volatility

Class project. Forecasts the next-21-day realised volatility of ten US sector ETFs from daily market and
macro data using an LSTM, a gated Transformer and an LSTM-Transformer hybrid, then stresses the trained
model with NGFS climate scenario paths (Net Zero 2050, Delayed Transition, Current Policies).

## Results (test period 2022 onwards)

| Model | Test R2 vs train mean (approx.) |
|---|---|
| Persistence | 0.28 |
| Ridge (flattened windows) | -0.34 |
| LSTM | 0.24 |
| Gated Transformer | 0.28 |
| LSTM-Transformer hybrid | 0.28 |
| HAR-RV (reference benchmark) | 0.43 |

The deep models explain roughly a quarter of out-of-sample volatility variation. A three-input HAR-RV
regression scores higher, so the deep models are competitive rather than best.

Scenario stress test (Transformer, average change in forecast volatility in 2050): Current Policies about 0%,
Net Zero 2050 about +4%, Delayed Transition about +8%. These magnitudes depend on assumed shock sizes
(`K_BMG`, `K_OIL`, `K_VIX` in the config cell), so the ranking of scenarios is the main finding.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Open `climate_risk_v2.py` in VS Code (Python + Jupyter extensions) and run the `# %%` cells in order.
The first run downloads daily data from Yahoo Finance and FRED and caches it in `data/`.
`data/ngfs_scenarios.csv` must be present (NGFS Phase 5 carbon price and GDP paths, from the IIASA
NGFS Scenario Explorer). Training takes roughly 25 to 50 minutes on a laptop CPU.

Set the environment variable `DEMO=1` to run the pipeline on synthetic data without internet.

## Outputs

Written to `outputs/`: `model_comparison.csv`, `per_sector_r2.csv`, `stress_vol_change.csv`,
`sensitivity.csv` and five charts.

## Limitations

Scenario effects come from assumed input shifts, not estimated relationships. The VIX is both an important
model input and one of the shocked variables. Feature gates are nearly uniform, so they give only a rough
indication of importance. See the report for details.
