# LSTM Stock Forecast

A PyTorch LSTM that predicts a stock's **next-day closing price**, tested honestly against simple baselines.

This started as the NeuralNine YouTube tutorial *"Stock Price Prediction in Python with PyTorch"*. That tutorial model looked impressive on a chart but had leakage and evaluation bugs. I rebuilt it step by step into a correct, reproducible project with fair measurement. In the end, **no model reliably beats the simple rule "tomorrow's close = today's close."** Showing that clearly is the main result of the project.

> ⚠️ **Educational project only.** This is not financial advice and not a trading system. Do not use these numbers to buy or sell anything.

---

## Highlights

- **Fixed the tutorial's bugs:** data leakage (the scaler saw test data), an off-by-one that dropped the last day, and plot dates shifted by one day.
- **Predicts log returns instead of raw prices.** The original model went flat once prices passed anything seen in training (~$300+). Predicting returns fixed this.
- **Proper training:** 70/10/20 chronological split, mini-batches, dropout, gradient clipping, early stopping, and a 3-model ensemble.
- **Honest evaluation:** every result is shown next to two baselines, *persistence* and *average move*, using RMSE, MAE, MAPE and directional accuracy.
- **Experiments:** feature groups (technical indicators, S&P 500, VIX), a 24-setup hyperparameter search, GRU and linear-regression comparisons, walk-forward testing, and other tickers (MSFT, SPY, NVDA).
- **Live forecast:** `predict_next_day()` downloads the latest data, retrains, and forecasts the next trading day.
- **Streamlit dashboard:** pick a ticker and compare the model against the baselines.

---

## Results (AAPL, test period 2025-05-28 → 2026-09-23, 333 days)

| Phase | Change | Test RMSE | Directional acc. | Persistence RMSE |
|---|---|---|---|---|
| v0 | Original tutorial (leaky, buggy) | $8.48* | — | — |
| 1 | Correctness fixes | $10.92 | 50.5% | $4.19 |
| 2 | Predict log returns | $4.29 | 51.7% | $4.19 |
| 3 | Proper training + early stopping | $4.18 | 52.9% | $4.19 |
| 4 | 3-model ensemble (extra features didn't help) | $4.17 | 48.0% | $4.19 |
| 5 | Best of 24 hyperparameter setups | $4.22 | 47.7% | $4.19 |

\*v0's number is not comparable: it used leaked data and a different test window.

**What this means**

- Phases 3–4 beat persistence by only 0.3–0.4%. That edge comes from learning the stock's average upward drift, and it ties the "average move" rule ($4.174).
- **Walk-forward test** (6 blocks, 756 days): LSTM $3.878 vs persistence $3.879 vs average move $3.873. That's a three-way tie.
- **Other tickers** (MSFT, SPY, NVDA): the model did worse than persistence on all three.
- Always guessing "up" gets 54.4% directional accuracy on the test period. No model did better.

Full experiment log and settings: [`RESULTS.md`](RESULTS.md).

---

## Project structure

```
├── stockpredict.ipynb              # Original tutorial model (v0, kept as reference)
├── enhanced_stockpredict_v2.ipynb  # Phases 1–6, explained step by step
├── stockpredict_pipeline.ipynb     # Thin notebook that runs the src/ pipeline
├── app.py                          # Streamlit dashboard
├── config.yaml                     # Final settings (model, split, training)
├── RESULTS.md                      # One row per experiment
├── src/
│   ├── data.py       # Pinned CSV snapshots + latest-data download (yfinance)
│   ├── features.py   # Returns and technical features, past data only
│   ├── model.py      # LSTM / GRU model
│   ├── train.py      # Dataset, early stopping, ensemble training
│   ├── evaluate.py   # Metrics and baseline comparison
│   └── forecast.py   # predict_next_day() and multi-day recursive forecast
├── data/             # Saved price snapshots (AAPL, MSFT, NVDA, SPY, ^GSPC, ^VIX)
└── checkpoints/      # Saved Phase 3 model
```

---

## Getting started

```bash
git clone https://github.com/<your-username>/lstm-stock-forecast.git
cd lstm-stock-forecast
pip install torch yfinance pandas numpy "scikit-learn>=1.4" matplotlib pyyaml streamlit jupyter
```

**Run the notebooks**

```bash
jupyter notebook enhanced_stockpredict_v2.ipynb
```

**Launch the dashboard**

```bash
streamlit run app.py
```

**Forecast tomorrow's close**

```python
from src import load_config
from src.forecast import predict_next_day

result = predict_next_day("AAPL", load_config())
print(result["forecast_for"], round(result["lstm_forecast"], 2))
```

Everything runs on CPU, so you don't need a GPU.

---

## Reproducibility

- **Pinned data:** experiments read saved CSVs in `data/` (2020-01-01 → 2026-09-23). Yahoo Finance returns slightly different adjusted prices on each download, so the saved files keep results identical. Delete them to refresh.
- **Fixed seeds** for `random`, `numpy` and `torch`. PyTorch is pinned to 4 CPU threads.
- **No leakage:** the split is chronological, scalers are fit on training data only, and features use only past data (checked by recomputing on truncated data).
- **Test set touched last:** all tuning uses the validation period only.

---

## What I learned

- A good-looking "actual vs predicted" chart can hide a model that just copies yesterday's price.
- Always compare against a simple baseline before claiming a model works.
- Validation gains often don't carry over to the test period. With only 166 validation days, picking from many setups rewards luck.
- Next-day stock prices are close to a random walk. Honest evaluation shows that, and adding complexity doesn't change it.

---

## Credits

- Based on [NeuralNine's "Stock Price Prediction in Python with PyTorch"](https://www.youtube.com/@NeuralNine) tutorial.
- Price data from Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance).
