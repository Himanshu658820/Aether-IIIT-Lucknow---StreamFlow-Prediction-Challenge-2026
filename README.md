---
title: Flood Prediction AI - Ganga-Brahmaputra Basin
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
license: mit
tags:
  - streamflow
  - hydrology
  - flood-prediction
  - lightgbm
  - xgboost
  - time-series
  - india
  - kaggle
---

# 🌊 Flood StreamFlow Prediction — v9

**IIIT Lucknow StreamFlow Prediction Challenge 2026**

A production-ready ML system for predicting **next-day river streamflow** across **367 Indian gauge stations** using an ensemble of LightGBM and XGBoost with three key innovations:

## 🚀 Key Innovations

| Innovation | Description |
|-----------|-------------|
| **Delta-Target** | Predicts `Δlog(Q)` instead of `log(Q_tomorrow)` — 10× lower target variance |
| **Flood-Weighted Loss** | High-flow rows get up to 5× gradient signal |
| **Per-Station Models** | 10 specialist models for the hardest gauge stations |
| **Scipy Ensemble** | KGE-optimised model blending with 40 random restarts |
| **Bias Correction** | Per-station multiplicative scale calibration |

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Validation KGE** | **0.999875** |
| Pearson r | 0.9999 |
| Variability (α) | 0.9999 |
| Bias (β) | 1.0000 |
| NSE | 0.999827 |
| Δ vs baseline | +0.001993 |
| Δ vs v8 | +0.002615 |

## 🏗 Model Architecture

- **LightGBM** (511 leaves, lr=0.03, GPU) — val KGE 0.999832
- **XGBoost** (max_depth=9, CUDA) — val KGE 0.999808
- **10 per-station LGB specialists** for worst stations
- **94 engineered features**: flow lags, rain ratios, monsoon flags, pre-flood alarm signals

## 📱 App Features

- **🏠 Home** — Hero dashboard with live 365-day demo
- **🔮 Predict** — Upload CSV → Download predictions
- **📊 Explorer** — Interactive charts (flow analysis, rain–flow correlation, station KGE)
- **🧠 How It Works** — Pipeline explainer with delta-target visualisation

## 📂 To Enable Live Predictions

Upload these files to the Space:
```
lgb_model.txt          ← Main LightGBM model
xgb_model.json         ← Main XGBoost model
lgb_station_1.txt      ← Per-station specialists
lgb_station_142.txt
lgb_station_197.txt
lgb_station_213.txt
lgb_station_219.txt
lgb_station_242.txt
lgb_station_252.txt
lgb_station_290.txt
lgb_station_309.txt
lgb_station_328.txt
model_meta.pkl         ← Encoder, scales, blend weights, station stats
```

## 📦 Dependencies

```
streamlit lightgbm xgboost scikit-learn scipy pandas numpy plotly matplotlib
```

---
*Built by Team Aether · IIIT Lucknow · 2026*