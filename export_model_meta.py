"""
export_model_meta.py
────────────────────
Run this ONCE on your local machine after training v9 to export the metadata
needed by the Streamlit app (encoder, station stats, blend weights, bias scales).

Usage:
  python export_model_meta.py

Produces:
  model_meta.pkl   ← upload this alongside lgb_model.txt and xgb_model.json
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Paths — edit these to match your local setup
TRAIN_PATH = r"E:\Kaggle\steamflow prediction\train_flood.csv"
MODEL_DIR  = r"E:\Kaggle\steamflow prediction\v9"
OUT_PATH   = os.path.join(MODEL_DIR, "model_meta.pkl")

_TARGET  = "streamflow_tomorrow_cumecs"
_STATIC  = [
    "dist_to_outlet_scaled", "upstream_area_scaled",
    "slope_scaled", "slope_uav_scaled",
    "forest_cover_scaled", "urban_cover_scaled",
]
_F32 = [
    "streamflow_today_cumecs", "streamflow_anomaly_zscore",
    "flow_rate_of_change", "flow_velocity_km_per_day",
    "antecedent_rain_3d_sum", "antecedent_rain_7d_sum",
    "antecedent_rain_15d_sum", "antecedent_rain_30d_sum",
    "antecedent_rain_60d", "antecedent_rain_ewm",
    "rainfall_anomaly_zscore",
    "upstream_rain_mean_scaled", "upstream_rain_weighted_scaled",
    "upstream_rain_lagged_dist_sink",
    "soil_saturation_score", "antecedent_saturation_interaction",
    "monsoon_cumulative_rain", "monsoon_intensity",
    "dist_to_outlet_scaled", "upstream_area_scaled",
    "slope_scaled", "slope_uav_scaled",
    "forest_cover_scaled", "urban_cover_scaled",
    "rain_soilmoisture_interaction", "rain_urban_interaction",
    "rain_slope_interaction", "rain_basinsize_interaction",
    "uparea_rain_interaction",
]
WORST_STATIONS = [328, 242, 1, 142, 197, 290, 252, 219, 309, 213]

print("Loading training data…")
dtype = {c: "float32" for c in _F32}
dtype.update({"row_id": "int32", "month": "int8",
              "day_of_year": "int16", "is_post_monsoon_saturated": "int8"})
train = pd.read_csv(TRAIN_PATH, dtype=dtype)
train[_TARGET] = train[_TARGET].astype("float32")
print(f"  {train.shape[0]:,} rows")

# ── Station encoder
combo = train[_STATIC].round(4).astype(str).agg("|".join, axis=1)
enc   = LabelEncoder()
train["station_id"] = enc.fit_transform(combo).astype("int16")
print(f"  {train['station_id'].nunique()} unique stations")

# ── Station stats
stats = (
    train.groupby("station_id")[_TARGET]
    .agg(station_mean_flow="mean",
         station_std_flow="std",
         station_p90_flow=lambda x: np.percentile(x, 90))
    .reset_index()
)
stats["station_std_flow"] = stats["station_std_flow"].fillna(0)

# ── Validation split (15%) to recompute bias scales
cut = train["row_id"].quantile(0.85)
val = train[train["row_id"] >= cut].copy()
print(f"  Val rows: {len(val):,}")

# Load models
import lightgbm as lgb
import xgboost as xgb

lgb_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgb_model.txt"))
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))

station_models = {}
for sid in WORST_STATIONS:
    p = os.path.join(MODEL_DIR, f"lgb_station_{sid}.txt")
    if os.path.exists(p):
        station_models[sid] = lgb.Booster(model_file=p)
        print(f"  Loaded station model: {sid}")

# ── Blend weights (from notebook output — LGB dominated)
blend_weights = {"lgb": 1.0, "xgb": 0.0}

# ── Bias scales (from notebook output)
# These are the min/max bias scales fitted during v9 training
# For a more accurate export, re-run the bias fitting below
# (simplified: use near-unity scales as placeholder)
station_scales = {}
for sid in val["station_id"].unique():
    station_scales[int(sid)] = 1.0  # placeholder — override with real scales if available

# Try to load from CSV if available
scales_csv = os.path.join(MODEL_DIR, "station_scales.csv")
if os.path.exists(scales_csv):
    sc_df = pd.read_csv(scales_csv)
    station_scales = dict(zip(sc_df["station_id"].astype(int), sc_df["scale"]))
    print(f"  Loaded {len(station_scales)} bias scales from CSV")

meta = {
    "enc":     enc,
    "stats":   stats,
    "weights": blend_weights,
    "scales":  station_scales,
}

with open(OUT_PATH, "wb") as f:
    pickle.dump(meta, f, protocol=4)

print(f"\n✅ Saved model_meta.pkl → {OUT_PATH}")
print("   Upload this file alongside lgb_model.txt and xgb_model.json to your HF Space.")
