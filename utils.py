"""
utils.py  ─  Core pipeline helpers for Streamflow v9
Extracted from v9.ipynb — all computation lives here so app.py stays clean.
"""

import gc
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_TARGET = "streamflow_tomorrow_cumecs"

WORST_STATIONS = [328, 242, 1, 142, 197, 290, 252, 219, 309, 213]

_STATIC = [
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

FEATURE_COLS = [
    "streamflow_today_cumecs", "streamflow_anomaly_zscore",
    "flow_rate_of_change", "flow_velocity_km_per_day",
    "antecedent_rain_3d_sum", "antecedent_rain_7d_sum",
    "antecedent_rain_15d_sum", "antecedent_rain_30d_sum",
    "antecedent_rain_60d", "antecedent_rain_ewm", "rainfall_anomaly_zscore",
    "upstream_rain_mean_scaled", "upstream_rain_weighted_scaled",
    "upstream_rain_lagged_dist_sink",
    "soil_saturation_score", "antecedent_saturation_interaction",
    "is_post_monsoon_saturated",
    "monsoon_intensity", "monsoon_cumulative_rain",
    "dist_to_outlet_scaled", "upstream_area_scaled",
    "slope_scaled", "slope_uav_scaled", "forest_cover_scaled", "urban_cover_scaled",
    "rain_soilmoisture_interaction", "rain_urban_interaction",
    "rain_slope_interaction", "rain_basinsize_interaction", "uparea_rain_interaction",
    "station_id", "station_mean_flow", "station_std_flow", "station_p90_flow",
    "log_flow", "log_flow_sq", "rel_rate", "flow_accel_sign", "flow_norm_station",
    "flow_lag_1", "flow_lag_2", "flow_lag_3", "flow_lag_4",
    "flow_lag_5", "flow_lag_6", "flow_lag_7",
    "rain_lag_1", "rain_lag_2", "rain_lag_3",
    "log_flow_lag_1", "log_flow_lag_2", "log_flow_lag_3",
    "log_flow_lag_4", "log_flow_lag_5", "log_flow_lag_6", "log_flow_lag_7",
    "flow_roll_mean_3d", "flow_roll_mean_7d", "flow_roll_max_7d",
    "flow_roll_std_7d", "rain_roll_std_7d",
    "flow_d1", "flow_d2", "flow_log_d1",
    "is_rising", "is_flooding", "is_receding",
    "flow_lag1_ratio", "flow_vs_roll7", "flow_spike_ratio", "flow_pct_of_7d_max",
    "is_worst_station",
    "week_of_year", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "rain_ratio_3_30", "rain_ratio_7_30", "rain_ratio_3_60", "rain_ratio_15_60",
    "rain_accel", "log_rain_3d", "log_rain_7d", "log_rain_30d", "log_rain_ewm",
    "ups_x_logflow", "ups_x_sat", "logflow_x_sat",
    "preflood_alarm", "rain_anom_x_sat",
    "in_peak_monsoon", "in_pre_monsoon", "rain_load_ratio",
]

LGB_CAT_COLS = [
    "is_post_monsoon_saturated", "flow_accel_sign",
    "in_peak_monsoon", "in_pre_monsoon",
    "is_rising", "is_flooding", "is_receding", "is_worst_station",
]

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def kge(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    r     = np.corrcoef(y_obs, y_sim)[0, 1]
    alpha = y_sim.std()  / (y_obs.std()  + 1e-10)
    beta  = y_sim.mean() / (y_obs.mean() + 1e-10)
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def kge_components(y_obs: np.ndarray, y_sim: np.ndarray) -> dict:
    r     = np.corrcoef(y_obs, y_sim)[0, 1]
    alpha = y_sim.std()  / (y_obs.std()  + 1e-10)
    beta  = y_sim.mean() / (y_obs.mean() + 1e-10)
    score = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return dict(KGE=round(score, 6), r=round(r, 4),
                alpha=round(alpha, 4), beta=round(beta, 4))


def nse(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    return float(1 - np.sum((y_obs - y_sim)**2) /
                 np.sum((y_obs - y_obs.mean())**2))


# ─────────────────────────────────────────────────────────────────────────────
# STATION ID
# ─────────────────────────────────────────────────────────────────────────────
def add_station_id(df: pd.DataFrame, enc=None):
    combo = df[_STATIC].round(4).astype(str).agg("|".join, axis=1)
    if enc is None:
        enc = LabelEncoder()
        df["station_id"] = enc.fit_transform(combo).astype("int16")
    else:
        known = set(enc.classes_)
        combo = combo.where(combo.isin(known), other=enc.classes_[0])
        df["station_id"] = enc.transform(combo).astype("int16")
    return df, enc


# ─────────────────────────────────────────────────────────────────────────────
# LAG FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id", "row_id"]).reset_index(drop=True)
    grp_flow = df.groupby("station_id")["streamflow_today_cumecs"]
    grp_rain = df.groupby("station_id")["antecedent_rain_3d_sum"]

    for lag in range(1, 8):
        df[f"flow_lag_{lag}"] = (
            grp_flow.shift(lag)
            .fillna(df["streamflow_today_cumecs"])
            .astype("float32"))

    for lag in range(1, 4):
        df[f"rain_lag_{lag}"] = (
            grp_rain.shift(lag)
            .fillna(df["antecedent_rain_3d_sum"])
            .astype("float32"))

    df["flow_roll_mean_3d"] = (
        grp_flow.transform(lambda x: x.rolling(3, min_periods=1).mean())
        .astype("float32"))
    df["flow_roll_mean_7d"] = (
        grp_flow.transform(lambda x: x.rolling(7, min_periods=1).mean())
        .astype("float32"))
    df["flow_roll_max_7d"] = (
        grp_flow.transform(lambda x: x.rolling(7, min_periods=1).max())
        .astype("float32"))
    df["flow_roll_std_7d"] = (
        grp_flow.transform(lambda x: x.rolling(7, min_periods=2).std())
        .fillna(0).astype("float32"))
    df["rain_roll_std_7d"] = (
        grp_rain.transform(lambda x: x.rolling(7, min_periods=2).std())
        .fillna(0).astype("float32"))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    Q   = df["streamflow_today_cumecs"].astype("float64")
    doy = df["day_of_year"].astype("float64")
    mon = df["month"].astype("float64")
    sat = df["soil_saturation_score"].astype("float64")
    ups = df["upstream_rain_mean_scaled"].astype("float64")

    df["log_flow"]          = np.log1p(Q).astype("float32")
    df["log_flow_sq"]       = (np.log1p(Q) ** 2).astype("float32")
    df["rel_rate"]          = (df["flow_rate_of_change"] / (Q + 1)
                               ).clip(-10, 10).astype("float32")
    df["flow_accel_sign"]   = np.sign(df["rel_rate"]).astype("int8")
    df["flow_norm_station"] = (
        (Q - df["station_mean_flow"].astype("float64")) /
        (df["station_std_flow"].astype("float64") + 1)
    ).clip(-5, 10).astype("float32")

    for lag in range(1, 8):
        col = f"flow_lag_{lag}"
        if col in df.columns:
            df[f"log_flow_lag_{lag}"] = np.log1p(
                df[col].astype("float64")).astype("float32")

    if "flow_lag_1" in df.columns:
        lag1 = df["flow_lag_1"].astype("float64")
        df["flow_d1"]     = (Q - lag1).astype("float32")
        df["flow_log_d1"] = (np.log1p(Q) - np.log1p(lag1)).astype("float32")

    if "flow_lag_1" in df.columns and "flow_lag_2" in df.columns:
        d1      = Q - df["flow_lag_1"].astype("float64")
        d1_prev = (df["flow_lag_1"].astype("float64") -
                   df["flow_lag_2"].astype("float64"))
        df["flow_d2"] = (d1 - d1_prev).clip(-5000, 5000).astype("float32")

    if "flow_lag_1" in df.columns:
        lag1 = df["flow_lag_1"].astype("float64")
        df["is_rising"]      = (Q > lag1 * 1.05).astype("int8")
        df["is_flooding"]    = (Q > df["station_p90_flow"].astype("float64")).astype("int8")
        df["is_receding"]    = (Q < lag1 * 0.95).astype("int8")
        df["flow_lag1_ratio"] = (Q / (lag1 + 1)).clip(0, 10).astype("float32")

    if "flow_roll_mean_7d" in df.columns:
        df["flow_vs_roll7"]    = (Q - df["flow_roll_mean_7d"].astype("float64")).astype("float32")
        df["flow_spike_ratio"] = (Q / (df["flow_roll_mean_7d"].astype("float64") + 1)).clip(0, 20).astype("float32")

    if "flow_roll_max_7d" in df.columns:
        df["flow_pct_of_7d_max"] = (Q / (df["flow_roll_max_7d"].astype("float64") + 1)).clip(0, 1).astype("float32")

    df["doy_sin"]      = np.sin(2 * np.pi * doy / 365).astype("float32")
    df["doy_cos"]      = np.cos(2 * np.pi * doy / 365).astype("float32")
    df["month_sin"]    = np.sin(2 * np.pi * mon / 12).astype("float32")
    df["month_cos"]    = np.cos(2 * np.pi * mon / 12).astype("float32")
    df["week_of_year"] = (doy // 7).clip(0, 52).astype("int8")

    r3  = df["antecedent_rain_3d_sum"].astype("float64")  + 1e-6
    r7  = df["antecedent_rain_7d_sum"].astype("float64")  + 1e-6
    r15 = df["antecedent_rain_15d_sum"].astype("float64") + 1e-6
    r30 = df["antecedent_rain_30d_sum"].astype("float64") + 1e-6
    r60 = df["antecedent_rain_60d"].astype("float64")     + 1e-6

    df["rain_ratio_3_30"]  = (r3  / r30).clip(0, 20).astype("float32")
    df["rain_ratio_7_30"]  = (r7  / r30).clip(0, 20).astype("float32")
    df["rain_ratio_3_60"]  = (r3  / r60).clip(0, 20).astype("float32")
    df["rain_ratio_15_60"] = (r15 / r60).clip(0, 20).astype("float32")
    df["rain_accel"]       = ((r3 / 3) - (r7 / 7)).astype("float32")
    df["log_rain_3d"]  = np.log1p(df["antecedent_rain_3d_sum"].astype("float64")).astype("float32")
    df["log_rain_7d"]  = np.log1p(df["antecedent_rain_7d_sum"].astype("float64")).astype("float32")
    df["log_rain_30d"] = np.log1p(df["antecedent_rain_30d_sum"].astype("float64")).astype("float32")
    df["log_rain_ewm"] = np.log1p(df["antecedent_rain_ewm"].astype("float64")).astype("float32")

    df["ups_x_logflow"]   = (ups * np.log1p(Q)).astype("float32")
    df["ups_x_sat"]       = (ups * sat).astype("float32")
    df["logflow_x_sat"]   = (np.log1p(Q) * sat).astype("float32")
    df["preflood_alarm"]  = (
        df["streamflow_anomaly_zscore"].astype("float64") *
        df["rel_rate"].astype("float64") * sat
    ).clip(-50, 50).astype("float32")
    df["rain_anom_x_sat"] = (df["rainfall_anomaly_zscore"].astype("float64") * sat).astype("float32")

    df["in_peak_monsoon"] = ((doy >= 150) & (doy <= 270)).astype("int8")
    df["in_pre_monsoon"]  = ((doy >= 90)  & (doy < 150)).astype("int8")
    df["rain_load_ratio"] = (
        df["monsoon_cumulative_rain"].astype("float64") /
        (df["station_mean_flow"].astype("float64") + 1)
    ).clip(0, 100).astype("float32")

    df["is_worst_station"] = df["station_id"].isin(WORST_STATIONS).astype("int8")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DELTA TARGET & RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_from_delta(delta_pred: np.ndarray,
                            q_today: np.ndarray) -> np.ndarray:
    log_today = np.log1p(q_today.astype("float64"))
    return np.expm1(log_today + delta_pred.astype("float64")).clip(0)


# ─────────────────────────────────────────────────────────────────────────────
# BIAS CORRECTION
# ─────────────────────────────────────────────────────────────────────────────
def apply_station_bias(df: pd.DataFrame, pred: np.ndarray,
                       scales: dict) -> np.ndarray:
    out = pred.copy().astype("float64")
    for sid, scale in scales.items():
        out[df["station_id"].values == sid] *= scale
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_COLS if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# FULL PREDICTION PIPELINE  (for uploaded CSV)
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction_pipeline(test_df: pd.DataFrame,
                             lgb_model,
                             xgb_model,
                             station_models: dict,
                             station_scales: dict,
                             blend_weights: dict,
                             station_stats: pd.DataFrame,
                             enc: LabelEncoder,
                             progress_cb=None) -> pd.DataFrame:
    """
    End-to-end prediction on a test CSV.
    Returns DataFrame with row_id + streamflow_tomorrow_cumecs.
    """
    def _prog(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)

    _prog("Assigning station IDs…", 0.05)
    test_df, _ = add_station_id(test_df, enc=enc)

    _prog("Merging station statistics…", 0.15)
    for col in ["station_mean_flow", "station_std_flow", "station_p90_flow"]:
        if col in test_df.columns:
            test_df.drop(columns=[col], inplace=True)
    test_df = test_df.merge(station_stats, on="station_id", how="left")
    gm   = float(station_stats["station_mean_flow"].mean())
    gs   = float(station_stats["station_std_flow"].mean())
    gp90 = float(station_stats["station_p90_flow"].mean())
    test_df["station_mean_flow"] = test_df["station_mean_flow"].fillna(gm).astype("float32")
    test_df["station_std_flow"]  = test_df["station_std_flow"].fillna(gs).astype("float32")
    test_df["station_p90_flow"]  = test_df["station_p90_flow"].fillna(gp90).astype("float32")

    _prog("Adding lag features…", 0.25)
    test_df = add_lag_features(test_df)

    _prog("Engineering features…", 0.40)
    test_df = engineer_features(test_df)

    feat_cols = get_feature_cols(test_df)
    X_test = test_df[feat_cols]
    q_today = test_df["streamflow_today_cumecs"].values.astype("float64")

    _prog("Running LightGBM inference…", 0.55)
    lgb_delta = lgb_model.predict(X_test)
    lgb_q     = reconstruct_from_delta(lgb_delta, q_today)

    _prog("Running XGBoost inference…", 0.70)
    xgb_delta = xgb_model.predict(X_test)
    xgb_q     = reconstruct_from_delta(xgb_delta, q_today)

    _prog("Blending models…", 0.80)
    w_lgb = blend_weights.get("lgb", 0.5)
    w_xgb = blend_weights.get("xgb", 0.5)
    blend_q = w_lgb * lgb_q + w_xgb * xgb_q

    _prog("Applying per-station models…", 0.88)
    final_q = blend_q.copy()
    for sid, sm in station_models.items():
        mask = test_df["station_id"].values == sid
        if not mask.any():
            continue
        delta_s = sm.predict(X_test[mask])
        q_s     = reconstruct_from_delta(delta_s, q_today[mask])
        final_q[mask] = 0.5 * blend_q[mask] + 0.5 * q_s

    _prog("Applying bias correction…", 0.95)
    final_q = apply_station_bias(test_df, final_q, station_scales)

    _prog("Done!", 1.0)
    result = pd.DataFrame({
        "row_id":                     test_df["row_id"].values,
        "streamflow_tomorrow_cumecs": final_q,
    })
    return result, test_df
