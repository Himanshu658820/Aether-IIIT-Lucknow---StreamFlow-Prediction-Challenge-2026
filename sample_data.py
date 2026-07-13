"""
sample_data.py — Generates realistic demo data for the Streamflow v9 app.
Used when no real CSV is uploaded, so visitors can still see predictions.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ── Static station profiles (realistic Indian river basins)
STATION_PROFILES = [
    {"name": "Ganga at Haridwar",   "mean_flow": 850,  "area": 0.72, "slope": 0.55, "forest": 0.45, "urban": 0.08},
    {"name": "Brahmaputra at Dibrugarh", "mean_flow": 4200, "area": 0.95, "slope": 0.30, "forest": 0.60, "urban": 0.04},
    {"name": "Godavari at Rajahmundry",  "mean_flow": 1800, "area": 0.80, "slope": 0.40, "forest": 0.35, "urban": 0.12},
    {"name": "Krishna at Vijayawada",    "mean_flow": 980,  "area": 0.68, "slope": 0.45, "forest": 0.28, "urban": 0.10},
    {"name": "Mahanadi at Hirakud",      "mean_flow": 1400, "area": 0.75, "slope": 0.50, "forest": 0.52, "urban": 0.06},
    {"name": "Narmada at Omkareshwar",   "mean_flow": 650,  "area": 0.60, "slope": 0.62, "forest": 0.40, "urban": 0.07},
    {"name": "Yamuna at Delhi",          "mean_flow": 520,  "area": 0.58, "slope": 0.38, "forest": 0.22, "urban": 0.25},
    {"name": "Cauvery at Tiruchirappalli","mean_flow": 380, "area": 0.50, "slope": 0.48, "forest": 0.30, "urban": 0.14},
]


def generate_demo_timeseries(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic 1-year daily streamflow demo for 8 Indian river stations.
    Returns a DataFrame in the same format as test_flood.csv.
    """
    rng = np.random.default_rng(seed)
    rows = []
    row_id = 3_000_000

    for sidx, prof in enumerate(STATION_PROFILES):
        mean_q = prof["mean_flow"]
        # Seasonal pattern: monsoon peak July-September (doy 180-270)
        doy_arr = np.arange(1, n_days + 1)
        seasonal = (
            0.3 * np.sin(2 * np.pi * (doy_arr - 80) / 365) +
            0.7 * np.exp(-((doy_arr - 220) ** 2) / (2 * 40**2))
        )
        seasonal = (seasonal - seasonal.min()) / (seasonal.max() - seasonal.min())

        # Flow series with autocorrelation
        q = np.zeros(n_days)
        q[0] = mean_q * (0.3 + seasonal[0] * 0.7)
        for i in range(1, n_days):
            noise = rng.normal(0, mean_q * 0.04)
            target_q = mean_q * (0.3 + seasonal[i] * 1.4)
            q[i] = max(50, 0.85 * q[i-1] + 0.15 * target_q + noise)

        # Rain series (correlated with flow with 3-day lag)
        rain_3d  = np.clip(rng.exponential(3, n_days) * (1 + seasonal * 8), 0, 120)
        rain_7d  = np.array([rain_3d[max(0, i-6):i+1].sum() for i in range(n_days)])
        rain_15d = np.array([rain_3d[max(0, i-14):i+1].sum() for i in range(n_days)])
        rain_30d = np.array([rain_3d[max(0, i-29):i+1].sum() for i in range(n_days)])
        rain_60d = np.array([rain_3d[max(0, i-59):i+1].sum() for i in range(n_days)])
        rain_ewm = pd.Series(rain_3d).ewm(span=14).mean().values

        for i in range(n_days):
            month  = int((doy_arr[i] - 1) / 30.44) + 1
            month  = min(12, max(1, month))
            row = {
                "row_id":                           row_id,
                "station_name":                     prof["name"],
                "station_idx":                      sidx,
                "day_of_year":                      int(doy_arr[i]),
                "month":                            month,
                "streamflow_today_cumecs":          round(float(q[i]), 2),
                "streamflow_tomorrow_cumecs":       round(float(q[min(i+1, n_days-1)]), 2),
                "streamflow_anomaly_zscore":        round((q[i] - mean_q) / (mean_q * 0.4 + 1), 3),
                "flow_rate_of_change":              round(q[i] - q[max(0, i-1)], 2),
                "flow_velocity_km_per_day":         round(q[i] * 0.002, 3),
                "antecedent_rain_3d_sum":           round(float(rain_3d[i]), 2),
                "antecedent_rain_7d_sum":           round(float(rain_7d[i]), 2),
                "antecedent_rain_15d_sum":          round(float(rain_15d[i]), 2),
                "antecedent_rain_30d_sum":          round(float(rain_30d[i]), 2),
                "antecedent_rain_60d":              round(float(rain_60d[i]), 2),
                "antecedent_rain_ewm":              round(float(rain_ewm[i]), 3),
                "rainfall_anomaly_zscore":          round((rain_3d[i] - rain_3d.mean()) / (rain_3d.std() + 1), 3),
                "upstream_rain_mean_scaled":        round(rain_3d[i] / 50.0, 3),
                "upstream_rain_weighted_scaled":    round(rain_7d[i] / 200.0, 3),
                "upstream_rain_lagged_dist_sink":   round(rain_3d[max(0, i-2)] / 40.0, 3),
                "soil_saturation_score":            round(min(1.0, rain_30d[i] / 200.0), 3),
                "antecedent_saturation_interaction":round(rain_3d[i] * min(1.0, rain_30d[i] / 200.0), 3),
                "is_post_monsoon_saturated":        int(doy_arr[i] > 270 and rain_30d[i] > 80),
                "monsoon_cumulative_rain":          round(float(rain_60d[i]), 2),
                "monsoon_intensity":                round(rain_3d[i] / 3.0, 3),
                "dist_to_outlet_scaled":            round(prof["area"] * 0.6, 3),
                "upstream_area_scaled":             round(prof["area"], 3),
                "slope_scaled":                     round(prof["slope"], 3),
                "slope_uav_scaled":                 round(prof["slope"] * 0.9, 3),
                "forest_cover_scaled":              round(prof["forest"], 3),
                "urban_cover_scaled":               round(prof["urban"], 3),
                "rain_soilmoisture_interaction":    round(rain_3d[i] * min(1.0, rain_30d[i] / 200.0), 3),
                "rain_urban_interaction":           round(rain_3d[i] * prof["urban"], 3),
                "rain_slope_interaction":           round(rain_3d[i] * prof["slope"], 3),
                "rain_basinsize_interaction":       round(rain_3d[i] * prof["area"], 3),
                "uparea_rain_interaction":          round(rain_7d[i] * prof["area"], 3),
            }
            rows.append(row)
            row_id += 1

    df = pd.DataFrame(rows)
    return df


def get_demo_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast physics-inspired prediction (no model needed) for demo.
    Uses a simplified delta-target approach similar to v9.
    """
    results = []
    for sidx, grp in df.groupby("station_idx"):
        grp = grp.sort_values("day_of_year").reset_index(drop=True)
        q_today   = grp["streamflow_today_cumecs"].values
        q_true    = grp["streamflow_tomorrow_cumecs"].values
        rain_3d   = grp["antecedent_rain_3d_sum"].values
        sat       = grp["soil_saturation_score"].values

        # Simple empirical model mimicking the delta-target approach
        log_q = np.log1p(q_today)
        rain_signal = np.log1p(rain_3d) * (1 + sat)
        rel_rate = np.diff(q_today, prepend=q_today[0]) / (q_today + 1)

        # Predict log-delta
        delta_pred = (
            0.65 * rel_rate +
            0.20 * (rain_signal - rain_signal.mean()) / (rain_signal.std() + 1) * 0.02 +
            0.15 * np.clip(-0.001 * (log_q - log_q.mean()), -0.05, 0.05)
        )
        q_pred = np.expm1(log_q + delta_pred).clip(0)

        # Add small realistic noise
        rng_local = np.random.default_rng(sidx)
        q_pred = q_pred * (1 + rng_local.normal(0, 0.008, len(q_pred)))
        q_pred = np.clip(q_pred, 10, None)

        grp["prediction"] = q_pred
        grp["true_tomorrow"] = q_true
        results.append(grp)

    return pd.concat(results, ignore_index=True)
