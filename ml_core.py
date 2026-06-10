"""
ml_core.py
==========
Production version — ZERO training at import time.
All heavy lifting is done offline by train_models.py.

Public API:
    load_artifacts(artifacts_dir)  → results dict (mirrors old train_all_models output)
    load_dashboard_data(artifacts_dir) → pd.DataFrame
    predict_new_song(...)          → {"predicted_score": float, "predicted_category": str}
    AUDIO_FEATURES                 → list[str]
"""

import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]

ARTIFACTS_DIR = "artifacts"


# ── Artifact loading ──────────────────────────────────────────────────────

def _artifacts_path(artifacts_dir: str, filename: str) -> str:
    return os.path.join(artifacts_dir, filename)


def load_artifacts(artifacts_dir: str = ARTIFACTS_DIR) -> dict:
    """
    Load all pre-trained artifacts from disk.
    Returns a results dict with the same keys that train_all_models() used to return,
    so every page in app.py works without modification.
    """
    required = [
        "best_rf.pkl", "scaler.pkl", "encoder.pkl",
        "final_features.pkl", "model_results.pkl",
    ]
    missing = [
        f for f in required
        if not os.path.exists(_artifacts_path(artifacts_dir, f))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {missing}\n"
            f"Run `python train_models.py` first to generate them."
        )

    best_rf        = joblib.load(_artifacts_path(artifacts_dir, "best_rf.pkl"))
    scaler         = joblib.load(_artifacts_path(artifacts_dir, "scaler.pkl"))
    le             = joblib.load(_artifacts_path(artifacts_dir, "encoder.pkl"))
    final_features = joblib.load(_artifacts_path(artifacts_dir, "final_features.pkl"))
    slim_results   = joblib.load(_artifacts_path(artifacts_dir, "model_results.pkl"))

    # Reconstruct the objects that app.py expects
    slim_results["feat_imp"] = pd.Series(slim_results["feat_imp"]).sort_values(
        ascending=False
    )
    slim_results["cm"] = np.array(slim_results["cm"])

    # Inject the live model objects back
    slim_results["best_rf"] = best_rf
    slim_results["scaler"]  = scaler
    slim_results["le"]      = le

    return slim_results, final_features


def load_dashboard_data(artifacts_dir: str = ARTIFACTS_DIR) -> pd.DataFrame:
    """Load the pre-built dashboard Parquet snapshot."""
    path = _artifacts_path(artifacts_dir, "dashboard_data.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"dashboard_data.parquet not found at {path}\n"
            "Run `python train_models.py` first."
        )
    return pd.read_parquet(path)


# ── Prediction (preserved exactly from notebook) ─────────────────────────

def predict_new_song(
    final_features, scaler, best_rf, le,
    danceability=0.7, energy=0.8, loudness=-5.0,
    speechiness=0.05, acousticness=0.1, instrumentalness=0.0,
    liveness=0.1, valence=0.6, tempo=120.0, duration_min=3.5,
    explicit=0, track_age=2, artist_popularity=70,
    followers_log=12.0, is_modern=1, is_popular_artist=1,
    artist_genre_count=3,
):
    feature_values = {
        "danceability":        danceability,
        "energy":              energy,
        "loudness":            loudness,
        "speechiness":         speechiness,
        "acousticness":        acousticness,
        "instrumentalness":    instrumentalness,
        "liveness":            liveness,
        "valence":             valence,
        "tempo":               tempo,
        "duration_min":        duration_min,
        "explicit":            explicit,
        "track_age":           track_age,
        "artist_popularity":   artist_popularity,
        "followers_log":       followers_log,
        "dance_energy_index":  danceability * energy,
        "mood_score":          valence * energy,
        "vocal_density":       max(0, 1 - instrumentalness - speechiness),
        "loudness_norm":       max(0, min(1, (loudness + 60) / 60)),
        "is_modern":           is_modern,
        "is_popular_artist":   is_popular_artist,
        "artist_genre_count":  artist_genre_count,
        "instrumentalness_log": np.log1p(instrumentalness),
        "speechiness_log":     np.log1p(speechiness),
        "liveness_log":        np.log1p(liveness),
        "acoustic_instrumental": acousticness * instrumentalness,
    }

    row      = {k: feature_values.get(k, 0) for k in final_features}
    input_df = pd.DataFrame([row])
    input_sc = scaler.transform(input_df)
    score    = float(np.clip(best_rf.predict(input_sc)[0], 0, 100))

    if score >= 70:
        cat = "Hit 🔥"
    elif score >= 40:
        cat = "Mainstream 🎵"
    elif score >= 10:
        cat = "Emerging 🌱"
    else:
        cat = "Underground 🎸"

    return {"predicted_score": round(score, 2), "predicted_category": cat}