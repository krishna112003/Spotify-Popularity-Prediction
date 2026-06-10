"""
train_models.py
===============
Run this ONCE locally before deploying to Streamlit Cloud.
Produces all artifacts that app.py loads at startup — no training at runtime.

Usage:
    python train_models.py
    python train_models.py --tracks data/tracks.csv --artists data/artists.csv

Outputs (written to ./artifacts/):
    best_rf.pkl            — tuned RandomForestRegressor
    scaler.pkl             — fitted StandardScaler
    encoder.pkl            — fitted LabelEncoder
    final_features.pkl     — list[str] of selected feature names
    model_results.pkl      — reg_results, clf_results, feat_imp, cm, class_names, etc.
    dashboard_data.parquet — lean 150k-row snapshot for all dashboard pages
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST = True
except ImportError:
    XGBOOST = False
    print("⚠  XGBoost not found — XGBoost models will be skipped.")

# ── Constants ──────────────────────────────────────────────────────────────
CURRENT_YEAR = 2021
ARTIFACTS_DIR = "artifacts"

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _first(s, default="Unknown"):
    try:
        p = eval(s)
        return p[0] if p else default
    except Exception:
        return default


def _genres(s):
    try:
        p = eval(s)
        return p if isinstance(p, list) else []
    except Exception:
        return []


# ── Pipeline steps ─────────────────────────────────────────────────────────

def load_and_clean(tracks_path: str, artists_path: str):
    print("📂  Loading CSVs…")
    df_tracks  = pd.read_csv(tracks_path)
    df_artists = pd.read_csv(artists_path)

    df_tracks  = df_tracks.drop_duplicates(subset="id").reset_index(drop=True)
    df_artists = df_artists.drop_duplicates(subset="id").reset_index(drop=True)

    df_tracks = df_tracks.dropna(subset=["popularity", "name"])
    df_tracks["explicit"] = df_tracks["explicit"].astype(int)

    df_tracks["release_year"] = pd.to_datetime(
        df_tracks["release_date"], errors="coerce"
    ).dt.year
    mask = df_tracks["release_year"].isna()
    df_tracks.loc[mask, "release_year"] = (
        df_tracks.loc[mask, "release_date"]
        .str[:4]
        .apply(pd.to_numeric, errors="coerce")
    )
    df_tracks["release_year"] = df_tracks["release_year"].fillna(0).astype(int)

    df_tracks["artist_name"]   = df_tracks["artists"].apply(_first)
    df_tracks["artist_id_key"] = df_tracks["id_artists"].apply(
        lambda s: _first(s, None)
    )

    for c in ["name", "artist_name"]:
        df_tracks[c] = df_tracks[c].str.strip()

    df_artists["genres_list"]   = df_artists["genres"].apply(_genres)
    df_artists["primary_genre"] = df_artists["genres_list"].apply(
        lambda x: x[0] if x else "unknown"
    )
    df_artists["genre_count"]   = df_artists["genres_list"].apply(len)

    df_tracks = df_tracks[df_tracks["tempo"] > 0]
    df_tracks = df_tracks[df_tracks["duration_ms"] > 0]
    df_tracks = df_tracks[df_tracks["release_year"].between(1920, 2021)]

    print(f"   tracks: {len(df_tracks):,}  |  artists: {len(df_artists):,}")
    return df_tracks, df_artists


def merge_datasets(df_tracks, df_artists):
    slim = df_artists[
        ["id", "followers", "popularity", "primary_genre", "genre_count"]
    ].copy()
    slim.columns = [
        "artist_id_key", "artist_followers", "artist_popularity",
        "primary_genre", "artist_genre_count",
    ]
    df = df_tracks.merge(slim, on="artist_id_key", how="left")
    df["artist_followers"]   = df["artist_followers"].fillna(0)
    df["artist_popularity"]  = df["artist_popularity"].fillna(0)
    df["primary_genre"]      = df["primary_genre"].fillna("unknown")
    df["artist_genre_count"] = df["artist_genre_count"].fillna(0)
    return df


def preprocess(df):
    df = df.copy()
    df["duration_min"]     = df["duration_ms"] / 60000
    df["track_age"]        = (CURRENT_YEAR - df["release_year"]).clip(lower=0)
    df["loudness_shifted"] = df["loudness"] - df["loudness"].min()

    high_skew = [
        c for c in AUDIO_FEATURES
        if c in df.columns and skew(df[c].dropna()) > 1
    ]
    for f in high_skew:
        df[f"{f}_log"] = np.log1p(df[f])

    return df


def feature_engineering(df):
    df = df.copy()

    def _cat_pop(p):
        if p >= 70:   return "Hit"
        elif p >= 40: return "Mainstream"
        elif p >= 10: return "Emerging"
        else:         return "Underground"

    df["popularity_category"]   = df["popularity"].apply(_cat_pop)
    df["energy_category"]       = pd.cut(
        df["energy"], bins=[0, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"]
    )
    df["dance_energy_index"]    = df["danceability"] * df["energy"]
    df["acoustic_instrumental"] = df["acousticness"] * df["instrumentalness"]
    df["mood_score"]            = df["valence"] * df["energy"]
    df["vocal_density"]         = (
        1 - df["instrumentalness"] - df["speechiness"]
    ).clip(0, 1)
    df["followers_log"]         = np.log1p(df["artist_followers"])
    df["is_modern"]             = (df["release_year"] >= 2010).astype(int)
    df["is_popular_artist"]     = (df["artist_popularity"] >= 60).astype(int)
    df["loudness_norm"]         = (df["loudness"] - df["loudness"].min()) / (
        df["loudness"].max() - df["loudness"].min()
    )
    return df


def select_features(df):
    candidates = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_min",
        "explicit", "track_age", "artist_popularity", "followers_log",
        "dance_energy_index", "mood_score", "vocal_density", "loudness_norm",
        "is_modern", "is_popular_artist", "artist_genre_count",
        "instrumentalness_log", "speechiness_log", "liveness_log",
    ]
    candidates = [c for c in candidates if c in df.columns]

    corr     = df[candidates + ["popularity"]].corr()["popularity"].drop("popularity")
    top_corr = set(corr.abs().sort_values(ascending=False).head(15).index)

    dmi = df[candidates + ["popularity"]].dropna()
    mi  = mutual_info_regression(dmi[candidates], dmi["popularity"], random_state=42)
    top_mi = set(
        pd.Series(mi, index=candidates).sort_values(ascending=False).head(15).index
    )

    return sorted(top_corr | top_mi), corr


def train_all_models(df, final_features):
    print("🤖  Training models…")
    df_m = df[final_features + ["popularity", "popularity_category"]].dropna().copy()

    # ── Sample to 100k rows to keep artifact size deployable ──────────
    if len(df_m) > 100_000:
        df_m = df_m.sample(100_000, random_state=42).reset_index(drop=True)
        print(f"   Sampled to {len(df_m):,} rows for training")

    X      = df_m[final_features]
    y_reg  = df_m["popularity"]

    le    = LabelEncoder()
    y_clf = le.fit_transform(df_m["popularity_category"])

    X_tr, X_te, y_tr_r, y_te_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_tr_c, y_te_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    sc      = StandardScaler()
    Xtr_sc  = sc.fit_transform(X_tr)
    Xte_sc  = sc.transform(X_te)

    # ── Regression ────────────────────────────────────────────────────
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Lasso Regression":  Lasso(alpha=0.1),
        "ElasticNet":        ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if XGBOOST:
        reg_models["XGBoost"] = XGBRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42,
            eval_metric="rmse", verbosity=0,
        )

    reg_results, reg_trained = {}, {}
    for name, mdl in reg_models.items():
        print(f"   [REG] {name}")
        mdl.fit(Xtr_sc, y_tr_r)
        p = mdl.predict(Xte_sc)
        reg_results[name] = dict(
            MAE=round(mean_absolute_error(y_te_r, p), 4),
            MSE=round(mean_squared_error(y_te_r, p), 4),
            RMSE=round(np.sqrt(mean_squared_error(y_te_r, p)), 4),
            R2=round(r2_score(y_te_r, p), 4),
        )
        reg_trained[name] = mdl

    # ── Classification ─────────────────────────────────────────────────
    clf_models = {
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
    }
    if XGBOOST:
        clf_models["XGBoost"] = XGBClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42,
            eval_metric="mlogloss", verbosity=0, use_label_encoder=False,
        )

    clf_results, clf_trained = {}, {}
    for name, mdl in clf_models.items():
        print(f"   [CLF] {name}")
        mdl.fit(Xtr_sc, y_tr_c)
        p = mdl.predict(Xte_sc)
        clf_results[name] = dict(
            Accuracy=round(accuracy_score(y_te_c, p), 4),
            Precision=round(
                precision_score(y_te_c, p, average="weighted", zero_division=0), 4
            ),
            Recall=round(
                recall_score(y_te_c, p, average="weighted", zero_division=0), 4
            ),
            F1=round(f1_score(y_te_c, p, average="weighted", zero_division=0), 4),
        )
        clf_trained[name] = mdl

    # ── Tuned Best RF ──────────────────────────────────────────────────
    print("🔧  Tuning Random Forest…")
    rf_base    = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators":      [50, 75, 100],
        "max_depth":         [8, 12, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
        "max_features":      ["sqrt"],
    }
    rs = RandomizedSearchCV(
        rf_base, param_grid, n_iter=3, cv=2,
        scoring="r2", n_jobs=1, random_state=42, verbose=0,
    )
    rs.fit(Xtr_sc, y_tr_r)
    best_rf = rs.best_estimator_
    p_best  = best_rf.predict(Xte_sc)
    tuned_metrics = dict(
        MAE=round(mean_absolute_error(y_te_r, p_best), 4),
        RMSE=round(np.sqrt(mean_squared_error(y_te_r, p_best)), 4),
        R2=round(r2_score(y_te_r, p_best), 4),
        best_params=rs.best_params_,
        best_cv_r2=round(rs.best_score_, 4),
    )

    feat_imp = pd.Series(
        best_rf.feature_importances_, index=final_features
    ).sort_values(ascending=False)

    best_clf_name = max(clf_results, key=lambda k: clf_results[k]["F1"])
    best_clf      = clf_trained[best_clf_name]
    cm            = confusion_matrix(y_te_c, best_clf.predict(Xte_sc))

    return dict(
        scaler=sc,
        le=le,
        reg_results=reg_results,
        reg_trained=reg_trained,
        clf_results=clf_results,
        clf_trained=clf_trained,
        best_rf=best_rf,
        tuned_metrics=tuned_metrics,
        feat_imp=feat_imp,
        best_clf_name=best_clf_name,
        cm=cm,
        class_names=le.classes_.tolist(),
    )


def build_dashboard_data(df):
    """
    Lean Parquet snapshot: only columns used by any dashboard page,
    downsampled to ≤150k rows so the file stays under ~20 MB.
    energy_category is converted to str to avoid Parquet Categorical issues
    when pyarrow version mismatches between train and serve environments.
    """
    keep = [
        "name", "artist_name", "popularity", "popularity_category",
        "release_year", "primary_genre", "explicit", "duration_min",
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo",
        "track_age", "artist_popularity", "followers_log",
        "dance_energy_index", "mood_score", "vocal_density", "loudness_norm",
        "is_modern", "is_popular_artist", "artist_genre_count",
    ]
    keep = [c for c in keep if c in df.columns]
    slim = df[keep].copy()

    # Convert any remaining Categorical columns to plain strings
    # (avoids pyarrow ArrowInvalid errors on column type mismatch)
    for col in slim.select_dtypes(include="category").columns:
        slim[col] = slim[col].astype(str)

    if len(slim) > 150_000:
        slim = slim.sample(150_000, random_state=42).reset_index(drop=True)

    return slim


# ══════════════════════════════════════════════════════════════════════════
# save_artifacts  —  THE FUNCTION BEING AUDITED
# Every file ml_core.py expects must be produced here.
# ══════════════════════════════════════════════════════════════════════════

def save_artifacts(results, final_features, dashboard_df):
    """
    Files written
    ─────────────
    artifacts/best_rf.pkl          RandomForestRegressor (tuned)
    artifacts/scaler.pkl           StandardScaler fitted on training data
    artifacts/encoder.pkl          LabelEncoder for popularity_category
    artifacts/final_features.pkl   list[str] — feature column order
    artifacts/model_results.pkl    dict with reg_results, clf_results, feat_imp,
                                   best_clf_name, cm, class_names, tuned_metrics
    artifacts/dashboard_data.parquet   lean DataFrame snapshot
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. best_rf.pkl ─────────────────────────────────────────────────
    joblib.dump(results["best_rf"],  os.path.join(ARTIFACTS_DIR, "best_rf.pkl"))

    # ── 2. scaler.pkl ──────────────────────────────────────────────────
    joblib.dump(results["scaler"],   os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

    # ── 3. encoder.pkl ─────────────────────────────────────────────────
    joblib.dump(results["le"],       os.path.join(ARTIFACTS_DIR, "encoder.pkl"))

    # ── 4. final_features.pkl ──────────────────────────────────────────
    joblib.dump(final_features,      os.path.join(ARTIFACTS_DIR, "final_features.pkl"))

    # ── 5. model_results.pkl ───────────────────────────────────────────
    # Strip the large sklearn model objects that are saved separately above.
    # Convert numpy/pandas types to plain Python so joblib stays portable.
    slim = {
        k: v for k, v in results.items()
        if k not in ("scaler", "le", "best_rf", "reg_trained", "clf_trained")
    }

    # feat_imp: pd.Series → dict  (ml_core reconstructs as pd.Series)
    slim["feat_imp"] = slim["feat_imp"].to_dict()

    # cm: np.ndarray → nested list  (ml_core reconstructs as np.array)
    slim["cm"] = slim["cm"].tolist()

    # tuned_metrics: already plain Python (floats + dict), safe as-is
    # best_clf_name: str, safe as-is
    # class_names: list[str], safe as-is
    # reg_results / clf_results: nested dicts of Python floats, safe as-is

    joblib.dump(slim, os.path.join(ARTIFACTS_DIR, "model_results.pkl"))

    # ── 6. dashboard_data.parquet ──────────────────────────────────────
    # Requires pyarrow (listed in requirements.txt).
    # engine="pyarrow" is explicit — never falls back to fastparquet.
    dashboard_df.to_parquet(
        os.path.join(ARTIFACTS_DIR, "dashboard_data.parquet"),
        index=False,
        engine="pyarrow",
    )

    # ── Verification summary ───────────────────────────────────────────
    expected = [
        "best_rf.pkl", "scaler.pkl", "encoder.pkl",
        "final_features.pkl", "model_results.pkl", "dashboard_data.parquet",
    ]
    print("\n✅  Artifact verification:")
    all_ok = True
    for fname in expected:
        path = os.path.join(ARTIFACTS_DIR, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1_048_576
            print(f"   ✓  {fname:<35} {size_mb:6.1f} MB")
        else:
            print(f"   ✗  {fname}  — MISSING")
            all_ok = False

    if not all_ok:
        raise RuntimeError("Some artifacts were not written. Check errors above.")

    return True


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Spotify ML pipeline offline.")
    parser.add_argument("--tracks",  default="data/tracks.csv",  help="Path to tracks.csv")
    parser.add_argument("--artists", default="data/artists.csv", help="Path to artists.csv")
    args = parser.parse_args()

    for label, path in [("tracks", args.tracks), ("artists", args.artists)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} file not found: {path}\n"
                f"Usage: python train_models.py --tracks <path> --artists <path>"
            )

    print("=" * 60)
    print("  Spotify ML — Offline Training Pipeline")
    print("=" * 60)

    df_tracks, df_artists = load_and_clean(args.tracks, args.artists)
    df = merge_datasets(df_tracks, df_artists)
    df = preprocess(df)
    df = feature_engineering(df)

    print("🔬  Selecting features…")
    final_features, _ = select_features(df)
    print(f"   {len(final_features)} features selected: {final_features}")

    results = train_all_models(df, final_features)

    print("\n📊  Results summary:")
    print("  Regression (R²):")
    for name, m in sorted(results["reg_results"].items(),
                           key=lambda x: x[1]["R2"], reverse=True):
        print(f"    {name:<28} R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")
    print("  Classification (F1):")
    for name, m in sorted(results["clf_results"].items(),
                           key=lambda x: x[1]["F1"], reverse=True):
        print(f"    {name:<28} F1={m['F1']:.4f}  Acc={m['Accuracy']:.4f}")
    print(f"\n  Tuned RF  R²={results['tuned_metrics']['R2']:.4f}"
          f"  RMSE={results['tuned_metrics']['RMSE']:.4f}")

    print("\n💾  Building dashboard snapshot…")
    dashboard_df = build_dashboard_data(df)
    print(f"   {len(dashboard_df):,} rows × {len(dashboard_df.columns)} columns")

    save_artifacts(results, final_features, dashboard_df)

    print("\n🎉  Done.  Commit the artifacts/ folder, then deploy to Streamlit Cloud.")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()