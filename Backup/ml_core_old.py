"""
components/ml_core.py
All ML logic preserved exactly from the notebook.
Returns trained models, results dicts, scalers, etc.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from scipy.stats import skew

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST = True
except ImportError:
    XGBOOST = False


# ── Constants ─────────────────────────────────────────────────────────────
CURRENT_YEAR = 2021

AUDIO_FEATURES = [
    "danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo",
]


# ── Data loading & cleaning ───────────────────────────────────────────────
def load_and_clean(tracks_path: str, artists_path: str):
    df_tracks  = pd.read_csv(tracks_path)
    df_artists = pd.read_csv(artists_path)

    print("\nTRACKS COLUMNS:")
    print(df_tracks.columns.tolist())

    print("\nARTISTS COLUMNS:")
    print(df_artists.columns.tolist())

    # Deduplication
    df_tracks  = df_tracks.drop_duplicates(subset="id").reset_index(drop=True)
    df_artists = df_artists.drop_duplicates(subset="id").reset_index(drop=True)

    # Drop missing target
    df_tracks = df_tracks.dropna(subset=["popularity","name"])
    df_tracks["explicit"] = df_tracks["explicit"].astype(int)

    # release_year
    df_tracks["release_year"] = pd.to_datetime(
        df_tracks["release_date"], errors="coerce").dt.year
    mask = df_tracks["release_year"].isna()
    df_tracks.loc[mask,"release_year"] = (
        df_tracks.loc[mask,"release_date"].str[:4]
        .apply(pd.to_numeric, errors="coerce"))
    df_tracks["release_year"] = df_tracks["release_year"].fillna(0).astype(int)

    # Artist parsing
    def _first(s, default="Unknown"):
        try:
            p = eval(s); return p[0] if p else default
        except: return default

    df_tracks["artist_name"]   = df_tracks["artists"].apply(_first)
    df_tracks["artist_id_key"] = df_tracks["id_artists"].apply(lambda s: _first(s, None))

    # Strip whitespace
    for c in ["name","artist_name"]:
        df_tracks[c] = df_tracks[c].str.strip()

    # Artists genre
    def _genres(s):
        try:
            p = eval(s); return p if isinstance(p, list) else []
        except: return []

    df_artists["genres_list"]   = df_artists["genres"].apply(_genres)
    df_artists["primary_genre"] = df_artists["genres_list"].apply(
        lambda x: x[0] if x else "unknown")
    df_artists["genre_count"]   = df_artists["genres_list"].apply(len)

    # Sanity filters
    df_tracks = df_tracks[df_tracks["tempo"] > 0]
    df_tracks = df_tracks[df_tracks["duration_ms"] > 0]
    df_tracks = df_tracks[df_tracks["release_year"].between(1920, 2021)]

    return df_tracks, df_artists


def merge_datasets(df_tracks, df_artists):
    slim = df_artists[["id","followers","popularity","primary_genre","genre_count"]].copy()
    slim.columns = ["artist_id_key","artist_followers","artist_popularity",
                    "primary_genre","artist_genre_count"]
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

    # Log-transform high-skew features
    high_skew = [c for c in AUDIO_FEATURES
                 if c in df.columns and skew(df[c].dropna()) > 1]
    for f in high_skew:
        df[f"{f}_log"] = np.log1p(df[f])

    return df


def feature_engineering(df):
    df = df.copy()

    def _cat_pop(p):
        if p >= 70:  return "Hit"
        elif p >= 40: return "Mainstream"
        elif p >= 10: return "Emerging"
        else:         return "Underground"

    df["popularity_category"]    = df["popularity"].apply(_cat_pop)
    df["energy_category"]        = pd.cut(df["energy"], bins=[0,.33,.66,1.],
                                          labels=["Low","Medium","High"])
    df["dance_energy_index"]     = df["danceability"] * df["energy"]
    df["acoustic_instrumental"]  = df["acousticness"] * df["instrumentalness"]
    df["mood_score"]             = df["valence"] * df["energy"]
    df["vocal_density"]          = (1 - df["instrumentalness"] - df["speechiness"]).clip(0,1)
    df["followers_log"]          = np.log1p(df["artist_followers"])
    df["is_modern"]              = (df["release_year"] >= 2010).astype(int)
    df["is_popular_artist"]      = (df["artist_popularity"] >= 60).astype(int)
    df["loudness_norm"]          = (df["loudness"] - df["loudness"].min()) / \
                                   (df["loudness"].max() - df["loudness"].min())
    return df


def select_features(df):
    candidates = [
        "danceability","energy","loudness","speechiness","acousticness",
        "instrumentalness","liveness","valence","tempo","duration_min",
        "explicit","track_age","artist_popularity","followers_log",
        "dance_energy_index","mood_score","vocal_density","loudness_norm",
        "is_modern","is_popular_artist","artist_genre_count",
        "instrumentalness_log","speechiness_log","liveness_log",
    ]
    candidates = [c for c in candidates if c in df.columns]

    corr = df[candidates + ["popularity"]].corr()["popularity"].drop("popularity")
    top_corr = set(corr.abs().sort_values(ascending=False).head(15).index)

    dmi = df[candidates + ["popularity"]].dropna()
    mi  = mutual_info_regression(dmi[candidates], dmi["popularity"], random_state=42)
    top_mi = set(pd.Series(mi, index=candidates).sort_values(ascending=False).head(15).index)

    return sorted(top_corr | top_mi), corr


def train_all_models(df, final_features):
    df_m = df[final_features + ["popularity","popularity_category"]].dropna().copy()

    X      = df_m[final_features]
    y_reg  = df_m["popularity"]

    le = LabelEncoder()
    y_clf = le.fit_transform(df_m["popularity_category"])

    X_tr, X_te, y_tr_r, y_te_r = train_test_split(X, y_reg,  test_size=.2, random_state=42)
    _,    _,    y_tr_c, y_te_c = train_test_split(X, y_clf,  test_size=.2, random_state=42)

    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_tr)
    Xte_sc = sc.transform(X_te)

    # ── Regression ─────────────────────────────────────────────────────
    reg_models = {
        "Linear Regression":       LinearRegression(),
        "Ridge Regression":        Ridge(alpha=1.0),
        "Lasso Regression":        Lasso(alpha=0.1),
        "ElasticNet":              ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Decision Tree":           DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":           RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if XGBOOST:
        reg_models["XGBoost"] = XGBRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42,
            eval_metric="rmse", verbosity=0)

    reg_results, reg_trained = {}, {}
    for name, mdl in reg_models.items():
        mdl.fit(Xtr_sc, y_tr_r)
        p = mdl.predict(Xte_sc)
        reg_results[name] = dict(
            MAE=round(mean_absolute_error(y_te_r, p),4),
            MSE=round(mean_squared_error(y_te_r, p),4),
            RMSE=round(np.sqrt(mean_squared_error(y_te_r, p)),4),
            R2=round(r2_score(y_te_r, p),4),
        )
        reg_trained[name] = mdl

    # ── Classification ─────────────────────────────────────────────────
    clf_models = {
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
    }
    if XGBOOST:
        clf_models["XGBoost"] = XGBClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42,
            eval_metric="mlogloss", verbosity=0, use_label_encoder=False)

    clf_results, clf_trained = {}, {}
    for name, mdl in clf_models.items():
        mdl.fit(Xtr_sc, y_tr_c)
        p = mdl.predict(Xte_sc)
        clf_results[name] = dict(
            Accuracy=round(accuracy_score(y_te_c, p),4),
            Precision=round(precision_score(y_te_c, p, average="weighted", zero_division=0),4),
            Recall=round(recall_score(y_te_c, p, average="weighted", zero_division=0),4),
            F1=round(f1_score(y_te_c, p, average="weighted", zero_division=0),4),
        )
        clf_trained[name] = mdl

    # ── Tuned Best RF ──────────────────────────────────────────────────
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators":[100,150,200], "max_depth":[10,20,None],
        "min_samples_split":[2,5], "min_samples_leaf":[1,2],
        "max_features":["sqrt"],
    }
    rs = RandomizedSearchCV(rf_base, param_grid, n_iter=3, cv=2,
                            scoring="r2", n_jobs=1, random_state=42, verbose=0)
    rs.fit(Xtr_sc, y_tr_r)
    best_rf = rs.best_estimator_
    p_best  = best_rf.predict(Xte_sc)
    tuned_metrics = dict(
        MAE=round(mean_absolute_error(y_te_r, p_best),4),
        RMSE=round(np.sqrt(mean_squared_error(y_te_r, p_best)),4),
        R2=round(r2_score(y_te_r, p_best),4),
        best_params=rs.best_params_,
        best_cv_r2=round(rs.best_score_,4),
    )

    # Feature importance
    feat_imp = pd.Series(best_rf.feature_importances_, index=final_features).sort_values(ascending=False)

    # Confusion matrix
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
        X_test=pd.DataFrame(Xte_sc, columns=final_features),
        y_test_reg=y_te_r.values,
        y_test_clf=y_te_c,
    )


# ── Prediction function (preserved exactly from notebook) ────────────────
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
        "danceability": danceability, "energy": energy, "loudness": loudness,
        "speechiness": speechiness, "acousticness": acousticness,
        "instrumentalness": instrumentalness, "liveness": liveness,
        "valence": valence, "tempo": tempo, "duration_min": duration_min,
        "explicit": explicit, "track_age": track_age,
        "artist_popularity": artist_popularity, "followers_log": followers_log,
        "dance_energy_index": danceability * energy,
        "mood_score": valence * energy,
        "vocal_density": max(0, 1 - instrumentalness - speechiness),
        "loudness_norm": max(0, min(1, (loudness + 60) / 60)),
        "is_modern": is_modern, "is_popular_artist": is_popular_artist,
        "artist_genre_count": artist_genre_count,
        "instrumentalness_log": np.log1p(instrumentalness),
        "speechiness_log": np.log1p(speechiness),
        "liveness_log": np.log1p(liveness),
        "acoustic_instrumental": acousticness * instrumentalness,
    }
    row      = {k: feature_values.get(k, 0) for k in final_features}
    input_df = pd.DataFrame([row])
    input_sc = scaler.transform(input_df)
    score    = float(np.clip(best_rf.predict(input_sc)[0], 0, 100))

    if score >= 70:  cat = "Hit 🔥"
    elif score >= 40: cat = "Mainstream 🎵"
    elif score >= 10: cat = "Emerging 🌱"
    else:             cat = "Underground 🎸"

    return {"predicted_score": round(score,2), "predicted_category": cat}