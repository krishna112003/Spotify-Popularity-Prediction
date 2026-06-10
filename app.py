import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

# ── path setup ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from ui import (
    metric_card, section_header, glass_card, fancy_divider,
    pill, page_hero, leader_row, insight_card,
    prediction_result_card, gauge_chart, bar_chart, heatmap_chart,
    scatter_chart, histogram_chart, radar_chart, line_chart,
    COLOR_SEQ, PLOTLY_BASE, apply_chart_style,
)
from ml_core import load_artifacts, load_dashboard_data, predict_new_song, AUDIO_FEATURES

# ── Streamlit page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────
css_path = os.path.join(ROOT, "assets", "style.css")
if not os.path.exists(css_path):
    css_path = os.path.join(ROOT, "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")


# ══════════════════════════════════════════════════════════════════════════
# CACHED LOADERS  — no training, no downloading
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_models():
    """Load pre-trained models and metrics from disk. ~1–3 s on cold start."""
    return load_artifacts(ARTIFACTS_DIR)


@st.cache_data(show_spinner=False)
def get_dashboard_data():
    """Load pre-built Parquet snapshot from disk. ~1–2 s on cold start."""
    df = load_dashboard_data(ARTIFACTS_DIR)
    # Re-derive decade column used in Analytics page
    df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"
    return df


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar(artifacts_ok: bool):
    with st.sidebar:
        st.markdown("""
        <div class="nav-logo">
          <div class="nav-logo-text">Spotify<span>Analytics</span></div>
          <div class="nav-version">Premium Intelligence Platform · v2.0</div>
        </div>""", unsafe_allow_html=True)

        page = option_menu(
            menu_title=None,
            options=[
                "Overview",
                "Data Explorer",
                "Analytics",
                "Model Lab",
                "Model Comparison",
                "Prediction Studio",
                "AI Insights",
            ],
            icons=[
                "house-fill", "table", "bar-chart-fill", "cpu-fill",
                "trophy-fill", "magic", "lightbulb-fill",
            ],
            default_index=0,
            styles={
                "container": {"padding": "0", "background": "transparent"},
                "icon": {"color": "#1DB954", "font-size": "13px"},
                "nav-link": {
                    "font-family": "Inter, sans-serif",
                    "font-size": "13px",
                    "font-weight": "500",
                    "color": "#94a3b8",
                    "padding": "0.6rem 1rem",
                    "border-radius": "8px",
                    "margin": "1px 0",
                    "--hover-color": "rgba(255,255,255,0.06)",
                },
                "nav-link-selected": {
                    "background": "rgba(29,185,84,0.12)",
                    "color": "#f8fafc",
                    "font-weight": "600",
                    "border": "1px solid rgba(29,185,84,0.2)",
                },
            },
        )

        st.markdown('<div class="fancy-divider" style="margin:1.5rem 0"></div>',
                    unsafe_allow_html=True)

        if artifacts_ok:
            st.markdown("""
            <div style="padding:.6rem 1rem;background:rgba(29,185,84,0.08);
                 border:1px solid rgba(29,185,84,0.2);border-radius:8px;
                 font-size:.75rem;color:#4ade80">
              ✓ Models loaded
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding:.6rem 1rem;background:rgba(239,68,68,0.08);
                 border:1px solid rgba(239,68,68,0.2);border-radius:8px;
                 font-size:.75rem;color:#f87171">
              ⚠ Run train_models.py first
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="position:absolute;bottom:1.5rem;left:1rem;right:1rem;
             font-size:.68rem;color:#334155;text-align:center;line-height:1.6">
          150k+ Spotify Tracks · 1921–2020<br>
          7+ ML Models · Real-time Prediction
        </div>""", unsafe_allow_html=True)

    return page


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW (Landing)
# ══════════════════════════════════════════════════════════════════════════

def page_overview(df, results, final_features):
    page_hero(
        eyebrow="🎵 Spotify Intelligence Platform",
        title="Music Analytics &amp; Popularity Prediction",
        subtitle="150,000+ tracks · 7+ ML models · Real-time inference · 1921–2020 dataset",
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Tracks", f"{len(df):,}", color="green", icon="🎵")
    with c2: metric_card("Avg Popularity", f"{df['popularity'].mean():.1f}", color="purple", icon="📊")
    with c3: metric_card("Unique Artists", f"{df['artist_name'].nunique():,}", color="blue", icon="🎤")
    with c4:
        best_r2 = max(v["R2"] for v in results["reg_results"].values())
        metric_card("Best R²", f"{best_r2:.3f}", color="amber", icon="🏆")
    with c5:
        best_f1 = max(v["F1"] for v in results["clf_results"].values())
        metric_card("Best F1", f"{best_f1:.3f}", color="pink", icon="🎯")

    fancy_divider()

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        section_header("Popularity Distribution", badge="TARGET VARIABLE", icon="📈")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["popularity"], nbinsx=50,
            marker=dict(color="#1DB954", opacity=0.75,
                        line=dict(color="rgba(0,0,0,0.2)", width=0.4)),
            name="Tracks",
        ))
        apply_chart_style(fig, "", 320)
        fig.update_layout(bargap=0.04)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section_header("Popularity Categories", badge="DISTRIBUTION", icon="🍩")
        cat_counts = df["popularity_category"].value_counts()
        order = ["Hit", "Mainstream", "Emerging", "Underground"]
        cat_counts = cat_counts.reindex([c for c in order if c in cat_counts.index])
        fig2 = go.Figure(go.Pie(
            labels=cat_counts.index.tolist(),
            values=cat_counts.values.tolist(),
            hole=0.6,
            marker=dict(colors=["#1DB954", "#3b82f6", "#f59e0b", "#7c3aed"],
                        line=dict(color="#050508", width=3)),
            textfont=dict(color="#f1f5f9", size=12),
            hovertemplate="<b>%{label}</b><br>%{value:,} tracks<br>%{percent}<extra></extra>",
        ))
        fig2.update_layout(
            **PLOTLY_BASE, height=320,
            annotations=[dict(
                text=f"{len(df):,}<br>tracks",
                x=0.5, y=0.5, font_size=14, font_color="#f8fafc", showarrow=False,
            )]
        )
        st.plotly_chart(fig2, use_container_width=True)

    fancy_divider()

    section_header("Model Performance Snapshot", badge="7+ MODELS", icon="🤖")
    col_r, col_c = st.columns(2, gap="large")

    with col_r:
        reg_df = pd.DataFrame(results["reg_results"]).T.sort_values("R2", ascending=False)
        fig3 = bar_chart(reg_df.index.tolist(), reg_df["R2"].tolist(),
                         title="Regression — R² Score", color="#1DB954", height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col_c:
        clf_df = pd.DataFrame(results["clf_results"]).T.sort_values("F1", ascending=False)
        fig4 = bar_chart(clf_df.index.tolist(), clf_df["F1"].tolist(),
                         title="Classification — F1 Score", color="#7c3aed", height=320)
        st.plotly_chart(fig4, use_container_width=True)

    fancy_divider()

    section_header("Dataset Profile", badge="1921 – 2020", icon="📀")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        years = df["release_year"].value_counts().sort_index()
        peak  = int(years.idxmax())
        glass_card(f"""
            <div class="metric-label">📅 Year Coverage</div>
            <div class="metric-value" style="font-size:1.6rem">100 yrs</div>
            <div style="color:#475569;font-size:.78rem;margin-top:.3rem">
              1921 – 2020 · Peak year: {peak}
            </div>""")
    with s2:
        avg_dur = df["duration_min"].mean()
        glass_card(f"""
            <div class="metric-label">⏱ Avg Duration</div>
            <div class="metric-value" style="font-size:1.6rem">{avg_dur:.1f} min</div>
            <div style="color:#475569;font-size:.78rem;margin-top:.3rem">
              Optimal: 2.5–4 min for popularity
            </div>""")
    with s3:
        expl_pct = df["explicit"].mean() * 100
        glass_card(f"""
            <div class="metric-label">🔞 Explicit Rate</div>
            <div class="metric-value" style="font-size:1.6rem">{expl_pct:.1f}%</div>
            <div style="color:#475569;font-size:.78rem;margin-top:.3rem">
              Slightly ↑ popularity correlation
            </div>""")
    with s4:
        genres = df["primary_genre"].nunique()
        glass_card(f"""
            <div class="metric-label">🎼 Unique Genres</div>
            <div class="metric-value" style="font-size:1.6rem">{genres:,}</div>
            <div style="color:#475569;font-size:.78rem;margin-top:.3rem">
              Cross-genre artists score higher
            </div>""")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════

def page_data_explorer(df):
    page_hero(
        eyebrow="📋 Data Explorer",
        title="Interactive Dataset Browser",
        subtitle="Search, filter, sort and download the Spotify dataset.",
    )

    fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
    with fc1:
        search = st.text_input("🔍 Search track or artist", placeholder="e.g. Bohemian Rhapsody…")
    with fc2:
        cat_filter = st.multiselect("Category",
                                    ["Hit", "Mainstream", "Emerging", "Underground"],
                                    default=["Hit", "Mainstream", "Emerging", "Underground"])
    with fc3:
        year_min, year_max = int(df["release_year"].min()), int(df["release_year"].max())
        year_range = st.slider("Release Year", year_min, year_max, (2000, 2020))
    with fc4:
        sort_col = st.selectbox("Sort by", ["popularity", "danceability", "energy",
                                             "valence", "tempo", "acousticness"])

    view = df.copy()
    if search:
        mask = (view["name"].str.contains(search, case=False, na=False) |
                view["artist_name"].str.contains(search, case=False, na=False))
        view = view[mask]
    view = view[view["popularity_category"].isin(cat_filter)]
    view = view[view["release_year"].between(*year_range)]
    view = view.sort_values(sort_col, ascending=False)

    st.markdown(f"""
    <div style="display:flex;gap:1rem;margin:.75rem 0;flex-wrap:wrap">
      <span class="pill pill-green">✓ {len(view):,} tracks</span>
      <span class="pill pill-purple">Avg Popularity: {view['popularity'].mean():.1f}</span>
      <span class="pill pill-blue">Avg Energy: {view['energy'].mean():.2f}</span>
      <span class="pill pill-amber">Avg Danceability: {view['danceability'].mean():.2f}</span>
    </div>""", unsafe_allow_html=True)

    display_cols = ["name", "artist_name", "popularity", "popularity_category",
                    "release_year", "danceability", "energy", "valence",
                    "tempo", "duration_min", "explicit", "primary_genre"]
    display_cols = [c for c in display_cols if c in view.columns]
    show_df = view[display_cols].head(500).reset_index(drop=True)

    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
        gb = GridOptionsBuilder.from_dataframe(show_df)
        gb.configure_default_column(
            resizable=True, sortable=True, filter=True,
            cellStyle={"color": "#f1f5f9", "background": "transparent", "fontSize": "12px"},
        )
        gb.configure_column("popularity", type=["numericColumn"],
                            cellStyle={"color": "#1DB954", "fontWeight": "600"})
        gb.configure_column("popularity_category",
                            cellStyle={"color": "#a78bfa", "fontWeight": "500"})
        gb.configure_selection("multiple", use_checkbox=False)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        go_opts = gb.build()
        go_opts["rowStyle"] = {"background": "transparent"}
        AgGrid(show_df, gridOptions=go_opts,
               height=480, theme="streamlit",
               update_mode=GridUpdateMode.NO_UPDATE,
               data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
               allow_unsafe_jscode=True,
               custom_css={
                   ".ag-root-wrapper": {
                       "border": "1px solid rgba(255,255,255,0.08)!important",
                       "border-radius": "12px!important",
                   },
                   ".ag-header": {"background": "rgba(255,255,255,0.03)!important"},
               })
    except ImportError:
        st.dataframe(show_df, use_container_width=True, height=480)

    fancy_divider()
    dl_col, _ = st.columns([1, 3])
    with dl_col:
        csv_data = show_df.to_csv(index=False).encode()
        st.download_button("⬇ Download Filtered CSV", csv_data,
                           "spotify_filtered.csv", "text/csv")

    fancy_divider()
    section_header("Quick Visuals", badge="FILTERED DATA", icon="📊")
    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        fig = histogram_chart(view["popularity"], "Popularity Distribution", "#1DB954", 40, 280)
        st.plotly_chart(fig, use_container_width=True)
    with vc2:
        fig = histogram_chart(view["energy"], "Energy Distribution", "#7c3aed", 40, 280)
        st.plotly_chart(fig, use_container_width=True)
    with vc3:
        fig = histogram_chart(view["danceability"], "Danceability Distribution", "#3b82f6", 40, 280)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def page_analytics(df):
    page_hero(
        eyebrow="📊 Analytics Dashboard",
        title="Deep-Dive Audio Intelligence",
        subtitle="Correlations, distributions, outliers, and feature relationships.",
    )

    tabs = st.tabs(["📈 KPIs & Trends", "🔥 Correlation", "🎸 Audio Features", "🔎 Scatter Lab"])

    with tabs[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        stats = {
            "Avg Popularity":   (df["popularity"].mean(),    "green",  "📊"),
            "Avg Danceability": (df["danceability"].mean(),  "purple", "💃"),
            "Avg Energy":       (df["energy"].mean(),        "blue",   "⚡"),
            "Avg Valence":      (df["valence"].mean(),       "amber",  "😊"),
            "Explicit %":       (df["explicit"].mean() * 100,"pink",   "🔞"),
            "Avg Tempo":        (df["tempo"].mean(),         "green",  "🥁"),
        }
        for col, (key, (val, color, icon)) in zip([k1, k2, k3, k4, k5, k6], stats.items()):
            with col:
                fmt = f"{val:.0f}" if "Tempo" in key or "Popularity" in key else f"{val:.2f}"
                if "%" in key:
                    fmt = f"{val:.1f}%"
                metric_card(key, fmt, color=color, icon=icon)

        fancy_divider()
        section_header("Popularity by Decade", badge="TREND", icon="📅")
        df_plot = df.copy()
        decade_pop = df_plot.groupby("decade")["popularity"].mean().reset_index()
        decade_pop = decade_pop[decade_pop["decade"] != "0s"].sort_values("decade")

        fig = go.Figure(go.Bar(
            x=decade_pop["decade"], y=decade_pop["popularity"],
            marker=dict(
                color=decade_pop["popularity"],
                colorscale=[[0, "#7c3aed"], [0.5, "#3b82f6"], [1, "#1DB954"]],
                opacity=0.85,
                line=dict(color="rgba(0,0,0,0.2)", width=0.5),
            ),
            hovertemplate="<b>%{x}</b><br>Avg Popularity: %{y:.2f}<extra></extra>",
        ))
        apply_chart_style(fig, "", 340)
        st.plotly_chart(fig, use_container_width=True)

        fancy_divider()
        lc1, lc2 = st.columns(2)
        with lc1:
            section_header("Avg Popularity by Category", badge="CLASSES", icon="🏷")
            cat_pop = df_plot.groupby("popularity_category")["popularity"].mean().reindex(
                ["Hit", "Mainstream", "Emerging", "Underground"]
            )
            fig2 = bar_chart(cat_pop.index.tolist(), cat_pop.values.tolist(),
                             color="#1DB954", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        with lc2:
            section_header("Track Count by Decade", badge="VOLUME", icon="📀")
            decade_ct = df_plot.groupby("decade").size().reset_index(name="count")
            decade_ct = decade_ct[decade_ct["decade"] != "0s"].sort_values("decade")
            fig3 = bar_chart(decade_ct["decade"].tolist(), decade_ct["count"].tolist(),
                             color="#7c3aed", height=300)
            st.plotly_chart(fig3, use_container_width=True)

    with tabs[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Feature Correlation Matrix", badge="PEARSON", icon="🔥")
        num_cols = ["popularity", "danceability", "energy", "loudness", "speechiness",
                    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
                    "duration_min", "explicit", "track_age", "artist_popularity"]
        num_cols = [c for c in num_cols if c in df.columns]
        corr_mat = df[num_cols].corr()
        fig = heatmap_chart(corr_mat, "")
        st.plotly_chart(fig, use_container_width=True)

        fancy_divider()
        section_header("Feature Correlation with Popularity", badge="TARGET", icon="🎯")
        corr_pop = corr_mat["popularity"].drop("popularity").sort_values()
        colors = ["#f87171" if v < 0 else "#1DB954" for v in corr_pop.values]
        fig2 = go.Figure(go.Bar(
            x=corr_pop.values, y=corr_pop.index,
            orientation="h",
            marker=dict(color=colors, opacity=0.8),
            hovertemplate="<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>",
        ))
        apply_chart_style(fig2, "", 400)
        fig2.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Audio Feature Distributions", badge="9 FEATURES", icon="🎸")
        feats = [c for c in AUDIO_FEATURES if c in df.columns]
        feat_colors = COLOR_SEQ[:len(feats)]

        cols = st.columns(3)
        for i, feat in enumerate(feats):
            with cols[i % 3]:
                fig = histogram_chart(df[feat].dropna(), feat.capitalize(),
                                      feat_colors[i], 35, 240)
                st.plotly_chart(fig, use_container_width=True)

        fancy_divider()
        section_header("Radar: Avg Audio Profile", badge="SPIDER CHART", icon="🕸")
        cats = ["Hit", "Mainstream", "Emerging", "Underground"]
        radar_feats = ["danceability", "energy", "valence", "acousticness", "liveness"]
        radar_feats = [f for f in radar_feats if f in df.columns]
        fig_r = go.Figure()
        radar_colors = ["#1DB954", "#3b82f6", "#f59e0b", "#7c3aed"]
        for i, cat in enumerate(cats):
            sub = df[df["popularity_category"] == cat]
            if len(sub) == 0:
                continue
            vals = [sub[f].mean() for f in radar_feats]
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_feats + [radar_feats[0]],
                name=cat, fill="toself",
                fillcolor=f"rgba({int(radar_colors[i][1:3],16)},"
                           f"{int(radar_colors[i][3:5],16)},"
                           f"{int(radar_colors[i][5:],16)},0.1)",
                line=dict(color=radar_colors[i], width=2),
            ))
        fig_r.update_layout(**PLOTLY_BASE, height=420,
            polar=dict(
                bgcolor="rgba(255,255,255,0.02)",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="rgba(255,255,255,0.08)",
                                tickfont=dict(color="#475569", size=9)),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                                 tickfont=dict(color="#94a3b8", size=10)),
            ))
        st.plotly_chart(fig_r, use_container_width=True)

    with tabs[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Scatter Lab", badge="EXPLORE", icon="🔎")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            x_feat = st.selectbox("X axis", AUDIO_FEATURES, index=0)
        with sc2:
            y_feat = st.selectbox("Y axis", AUDIO_FEATURES, index=1)
        with sc3:
            color_feat = st.selectbox("Colour by", ["popularity_category", "explicit",
                                                      "is_modern", "is_popular_artist"])
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig_sc = scatter_chart(sample, x_feat, y_feat, color=color_feat,
                               title=f"{x_feat} vs {y_feat}", height=460)
        st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL LAB
# ══════════════════════════════════════════════════════════════════════════

def page_model_lab(results, final_features):
    page_hero(
        eyebrow="🧪 Model Lab",
        title="Model Performance &amp; Diagnostics",
        subtitle="Feature importance, confusion matrix, regression metrics.",
    )

    tabs = st.tabs(["📊 Feature Importance", "🧩 Confusion Matrix",
                    "📉 Regression", "📋 Classification"])

    with tabs[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Tuned RF — Feature Importance", badge="TOP FEATURES", icon="🌲")
        fi = results["feat_imp"]

        # Top-10 bar chart
        top10 = fi.head(10)
        fig = bar_chart(top10.index.tolist(), top10.values.tolist(),
                        title="Top 10 Features", color="#1DB954", height=360)
        st.plotly_chart(fig, use_container_width=True)

        fancy_divider()
        section_header("All Features Ranked", badge="IMPORTANCE", icon="📋")
        for feat, imp in fi.items():
            pct = imp / fi.sum() * 100
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem">
              <span style="font-size:.8rem;color:#94a3b8;width:11rem;flex-shrink:0">{feat}</span>
              <div style="flex:1;background:rgba(255,255,255,0.05);
                   border-radius:999px;height:6px;overflow:hidden">
                <div style="width:{pct*3:.0f}%;height:100%;
                     background:linear-gradient(90deg,#1DB954,#7c3aed);
                     border-radius:999px"></div>
              </div>
              <span style="color:#1DB954;font-weight:700;font-size:.85rem;
                   width:4rem;text-align:right">{imp:.4f}</span>
            </div>""", unsafe_allow_html=True)

    with tabs[1]:
        section_header(f"Confusion Matrix — {results['best_clf_name']}",
                       badge="BEST CLASSIFIER", icon="🧩")
        cm   = results["cm"]
        lbls = results["class_names"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig = go.Figure(go.Heatmap(
            z=cm_norm, x=lbls, y=lbls,
            colorscale=[[0, "#050508"], [0.5, "rgba(29,185,84,0.3)"], [1, "#1DB954"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=14, color="#f8fafc"),
            hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>"
                          "Count: %{text}<br>Rate: %{z:.2%}<extra></extra>",
            showscale=True,
            colorbar=dict(tickfont=dict(color="#94a3b8", size=10),
                          outlinecolor="rgba(255,255,255,0.08)", outlinewidth=1),
        ))
        apply_chart_style(fig, "", 480)
        fig.update_xaxes(title_text="Predicted", title_font=dict(color="#94a3b8"))
        fig.update_yaxes(title_text="Actual",    title_font=dict(color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        section_header("All Regression Models", badge="COMPARISON", icon="📊")
        reg_df = pd.DataFrame(results["reg_results"]).T.sort_values("R2", ascending=False)
        reg_df.index.name = "Model"
        reg_df = reg_df.reset_index()

        fig = go.Figure()
        for metric, color in [("R2", "#1DB954"), ("MAE", "#f59e0b"), ("RMSE", "#7c3aed")]:
            fig.add_trace(go.Bar(
                name=metric, x=reg_df["Model"], y=reg_df[metric],
                marker=dict(color=color, opacity=0.8),
            ))
        apply_chart_style(fig, "Regression Model Metrics", 420)
        fig.update_layout(barmode="group", bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            reg_df.set_index("Model").style
            .background_gradient(cmap="YlGn", subset=["R2"])
            .background_gradient(cmap="YlOrRd_r", subset=["MAE", "RMSE"])
            .format("{:.4f}"),
            use_container_width=True,
        )

    with tabs[3]:
        section_header("All Classification Models", badge="COMPARISON", icon="📋")
        clf_df = pd.DataFrame(results["clf_results"]).T.sort_values("F1", ascending=False)
        clf_df.index.name = "Model"
        clf_df = clf_df.reset_index()

        fig = go.Figure()
        for metric, color in [("Accuracy", "#1DB954"), ("Precision", "#3b82f6"),
                               ("Recall", "#f59e0b"), ("F1", "#7c3aed")]:
            fig.add_trace(go.Bar(
                name=metric, x=clf_df["Model"], y=clf_df[metric],
                marker=dict(color=color, opacity=0.8),
            ))
        apply_chart_style(fig, "Classification Model Metrics", 420)
        fig.update_layout(barmode="group", bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            clf_df.set_index("Model").style
            .background_gradient(cmap="Blues", subset=["Accuracy", "F1"])
            .format("{:.4f}"),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON (Leaderboard)
# ══════════════════════════════════════════════════════════════════════════

def page_model_comparison(results):
    page_hero(
        eyebrow="🏆 Model Comparison",
        title="Leaderboard &amp; Rankings",
        subtitle="Head-to-head battle across every regression and classification model.",
    )

    left, right = st.columns(2, gap="large")

    with left:
        section_header("Regression Leaderboard", badge="R² RANKING", icon="🥇")
        reg_df = pd.DataFrame(results["reg_results"]).T.sort_values("R2", ascending=False)
        for i, (name, row) in enumerate(reg_df.iterrows()):
            leader_row(i + 1, name, row["R2"], "R²")

        fancy_divider()
        section_header("R² Score Bars", badge="VISUAL", icon="📊")
        fig = go.Figure(go.Bar(
            x=reg_df["R2"].values,
            y=reg_df.index.tolist(),
            orientation="h",
            marker=dict(
                color=reg_df["R2"].values,
                colorscale=[[0, "#7c3aed"], [1, "#1DB954"]],
                opacity=0.85,
                line=dict(color="rgba(0,0,0,0.15)", width=0.5),
            ),
            hovertemplate="<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>",
        ))
        apply_chart_style(fig, "", 340)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section_header("Classification Leaderboard", badge="F1 RANKING", icon="🎯")
        clf_df = pd.DataFrame(results["clf_results"]).T.sort_values("F1", ascending=False)
        for i, (name, row) in enumerate(clf_df.iterrows()):
            leader_row(i + 1, name, row["F1"], "F1")

        fancy_divider()
        section_header("F1 Score Bars", badge="VISUAL", icon="📊")
        fig2 = go.Figure(go.Bar(
            x=clf_df["F1"].values,
            y=clf_df.index.tolist(),
            orientation="h",
            marker=dict(
                color=clf_df["F1"].values,
                colorscale=[[0, "#3b82f6"], [1, "#7c3aed"]],
                opacity=0.85,
                line=dict(color="rgba(0,0,0,0.15)", width=0.5),
            ),
            hovertemplate="<b>%{y}</b><br>F1: %{x:.4f}<extra></extra>",
        ))
        apply_chart_style(fig2, "", 340)
        st.plotly_chart(fig2, use_container_width=True)

    fancy_divider()
    section_header("🏅 Champion Models", badge="BEST IN CLASS", icon="")
    h1, h2 = st.columns(2)
    best_reg_name = reg_df["R2"].idxmax()
    best_clf_name = clf_df["F1"].idxmax()
    best_reg_r2   = reg_df.loc[best_reg_name, "R2"]
    best_clf_f1   = clf_df.loc[best_clf_name, "F1"]

    with h1:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;border-color:rgba(29,185,84,0.3);
             background:linear-gradient(135deg,rgba(29,185,84,0.06),rgba(10,10,15,0.8))">
          <div style="font-size:2rem;margin-bottom:.5rem">🥇</div>
          <div style="font-size:.72rem;font-weight:700;letter-spacing:.1em;
               color:#1DB954;text-transform:uppercase;margin-bottom:.4rem">Best Regressor</div>
          <div style="font-size:1.4rem;font-weight:800;color:#f8fafc">{best_reg_name}</div>
          <div style="font-size:2.5rem;font-weight:900;font-family:'Space Grotesk',sans-serif;
               color:#1DB954;margin:.4rem 0">{best_reg_r2:.4f}</div>
          <div style="font-size:.75rem;color:#475569">R² Score</div>
        </div>""", unsafe_allow_html=True)

    with h2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;border-color:rgba(124,58,237,0.3);
             background:linear-gradient(135deg,rgba(124,58,237,0.06),rgba(10,10,15,0.8))">
          <div style="font-size:2rem;margin-bottom:.5rem">🥇</div>
          <div style="font-size:.72rem;font-weight:700;letter-spacing:.1em;
               color:#a78bfa;text-transform:uppercase;margin-bottom:.4rem">Best Classifier</div>
          <div style="font-size:1.4rem;font-weight:800;color:#f8fafc">{best_clf_name}</div>
          <div style="font-size:2.5rem;font-weight:900;font-family:'Space Grotesk',sans-serif;
               color:#a78bfa;margin:.4rem 0">{best_clf_f1:.4f}</div>
          <div style="font-size:.75rem;color:#475569">Weighted F1 Score</div>
        </div>""", unsafe_allow_html=True)

    fancy_divider()
    section_header("Top-3 Regressors — Multi-Metric Radar", badge="R² · RMSE · MAE", icon="🕸")
    top3 = reg_df.head(3)
    fig_radar = go.Figure()
    radar_metrics = ["R2", "MAE", "RMSE"]
    radar_colors  = ["#1DB954", "#3b82f6", "#7c3aed"]
    for i, (name, row) in enumerate(top3.iterrows()):
        vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_metrics + [radar_metrics[0]], name=name,
            fill="toself",
            fillcolor=["rgba(29,185,84,0.15)",
                       "rgba(59,130,246,0.15)",
                       "rgba(124,58,237,0.15)"][i],
            line=dict(color=radar_colors[i], width=2),
        ))
    fig_radar.update_layout(**PLOTLY_BASE, height=380,
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.08)",
                            tickfont=dict(color="#475569", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                             tickfont=dict(color="#94a3b8", size=11)),
        ))
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICTION STUDIO
# ══════════════════════════════════════════════════════════════════════════

def page_prediction_studio(results, final_features):
    page_hero(
        eyebrow="🔮 Prediction Studio",
        title="Real-Time Song Popularity Predictor",
        subtitle="Tune audio features and artist metrics · Get instant AI-powered popularity score.",
    )

    form_col, result_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        section_header("Audio Features", badge="SPOTIFY METRICS", icon="🎛")

        f1, f2 = st.columns(2)
        with f1:
            danceability     = st.slider("💃 Danceability",    0.0, 1.0, 0.70, 0.01)
            energy           = st.slider("⚡ Energy",           0.0, 1.0, 0.80, 0.01)
            valence          = st.slider("😊 Valence",          0.0, 1.0, 0.60, 0.01)
            speechiness      = st.slider("🗣 Speechiness",      0.0, 1.0, 0.05, 0.01)
            acousticness     = st.slider("🎸 Acousticness",     0.0, 1.0, 0.10, 0.01)
        with f2:
            liveness         = st.slider("🎤 Liveness",         0.0, 1.0, 0.10, 0.01)
            instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.00, 0.01)
            loudness         = st.slider("🔊 Loudness (dB)",  -60.0, 0.0, -5.0, 0.5)
            tempo            = st.slider("🥁 Tempo (BPM)",    60.0, 200.0, 120.0, 1.0)
            duration_min     = st.slider("⏱ Duration (min)",   0.5, 10.0,   3.5, 0.1)

        fancy_divider()
        section_header("Artist & Track Metadata", badge="CONTEXT", icon="🎤")

        a1, a2 = st.columns(2)
        with a1:
            artist_popularity  = st.slider("🌟 Artist Popularity",   0, 100, 70)
            followers_log      = st.slider("👥 Followers (log)",     0.0, 20.0, 12.0, 0.1)
            artist_genre_count = st.slider("🎼 Genre Count",          0, 10, 3)
        with a2:
            track_age = st.slider("📅 Track Age (years)", 0, 100, 2)
            explicit  = st.selectbox("🔞 Explicit", [0, 1],
                                     format_func=lambda x: "Yes" if x else "No")
            is_modern  = st.selectbox("🆕 Modern (post-2010)", [1, 0],
                                      format_func=lambda x: "Yes" if x else "No")
            is_pop_art = st.selectbox("⭐ Popular Artist", [1, 0],
                                      format_func=lambda x: "Yes" if x else "No")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Popularity", use_container_width=True)

    with result_col:
        section_header("Prediction Result", badge="AI INFERENCE", icon="✨")

        if predict_btn or "last_pred" in st.session_state:
            if predict_btn:
                pred = predict_new_song(
                    final_features=final_features,
                    scaler=results["scaler"],
                    best_rf=results["best_rf"],
                    le=results["le"],
                    danceability=danceability,
                    energy=energy,
                    loudness=loudness,
                    speechiness=speechiness,
                    acousticness=acousticness,
                    instrumentalness=instrumentalness,
                    liveness=liveness,
                    valence=valence,
                    tempo=tempo,
                    duration_min=duration_min,
                    explicit=explicit,
                    track_age=track_age,
                    artist_popularity=artist_popularity,
                    followers_log=followers_log,
                    is_modern=is_modern,
                    is_popular_artist=is_pop_art,
                    artist_genre_count=artist_genre_count,
                )
                st.session_state["last_pred"] = pred
            else:
                pred = st.session_state["last_pred"]

            score = pred["predicted_score"]
            cat   = pred["predicted_category"]
            prediction_result_card(score, cat)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge = gauge_chart(score, "Popularity Score", 100)
            st.plotly_chart(fig_gauge, use_container_width=True)

            fancy_divider()
            section_header("Input Summary", badge="YOUR SONG", icon="📋")
            inputs = {
                "Danceability": danceability if predict_btn else "-",
                "Energy":       energy       if predict_btn else "-",
                "Valence":      valence      if predict_btn else "-",
                "Artist Pop.":  artist_popularity / 100 if predict_btn else "-",
            }
            for feat_name, val in inputs.items():
                if isinstance(val, float):
                    pct = int(val * 100)
                    st.markdown(f"""
                    <div style="margin-bottom:.6rem">
                      <div style="display:flex;justify-content:space-between;
                           font-size:.75rem;color:#94a3b8;margin-bottom:.3rem">
                        <span>{feat_name}</span><span>{val:.2f}</span>
                      </div>
                      <div style="background:rgba(255,255,255,0.05);
                           border-radius:999px;height:4px;overflow:hidden">
                        <div style="width:{pct}%;height:100%;
                             background:linear-gradient(90deg,#1DB954,#7c3aed);
                             border-radius:999px"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;
                 background:rgba(255,255,255,0.02);border-radius:16px;
                 border:1px dashed rgba(255,255,255,0.08)">
              <div style="font-size:3rem;margin-bottom:1rem">🎵</div>
              <div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:.5rem">
                Configure your song
              </div>
              <div style="font-size:.85rem;color:#475569">
                Adjust the sliders on the left and click<br>
                <b style="color:#1DB954">Predict Popularity</b> to see the result.
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════

def page_ai_insights(df, results, final_features):
    page_hero(
        eyebrow="💡 AI Insights",
        title="Automated Intelligence &amp; Recommendations",
        subtitle="15 business insights + 10 Spotify recommendations derived from 150k+ track analysis.",
    )

    tabs = st.tabs(["🔍 Key Findings", "💼 Business Recommendations", "🔬 Model Intelligence"])

    with tabs[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        insights = [
            ("📅", "Recency is the #1 Popularity Driver",
             "track_age and is_modern rank as the top predictors. Spotify's algorithm rewards freshly uploaded tracks. A&R teams should prioritize promoting new releases in the first 30 days post-launch."),
            ("⭐", "Artist Brand Beats Audio Features",
             "artist_popularity and followers_log consistently outrank danceability or energy — audience trust in an artist brand matters more than the track's raw sound profile."),
            ("💃", "Danceability × Energy = Hits",
             "The engineered dance_energy_index is one of the strongest features. Tracks that are simultaneously danceable AND energetic outperform either dimension alone — validating pop/EDM chart dominance."),
            ("🔊", "Loudness Proxies Production Quality",
             "Louder (less negative dB) tracks score higher. Louder masters are perceived as more polished and radio-ready, and Spotify's normalization doesn't fully eliminate the bias."),
            ("🎸", "High Acousticness = Low Modern Popularity",
             "Acoustic tracks score significantly lower in 2010–2020. This reflects electronic/hip-hop dominance, not a universal truth about acoustic quality."),
            ("🔞", "Explicit Tracks Are Slightly More Popular",
             "Reflects Spotify's 18–34 demographic who skew toward hip-hop, trap, and R&B — all genres with high explicit rates and strong streaming numbers."),
            ("📊", "Underground Tracks Dominate Volume (~50%+)",
             "Long-tail music economics: a tiny fraction of artists capture most listens. Algorithms must be carefully designed to surface emerging artists equitably."),
            ("🥁", "Tempo Is Surprisingly Weak",
             "Tracks across all BPM ranges achieve hits. The 120–130 BPM sweet spot is common (natural for dancing), but extreme tempos are not penalized in the model."),
            ("🎼", "Genre Diversity Expands Reach",
             "Artists tagged with more genres (artist_genre_count) tend to have higher artist_popularity. Cross-genre appeal broadens listener base and boosts algorithmic surface area."),
            ("📀", "2010s: Most Tracks & Highest Avg Popularity",
             "Streaming-era production democratized music creation, but Spotify's discovery engine amplifies the best-performing tracks of the decade disproportionately."),
            ("🗣", "Speechiness > 0.66 = Classifier Red Flag",
             "Tracks above this threshold are almost never in the 'Hit' category — podcasts and spoken-word uploads contaminate the dataset and dilute audio feature models."),
            ("🎤", "Liveness Penalizes Popularity",
             "Live recordings score lower — listeners prefer studio-quality tracks for on-demand listening. Exception: viral live performances from global superstars."),
            ("😊", "Valence Has Mild Positive Correlation",
             "Happy-sounding tracks outperform sad ones marginally, but dark/melancholic hits exist (Billie Eilish, The Weeknd), so emotional tone alone doesn't determine success."),
            ("⏱", "Optimal Duration: 2.5–4 Minutes",
             "Both very short (<1.5 min) and very long (>7 min) tracks score lower — catering to playlist behavior and streaming attention spans."),
            ("📉", "1990–2000 Dip from Stream-Based Bias",
             "90s tracks have fewer streams relative to cultural impact — digitized post-peak. Systematic underestimation creates a bias that needs historical correction in the model."),
        ]
        for icon, title, text in insights:
            insight_card(icon, title, text)

    with tabs[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        recs = [
            ("🚀", "Freshness Boost Algorithm",
             "Introduce a time-decay multiplier that temporarily boosts songs uploaded within the last 30 days in recommendation feeds, counteracting the popularity gap between new and established tracks."),
            ("📊", "Pre-Release Popularity Scoring API",
             "Offer record labels and independent artists a 'predicted popularity score' dashboard using this ML model, enabling data-driven promotion decisions before a song launches."),
            ("🎼", "Genre-Aware Normalization",
             "Popularity scores should be normalized within genre clusters (hip-hop vs. classical vs. folk), as current scoring systematically favors high-energy/vocal genres."),
            ("💃", "Dance-Energy Playlist Optimization",
             "Auto-curate playlists by maximizing dance_energy_index for workout and party contexts, using the feature importance results from the Random Forest model."),
            ("😊", "Mood Score Feature in UI",
             "Expose mood_score (valence × energy) in Spotify's public-facing audio features so users and curators can filter playlists by mood quadrant (Happy-Energetic, Calm-Positive, etc)."),
            ("🌱", "Emerging Artist Support",
             "Use classification probabilities from the model to proactively surface 'Emerging' songs predicted to become 'Mainstream' within 30 days, routing them into editorial playlists."),
            ("🔞", "Explicit Content Demographic Targeting",
             "Serve explicit tracks more aggressively in markets and demographic segments where correlation with popularity is validated, improving click-through rates and discovery."),
            ("🎤", "Label Partnership Dashboards",
             "Offer A&R teams an artist-level 'audio fingerprint' comparison — showing how a new signing's catalog compares to genre-defining artists across all 9 audio features."),
            ("📅", "Streaming Era Bias Correction",
             "For pre-2000 catalog tracks, apply a historical popularity correction coefficient to prevent under-representation of legacy classics in algorithmic recommendations."),
            ("📡", "Real-Time Trend Monitoring",
             "Deploy a rolling 30-day sliding window version of this model to detect audio feature trends before they peak — giving Spotify's editorial team a 2–4 week head start on emerging genre shifts."),
        ]
        for icon, title, text in recs:
            insight_card(icon, title, text)

    with tabs[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Feature Selection — Final Set",
                       badge=f"{len(final_features)} FEATURES", icon="🔬")

        fi = results["feat_imp"]
        fi_df = fi.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fi_df["Rank"] = range(1, len(fi_df) + 1)
        fi_df["Pct"] = (fi_df["Importance"] / fi_df["Importance"].sum() * 100).round(2)

        fig = go.Figure(go.Treemap(
            labels=fi_df["Feature"].tolist(),
            parents=[""] * len(fi_df),
            values=fi_df["Importance"].tolist(),
            texttemplate="<b>%{label}</b><br>%{value:.4f}",
            marker=dict(
                colorscale=[[0, "#050508"], [0.3, "#7c3aed"], [0.7, "#3b82f6"], [1, "#1DB954"]],
                line=dict(color="#050508", width=2),
                pad=dict(t=20, l=5, r=5, b=5),
            ),
            textfont=dict(color="#f8fafc", size=12),
            hovertemplate="<b>%{label}</b><br>Importance: %{value:.4f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_BASE, height=480)
        st.plotly_chart(fig, use_container_width=True)

        fancy_divider()
        section_header("Feature Importance Table", badge="RANKED", icon="📋")
        st.dataframe(
            fi_df.set_index("Rank").style
            .bar(subset=["Importance"], color="#1DB95444")
            .bar(subset=["Pct"], color="#7c3aed44")
            .format({"Importance": "{:.5f}", "Pct": "{:.2f}%"}),
            use_container_width=True,
            height=420,
        )


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Check artifacts exist ──────────────────────────────────────────
    artifacts_ok = os.path.exists(os.path.join(ARTIFACTS_DIR, "best_rf.pkl"))

    page = render_sidebar(artifacts_ok)

    if not artifacts_ok:
        st.error(
            "**Artifacts not found.** "
            "Run `python train_models.py` locally, then commit the `artifacts/` folder to your repo.",
            icon="🚨",
        )
        st.code("python train_models.py --tracks data/tracks.csv --artists data/artists.csv")
        st.stop()

    # ── Load (fast — from disk, cached after first hit) ────────────────
    with st.spinner("Loading models…"):
        results, final_features = get_models()

    with st.spinner("Loading dataset…"):
        df = get_dashboard_data()

    # ── Route to page ──────────────────────────────────────────────────
    if page == "Overview":
        page_overview(df, results, final_features)
    elif page == "Data Explorer":
        page_data_explorer(df)
    elif page == "Analytics":
        page_analytics(df)
    elif page == "Model Lab":
        page_model_lab(results, final_features)
    elif page == "Model Comparison":
        page_model_comparison(results)
    elif page == "Prediction Studio":
        page_prediction_studio(results, final_features)
    elif page == "AI Insights":
        page_ai_insights(df, results, final_features)


if __name__ == "__main__":
    main()