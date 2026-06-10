"""
components/ui.py — Premium reusable UI building blocks
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── Plotly base layout ────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.08)",
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.08)",
        showgrid=True,
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
    ),
)

COLOR_SEQ = ["#1DB954","#7c3aed","#3b82f6","#f59e0b","#ec4899",
             "#06b6d4","#84cc16","#f97316","#a855f7","#14b8a6"]

def apply_chart_style(fig, title="", height=400):
    fig.update_layout(**PLOTLY_BASE, title=dict(
        text=title, font=dict(size=14, color="#f1f5f9"),
        x=0, xanchor="left", pad=dict(l=4),
    ), height=height)
    return fig


# ── HTML helpers ─────────────────────────────────────────────────────────
def metric_card(label, value, delta=None, color="green", icon=""):
    delta_html = ""
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        cls  = "pos" if delta >= 0 else "neg"
        delta_html = f'<div class="metric-delta {cls}">{sign}{delta}</div>'
    st.markdown(f"""
    <div class="metric-card {color} fade-up">
      <div class="metric-label">{icon} {label}</div>
      <div class="metric-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def section_header(title, badge=None, icon=""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header fade-up">
      <span class="section-title">{icon} {title}</span>
      {badge_html}
    </div>""", unsafe_allow_html=True)


def glass_card(content_html, extra_class=""):
    st.markdown(f'<div class="glass-card {extra_class}">{content_html}</div>',
                unsafe_allow_html=True)


def fancy_divider():
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)


def pill(text, color="green"):
    dot = {"green":"🟢","purple":"🟣","blue":"🔵","amber":"🟡"}.get(color,"●")
    st.markdown(f'<span class="pill pill-{color}">{dot} {text}</span>',
                unsafe_allow_html=True)


def page_hero(eyebrow, title, subtitle):
    st.markdown(f"""
    <div class="page-hero fade-up">
      <div class="hero-eyebrow">{eyebrow}</div>
      <div class="hero-title">{title}</div>
      <div class="hero-subtitle">{subtitle}</div>
    </div>""", unsafe_allow_html=True)


def leader_row(rank, name, score, score_label="R²", medal=None):
    medal_class = {1:"gold",2:"silver",3:"bronze"}.get(rank, "")
    medal_icon  = {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"#{rank}")
    st.markdown(f"""
    <div class="leader-row {medal_class} fade-up">
      <div class="leader-rank">{medal_icon}</div>
      <div class="leader-name">{name}</div>
      <div class="leader-score" style="color:#1DB954">{score:.4f} <span style="color:#475569;font-size:.75rem">{score_label}</span></div>
    </div>""", unsafe_allow_html=True)


def insight_card(icon, title, text):
    st.markdown(f"""
    <div class="insight-card fade-up">
      <div class="insight-icon">{icon}</div>
      <div class="insight-body">
        <div class="insight-title">{title}</div>
        <div class="insight-text">{text}</div>
      </div>
    </div>""", unsafe_allow_html=True)


def prediction_result_card(score, category):
    pct = int(score)
    color_map = {
        "Hit":"#1DB954","Mainstream":"#3b82f6",
        "Emerging":"#f59e0b","Underground":"#7c3aed"
    }
    cat_clean = category.split()[0]
    clr = color_map.get(cat_clean, "#1DB954")
    st.markdown(f"""
    <div class="prediction-card">
      <div class="prediction-score">{score:.1f}</div>
      <div class="prediction-category">{category}</div>
      <div class="prediction-label">Predicted Popularity Score / 100</div>
      <div style="margin-top:1.5rem">
        <div style="display:flex;justify-content:space-between;font-size:.75rem;color:#475569;margin-bottom:.4rem">
          <span>Popularity</span><span>{pct}%</span>
        </div>
        <div class="confidence-bar-wrap">
          <div class="confidence-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{clr},{clr}aa)"></div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── Chart builders ────────────────────────────────────────────────────────

def gauge_chart(value, title="Score", max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x":[0,1],"y":[0,1]},
        title={"text":title,"font":{"color":"#94a3b8","size":13}},
        number={"font":{"color":"#f8fafc","size":48,"family":"Space Grotesk"}},
        gauge=dict(
            axis=dict(range=[0,max_val], tickcolor="#475569",
                      tickfont=dict(color="#475569",size=10)),
            bar=dict(color="#1DB954", thickness=0.25),
            bgcolor="rgba(255,255,255,0.04)",
            borderwidth=0,
            steps=[
                dict(range=[0,10],   color="rgba(124,58,237,0.15)"),
                dict(range=[10,40],  color="rgba(245,158,11,0.15)"),
                dict(range=[40,70],  color="rgba(59,130,246,0.15)"),
                dict(range=[70,100], color="rgba(29,185,84,0.15)"),
            ],
            threshold=dict(line=dict(color="#f8fafc",width=2), value=value),
        ),
    ))
    fig.update_layout(**PLOTLY_BASE, height=280)
    return fig


def bar_chart(x, y, title="", color="#1DB954", orientation="v", height=360):
    fig = go.Figure(go.Bar(
        x=x if orientation=="v" else y,
        y=y if orientation=="v" else x,
        orientation=orientation,
        marker=dict(color=color, opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.06)",width=1)),
        hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>" if orientation=="v"
                      else "<b>%{y}</b><br>%{x:.4f}<extra></extra>",
    ))
    apply_chart_style(fig, title, height)
    return fig


def heatmap_chart(df_corr, title="Correlation Heatmap"):
    fig = go.Figure(go.Heatmap(
        z=df_corr.values,
        x=df_corr.columns.tolist(),
        y=df_corr.index.tolist(),
        colorscale=[
            [0.0, "#7c3aed"],[0.5, "#0a0a0f"],[1.0, "#1DB954"]
        ],
        zmid=0,
        text=np.round(df_corr.values,2),
        texttemplate="%{text}",
        textfont=dict(size=9, color="#94a3b8"),
        hoverongaps=False,
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#94a3b8",size=10),
            outlinecolor="rgba(255,255,255,0.08)",
            outlinewidth=1,
        ),
    ))
    apply_chart_style(fig, title, 500)
    return fig


def scatter_chart(df, x, y, color=None, size=None, title="", height=420):
    kw = dict(color=color) if color else {}
    if size: kw["size"] = size
    fig = px.scatter(df, x=x, y=y, **kw,
                     color_discrete_sequence=COLOR_SEQ,
                     opacity=0.65,
                     hover_data={x:True, y:True})
    apply_chart_style(fig, title, height)
    return fig


def histogram_chart(series, title="", color="#1DB954", bins=40, height=320):
    fig = go.Figure(go.Histogram(
        x=series, nbinsx=bins,
        marker=dict(color=color, opacity=0.8,
                    line=dict(color="rgba(0,0,0,0.3)",width=0.5)),
    ))
    apply_chart_style(fig, title, height)
    return fig


def radar_chart(categories, values, title="", height=380):
    cats = categories + [categories[0]]
    vals = list(values) + [values[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(29,185,84,0.12)",
        line=dict(color="#1DB954", width=2),
        marker=dict(color="#1DB954", size=4),
    ))
    fig.update_layout(**PLOTLY_BASE,
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,0.08)",
                            tickfont=dict(color="#475569",size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                             tickfont=dict(color="#94a3b8",size=10)),
        ),
        title=dict(text=title, font=dict(size=14,color="#f1f5f9")),
        height=height,
    )
    return fig


def line_chart(df, x, y_cols, title="", height=380):
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], name=col, mode="lines",
            line=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)], width=2),
            fill="tozeroy" if i == 0 else "none",
            fillcolor=f"rgba({int(COLOR_SEQ[i%len(COLOR_SEQ)][1:3],16)},"
                      f"{int(COLOR_SEQ[i%len(COLOR_SEQ)][3:5],16)},"
                      f"{int(COLOR_SEQ[i%len(COLOR_SEQ)][5:],16)},0.06)",
        ))
    apply_chart_style(fig, title, height)
    return fig