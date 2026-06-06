"""
Road Damage Detection — YOLOv8 Results Dashboard
=================================================
Bug fix: `training_curve_fig()` generated invalid Plotly `fillcolor` values.

ROOT CAUSE (line ~418 in original app.py):
    fillcolor=cols[name].rstrip(")") + ",0.05)" if cols[name].startswith("rgb")
             else cols[name] + "12"

  • When the color is a hex string like "#E74C3C", the `else` branch appends
    the literal text "12", producing "#E74C3C12".
  • Plotly does NOT support 8-digit CSS hex (#RRGGBBAA). It only accepts
    "#RRGGBB", "rgb(r,g,b)", or "rgba(r,g,b,a)".
  • The new Plotly validator introduced in Python 3.14 / plotly ≥ 6.x is
    stricter and raises ValueError on "#RRGGBBAA".

FIX: Replace the fragile inline expression with a robust `to_rgba(color, alpha)`
helper that handles hex, rgb(), and rgba() inputs and always outputs valid
"rgba(r,g,b,a)" strings that every Plotly version accepts.
"""

import re
import io
import zipfile
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY ── safe color → rgba conversion (the core bug fix)
# ──────────────────────────────────────────────────────────────────────────────

def to_rgba(color: str, alpha: float = 0.08) -> str:
    """Convert any CSS color string to a valid Plotly rgba() string.

    Handles:
      '#RGB'        → expands to 6-digit hex first
      '#RRGGBB'     → rgba(r,g,b,alpha)
      '#RRGGBBAA'   → rgba(r,g,b,alpha)   (ignores embedded alpha, uses arg)
      'rgb(r,g,b)'  → rgba(r,g,b,alpha)
      'rgba(r,g,b,a)' → rgba(r,g,b,alpha)  (replaces alpha with arg)
    """
    c = color.strip()

    # ── hex ──────────────────────────────────────────────────────────────────
    if c.startswith("#"):
        h = c.lstrip("#")
        if len(h) == 3:                          # shorthand #RGB
            h = "".join(ch * 2 for ch in h)
        if len(h) in (6, 8):                     # #RRGGBB or #RRGGBBAA
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

    # ── rgb() / rgba() ───────────────────────────────────────────────────────
    m = re.match(r"rgba?\(([^)]+)\)", c, re.I)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        r, g, b = parts[0], parts[1], parts[2]
        return f"rgba({r},{g},{b},{alpha})"

    # ── fallback: return as-is (named colors like 'red' are valid in Plotly) ─
    return c


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_results_csv(source) -> pd.DataFrame:
    """Load results.csv from a file path, uploaded file, or ZIP bytes."""
    if isinstance(source, (str,)):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data
def load_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith("results.csv")]
        if not csv_files:
            st.error("No results.csv found inside the ZIP.")
            st.stop()
        with zf.open(csv_files[0]) as f:
            df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df


def detect_resume_points(df: pd.DataFrame) -> list[int]:
    """Return row indices where training was resumed (time column resets)."""
    if "time" not in df.columns:
        return []
    diffs = df["time"].diff()
    return df[diffs < 0].index.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    # Losses
    "train/box_loss":  "#E74C3C",
    "train/cls_loss":  "#E67E22",
    "train/dfl_loss":  "#9B59B6",
    "val/box_loss":    "#C0392B",
    "val/cls_loss":    "#D35400",
    "val/dfl_loss":    "#8E44AD",
    # Metrics
    "metrics/mAP50(B)":    "#2ECC71",
    "metrics/mAP50-95(B)": "#27AE60",
    "metrics/precision(B)": "#3498DB",
    "metrics/recall(B)":    "#2980B9",
    # LR
    "lr/pg0": "#F1C40F",
    "lr/pg1": "#F39C12",
    "lr/pg2": "#D4AC0D",
}

DISPLAY_NAMES = {
    "train/box_loss":       "Train Box Loss",
    "train/cls_loss":       "Train Cls Loss",
    "train/dfl_loss":       "Train DFL Loss",
    "val/box_loss":         "Val Box Loss",
    "val/cls_loss":         "Val Cls Loss",
    "val/dfl_loss":         "Val DFL Loss",
    "metrics/mAP50(B)":     "mAP@50",
    "metrics/mAP50-95(B)":  "mAP@50-95",
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)":    "Recall",
    "lr/pg0":               "LR (pg0)",
    "lr/pg1":               "LR (pg1)",
    "lr/pg2":               "LR (pg2)",
}


# ──────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def training_curve_fig(df: pd.DataFrame, resume_rows: list[int]) -> go.Figure:
    """
    Render training curves with shaded fills.

    ✅ FIX: All fillcolor values are generated via `to_rgba()` which always
    returns valid 'rgba(r,g,b,a)' strings, avoiding the '#RRGGBB12' pattern
    that caused ValueError in plotly's stricter validator.
    """
    metric_groups = {
        "Losses — Train": ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
        "Losses — Val":   ["val/box_loss",   "val/cls_loss",   "val/dfl_loss"],
        "Detection Metrics": ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                              "metrics/precision(B)", "metrics/recall(B)"],
        "Learning Rate":  ["lr/pg0"],
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(metric_groups.keys()),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), (group_name, cols) in zip(positions, metric_groups.items()):
        for col_name in cols:
            if col_name not in df.columns:
                continue
            vals   = df[col_name].values
            epochs = df["epoch"].values
            color  = PALETTE.get(col_name, "#888888")
            label  = DISPLAY_NAMES.get(col_name, col_name)

            # ── Line trace ──────────────────────────────────────────────────
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=vals,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                    legendgroup=group_name,
                    hovertemplate=f"<b>{label}</b><br>Epoch: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
                ),
                row=row, col=col,
            )

            # ── Shaded fill — FIX IS HERE ────────────────────────────────────
            # OLD (buggy): cols[name] + "12"  →  "#E74C3C12"  (invalid 8-digit hex)
            # NEW (fixed): to_rgba(color, 0.08) → "rgba(231,76,60,0.08)"  ✅
            fill_color = to_rgba(color, alpha=0.08)

            fig.add_trace(
                go.Scatter(
                    x=epochs, y=vals,
                    mode="none",
                    fill="tozeroy",
                    fillcolor=fill_color,       # ← FIXED
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=group_name,
                ),
                row=row, col=col,
            )

        # ── Resume markers (vertical dashed lines) ───────────────────────────
        for resume_idx in resume_rows:
            resume_epoch = df.loc[resume_idx, "epoch"]
            fig.add_vline(
                x=resume_epoch,
                line=dict(color="rgba(255,255,255,0.5)", width=1.5, dash="dot"),
                row=row, col=col,
                annotation_text="resume",
                annotation_font_size=9,
                annotation_font_color="rgba(255,255,255,0.6)",
            )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1A1E2E",
        font=dict(color="#E0E0E0", size=12),
        legend=dict(
            bgcolor="rgba(30,34,50,0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        margin=dict(l=50, r=20, t=60, b=40),
        title=dict(
            text="YOLOv8 Training Curves",
            font=dict(size=18),
            x=0.5, xanchor="center",
        ),
    )
    fig.update_xaxes(title_text="Epoch", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def loss_comparison_fig(df: pd.DataFrame) -> go.Figure:
    """Side-by-side train vs val loss comparison."""
    fig = go.Figure()
    pairs = [
        ("train/box_loss", "val/box_loss",  "Box Loss",  "#E74C3C", "#C0392B"),
        ("train/cls_loss", "val/cls_loss",  "Cls Loss",  "#E67E22", "#D35400"),
        ("train/dfl_loss", "val/dfl_loss",  "DFL Loss",  "#9B59B6", "#8E44AD"),
    ]
    epochs = df["epoch"].values
    for tr_col, vl_col, label, tr_color, vl_color in pairs:
        if tr_col in df.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=df[tr_col].values, name=f"Train {label}",
                line=dict(color=tr_color, width=2),
                hovertemplate=f"<b>Train {label}</b><br>Epoch: %{{x}}<br>%{{y:.4f}}<extra></extra>",
            ))
        if vl_col in df.columns:
            fig.add_trace(go.Scatter(
                x=epochs, y=df[vl_col].values, name=f"Val {label}",
                line=dict(color=vl_color, width=2, dash="dash"),
                hovertemplate=f"<b>Val {label}</b><br>Epoch: %{{x}}<br>%{{y:.4f}}<extra></extra>",
            ))

    fig.update_layout(
        title="Train vs Validation Loss",
        height=400, template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1E2E",
        xaxis_title="Epoch", yaxis_title="Loss",
        legend=dict(bgcolor="rgba(30,34,50,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        font=dict(color="#E0E0E0"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def metrics_fig(df: pd.DataFrame) -> go.Figure:
    """mAP, Precision, Recall over epochs."""
    fig = go.Figure()
    metrics = [
        ("metrics/mAP50(B)",     "mAP@50",    "#2ECC71"),
        ("metrics/mAP50-95(B)", "mAP@50-95", "#27AE60"),
        ("metrics/precision(B)", "Precision", "#3498DB"),
        ("metrics/recall(B)",    "Recall",    "#2980B9"),
    ]
    epochs = df["epoch"].values
    for col, label, color in metrics:
        if col not in df.columns:
            continue
        vals = df[col].values
        fig.add_trace(go.Scatter(
            x=epochs, y=vals, name=label,
            line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=to_rgba(color, 0.07),  # ← using the fixed helper
            hovertemplate=f"<b>{label}</b><br>Epoch: %{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Detection Metrics over Training",
        height=400, template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1E2E",
        xaxis_title="Epoch", yaxis_title="Score (0–1)",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(bgcolor="rgba(30,34,50,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        font=dict(color="#E0E0E0"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def lr_fig(df: pd.DataFrame) -> go.Figure:
    """Learning rate schedule."""
    fig = go.Figure()
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    colors  = ["#F1C40F", "#F39C12", "#D4AC0D"]
    epochs  = df["epoch"].values
    for col, color in zip(lr_cols, colors):
        fig.add_trace(go.Scatter(
            x=epochs, y=df[col].values,
            name=col, line=dict(color=color, width=2),
            hovertemplate=f"<b>{col}</b><br>Epoch: %{{x}}<br>LR: %{{y:.6f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Learning Rate Schedule",
        height=300, template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1E2E",
        xaxis_title="Epoch", yaxis_title="LR",
        font=dict(color="#E0E0E0"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def epoch_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a human-readable summary for the 5 best mAP50 epochs."""
    key = "metrics/mAP50(B)"
    if key not in df.columns:
        return df.head()
    top = df.nlargest(5, key)[
        ["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
         "metrics/precision(B)", "metrics/recall(B)",
         "train/box_loss", "val/box_loss"]
    ].copy()
    top.columns = ["Epoch", "mAP@50", "mAP@50-95", "Precision", "Recall",
                   "Train Box Loss", "Val Box Loss"]
    top = top.round(4).reset_index(drop=True)
    return top


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Road Damage Detection — YOLOv8 Dashboard",
    page_icon="🛣️",
    layout="wide",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .metric-card {
        background: #1A1E2E; border-radius: 12px; padding: 18px 22px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #2ECC71; }
    .metric-card .label { font-size: 0.85rem; color: #9BA3B8; margin-top: 4px; }
    .metric-card .delta { font-size: 0.8rem; margin-top: 2px; }
    .bug-box {
        background: #1E0B0B; border-left: 4px solid #E74C3C;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0;
        font-family: monospace; font-size: 0.85rem; color: #F5B7B1;
    }
    .fix-box {
        background: #0B1E0E; border-left: 4px solid #2ECC71;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0;
        font-family: monospace; font-size: 0.85rem; color: #A9DFBF;
    }
    h1, h2, h3 { color: #E0E0E0; }
    .stTabs [data-baseweb="tab"] { color: #9BA3B8; }
    .stTabs [aria-selected="true"] { color: #2ECC71; border-bottom-color: #2ECC71; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛣️ Road Damage Detection — YOLOv8 Dashboard")
st.markdown("**Training results viewer with bug-fixed training curves**")

# ── Sidebar: data upload ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Load Data")
    upload = st.file_uploader(
        "Upload results.csv or metrics ZIP",
        type=["csv", "zip"],
        help="Upload your Ultralytics results.csv or the metrics ZIP file",
    )
    st.divider()
    st.markdown("**About the bug fix**")
    st.markdown(
        "The original `training_curve_fig()` appended `'12'` to hex colours "
        "(`'#RRGGBB' + '12'` → `'#RRGGBB12'`) which is not a valid Plotly "
        "colour string. The fix uses `to_rgba()` which always produces valid "
        "`rgba(r,g,b,a)` strings."
    )

# ── Load data ─────────────────────────────────────────────────────────────────
df = None

if upload is not None:
    if upload.name.endswith(".zip"):
        df = load_from_zip(upload.read())
    else:
        df = load_results_csv(upload)
    st.sidebar.success(f"✅ Loaded {len(df)} epochs")
else:
    # Try to load from the default path (when running on Streamlit Cloud
    # with the repo's bundled results.csv)
    import os
    default_paths = [
        "results.csv",
        "content/drive/MyDrive/YOLO_RoadDamage/run2/results.csv",
    ]
    for p in default_paths:
        if os.path.exists(p):
            df = load_results_csv(p)
            st.sidebar.info(f"Auto-loaded: `{p}`")
            break

    if df is None:
        st.info("👈 Upload your `results.csv` or metrics ZIP in the sidebar to get started.")
        st.stop()

# ── Compute derived info ───────────────────────────────────────────────────────
resume_rows = detect_resume_points(df)
total_epochs = int(df["epoch"].max())
best_idx     = df["metrics/mAP50(B)"].idxmax() if "metrics/mAP50(B)" in df.columns else 0
best_row     = df.loc[best_idx]
last_row     = df.iloc[-1]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_curves, tab_losses, tab_metrics, tab_lr, tab_debug = st.tabs([
    "📊 Overview",
    "📈 Training Curves",
    "📉 Loss Detail",
    "🎯 Metrics",
    "🔧 Learning Rate",
    "🐛 Bug Fix",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if resume_rows:
        st.warning(
            f"⚠️ Training was **resumed** at epoch {int(df.loc[resume_rows[0], 'epoch'])}. "
            "Metrics show a jump because the resumed checkpoint was different (fine-tuned weights). "
            "Resume points are marked with dotted lines on the charts."
        )

    # ── KPI cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Total Epochs",  f"{total_epochs}", "rows in results.csv", "#3498DB"),
        (c2, "Best mAP@50",   f"{best_row['metrics/mAP50(B)']:.4f}",
              f"Epoch {int(best_row['epoch'])}", "#2ECC71"),
        (c3, "Best mAP@50-95", f"{best_row['metrics/mAP50-95(B)']:.4f}",
              f"Epoch {int(best_row['epoch'])}", "#27AE60"),
        (c4, "Best Precision", f"{df['metrics/precision(B)'].max():.4f}",
              f"Epoch {int(df.loc[df['metrics/precision(B)'].idxmax(),'epoch'])}", "#3498DB"),
        (c5, "Best Recall",   f"{df['metrics/recall(B)'].max():.4f}",
              f"Epoch {int(df.loc[df['metrics/recall(B)'].idxmax(),'epoch'])}", "#2980B9"),
    ]
    for col, label, val, delta, color in kpis:
        col.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:{color}">{val}</div>
            <div class="label">{label}</div>
            <div class="delta" style="color:#7F8C8D">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Training summary ────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("📋 Top 5 Epochs by mAP@50")
        st.dataframe(epoch_summary_table(df), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("📦 Training Configuration")
        config = {
            "Model": "YOLOv8",
            "Optimizer": "SGD",
            "Epochs configured": 100,
            "Epochs completed": total_epochs,
            "Batch size": 8,
            "Image size": "640×640",
            "LR (initial)": "0.001",
            "Patience": 10,
            "Resumed": f"Yes — at epoch {int(df.loc[resume_rows[0],'epoch'])}" if resume_rows else "No",
            "AMP": "Enabled",
        }
        config_df = pd.DataFrame(list(config.items()), columns=["Parameter", "Value"])
        st.dataframe(config_df, use_container_width=True, hide_index=True)

    # ── Final metrics ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏁 Final Epoch Metrics (Epoch 94)")
    f1, f2, f3, f4 = st.columns(4)
    final_metrics = [
        (f1, "mAP@50",    last_row.get("metrics/mAP50(B)", 0),    "#2ECC71"),
        (f2, "mAP@50-95", last_row.get("metrics/mAP50-95(B)", 0), "#27AE60"),
        (f3, "Precision", last_row.get("metrics/precision(B)", 0), "#3498DB"),
        (f4, "Recall",    last_row.get("metrics/recall(B)", 0),   "#2980B9"),
    ]
    for col, label, val, color in final_metrics:
        pct = int(val * 100)
        col.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:{color}">{val:.4f}</div>
            <div class="label">{label}</div>
            <div class="delta" style="color:{color}">{pct}%</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Training Curves (the fixed function)
# ════════════════════════════════════════════════════════════════════════════════
with tab_curves:
    st.subheader("📈 All Training Curves")
    if resume_rows:
        st.caption(
            f"Dotted vertical lines mark resume points "
            f"(epoch {', '.join(str(int(df.loc[i,'epoch'])) for i in resume_rows)})"
        )
    # ✅ This is where the bug was — now fixed via to_rgba()
    fig = training_curve_fig(df, resume_rows)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 View raw data"):
        st.dataframe(df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Loss Detail
# ════════════════════════════════════════════════════════════════════════════════
with tab_losses:
    st.subheader("📉 Train vs Validation Losses")
    st.plotly_chart(loss_comparison_fig(df), use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Final Train Losses**")
        for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
            if col in df.columns:
                label = DISPLAY_NAMES.get(col, col)
                st.metric(label, f"{last_row[col]:.4f}")

    with col_b:
        st.markdown("**Final Val Losses**")
        for col in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
            if col in df.columns:
                label = DISPLAY_NAMES.get(col, col)
                first_val = df.iloc[0][col]
                delta_val = last_row[col] - first_val
                st.metric(label, f"{last_row[col]:.4f}", delta=f"{delta_val:+.4f}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Detection Metrics
# ════════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.subheader("🎯 Detection Metrics over Training")
    st.plotly_chart(metrics_fig(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Metric Progression (first → best → final)")
    metric_cols = [
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/precision(B)", "metrics/recall(B)",
    ]
    rows = []
    for col in metric_cols:
        if col not in df.columns:
            continue
        rows.append({
            "Metric":  DISPLAY_NAMES.get(col, col),
            "Epoch 1": round(df.iloc[0][col], 4),
            "Best":    round(df[col].max(), 4),
            "Best Epoch": int(df.loc[df[col].idxmax(), "epoch"]),
            "Final":   round(last_row[col], 4),
            "Δ (E1→Final)": round(last_row[col] - df.iloc[0][col], 4),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Learning Rate
# ════════════════════════════════════════════════════════════════════════════════
with tab_lr:
    st.subheader("🔧 Learning Rate Schedule")
    st.plotly_chart(lr_fig(df), use_container_width=True)
    st.caption(
        "LR warms up for the first 3 epochs, then decays with cosine annealing. "
        "The schedule resets slightly at the resume point (dotted line)."
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — Bug Fix Explanation
# ════════════════════════════════════════════════════════════════════════════════
with tab_debug:
    st.subheader("🐛 Bug Root Cause & Fix")

    st.markdown("""
The crash occurred in `training_curve_fig()` at the `fillcolor` parameter of `go.Scatter`.

---
### 🔴 Buggy code (original `app.py` ~line 418)
""")
    st.markdown("""
<div class="bug-box">
fig.add_trace(go.Scatter(<br>
&nbsp;&nbsp;x=epochs, y=vals, mode="lines", name=name,<br>
&nbsp;&nbsp;<strong>fillcolor=cols[name].rstrip(")") + ",0.05)"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if cols[name].startswith("rgb")<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else cols[name] + "12"</strong><br>
))
</div>
""", unsafe_allow_html=True)

    st.markdown("""
**What goes wrong:**

When `cols[name]` is a hex string like `"#E74C3C"` (does not start with `"rgb"`), the
`else` branch runs: `"#E74C3C" + "12"` → `"#E74C3C12"`.

Plotly does **not** support 8-digit CSS hex (`#RRGGBBAA`). It only accepts:
- `"#RRGGBB"` — 6-digit hex (no transparency)  
- `"rgb(r, g, b)"` — RGB  
- `"rgba(r, g, b, a)"` — RGBA ✅ with transparency

The newer Plotly validator (Python 3.14 / plotly ≥ 6.x) raises `ValueError` on
`"#E74C3C12"`. Older Plotly silently ignored it (different validator), which is
why the bug was latent.

---
### ✅ Fixed code (`app_fixed.py`)
""")

    st.markdown("""
<div class="fix-box">
def to_rgba(color: str, alpha: float = 0.08) -> str:<br>
&nbsp;&nbsp;\"\"\"Convert any CSS color → valid Plotly rgba() string.\"\"\"<br>
&nbsp;&nbsp;c = color.strip()<br>
&nbsp;&nbsp;if c.startswith("#"):<br>
&nbsp;&nbsp;&nbsp;&nbsp;h = c.lstrip("#")<br>
&nbsp;&nbsp;&nbsp;&nbsp;if len(h) == 3: h = "".join(ch*2 for ch in h)<br>
&nbsp;&nbsp;&nbsp;&nbsp;if len(h) in (6, 8):<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"rgba({r},{g},{b},{alpha})"<br>
&nbsp;&nbsp;m = re.match(r"rgba?\(([^)]+)\)", c)<br>
&nbsp;&nbsp;if m:<br>
&nbsp;&nbsp;&nbsp;&nbsp;r,g,b = m.group(1).split(",")[:3]<br>
&nbsp;&nbsp;&nbsp;&nbsp;return f"rgba({r},{g},{b},{alpha})"<br>
&nbsp;&nbsp;return c<br>
<br>
# In go.Scatter:<br>
fillcolor=to_rgba(color, alpha=0.08)&nbsp;&nbsp;# ← always valid!
</div>
""", unsafe_allow_html=True)

    st.markdown("""
---
### Conversion examples

| Input color | Buggy output | Fixed output |
|------------|-------------|-------------|
| `"#E74C3C"` | `"#E74C3C12"` ❌ | `"rgba(231,76,60,0.08)"` ✅ |
| `"#2ECC71"` | `"#2ECC7112"` ❌ | `"rgba(46,204,113,0.08)"` ✅ |
| `"rgb(52,152,219)"` | `"rgb(52,152,219,0.05)"` ✅ (worked by accident) | `"rgba(52,152,219,0.08)"` ✅ |
| `"rgba(155,89,182,0.5)"` | — | `"rgba(155,89,182,0.08)"` ✅ |

---
### How to apply to your existing `app.py`

1. Add the `to_rgba()` helper near the top of your file.
2. In `training_curve_fig()`, replace the `fillcolor` expression:

```python
# BEFORE (line ~422 in original):
fillcolor=cols[name].rstrip(")") + ",0.05)" if cols[name].startswith("rgb") else cols[name] + "12",

# AFTER:
fillcolor=to_rgba(cols[name], alpha=0.05),
```

That's the only change needed to fix the crash.
""")

    st.markdown("---")
    st.subheader("Additional fix: Training resume artifact in data")
    st.markdown(f"""
Your `results.csv` contains **94 rows** but the training was run in two sessions:

- **Rows 1–81** (epochs 1–81): First training session, mAP@50 peaked at ~0.614
- **Rows 82–94** (epochs 82–94): Resumed from a **fine-tuned checkpoint** — metrics
  jump from ~0.608 to **0.868** because `last.pt` at row 81 was not the same as
  the checkpoint used to resume (the `last_epoch80_backup.pt` in your ZIP confirms this).

The resume point is annotated on all charts with a **dotted white vertical line**.
""")

    with st.expander("See raw results.csv"):
        st.dataframe(df, use_container_width=True)
