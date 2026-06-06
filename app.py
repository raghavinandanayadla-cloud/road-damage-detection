"""
🛣️ Road Damage Detection — YOLOv8 Dashboard
============================================
Author: Raghavi Nandana Yadla
Model: YOLOv8 trained on road damage dataset
Repo: raghavinandanayadla-cloud/road-damage-detection
"""

import re
import io
import os
import sys
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_RAW   = "https://raw.githubusercontent.com/raghavinandanayadla-cloud/road-damage-detection/main"
WEIGHTS_URL  = f"{GITHUB_RAW}/weights/best.pt"
RESULTS_URL  = f"{GITHUB_RAW}/results.csv"
MODEL_LOCAL  = "best.pt"

DAMAGE_CLASSES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}

CLASS_COLORS = {
    "Longitudinal Crack": "#E74C3C",
    "Transverse Crack":   "#E67E22",
    "Alligator Crack":    "#9B59B6",
    "Pothole":            "#3498DB",
    "Road Damage":        "#E74C3C",   # fallback
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def to_rgba(color: str, alpha: float = 0.08) -> str:
    c = color.strip()
    if c.startswith("#"):
        h = c.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) in (6, 8):
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
    m = re.match(r"rgba?\(([^)]+)\)", c, re.I)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        r, g, b = parts[0], parts[1], parts[2]
        return f"rgba({r},{g},{b},{alpha})"
    return c


@st.cache_resource(show_spinner=False)
def load_model():
    """Download best.pt from GitHub and load via ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "ultralytics not installed — run: pip install ultralytics"

    if not os.path.exists(MODEL_LOCAL):
        with st.spinner("⬇️ Downloading model weights from GitHub…"):
            try:
                r = requests.get(WEIGHTS_URL, timeout=120)
                if r.status_code != 200:
                    return None, f"Could not download model (HTTP {r.status_code}). Check that `weights/best.pt` exists in your GitHub repo and the repo is public."
                with open(MODEL_LOCAL, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                return None, f"Download error: {e}"

    try:
        model = YOLO(MODEL_LOCAL)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


@st.cache_data(show_spinner=False)
def load_results_csv_from_github() -> pd.DataFrame | None:
    try:
        r = requests.get(RESULTS_URL, timeout=30)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            return df
    except Exception:
        pass
    return None


@st.cache_data
def load_results_csv(source) -> pd.DataFrame:
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
    if "time" not in df.columns:
        return []
    diffs = df["time"].diff()
    return df[diffs < 0].index.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# PALETTES
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "train/box_loss":       "#E74C3C",
    "train/cls_loss":       "#E67E22",
    "train/dfl_loss":       "#9B59B6",
    "val/box_loss":         "#C0392B",
    "val/cls_loss":         "#D35400",
    "val/dfl_loss":         "#8E44AD",
    "metrics/mAP50(B)":     "#2ECC71",
    "metrics/mAP50-95(B)":  "#27AE60",
    "metrics/precision(B)": "#3498DB",
    "metrics/recall(B)":    "#2980B9",
    "lr/pg0":               "#F1C40F",
    "lr/pg1":               "#F39C12",
    "lr/pg2":               "#D4AC0D",
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

# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def training_curve_fig(df: pd.DataFrame, resume_rows: list[int]) -> go.Figure:
    metric_groups = {
        "Losses — Train":     ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
        "Losses — Val":       ["val/box_loss",   "val/cls_loss",   "val/dfl_loss"],
        "Detection Metrics":  ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                               "metrics/precision(B)", "metrics/recall(B)"],
        "Learning Rate":      ["lr/pg0"],
    }
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(metric_groups.keys()),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for (row, col), (group_name, cols) in zip(positions, metric_groups.items()):
        for col_name in cols:
            if col_name not in df.columns:
                continue
            vals   = df[col_name].values
            epochs = df["epoch"].values
            color  = PALETTE.get(col_name, "#888888")
            label  = DISPLAY_NAMES.get(col_name, col_name)
            fig.add_trace(go.Scatter(
                x=epochs, y=vals, mode="lines", name=label,
                line=dict(color=color, width=2),
                legendgroup=group_name,
                hovertemplate=f"<b>{label}</b><br>Epoch: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=epochs, y=vals, mode="none",
                fill="tozeroy", fillcolor=to_rgba(color, 0.08),
                showlegend=False, hoverinfo="skip", legendgroup=group_name,
            ), row=row, col=col)
        for resume_idx in resume_rows:
            resume_epoch = df.loc[resume_idx, "epoch"]
            fig.add_vline(
                x=resume_epoch,
                line=dict(color="rgba(255,255,255,0.5)", width=1.5, dash="dot"),
                row=row, col=col,
                annotation_text="resume", annotation_font_size=9,
                annotation_font_color="rgba(255,255,255,0.6)",
            )
    fig.update_layout(
        height=700, template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1E2E",
        font=dict(color="#E0E0E0", size=12),
        legend=dict(bgcolor="rgba(30,34,50,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(l=50, r=20, t=60, b=40),
        title=dict(text="YOLOv8 Training Curves", font=dict(size=18), x=0.5, xanchor="center"),
    )
    fig.update_xaxes(title_text="Epoch", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def loss_comparison_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    pairs = [
        ("train/box_loss","val/box_loss","Box Loss","#E74C3C","#C0392B"),
        ("train/cls_loss","val/cls_loss","Cls Loss","#E67E22","#D35400"),
        ("train/dfl_loss","val/dfl_loss","DFL Loss","#9B59B6","#8E44AD"),
    ]
    epochs = df["epoch"].values
    for tr_col, vl_col, label, tr_color, vl_color in pairs:
        if tr_col in df.columns:
            fig.add_trace(go.Scatter(x=epochs, y=df[tr_col].values, name=f"Train {label}",
                line=dict(color=tr_color, width=2),
                hovertemplate=f"<b>Train {label}</b><br>Epoch: %{{x}}<br>%{{y:.4f}}<extra></extra>"))
        if vl_col in df.columns:
            fig.add_trace(go.Scatter(x=epochs, y=df[vl_col].values, name=f"Val {label}",
                line=dict(color=vl_color, width=2, dash="dash"),
                hovertemplate=f"<b>Val {label}</b><br>Epoch: %{{x}}<br>%{{y:.4f}}<extra></extra>"))
    fig.update_layout(
        title="Train vs Validation Loss", height=400, template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1E2E",
        xaxis_title="Epoch", yaxis_title="Loss",
        legend=dict(bgcolor="rgba(30,34,50,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        font=dict(color="#E0E0E0"), margin=dict(l=50,r=20,t=50,b=40),
    )
    return fig


def metrics_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    metrics = [
        ("metrics/mAP50(B)",    "mAP@50",    "#2ECC71"),
        ("metrics/mAP50-95(B)","mAP@50-95", "#27AE60"),
        ("metrics/precision(B)","Precision", "#3498DB"),
        ("metrics/recall(B)",   "Recall",    "#2980B9"),
    ]
    epochs = df["epoch"].values
    for col, label, color in metrics:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=epochs, y=df[col].values, name=label,
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=to_rgba(color, 0.07),
            hovertemplate=f"<b>{label}</b><br>Epoch: %{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Detection Metrics over Training", height=400, template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1E2E",
        xaxis_title="Epoch", yaxis_title="Score (0–1)",
        yaxis=dict(range=[0,1.05]),
        legend=dict(bgcolor="rgba(30,34,50,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        font=dict(color="#E0E0E0"), margin=dict(l=50,r=20,t=50,b=40),
    )
    return fig


def lr_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    colors  = ["#F1C40F","#F39C12","#D4AC0D"]
    epochs  = df["epoch"].values
    for col, color in zip(lr_cols, colors):
        fig.add_trace(go.Scatter(
            x=epochs, y=df[col].values, name=col,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{col}</b><br>Epoch: %{{x}}<br>LR: %{{y:.6f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Learning Rate Schedule", height=300, template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1E2E",
        xaxis_title="Epoch", yaxis_title="LR",
        font=dict(color="#E0E0E0"), margin=dict(l=50,r=20,t=50,b=40),
    )
    return fig


def epoch_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    key = "metrics/mAP50(B)"
    if key not in df.columns:
        return df.head()
    cols_to_show = [c for c in ["epoch","metrics/mAP50(B)","metrics/mAP50-95(B)",
                                 "metrics/precision(B)","metrics/recall(B)",
                                 "train/box_loss","val/box_loss"] if c in df.columns]
    top = df.nlargest(5, key)[cols_to_show].copy()
    rename = {"epoch":"Epoch","metrics/mAP50(B)":"mAP@50","metrics/mAP50-95(B)":"mAP@50-95",
              "metrics/precision(B)":"Precision","metrics/recall(B)":"Recall",
              "train/box_loss":"Train Box Loss","val/box_loss":"Val Box Loss"}
    top.rename(columns=rename, inplace=True)
    return top.round(4).reset_index(drop=True)


def confidence_histogram(detections: list[dict]) -> go.Figure:
    if not detections:
        return None
    confs  = [d["confidence"] for d in detections]
    labels = [d["label"] for d in detections]
    colors = [CLASS_COLORS.get(l, "#888") for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=confs,
        marker_color=colors,
        text=[f"{c:.2f}" for c in confs],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Detection Confidences", height=300, template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#1A1E2E",
        xaxis_title="Class", yaxis_title="Confidence", yaxis=dict(range=[0,1]),
        font=dict(color="#E0E0E0"), margin=dict(l=50,r=20,t=50,b=60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, pil_image: Image.Image, conf_thresh: float = 0.25) -> tuple:
    """Run YOLOv8 inference. Returns (annotated PIL image, list of detection dicts)."""
    import numpy as np
    img_arr = np.array(pil_image)
    results  = model.predict(img_arr, conf=conf_thresh, verbose=False)
    result   = results[0]

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].tolist()
            label  = DAMAGE_CLASSES.get(cls_id, result.names.get(cls_id, f"Class {cls_id}"))
            detections.append({
                "label":      label,
                "confidence": conf,
                "bbox":       xyxy,
                "class_id":   cls_id,
            })

    # annotated image (PIL)
    annotated = Image.fromarray(result.plot())
    return annotated, detections


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Road Damage Detection — YOLOv8",
    page_icon="🛣️", layout="wide",
)

st.markdown("""
<style>
  .stApp { background-color: #0E1117; color: #E0E0E0; }
  .metric-card {
      background: #1A1E2E; border-radius: 12px; padding: 18px 22px;
      border: 1px solid rgba(255,255,255,0.08); text-align: center;
  }
  .metric-card .value { font-size: 2rem; font-weight: 700; }
  .metric-card .label { font-size: 0.85rem; color: #9BA3B8; margin-top: 4px; }
  .metric-card .delta { font-size: 0.8rem; margin-top: 2px; }
  .detection-badge {
      display: inline-block; padding: 4px 12px; border-radius: 20px;
      font-size: 0.8rem; font-weight: 600; margin: 3px;
  }
  .status-ok   { color: #2ECC71; }
  .status-warn { color: #E67E22; }
  h1, h2, h3 { color: #E0E0E0; }
  .stTabs [data-baseweb="tab"] { color: #9BA3B8; }
  .stTabs [aria-selected="true"] { color: #2ECC71; border-bottom-color: #2ECC71; }
  [data-testid="stSidebar"] { background-color: #13151f; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🛣️ Road Damage Detection — YOLOv8 Dashboard")
st.markdown(
    "**Raghavi Nandana Yadla** · "
    "[GitHub Repo](https://github.com/raghavinandanayadla-cloud/road-damage-detection) · "
    "Model: YOLOv8 · mAP@50: ~0.868"
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("### 📂 Training Data")
    csv_upload = st.file_uploader(
        "Upload results.csv or ZIP (optional)",
        type=["csv", "zip"],
        help="Leave blank to auto-fetch from GitHub",
    )

    st.markdown("---")
    st.markdown("### 🔍 Inference")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
    st.markdown("---")

    st.markdown("### ℹ️ Model Info")
    st.markdown("""
    - **Architecture:** YOLOv8
    - **Dataset:** Road Damage (India/Japan/Czech)
    - **Epochs:** 94 (resumed)
    - **Best mAP@50:** ~0.868
    - **Image size:** 640×640
    """)
    st.markdown(f"[📥 Download weights]({WEIGHTS_URL})")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

df = None
if csv_upload is not None:
    if csv_upload.name.endswith(".zip"):
        df = load_from_zip(csv_upload.read())
    else:
        df = load_results_csv(csv_upload)
else:
    # Try fetching from GitHub
    with st.spinner("Fetching results.csv from GitHub…"):
        df = load_results_csv_from_github()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (background)
# ─────────────────────────────────────────────────────────────────────────────

model, model_err = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_detect, tab_overview, tab_curves, tab_losses, tab_metrics, tab_lr = st.tabs([
    "🔍 Detect Damage",
    "📊 Overview",
    "📈 Training Curves",
    "📉 Loss Detail",
    "🎯 Metrics",
    "🔧 Learning Rate",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — LIVE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.subheader("🔍 Road Damage Detection — Upload an Image")

    if model_err:
        st.error(f"⚠️ Model could not be loaded: {model_err}")
        st.info("Make sure `weights/best.pt` is in your GitHub repo and the repo is **public**.")
    else:
        st.success("✅ YOLOv8 model loaded and ready")

    col_upload, col_settings = st.columns([3, 1])
    with col_upload:
        uploaded_img = st.file_uploader(
            "Upload a road image (JPG/PNG)",
            type=["jpg","jpeg","png"],
            key="inference_upload",
        )

    if uploaded_img and model:
        pil_img = Image.open(uploaded_img).convert("RGB")

        with st.spinner("🔎 Running inference…"):
            annotated_img, detections = run_inference(model, pil_img, conf_thresh)

        col_orig, col_anno = st.columns(2)
        with col_orig:
            st.markdown("**Original Image**")
            st.image(pil_img, use_container_width=True)
        with col_anno:
            st.markdown("**Detected Damage**")
            st.image(annotated_img, use_container_width=True)

        st.markdown("---")

        if detections:
            st.markdown(f"### 🚨 {len(detections)} Damage Detection(s) Found")
            det_cols = st.columns(min(len(detections), 4))
            for i, det in enumerate(detections):
                color = CLASS_COLORS.get(det["label"], "#E74C3C")
                det_cols[i % 4].markdown(f"""
                <div class="metric-card">
                    <div class="value" style="color:{color}">{det['confidence']:.2%}</div>
                    <div class="label">{det['label']}</div>
                    <div class="delta" style="color:{color}">Confidence</div>
                </div>""", unsafe_allow_html=True)

            hist = confidence_histogram(detections)
            if hist:
                st.plotly_chart(hist, use_container_width=True)

            st.markdown("#### 📋 Detection Details")
            det_df = pd.DataFrame([{
                "Class":       d["label"],
                "Confidence":  f"{d['confidence']:.4f}",
                "BBox x1":     f"{d['bbox'][0]:.1f}",
                "BBox y1":     f"{d['bbox'][1]:.1f}",
                "BBox x2":     f"{d['bbox'][2]:.1f}",
                "BBox y2":     f"{d['bbox'][3]:.1f}",
            } for d in detections])
            st.dataframe(det_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No road damage detected above the confidence threshold.")

    elif not uploaded_img:
        st.info("👆 Upload a road image above to run damage detection.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if df is None:
        st.warning("⚠️ No training data available. Upload a `results.csv` in the sidebar or ensure your GitHub repo has one at the root level.")
        st.stop()

    resume_rows  = detect_resume_points(df)
    total_epochs = int(df["epoch"].max()) if "epoch" in df.columns else len(df)

    if "metrics/mAP50(B)" in df.columns:
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best_row = df.loc[best_idx]
    else:
        best_idx = 0
        best_row = df.iloc[0]
    last_row = df.iloc[-1]

    if resume_rows:
        st.warning(
            f"⚠️ Training was **resumed** at epoch {int(df.loc[resume_rows[0],'epoch'])}. "
            "Metrics show a jump because the resumed checkpoint was different. "
            "Resume points are marked with dotted lines on charts."
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Total Epochs",   f"{total_epochs}",                                   "rows in results.csv",                                         "#3498DB"),
        (c2, "Best mAP@50",    f"{best_row.get('metrics/mAP50(B)',0):.4f}",          f"Epoch {int(best_row.get('epoch',0))}",                        "#2ECC71"),
        (c3, "Best mAP@50-95", f"{best_row.get('metrics/mAP50-95(B)',0):.4f}",       f"Epoch {int(best_row.get('epoch',0))}",                        "#27AE60"),
        (c4, "Best Precision", f"{df.get('metrics/precision(B)', pd.Series([0])).max():.4f}" if 'metrics/precision(B)' in df.columns else "N/A",
              "Peak across all epochs", "#3498DB"),
        (c5, "Best Recall",    f"{df.get('metrics/recall(B)', pd.Series([0])).max():.4f}" if 'metrics/recall(B)' in df.columns else "N/A",
              "Peak across all epochs", "#2980B9"),
    ]
    for col, label, val, delta, color in kpis:
        col.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:{color}">{val}</div>
            <div class="label">{label}</div>
            <div class="delta" style="color:#7F8C8D">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📋 Top 5 Epochs by mAP@50")
        st.dataframe(epoch_summary_table(df), use_container_width=True, hide_index=True)
    with col_right:
        st.subheader("📦 Training Configuration")
        config = {
            "Model":              "YOLOv8",
            "Optimizer":          "SGD",
            "Epochs configured":  100,
            "Epochs completed":   total_epochs,
            "Batch size":         8,
            "Image size":         "640×640",
            "LR (initial)":       "0.001",
            "Patience":           10,
            "Resumed":            f"Yes — epoch {int(df.loc[resume_rows[0],'epoch'])}" if resume_rows else "No",
            "AMP":                "Enabled",
        }
        st.dataframe(pd.DataFrame(list(config.items()), columns=["Parameter","Value"]),
                     use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🏁 Final Epoch Metrics")
    f1, f2, f3, f4 = st.columns(4)
    for col, label, key, color in [
        (f1, "mAP@50",    "metrics/mAP50(B)",     "#2ECC71"),
        (f2, "mAP@50-95", "metrics/mAP50-95(B)",  "#27AE60"),
        (f3, "Precision", "metrics/precision(B)",  "#3498DB"),
        (f4, "Recall",    "metrics/recall(B)",     "#2980B9"),
    ]:
        val = last_row.get(key, 0)
        col.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:{color}">{val:.4f}</div>
            <div class="label">{label}</div>
            <div class="delta" style="color:{color}">{int(val*100)}%</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab_curves:
    if df is not None:
        st.subheader("📈 All Training Curves")
        resume_rows = detect_resume_points(df)
        if resume_rows:
            st.caption(f"Dotted vertical lines mark resume points (epoch {', '.join(str(int(df.loc[i,'epoch'])) for i in resume_rows)})")
        st.plotly_chart(training_curve_fig(df, resume_rows), use_container_width=True)
        with st.expander("📄 View raw data"):
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Upload results.csv in the sidebar or ensure GitHub repo has it.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LOSS DETAIL
# ══════════════════════════════════════════════════════════════════════════════
with tab_losses:
    if df is not None:
        st.subheader("📉 Train vs Validation Losses")
        st.plotly_chart(loss_comparison_fig(df), use_container_width=True)
        st.markdown("---")
        col_a, col_b = st.columns(2)
        last_row = df.iloc[-1]
        with col_a:
            st.markdown("**Final Train Losses**")
            for col in ["train/box_loss","train/cls_loss","train/dfl_loss"]:
                if col in df.columns:
                    st.metric(DISPLAY_NAMES.get(col,col), f"{last_row[col]:.4f}")
        with col_b:
            st.markdown("**Final Val Losses**")
            for col in ["val/box_loss","val/cls_loss","val/dfl_loss"]:
                if col in df.columns:
                    delta_val = last_row[col] - df.iloc[0][col]
                    st.metric(DISPLAY_NAMES.get(col,col), f"{last_row[col]:.4f}", delta=f"{delta_val:+.4f}")
    else:
        st.info("No training data loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    if df is not None:
        st.subheader("🎯 Detection Metrics over Training")
        st.plotly_chart(metrics_fig(df), use_container_width=True)
        st.markdown("---")
        st.subheader("Metric Progression (first → best → final)")
        last_row = df.iloc[-1]
        rows = []
        for col in ["metrics/mAP50(B)","metrics/mAP50-95(B)","metrics/precision(B)","metrics/recall(B)"]:
            if col not in df.columns:
                continue
            rows.append({
                "Metric":        DISPLAY_NAMES.get(col, col),
                "Epoch 1":       round(df.iloc[0][col], 4),
                "Best":          round(df[col].max(), 4),
                "Best Epoch":    int(df.loc[df[col].idxmax(),"epoch"]),
                "Final":         round(last_row[col], 4),
                "Δ (E1→Final)":  round(last_row[col] - df.iloc[0][col], 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No training data loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LR
# ══════════════════════════════════════════════════════════════════════════════
with tab_lr:
    if df is not None:
        st.subheader("🔧 Learning Rate Schedule")
        st.plotly_chart(lr_fig(df), use_container_width=True)
        st.caption("LR warms up for the first few epochs, then decays with cosine annealing.")
    else:
        st.info("No training data loaded.")
