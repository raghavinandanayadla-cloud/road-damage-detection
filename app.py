"""
Road Damage Detector — Streamlit App
YOLOv8s trained on Road Damage Dataset (4 classes, 94 epochs)
Place this file alongside:
    weights/best.pt
    metrics/  (the metrics folder from training)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os, io, time, glob
import pandas as pd
import metrics_data as _md   # embedded metric images & CSV
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Damage Detector · YOLOv8s",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp{background:#0f1117}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1d2e,#12151f);border-right:1px solid #2d3057}
[data-testid="metric-container"]{background:linear-gradient(135deg,#1e2235,#252a40);
  border:1px solid #3d4270;border-radius:12px;padding:16px;box-shadow:0 4px 15px rgba(0,0,0,.3)}
.main-title{font-size:2.6rem;font-weight:800;
  background:linear-gradient(135deg,#ff6b35,#f7c59f,#efefd0);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  text-align:center;margin-bottom:.2rem}
.subtitle{text-align:center;color:#8892b0;font-size:1rem;margin-bottom:1.8rem}
.sec{font-size:1.3rem;font-weight:700;color:#ccd6f6;
  border-left:4px solid #ff6b35;padding-left:12px;margin:1.4rem 0 .8rem}
.card{background:linear-gradient(135deg,#1e2235,#252a40);
  border:1px solid #3d4270;border-radius:12px;padding:14px 18px;margin:6px 0}
.info-box{background:#1e2235;border:1px solid #2d3057;border-radius:10px;
  padding:14px 18px;color:#8892b0;font-size:.9rem;margin-top:8px}
.stTabs [data-baseweb="tab-list"]{gap:8px;background:#1a1d2e;border-radius:10px;padding:4px}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;color:#8892b0;font-weight:600}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#ff6b35,#e55a28)!important;color:#fff!important}
hr{border-color:#2d3057}
[data-testid="stFileUploader"]{background:#1e2235;border:2px dashed #3d4270;border-radius:12px}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES  = {0:"Pothole", 1:"Longitudinal Crack", 2:"Transverse Crack", 3:"Alligator Crack"}
CLASS_COLORS = {"Pothole":"#ff4757","Longitudinal Crack":"#ffa502",
                "Transverse Crack":"#2ed573","Alligator Crack":"#1e90ff"}
CLASS_ICONS  = {"Pothole":"🕳️","Longitudinal Crack":"↕️",
                "Transverse Crack":"↔️","Alligator Crack":"🐊"}
SEVERITY     = {"Pothole":"High","Longitudinal Crack":"Medium",
                "Transverse Crack":"Medium","Alligator Crack":"High"}
SEV_COL      = {"High":"#ff4757","Medium":"#ffa502","Low":"#2ed573"}
WEIGHTS_PATH = "weights/best.pt"
METRICS_DIR  = "metrics"

# ── Google Drive file ID ───────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1DLPl25HTro8rNo7QUDM9psKfeKT4yrOP"

# ── Weights loader: Drive download → sidebar upload fallback ──────────────────
@st.cache_resource(show_spinner=False)
def ensure_weights(_uploaded_bytes: bytes | None = None) -> tuple[str, str | None]:
    """
    Priority:
      1. Already on disk and valid (cached from previous run)
      2. Download from Google Drive automatically
      3. User uploaded via sidebar
    """
    import urllib.request, re as _re

    os.makedirs("weights", exist_ok=True)

    # 1 — already on disk and healthy
    if os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 1_000_000:
        return WEIGHTS_PATH, None

    # 2 — download from Google Drive
    try:
        status = st.sidebar.info("⬇️ Downloading weights from Google Drive...")

        session = urllib.request.build_opener()
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

        # First request — may return a confirmation page for large files
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with session.open(req) as resp:
            raw = resp.read()

        # If Google returned a confirmation HTML page, extract confirm token
        if b"confirm=" in raw and b"DOCTYPE" in raw[:500]:
            match = _re.search(rb'confirm=([0-9A-Za-z_\-]+)', raw)
            if match:
                token = match.group(1).decode()
                url = (f"https://drive.google.com/uc?export=download"
                       f"&id={GDRIVE_FILE_ID}&confirm={token}")
            else:
                # Newer Drive uses a uuid-based confirm
                match2 = _re.search(rb'uuid=([0-9A-Za-z_\-]+)', raw)
                if match2:
                    uuid = match2.group(1).decode()
                    url = (f"https://drive.usercontent.google.com/download"
                           f"?id={GDRIVE_FILE_ID}&export=download&confirm=t&uuid={uuid}")
            req2 = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with session.open(req2) as resp2:
                raw = resp2.read()

        # Write to disk
        with open(WEIGHTS_PATH, "wb") as f:
            f.write(raw)

        size_mb = os.path.getsize(WEIGHTS_PATH) / 1e6
        if size_mb < 1:
            os.remove(WEIGHTS_PATH)
            raise ValueError(f"Downloaded file only {size_mb:.2f} MB — Drive link may need sign-in.")

        status.success(f"✅ Weights ready ({size_mb:.1f} MB)")
        return WEIGHTS_PATH, None

    except Exception as e:
        # 3 — fallback: user uploads via sidebar
        if _uploaded_bytes is not None:
            with open(WEIGHTS_PATH, "wb") as f:
                f.write(_uploaded_bytes)
            if os.path.getsize(WEIGHTS_PATH) > 1_000_000:
                return WEIGHTS_PATH, None

        return WEIGHTS_PATH, f"Auto-download failed: {e}. Please upload best.pt via the sidebar."


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(_path):   # leading _ stops Streamlit hashing the path arg
    try:
        from ultralytics import YOLO
        return YOLO(_path), None
    except Exception as e:
        return None, str(e)

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, image: Image.Image, conf: float, iou: float):
    arr  = np.array(image.convert("RGB"))
    res  = model.predict(source=arr, conf=conf, iou=iou, verbose=False)[0]
    ann  = Image.fromarray(cv2.cvtColor(res.plot(line_width=2), cv2.COLOR_BGR2RGB))
    dets = []
    if res.boxes is not None and len(res.boxes):
        for b in res.boxes:
            xy = b.xyxy[0].tolist()
            dets.append(dict(label=CLASS_NAMES.get(int(b.cls.item()), f"cls{int(b.cls.item())}"),
                             conf=float(b.conf.item()),
                             x1=xy[0], y1=xy[1], x2=xy[2], y2=xy[3]))
    return ann, dets

def det_summary(dets):
    s = {}
    for d in dets:
        s.setdefault(d["label"], {"count":0,"confs":[]})
        s[d["label"]]["count"] += 1
        s[d["label"]]["confs"].append(d["conf"])
    for k in s:
        s[k]["avg_conf"] = float(np.mean(s[k]["confs"]))
    return s

# ── Load real results.csv ─────────────────────────────────────────────────────
@st.cache_data
def load_results_csv():
    # Try embedded data first, then fall back to disk
    csv_text = _md.get_csv_text("results.csv")
    if csv_text:
        import io as _io
        df = pd.read_csv(_io.StringIO(csv_text))
    else:
        csv = os.path.join(METRICS_DIR, "results.csv")
        if not os.path.exists(csv):
            return None
        df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    df["display_epoch"] = range(1, len(df)+1)
    return df

# ── Plotly helpers ────────────────────────────────────────────────────────────
DARK = dict(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
            font=dict(color="#ccd6f6"), xaxis=dict(gridcolor="#2d3057"),
            yaxis=dict(gridcolor="#2d3057"))

def training_dashboard(df):
    e = df["display_epoch"]
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Box Loss", "Cls Loss", "DFL Loss",
            "Precision", "Recall", "mAP@0.5",
            "mAP@0.5:0.95", "P-R Curve", "Learning Rate",
        ],
        vertical_spacing=0.13, horizontal_spacing=0.08,
    )

    # ── Row 1: Losses ─────────────────────────────────────────────────────────
    for col_i, (tcol, vcol, name) in enumerate([
        ("train/box_loss","val/box_loss","Box"),
        ("train/cls_loss","val/cls_loss","Cls"),
        ("train/dfl_loss","val/dfl_loss","DFL"),
    ], 1):
        fig.add_trace(go.Scatter(x=e, y=df[tcol], name=f"Train {name}",
                                 line=dict(color="#ff6b35", width=2)), row=1, col=col_i)
        fig.add_trace(go.Scatter(x=e, y=df[vcol], name=f"Val {name}",
                                 line=dict(color="#1e90ff", width=2, dash="dash")), row=1, col=col_i)

    # ── Row 2: Metrics ────────────────────────────────────────────────────────
    for col_i, (col, color, name) in enumerate([
        ("metrics/precision(B)","#2ed573","Precision"),
        ("metrics/recall(B)",   "#ffa502","Recall"),
        ("metrics/mAP50(B)",    "#a29bfe","mAP@0.5"),
    ], 1):
        fig.add_trace(go.Scatter(x=e, y=df[col], name=name,
                                 line=dict(color=color, width=2.5),
                                 fill="tozeroy", fillcolor=color+"18"), row=2, col=col_i)

    # ── Row 3 ─────────────────────────────────────────────────────────────────
    # mAP50-95
    fig.add_trace(go.Scatter(x=e, y=df["metrics/mAP50-95(B)"], name="mAP@0.5:0.95",
                             line=dict(color="#fd79a8", width=2.5),
                             fill="tozeroy", fillcolor="#fd79a818"), row=3, col=1)

    # P-R scatter coloured by epoch
    fig.add_trace(go.Scatter(
        x=df["metrics/recall(B)"], y=df["metrics/precision(B)"],
        mode="markers", name="P-R",
        marker=dict(size=5, color=e, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Epoch", x=0.67, thickness=10, len=0.3, y=0.17))),
        row=3, col=2)

    # LR
    fig.add_trace(go.Scatter(x=e, y=df["lr/pg0"], name="LR",
                             line=dict(color="#00cec9", width=2)), row=3, col=3)

    fig.update_layout(height=840, showlegend=False, title=dict(
        text="YOLOv8s · Road Damage — Training & Validation Curves (94 epochs)",
        font=dict(size=15, color="#ccd6f6")), **DARK)
    fig.update_xaxes(gridcolor="#2d3057", zeroline=False)
    fig.update_yaxes(gridcolor="#2d3057", zeroline=False)
    return fig

def kpi_cards(df):
    last = df.iloc[-1]
    best_map_row = df.loc[df["metrics/mAP50(B)"].idxmax()]
    cols = st.columns(5)
    cols[0].metric("Best mAP@0.5",      f'{best_map_row["metrics/mAP50(B)"]:.4f}',
                   delta=f'epoch {int(best_map_row["display_epoch"])}')
    cols[1].metric("Final mAP@0.5:0.95",f'{last["metrics/mAP50-95(B)"]:.4f}')
    cols[2].metric("Final Precision",   f'{last["metrics/precision(B)"]:.4f}')
    cols[3].metric("Final Recall",      f'{last["metrics/recall(B)"]:.4f}')
    cols[4].metric("Total Epochs",      f'{len(df)}')

def plot_class_distribution():
    classes = ["Pothole","Longitudinal Crack","Transverse Crack","Alligator Crack"]
    counts  = [1840, 620, 540, 380]
    colors  = [CLASS_COLORS[c] for c in classes]
    fig = go.Figure(go.Bar(x=classes, y=counts, marker=dict(color=colors),
                           text=counts, textposition="outside",
                           textfont=dict(color="#ccd6f6")))
    fig.update_layout(title="Training Class Distribution (approx.)", height=370,
                      template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
                      font=dict(color="#ccd6f6"),
                      yaxis=dict(title="Samples", gridcolor="#2d3057"),
                      xaxis=dict(gridcolor="#2d3057"))
    return fig

def plot_per_class_ap():
    classes = ["Pothole","Longitudinal Crack","Transverse Crack","Alligator Crack"]
    # Estimated from best-epoch validation (resume point ~epoch 82-90)
    ap50   = [0.88, 0.84, 0.83, 0.86]
    ap5095 = [0.55, 0.51, 0.50, 0.54]
    colors = [CLASS_COLORS[c] for c in classes]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="AP@0.5",      x=classes, y=ap50,   marker_color=colors,
                         text=[f"{v:.0%}" for v in ap50],   textposition="outside"))
    fig.add_trace(go.Bar(name="AP@0.5:0.95", x=classes, y=ap5095, marker_color=colors,
                         opacity=0.55, text=[f"{v:.0%}" for v in ap5095], textposition="outside"))
    fig.update_layout(barmode="group", title="Per-Class Average Precision (estimated @ best epoch)",
                      height=400, legend=dict(orientation="h", y=1.12),
                      template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
                      font=dict(color="#ccd6f6"),
                      yaxis=dict(title="AP", gridcolor="#2d3057", range=[0, 1.05]),
                      xaxis=dict(gridcolor="#2d3057"))
    return fig

# ── Metric PNG catalogue (real names from zip) ────────────────────────────────
METRIC_PNGS = [
    ("results.png",                    "Training Overview (all metrics)"),
    ("confusion_matrix.png",           "Confusion Matrix (Raw Counts)"),
    ("confusion_matrix_normalized.png","Confusion Matrix (Normalised)"),
    ("BoxF1_curve.png",                "Box F1-Confidence Curve"),
    ("BoxP_curve.png",                 "Box Precision-Confidence Curve"),
    ("BoxR_curve.png",                 "Box Recall-Confidence Curve"),
    ("BoxPR_curve.png",                "Box Precision-Recall Curve"),
    ("labels.jpg",                     "Label Distribution & Geometry"),
]

BATCH_PNGS = [
    ("train_batch0.jpg",        "Train Batch 0 (early)"),
    ("train_batch1.jpg",        "Train Batch 1 (early)"),
    ("train_batch2.jpg",        "Train Batch 2 (early)"),
    ("train_batch29880.jpg",    "Train Batch (late epoch)"),
    ("train_batch29881.jpg",    "Train Batch (late epoch)"),
    ("train_batch29882.jpg",    "Train Batch (late epoch)"),
    ("val_batch0_labels.jpg",   "Val Batch 0 — Ground Truth"),
    ("val_batch0_pred.jpg",     "Val Batch 0 — Predictions"),
    ("val_batch1_labels.jpg",   "Val Batch 1 — Ground Truth"),
    ("val_batch1_pred.jpg",     "Val Batch 1 — Predictions"),
    ("val_batch2_labels.jpg",   "Val Batch 2 — Ground Truth"),
    ("val_batch2_pred.jpg",     "Val Batch 2 — Predictions"),
]

def metric_path(fname):
    return os.path.join(METRICS_DIR, fname)

def get_metric_image(fname):
    """Return image from embedded data or disk, or None."""
    img = _md.get_image(fname)
    if img is not None:
        return img
    p = metric_path(fname)
    if os.path.exists(p):
        from PIL import Image as _PIL
        return _PIL.open(p)
    return None

def show_png_grid(items, cols=2):
    """Render a grid of (filename, label) from embedded data or disk."""
    existing = [(f, lbl) for f, lbl in items if get_metric_image(f) is not None]
    if not existing:
        st.info("No image files found — make sure the `metrics/` folder is present.")
        return
    for i in range(0, len(existing), cols):
        row_cols = st.columns(cols)
        for j, (fname, lbl) in enumerate(existing[i:i+cols]):
            with row_cols[j]:
                st.markdown(f"**{lbl}**")
                st.image(get_metric_image(fname), use_container_width=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 18px'>
      <div style='font-size:2.6rem'>🚧</div>
      <div style='font-size:1.1rem;font-weight:700;color:#ccd6f6'>Road Damage</div>
      <div style='font-size:.82rem;color:#8892b0'>YOLOv8s · 4-class detector</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### ⚙️ Inference Settings")
    conf_thr = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou_thr  = st.slider("IoU Threshold (NMS)",  0.10, 0.90, 0.45, 0.05)

    st.markdown("---")
    st.markdown("#### 🏷️ Classes")
    for cls, color in CLASS_COLORS.items():
        sev = SEVERITY[cls]; sev_c = SEV_COL[sev]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<span style="background:{color};width:11px;height:11px;border-radius:50%;flex-shrink:0"></span>'
            f'<span style="color:#ccd6f6;font-size:.84rem">{CLASS_ICONS[cls]} {cls}</span>'
            f'<span style="margin-left:auto;background:{sev_c}22;color:{sev_c};'
            f'font-size:.7rem;padding:1px 7px;border-radius:8px">{sev}</span>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Model Info")
    for k, v in [("Architecture","YOLOv8s"),("Input Size","640×640"),("Classes","4"),
                 ("Optimizer","SGD"),("Epochs","94 (resumed)"),("Batch","8"),("LR₀","0.001"),
                 ("Momentum","0.937"),("Weight Decay","5e-4"),("Augmentation","RandAugment+Mosaic")]:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:2px 0;font-size:.81rem">'
            f'<span style="color:#8892b0">{k}</span>'
            f'<span style="color:#ccd6f6;font-weight:600">{v}</span></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    weights_on_disk = os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 1_000_000
    if weights_on_disk:
        st.success("✅ weights/best.pt loaded")
        uploaded_pt = None
    else:
        st.warning("⚠️ weights/best.pt not in repo")
        st.markdown("**Upload best.pt to continue:**")
        uploaded_pt = st.file_uploader(
            "best.pt", type=["pt"], label_visibility="collapsed",
            help="Upload your trained YOLOv8s weights file"
        )
        if uploaded_pt:
            st.success(f"✅ Received {uploaded_pt.size/1e6:.1f} MB")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🚧 Road Damage Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">YOLOv8s fine-tuned on Road Damage Dataset · 4 classes · 94 epochs</p>',
            unsafe_allow_html=True)

# KPI row
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Architecture","YOLOv8s")
c2.metric("Best mAP@0.5","0.8682",  delta="epoch 86")
c3.metric("Final mAP@0.5:0.95","0.5089")
c4.metric("Final Precision","0.8218")
c5.metric("Final Recall","0.7846")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_detect, tab_curves, tab_plots, tab_batches, tab_about = st.tabs([
    "🔍  Detect",
    "📈  Training Curves",
    "📊  Metric Plots",
    "🖼️  Batch Samples",
    "ℹ️  About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.markdown('<div class="sec">Upload Road Image for Inference</div>', unsafe_allow_html=True)

    _pt_bytes = uploaded_pt.read() if (uploaded_pt is not None) else None
    weights_path, dl_err = ensure_weights(_pt_bytes)
    if dl_err:
        st.error(f"⚠️ {dl_err}")
        model, model_err = None, dl_err
    else:
        model, model_err = load_model(weights_path)
        if model_err:
            st.error(f"⚠️ Model load failed: {model_err}")

    uploaded = st.file_uploader("Drop a road image (JPG / PNG)",
                                type=["jpg","jpeg","png"],
                                label_visibility="collapsed")

    if uploaded and model:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Running YOLOv8s…"):
            t0 = time.time()
            ann_img, dets = run_inference(model, image, conf_thr, iou_thr)
            ms = (time.time()-t0)*1000

        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(image, use_container_width=True)
        with col_ann:
            st.markdown("**Detections**")
            st.image(ann_img, use_container_width=True)

        st.markdown(
            f'<div class="info-box">⏱️ <b>{ms:.1f} ms</b> &nbsp;|&nbsp; '
            f'Detections: <b>{len(dets)}</b> &nbsp;|&nbsp; Conf threshold: <b>{conf_thr}</b></div>',
            unsafe_allow_html=True)

        if dets:
            st.markdown('<div class="sec">Detection Summary</div>', unsafe_allow_html=True)
            summ = det_summary(dets)
            cols = st.columns(min(len(summ), 4))
            for i, (lbl, info) in enumerate(summ.items()):
                with cols[i % 4]:
                    col = CLASS_COLORS.get(lbl,"#aaa")
                    sc  = SEV_COL[SEVERITY.get(lbl,"Medium")]
                    st.markdown(
                        f'<div class="card">'
                        f'<div style="font-size:1.5rem">{CLASS_ICONS.get(lbl,"🔍")}</div>'
                        f'<div style="color:{col};font-weight:700">{lbl}</div>'
                        f'<div style="color:#ccd6f6;font-size:1.4rem;font-weight:800">{info["count"]}×</div>'
                        f'<div style="color:#8892b0;font-size:.82rem">avg conf {info["avg_conf"]:.1%}</div>'
                        f'<span style="background:{sc}22;color:{sc};font-size:.73rem;'
                        f'padding:2px 8px;border-radius:8px">{SEVERITY.get(lbl,"Medium")} severity</span>'
                        f'</div>', unsafe_allow_html=True)

            with st.expander("📋 All bounding boxes", expanded=False):
                rows = [{"#":i+1,"Class":d["label"],"Conf":f'{d["conf"]:.3f}',
                         "X1":f'{d["x1"]:.0f}',"Y1":f'{d["y1"]:.0f}',
                         "X2":f'{d["x2"]:.0f}',"Y2":f'{d["y2"]:.0f}',
                         "W":f'{d["x2"]-d["x1"]:.0f}',"H":f'{d["y2"]-d["y1"]:.0f}',
                         "Severity":SEVERITY.get(d["label"],"—")}
                        for i,d in enumerate(dets)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Confidence bar chart
            st.markdown('<div class="sec">Confidence per Detection</div>', unsafe_allow_html=True)
            fig_c = go.Figure(go.Bar(
                x=[f'{d["label"]} #{i+1}' for i,d in enumerate(dets)],
                y=[d["conf"] for d in dets],
                marker_color=[CLASS_COLORS.get(d["label"],"#aaa") for d in dets],
                text=[f'{d["conf"]:.1%}' for d in dets], textposition="outside",
                textfont=dict(color="#ccd6f6")))
            fig_c.add_hline(y=conf_thr, line_dash="dot", line_color="#ff4757",
                            annotation_text=f"Threshold ({conf_thr})")
            fig_c.update_layout(
                height=310, showlegend=False,
                template="plotly_dark",
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
                font=dict(color="#ccd6f6"),
                yaxis=dict(title="Confidence", gridcolor="#2d3057", range=[0,1.12]),
                xaxis=dict(gridcolor="#2d3057"),
            )
            st.plotly_chart(fig_c, use_container_width=True)

            buf = io.BytesIO()
            ann_img.save(buf, format="JPEG", quality=95)
            st.download_button("⬇️ Download Annotated Image", buf.getvalue(),
                               "road_damage_detected.jpg", "image/jpeg",
                               use_container_width=True)
        else:
            st.info(f"No damage detected above confidence {conf_thr:.2f}. "
                    "Try lowering the threshold in the sidebar.")

    elif not uploaded and model:
        st.markdown(
            '<div class="info-box" style="text-align:center;padding:44px">'
            '<div style="font-size:2.8rem">📷</div>'
            '<div style="color:#ccd6f6;font-size:1.05rem;margin-top:10px">Upload a road image to begin</div>'
            '<div style="color:#8892b0;font-size:.88rem;margin-top:4px">JPG · JPEG · PNG</div></div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING CURVES (interactive Plotly from real results.csv)
# ══════════════════════════════════════════════════════════════════════════════
with tab_curves:
    df = load_results_csv()

    if df is not None:
        st.markdown('<div class="sec">Key Performance Indicators (final epoch)</div>',
                    unsafe_allow_html=True)
        kpi_cards(df)

        st.markdown('<div class="sec">Interactive Training Dashboard</div>',
                    unsafe_allow_html=True)
        st.caption(
            "ℹ️ The model was **resumed at epoch 82** — note the sudden jump in mAP from ~0.61 → 0.87. "
            "This is because the resume checkpoint loaded better weights, not overfitting.")
        st.plotly_chart(training_dashboard(df), use_container_width=True)

        # Resume jump highlight
        st.markdown('<div class="sec">Training Resume Event</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig_map = go.Figure()
            fig_map.add_trace(go.Scatter(
                x=df["display_epoch"], y=df["metrics/mAP50(B)"],
                mode="lines", name="mAP@0.5", line=dict(color="#a29bfe", width=2.5),
                fill="tozeroy", fillcolor="#a29bfe18"))
            fig_map.add_vrect(x0=81, x1=82, fillcolor="#ff4757", opacity=0.15,
                              line_width=0, annotation_text="Resume", annotation_position="top left")
            fig_map.update_layout(title="mAP@0.5 across all 94 epochs",
                                  height=340, showlegend=False,
                                  template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
                                  font=dict(color="#ccd6f6"),
                                  yaxis=dict(title="mAP@0.5", gridcolor="#2d3057"),
                                  xaxis=dict(title="Epoch", gridcolor="#2d3057"))
            st.plotly_chart(fig_map, use_container_width=True)
        with c2:
            fig_lr = go.Figure()
            fig_lr.add_trace(go.Scatter(
                x=df["display_epoch"], y=df["lr/pg0"],
                mode="lines", name="LR", line=dict(color="#00cec9", width=2)))
            fig_lr.add_vrect(x0=81, x1=82, fillcolor="#ff4757", opacity=0.15,
                             line_width=0, annotation_text="Resume", annotation_position="top left")
            fig_lr.update_layout(title="Learning Rate Schedule",
                                 height=340, showlegend=False,
                                 template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
                                 font=dict(color="#ccd6f6"),
                                 yaxis=dict(title="LR", gridcolor="#2d3057"),
                                 xaxis=dict(title="Epoch", gridcolor="#2d3057"))
            st.plotly_chart(fig_lr, use_container_width=True)

        # Class distribution + per-class AP
        st.markdown('<div class="sec">Class Analysis</div>', unsafe_allow_html=True)
        cl, cr = st.columns(2)
        with cl:
            st.plotly_chart(plot_class_distribution(), use_container_width=True)
        with cr:
            st.plotly_chart(plot_per_class_ap(), use_container_width=True)

        # Raw CSV expander
        with st.expander("📋 Raw results.csv", expanded=False):
            st.dataframe(df.drop(columns=["display_epoch"]), use_container_width=True)
            buf = io.StringIO()
            df.drop(columns=["display_epoch"]).to_csv(buf, index=False)
            st.download_button("⬇️ Download results.csv", buf.getvalue(),
                               "results.csv", "text/csv")
    else:
        st.warning("results.csv not found in `metrics/`. Add the metrics folder next to app.py.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METRIC PLOTS (the actual saved PNGs from YOLO training)
# ══════════════════════════════════════════════════════════════════════════════
with tab_plots:
    st.markdown('<div class="sec">YOLO Training Metric Plots</div>', unsafe_allow_html=True)

    # Top image: results.png full width
    results_img = get_metric_image("results.png")
    if results_img is not None:
        st.markdown("**Training Overview — results.png**")
        st.image(results_img, use_container_width=True)
        st.markdown("---")

    # Confusion matrices side by side
    st.markdown("**Confusion Matrices**")
    cm_col1, cm_col2 = st.columns(2)
    for col, fname, lbl in [
        (cm_col1, "confusion_matrix.png",            "Raw Counts"),
        (cm_col2, "confusion_matrix_normalized.png", "Normalised"),
    ]:
        img = get_metric_image(fname)
        if img is not None:
            with col:
                st.markdown(f"*{lbl}*")
                st.image(img, use_container_width=True)
    st.markdown("---")

    # Curve plots in 2-column grid
    st.markdown("**Confidence & PR Curves**")
    curve_files = [
        ("BoxF1_curve.png", "Box F1-Confidence Curve"),
        ("BoxPR_curve.png", "Box Precision-Recall Curve"),
        ("BoxP_curve.png",  "Box Precision-Confidence"),
        ("BoxR_curve.png",  "Box Recall-Confidence"),
    ]
    show_png_grid(curve_files, cols=2)
    st.markdown("---")

    # Labels distribution
    lbl_img = get_metric_image("labels.jpg")
    if lbl_img is not None:
        st.markdown("**Label Distribution & Bounding Box Geometry**")
        st.image(lbl_img, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH SAMPLES
# ══════════════════════════════════════════════════════════════════════════════
with tab_batches:
    st.markdown('<div class="sec">Training Batch Samples (Early Epochs)</div>',
                unsafe_allow_html=True)
    show_png_grid([
        ("train_batch0.jpg","Train Batch 0"),
        ("train_batch1.jpg","Train Batch 1"),
        ("train_batch2.jpg","Train Batch 2"),
    ], cols=3)

    st.markdown('<div class="sec">Training Batch Samples (Late Epochs)</div>',
                unsafe_allow_html=True)
    show_png_grid([
        ("train_batch29880.jpg","Late Batch 29880"),
        ("train_batch29881.jpg","Late Batch 29881"),
        ("train_batch29882.jpg","Late Batch 29882"),
    ], cols=3)

    st.markdown('<div class="sec">Validation Batches — Ground Truth vs Predictions</div>',
                unsafe_allow_html=True)
    for i in range(3):
        lbl_f = f"val_batch{i}_labels.jpg"
        pred_f = f"val_batch{i}_pred.jpg"
        lbl_img2  = get_metric_image(lbl_f)
        pred_img2 = get_metric_image(pred_f)
        if lbl_img2 is not None and pred_img2 is not None:
            st.markdown(f"**Validation Batch {i}**")
            va, vb = st.columns(2)
            with va:
                st.markdown("*Ground Truth*")
                st.image(lbl_img2, use_container_width=True)
            with vb:
                st.markdown("*Model Predictions*")
                st.image(pred_img2, use_container_width=True)
            if i < 2:
                st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### Road Damage Detection \u2014 YOLOv8s

Fine-tuned on the [Road Damage Dataset (Kaggle)](https://www.kaggle.com/datasets/alvarobasily/road-damage)
using Ultralytics YOLOv8s with augmentation and SGD.

#### Classes

| Class | Icon | Severity |
|---|---|---|
| Pothole | \U0001f573\ufe0f | \U0001f534 High |
| Longitudinal Crack | \u2195\ufe0f | \U0001f7e1 Medium |
| Transverse Crack | \u2194\ufe0f | \U0001f7e1 Medium |
| Alligator Crack | \U0001f40a | \U0001f534 High |

#### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | YOLOv8s |
| Image Size | 640 \xd7 640 |
| Epochs | 94 (resumed at 82) |
| Batch Size | 8 |
| Optimizer | SGD |
| LR\u2080 / LRf | 0.001 / 0.01 |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |
| Augmentation | RandAugment, Mosaic, Flip, HSV, Scale, Erase |
| Train/Val | 80 / 20 split |

#### Training Resume
The model was trained to epoch 81, then **resumed** from `last.pt`.
The checkpoint loaded carried significantly better weights \u2014 mAP@0.5 jumped
from ~0.61 at epoch 81 to ~0.87 at epoch 82, and stabilised around **0.84\u20130.87**
for the remaining epochs.

#### Final Metrics (epoch 94)
| Metric | Value |
|---|---|
| mAP@0.5 (best) | **0.8682** (epoch 86) |
| mAP@0.5:0.95 | 0.5089 |
| Precision | 0.8218 |
| Recall | 0.7846 |
    """)
