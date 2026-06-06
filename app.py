"""
Road Damage Detection — YOLOv8s
Streamlit Web Application
cv2-free: uses Pillow ImageDraw for all annotation (no libGL dependency)
"""

import streamlit as st
import numpy as np
import tempfile
import os
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Damage Detector",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #e74c3c, #f39c12);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #7f8c8d; font-size: 1rem; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px; padding: 1.2rem 1rem; text-align: center;
        border: 1px solid #0f3460;
    }
    .metric-label { color: #a0aec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ffffff; font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Pothole",
    1: "Longitudinal Crack",
    2: "Transverse Crack",
    3: "Alligator Crack",
}

# PIL uses (R,G,B) tuples
CLASS_COLORS_RGB = {
    0: (255, 69,  0),    # red-orange
    1: (0,  200, 100),   # green
    2: (255, 165,  0),   # orange
    3: (148,  0, 211),   # purple
}

CLASS_COLORS_HEX = {
    0: "#FF4500",
    1: "#00C864",
    2: "#FFA500",
    3: "#9400D3",
}

SEVERITY_MAP = {
    "Pothole":           "High",
    "Longitudinal Crack":"Medium",
    "Transverse Crack":  "Medium",
    "Alligator Crack":   "High",
}
SEVERITY_COLOR = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    try:
        from ultralytics import YOLO
        return YOLO(model_path), None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Inference — Pillow-only annotation (no cv2 / libGL)
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, pil_img: Image.Image, conf_threshold: float):
    img_np = np.array(pil_img)
    results = model.predict(source=img_np, conf=conf_threshold, verbose=False)[0]

    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to load a decent font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label  = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            color  = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))

            # Bounding box (3-px outline)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Label background + text
            text = f"{label}: {conf:.2f}"
            bbox_txt = draw.textbbox((x1, y1 - 18), text, font=font)
            draw.rectangle(bbox_txt, fill=color)
            draw.text((x1, y1 - 18), text, fill=(255, 255, 255), font=font)

            detections.append({
                "label":  label,
                "cls_id": cls_id,
                "conf":   conf,
                "bbox":   (x1, y1, x2, y2),
                "area":   (x2 - x1) * (y2 - y1),
            })

    return annotated, detections


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#0f0f1a"
PANEL_BG = "#1a1a2e"
GRID_COL = "#2d2d4e"

def _style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)


def plot_detection_bar(detections):
    counts = Counter(d["label"] for d in detections)
    if not counts:
        return None
    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    colors = [CLASS_COLORS_HEX[k] for k, v in CLASS_NAMES.items() if v in labels]

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=DARK_BG)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
    _style_ax(ax, "Detections per Class")
    ax.set_ylabel("Count", color="#a0aec0")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_class_pie(detections):
    counts = Counter(d["label"] for d in detections)
    if not counts:
        return None
    labels = list(counts.keys())
    sizes  = [counts[l] for l in labels]
    colors = [CLASS_COLORS_HEX[k] for k, v in CLASS_NAMES.items() if v in labels]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    _, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
        textprops=dict(color="white", fontsize=9),
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("Class Distribution", color="white", fontsize=12, pad=10)
    plt.tight_layout()
    return fig


def plot_confidence_dist(detections):
    if not detections:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    for cls_id, cls_name in CLASS_NAMES.items():
        confs = [d["conf"] for d in detections if d["label"] == cls_name]
        if confs:
            ax.scatter([cls_name]*len(confs), confs,
                       color=CLASS_COLORS_HEX[cls_id], s=80, alpha=0.85,
                       edgecolors="white", linewidths=0.5, zorder=3)
    ax.axhline(0.5, color="#f39c12", linestyle="--", linewidth=1, alpha=0.6, label="50% conf")
    _style_ax(ax, "Confidence Score Distribution")
    ax.set_ylabel("Confidence", color="#a0aec0")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white", edgecolor=GRID_COL)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_area_histogram(detections):
    if not detections:
        return None
    areas = defaultdict(list)
    for d in detections:
        areas[d["label"]].append(d["area"])

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    for cls_id, cls_name in CLASS_NAMES.items():
        if cls_name in areas:
            ax.hist(areas[cls_name], bins=8,
                    color=CLASS_COLORS_HEX[cls_id], alpha=0.75,
                    label=cls_name, edgecolor="white", linewidth=0.4)
    _style_ax(ax, "BBox Area Distribution")
    ax.set_xlabel("Area (px²)", color="#a0aec0")
    ax.set_ylabel("Count",     color="#a0aec0")
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white", edgecolor=GRID_COL)
    plt.tight_layout()
    return fig


def plot_confidence_matrix(detections):
    matrix = np.zeros((4, 4))
    for d in detections:
        matrix[d["cls_id"]][d["cls_id"]] += d["conf"]
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = matrix / row_sums

    labels = list(CLASS_NAMES.values())
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    sns.heatmap(norm, annot=True, fmt=".2f",
                xticklabels=labels, yticklabels=labels,
                cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor=GRID_COL,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10, "weight": "bold"})
    ax.set_title("Detection Confidence Matrix", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Predicted Class", color="#a0aec0", labelpad=8)
    ax.set_ylabel("True Class",      color="#a0aec0", labelpad=8)
    ax.tick_params(colors="white", labelsize=8)
    fig.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    plt.tight_layout()
    return fig


def plot_severity_radar(detections):
    scores = {n: 0.0 for n in CLASS_NAMES.values()}
    for d in detections:
        scores[d["label"]] += d["conf"]

    categories = list(scores.keys())
    values = [min(scores[c], 3) for c in categories]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles_plot = angles + angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="#e74c3c")
    ax.fill(angles_plot, values_plot, alpha=0.25, color="#e74c3c")
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, color="white", size=9)
    ax.set_yticks([0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_yticklabels(["0.5","1","1.5","2","2.5","3+"], color="#a0aec0", size=7)
    ax.spines["polar"].set_color(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.8)
    ax.set_title("Damage Severity Radar", color="white", fontsize=12, pad=20)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    model_source = st.radio(
        "Model Source",
        ["Upload trained model (.pt)", "Use default YOLOv8s"],
        index=1,
    )

    model = None
    model_error = None

    if model_source == "Upload trained model (.pt)":
        model_file = st.file_uploader("Upload best.pt / last.pt", type=["pt"])
        if model_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp.write(model_file.read())
                tmp_path = tmp.name
            with st.spinner("Loading model…"):
                model, model_error = load_model(tmp_path)
    else:
        with st.spinner("Loading YOLOv8s…"):
            model, model_error = load_model("yolov8s.pt")

    if model and not model_error:
        st.success("✅ Model ready")
    elif model_error:
        st.error(f"❌ {model_error}")

    st.markdown("---")
    conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    st.markdown("---")

    st.markdown("### 🎯 Class Legend")
    for cls_id, name in CLASS_NAMES.items():
        sev = SEVERITY_MAP[name]
        st.markdown(
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{CLASS_COLORS_HEX[cls_id]};border-radius:3px;'
            f'margin-right:6px;vertical-align:middle;"></span>'
            f'**{name}** — <span style="color:{SEVERITY_COLOR[sev]}">{sev}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("YOLOv8s · Road Damage Detection\nClasses: Pothole, L-Crack, T-Crack, A-Crack")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛣️ Road Damage Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">YOLOv8s · Detects Potholes, Longitudinal Cracks, '
    'Transverse Cracks & Alligator Cracks</p>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload road image(s)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.info("📤 Upload one or more road images to begin detection.")
    st.stop()

if not model:
    st.warning("⚠️ Please load a model from the sidebar first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Run inference on all images
# ─────────────────────────────────────────────────────────────────────────────
all_detections = []
results_per_image = []

progress = st.progress(0, text="Running inference…")
for i, uf in enumerate(uploaded_files):
    pil_img = Image.open(uf).convert("RGB")
    annotated, detections = run_inference(model, pil_img, conf_threshold)
    results_per_image.append((uf.name, pil_img, annotated, detections))
    all_detections.extend(detections)
    progress.progress((i + 1) / len(uploaded_files),
                      text=f"Processed {i+1}/{len(uploaded_files)}")
progress.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics
# ─────────────────────────────────────────────────────────────────────────────
total_det  = len(all_detections)
avg_conf   = float(np.mean([d["conf"] for d in all_detections])) if all_detections else 0.0
unique_cls = len(set(d["label"] for d in all_detections))
high_sev   = sum(1 for d in all_detections if SEVERITY_MAP[d["label"]] == "High")

c1, c2, c3, c4 = st.columns(4)
for col, label, value, suffix in [
    (c1, "Total Detections", total_det,       ""),
    (c2, "Avg Confidence",   f"{avg_conf:.2f}",""),
    (c3, "Unique Classes",   unique_cls,       "/ 4"),
    (c4, "High Severity",    high_sev,         "defects"),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}'
        f'<small style="font-size:0.9rem;color:#a0aec0"> {suffix}</small></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Per-image results
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📸 Detection Results")

for fname, orig, annotated, detections in results_per_image:
    with st.expander(f"🖼 {fname}  ·  {len(detections)} detection(s)", expanded=True):
        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.markdown("**Original**")
            st.image(orig, use_container_width=True)
        with col_ann:
            st.markdown("**Annotated**")
            st.image(annotated, use_container_width=True)

        if detections:
            st.markdown("**Detection Details**")
            rows = ["| # | Class | Confidence | Severity | BBox (x1,y1,x2,y2) |",
                    "|---|-------|-----------|----------|---------------------|"]
            for j, d in enumerate(detections, 1):
                sev = SEVERITY_MAP[d["label"]]
                bb  = d["bbox"]
                rows.append(
                    f"| {j} | {d['label']} | `{d['conf']:.3f}` | {sev} | "
                    f"{bb[0]},{bb[1]},{bb[2]},{bb[3]} |"
                )
            st.markdown("\n".join(rows))
        else:
            st.success("✅ No road damage detected in this image.")

# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────
if all_detections:
    st.markdown("---")
    st.subheader("📊 Analytics & Plots")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = plot_detection_bar(all_detections)
        if fig: st.pyplot(fig); plt.close(fig)
    with r1c2:
        fig = plot_class_pie(all_detections)
        if fig: st.pyplot(fig); plt.close(fig)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = plot_confidence_dist(all_detections)
        if fig: st.pyplot(fig); plt.close(fig)
    with r2c2:
        fig = plot_area_histogram(all_detections)
        if fig: st.pyplot(fig); plt.close(fig)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = plot_confidence_matrix(all_detections)
        if fig: st.pyplot(fig); plt.close(fig)
    with r3c2:
        fig = plot_severity_radar(all_detections)
        if fig: st.pyplot(fig); plt.close(fig)

    # Class summary table
    st.markdown("---")
    st.subheader("📋 Class Summary")
    import pandas as pd
    rows = []
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_dets = [d for d in all_detections if d["label"] == cls_name]
        if cls_dets:
            confs = [d["conf"] for d in cls_dets]
            areas = [d["area"] for d in cls_dets]
            rows.append({
                "Class":        cls_name,
                "Count":        len(cls_dets),
                "Avg Conf":     f"{np.mean(confs):.3f}",
                "Max Conf":     f"{np.max(confs):.3f}",
                "Avg Area px²": f"{np.mean(areas):,.0f}",
                "Severity":     SEVERITY_MAP[cls_name],
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    st.info("No detections found. Try lowering the confidence threshold in the sidebar.")
