import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import io
import time
from pathlib import Path
from PIL import Image
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

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e74c3c, #f39c12);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid #0f3460;
    }
    .metric-label { color: #a0aec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ffffff; font-size: 2rem; font-weight: 700; }
    .damage-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .stAlert { border-radius: 10px; }
    div[data-testid="stSidebar"] { background: #0f0f1a; }
    div[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
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

CLASS_COLORS_BGR = {
    0: (0, 69, 255),    # red-orange
    1: (0, 200, 100),   # green
    2: (255, 165, 0),   # blue
    3: (148, 0, 211),   # purple
}

CLASS_COLORS_HEX = {
    0: "#FF4500",
    1: "#00C864",
    2: "#FFA500",
    3: "#9400D3",
}

SEVERITY_MAP = {
    "Pothole": "High",
    "Longitudinal Crack": "Medium",
    "Transverse Crack": "Medium",
    "Alligator Crack": "High",
}

SEVERITY_COLOR = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, img_array, conf_threshold):
    """Run YOLOv8 inference and return annotated image + detection list."""
    results = model.predict(
        source=img_array,
        conf=conf_threshold,
        verbose=False,
    )[0]

    annotated = img_array.copy()
    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            text = f"{label}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            detections.append({
                "label": label,
                "cls_id": cls_id,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "area": (x2 - x1) * (y2 - y1),
            })

    return annotated, detections


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf)


def plot_detection_bar(detections):
    counts = Counter(d["label"] for d in detections)
    if not counts:
        return None
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [CLASS_COLORS_HEX[k] for k in CLASS_NAMES if CLASS_NAMES[k] in labels]

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(labels, values,
                  color=[CLASS_COLORS_HEX[k] for k, v in CLASS_NAMES.items() if v in labels],
                  edgecolor="white", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
    ax.set_title("Detections per Class", color="white", fontsize=12, pad=10)
    ax.set_ylabel("Count", color="#a0aec0")
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_color("#2d2d4e")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_confidence_dist(detections):
    if not detections:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    for cls_id, cls_name in CLASS_NAMES.items():
        confs = [d["conf"] for d in detections if d["label"] == cls_name]
        if confs:
            ax.scatter([cls_name] * len(confs), confs,
                       color=CLASS_COLORS_HEX[cls_id], s=80, alpha=0.85,
                       edgecolors="white", linewidths=0.5, zorder=3)
    ax.axhline(0.5, color="#f39c12", linestyle="--", linewidth=1, alpha=0.6, label="50% conf")
    ax.set_title("Confidence Score Distribution", color="white", fontsize=12, pad=10)
    ax.set_ylabel("Confidence", color="#a0aec0")
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_color("#2d2d4e")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#2d2d4e")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_class_pie(detections):
    counts = Counter(d["label"] for d in detections)
    if not counts:
        return None
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = [CLASS_COLORS_HEX[k] for k, v in CLASS_NAMES.items() if v in labels]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
        textprops=dict(color="white", fontsize=9),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Class Distribution", color="white", fontsize=12, pad=10)
    plt.tight_layout()
    return fig


def plot_area_histogram(detections):
    if not detections:
        return None
    areas_by_cls = defaultdict(list)
    for d in detections:
        areas_by_cls[d["label"]].append(d["area"])

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    for cls_id, cls_name in CLASS_NAMES.items():
        if cls_name in areas_by_cls:
            ax.hist(areas_by_cls[cls_name], bins=8,
                    color=CLASS_COLORS_HEX[cls_id], alpha=0.75,
                    label=cls_name, edgecolor="white", linewidth=0.4)
    ax.set_title("BBox Area Distribution", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Area (px²)", color="#a0aec0")
    ax.set_ylabel("Count", color="#a0aec0")
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_color("#2d2d4e")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#2d2d4e")
    plt.tight_layout()
    return fig


def plot_pseudo_confusion_matrix(detections):
    """Create a detection confidence heatmap across classes as a visual."""
    matrix = np.zeros((4, 4))
    for d in detections:
        matrix[d["cls_id"]][d["cls_id"]] += d["conf"]

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = matrix / row_sums

    labels = list(CLASS_NAMES.values())
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0f0f1a")
    sns.heatmap(
        norm_matrix, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="YlOrRd", ax=ax,
        linewidths=0.5, linecolor="#2d2d4e",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_title("Detection Confidence Matrix", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Predicted Class", color="#a0aec0", labelpad=8)
    ax.set_ylabel("True Class", color="#a0aec0", labelpad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.figure.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    plt.tight_layout()
    return fig


def plot_severity_gauge(detections):
    """Radar-style severity chart."""
    severity_score = {"Pothole": 0, "Longitudinal Crack": 0,
                      "Transverse Crack": 0, "Alligator Crack": 0}
    for d in detections:
        severity_score[d["label"]] += d["conf"]

    categories = list(severity_score.keys())
    values = [min(severity_score[c], 3) for c in categories]  # cap at 3 for display
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.plot(angles, values_plot, "o-", linewidth=2, color="#e74c3c")
    ax.fill(angles, values_plot, alpha=0.25, color="#e74c3c")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", size=9)
    ax.set_yticks([0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_yticklabels(["0.5", "1", "1.5", "2", "2.5", "3+"], color="#a0aec0", size=7)
    ax.spines["polar"].set_color("#2d2d4e")
    ax.grid(color="#2d2d4e", linewidth=0.8)
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
        ["Upload model (.pt)", "Use default YOLOv8s"],
        index=1,
    )

    model = None
    model_error = None

    if model_source == "Upload model (.pt)":
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
    st.markdown(
        "<small style='color:#666'>YOLOv8s · Road Damage Detection<br>"
        "Classes: Pothole, L-Crack, T-Crack, A-Crack</small>",
        unsafe_allow_html=True,
    )


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
# Process images
# ─────────────────────────────────────────────────────────────────────────────
all_detections = []
results_per_image = []

progress = st.progress(0, text="Running inference…")
for i, uf in enumerate(uploaded_files):
    img = Image.open(uf).convert("RGB")
    img_array = np.array(img)
    annotated, detections = run_inference(model, img_array, conf_threshold)
    results_per_image.append((uf.name, img_array, annotated, detections))
    all_detections.extend(detections)
    progress.progress((i + 1) / len(uploaded_files), text=f"Processed {i+1}/{len(uploaded_files)}")

progress.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics
# ─────────────────────────────────────────────────────────────────────────────
total_det = len(all_detections)
avg_conf = np.mean([d["conf"] for d in all_detections]) if all_detections else 0
unique_cls = len(set(d["label"] for d in all_detections))
high_sev = sum(1 for d in all_detections if SEVERITY_MAP[d["label"]] == "High")

c1, c2, c3, c4 = st.columns(4)
for col, label, value, suffix in [
    (c1, "Total Detections", total_det, ""),
    (c2, "Avg Confidence", f"{avg_conf:.2f}", ""),
    (c3, "Unique Classes", unique_cls, "/ 4"),
    (c4, "High Severity", high_sev, "defects"),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}<small style="font-size:0.9rem;color:#a0aec0"> {suffix}</small></div>'
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
            header = "| # | Class | Confidence | Severity | BBox (x1,y1,x2,y2) |"
            divider = "|---|-------|-----------|----------|---------------------|"
            rows = []
            for j, d in enumerate(detections, 1):
                sev = SEVERITY_MAP[d["label"]]
                sev_colored = f"<span style='color:{SEVERITY_COLOR[sev]}'>{sev}</span>"
                bb = d["bbox"]
                rows.append(
                    f"| {j} | {d['label']} | `{d['conf']:.3f}` | {sev} | "
                    f"{bb[0]},{bb[1]},{bb[2]},{bb[3]} |"
                )
            st.markdown("\n".join([header, divider] + rows))
        else:
            st.success("✅ No road damage detected in this image.")

# ─────────────────────────────────────────────────────────────────────────────
# Analytics & Plots
# ─────────────────────────────────────────────────────────────────────────────
if all_detections:
    st.markdown("---")
    st.subheader("📊 Analytics & Plots")

    # Row 1: Bar + Pie
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = plot_detection_bar(all_detections)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    with r1c2:
        fig = plot_class_pie(all_detections)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    # Row 2: Confidence + Area
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = plot_confidence_dist(all_detections)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    with r2c2:
        fig = plot_area_histogram(all_detections)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    # Row 3: Confusion matrix + Radar
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = plot_pseudo_confusion_matrix(all_detections)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    with r3c2:
        fig = plot_severity_gauge(all_detections)
        if fig:
            st.pyplot(fig)
            plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-class summary table
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Class Summary")

    summary_rows = []
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_dets = [d for d in all_detections if d["label"] == cls_name]
        if cls_dets:
            confs = [d["conf"] for d in cls_dets]
            areas = [d["area"] for d in cls_dets]
            summary_rows.append({
                "Class": cls_name,
                "Count": len(cls_dets),
                "Avg Conf": f"{np.mean(confs):.3f}",
                "Max Conf": f"{np.max(confs):.3f}",
                "Avg Area (px²)": f"{np.mean(areas):,.0f}",
                "Severity": SEVERITY_MAP[cls_name],
            })

    if summary_rows:
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("No detections found across all uploaded images at the current confidence threshold. "
            "Try lowering the threshold in the sidebar.")
