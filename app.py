"""
Road Damage Detection — YOLOv8s
Streamlit Web Application
• No cv2 / libGL dependency (Pillow ImageDraw for annotations)
• Bundled metrics tab: reads results.csv + saved PNG plots from zip
"""

import streamlit as st
import numpy as np
import tempfile, os, io, zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Road Damage Detector", page_icon="🛣️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.4rem;font-weight:800;
  background:linear-gradient(135deg,#e74c3c,#f39c12);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem}
.sub-header{color:#7f8c8d;font-size:1rem;margin-bottom:2rem}
.metric-card{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;
  padding:1.2rem 1rem;text-align:center;border:1px solid #0f3460}
.metric-label{color:#a0aec0;font-size:.8rem;text-transform:uppercase;letter-spacing:1px}
.metric-value{color:#fff;font-size:2rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES   = {0:"Pothole",1:"Longitudinal Crack",2:"Transverse Crack",3:"Alligator Crack"}
CLASS_COLORS  = {0:(255,69,0),1:(0,200,100),2:(255,165,0),3:(148,0,211)}
CLASS_HEX     = {0:"#FF4500",1:"#00C864",2:"#FFA500",3:"#9400D3"}
SEVERITY_MAP  = {"Pothole":"High","Longitudinal Crack":"Medium",
                 "Transverse Crack":"Medium","Alligator Crack":"High"}
SEVERITY_COL  = {"High":"#e74c3c","Medium":"#f39c12","Low":"#2ecc71"}
DARK,PANEL,GRID = "#0f0f1a","#1a1a2e","#2d2d4e"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) — safe for plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        from ultralytics import YOLO
        return YOLO(path), None
    except Exception as e:
        return None, str(e)


def run_inference(model, pil_img, conf):
    img_np = np.array(pil_img)
    results = model.predict(source=img_np, conf=conf, verbose=False)[0]
    ann = pil_img.copy()
    draw = ImageDraw.Draw(ann)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            cid  = int(box.cls[0]); conf_v = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            label = CLASS_NAMES.get(cid, f"Class {cid}")
            color = CLASS_COLORS.get(cid, (255,255,255))
            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            text = f"{label}: {conf_v:.2f}"
            tb = draw.textbbox((x1, y1-18), text, font=font)
            draw.rectangle(tb, fill=color)
            draw.text((x1, y1-18), text, fill=(255,255,255), font=font)
            detections.append({"label":label,"cls_id":cid,"conf":conf_v,
                                "bbox":(x1,y1,x2,y2),"area":(x2-x1)*(y2-y1)})
    return ann, detections

# ─────────────────────────────────────────────────────────────────────────────
# Detection plots (matplotlib, no cv2)
# ─────────────────────────────────────────────────────────────────────────────
def _ax(ax, title):
    ax.set_facecolor(PANEL); ax.set_title(title,color="white",fontsize=12,pad=10)
    ax.tick_params(colors="white",labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)

def plot_bar(dets):
    c = Counter(d["label"] for d in dets)
    if not c: return None
    lbl,val = list(c.keys()),[c[l] for l in c]
    clr = [CLASS_HEX[k] for k,v in CLASS_NAMES.items() if v in lbl]
    fig,ax = plt.subplots(figsize=(6,3.5),facecolor=DARK)
    bars = ax.bar(lbl,val,color=clr,edgecolor="white",linewidth=.5,width=.5)
    for b,v in zip(bars,val):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.05,str(v),
                ha="center",va="bottom",color="white",fontsize=10,fontweight="bold")
    _ax(ax,"Detections per Class"); ax.set_ylabel("Count",color="#a0aec0")
    plt.xticks(rotation=15,ha="right"); plt.tight_layout(); return fig

def plot_pie(dets):
    c = Counter(d["label"] for d in dets)
    if not c: return None
    lbl,sz = list(c.keys()),[c[l] for l in c]
    clr = [CLASS_HEX[k] for k,v in CLASS_NAMES.items() if v in lbl]
    fig,ax = plt.subplots(figsize=(5,4),facecolor=DARK); ax.set_facecolor(DARK)
    _,texts,ats = ax.pie(sz,labels=lbl,colors=clr,autopct="%1.0f%%",startangle=140,
        wedgeprops=dict(edgecolor="white",linewidth=1.2),textprops=dict(color="white",fontsize=9))
    for a in ats: a.set_color("white"); a.set_fontweight("bold")
    ax.set_title("Class Distribution",color="white",fontsize=12,pad=10)
    plt.tight_layout(); return fig

def plot_conf(dets):
    if not dets: return None
    fig,ax = plt.subplots(figsize=(6,3.5),facecolor=DARK)
    ax.set_facecolor(PANEL)
    for cid,cn in CLASS_NAMES.items():
        cs=[d["conf"] for d in dets if d["label"]==cn]
        if cs: ax.scatter([cn]*len(cs),cs,color=CLASS_HEX[cid],s=80,alpha=.85,
                          edgecolors="white",linewidths=.5,zorder=3)
    ax.axhline(.5,color="#f39c12",linestyle="--",linewidth=1,alpha=.6,label="50%")
    _ax(ax,"Confidence Score Distribution"); ax.set_ylabel("Confidence",color="#a0aec0")
    ax.set_ylim(0,1.05); ax.legend(fontsize=8,facecolor=PANEL,labelcolor="white",edgecolor=GRID)
    plt.xticks(rotation=15,ha="right"); plt.tight_layout(); return fig

def plot_area(dets):
    if not dets: return None
    areas = defaultdict(list)
    for d in dets: areas[d["label"]].append(d["area"])
    fig,ax = plt.subplots(figsize=(6,3.5),facecolor=DARK); ax.set_facecolor(PANEL)
    for cid,cn in CLASS_NAMES.items():
        if cn in areas:
            ax.hist(areas[cn],bins=8,color=CLASS_HEX[cid],alpha=.75,
                    label=cn,edgecolor="white",linewidth=.4)
    _ax(ax,"BBox Area Distribution"); ax.set_xlabel("Area (px²)",color="#a0aec0")
    ax.set_ylabel("Count",color="#a0aec0")
    ax.legend(fontsize=8,facecolor=PANEL,labelcolor="white",edgecolor=GRID)
    plt.tight_layout(); return fig

def plot_heatmap(dets):
    m = np.zeros((4,4))
    for d in dets: m[d["cls_id"]][d["cls_id"]] += d["conf"]
    rs = m.sum(axis=1,keepdims=True); rs[rs==0]=1; m/=rs
    lbl=list(CLASS_NAMES.values())
    fig,ax = plt.subplots(figsize=(6,5),facecolor=DARK)
    sns.heatmap(m,annot=True,fmt=".2f",xticklabels=lbl,yticklabels=lbl,
                cmap="YlOrRd",ax=ax,linewidths=.5,linecolor=GRID,
                cbar_kws={"shrink":.8},annot_kws={"size":10,"weight":"bold"})
    ax.set_title("Detection Confidence Matrix",color="white",fontsize=12,pad=10)
    ax.set_xlabel("Predicted",color="#a0aec0",labelpad=8)
    ax.set_ylabel("True",color="#a0aec0",labelpad=8)
    ax.tick_params(colors="white",labelsize=8)
    fig.set_facecolor(DARK); ax.set_facecolor(PANEL); plt.tight_layout(); return fig

def plot_radar(dets):
    scores={n:0. for n in CLASS_NAMES.values()}
    for d in dets: scores[d["label"]] += d["conf"]
    cats=list(scores.keys()); vals=[min(scores[c],3) for c in cats]
    N=len(cats); angles=[n/N*2*np.pi for n in range(N)]
    ap=angles+angles[:1]; vp=vals+vals[:1]
    fig,ax=plt.subplots(figsize=(5,5),subplot_kw=dict(polar=True),facecolor=DARK)
    ax.set_facecolor(PANEL); ax.plot(ap,vp,"o-",lw=2,color="#e74c3c")
    ax.fill(ap,vp,alpha=.25,color="#e74c3c"); ax.set_xticks(angles)
    ax.set_xticklabels(cats,color="white",size=9)
    ax.set_yticks([.5,1,1.5,2,2.5,3]); ax.set_yticklabels([".5","1","1.5","2","2.5","3+"],color="#a0aec0",size=7)
    ax.spines["polar"].set_color(GRID); ax.grid(color=GRID,lw=.8)
    ax.set_title("Damage Severity Radar",color="white",fontsize=12,pad=20)
    plt.tight_layout(); return fig

# ─────────────────────────────────────────────────────────────────────────────
# Training metrics plots (Plotly) — FIXED fillcolor
# ─────────────────────────────────────────────────────────────────────────────
def training_curve_fig(df: pd.DataFrame) -> go.Figure:
    """Interactive training/val loss + metrics curves. fillcolor uses rgba() — no crash."""
    epochs = df["epoch"].tolist()

    CURVE_COLS = {
        "train/box_loss":         CLASS_HEX[0],
        "train/cls_loss":         CLASS_HEX[1],
        "train/dfl_loss":         CLASS_HEX[2],
        "val/box_loss":           "#e74c3c",
        "val/cls_loss":           "#3498db",
        "val/dfl_loss":           "#9b59b6",
        "metrics/mAP50(B)":       "#2ecc71",
        "metrics/mAP50-95(B)":    "#f39c12",
        "metrics/precision(B)":   "#1abc9c",
        "metrics/recall(B)":      "#e67e22",
    }

    # Two subplots: losses | metrics
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Losses per Epoch", "Metrics per Epoch"],
        horizontal_spacing=0.08,
    )

    loss_cols   = [c for c in CURVE_COLS if "loss" in c]
    metric_cols = [c for c in CURVE_COLS if "loss" not in c]

    for col, is_loss in [(loss_cols, True), (metric_cols, False)]:
        row, plot_col = (1, 1) if is_loss else (1, 2)
        for name in col:
            if name not in df.columns:
                continue
            vals = df[name].tolist()
            hex_c = CURVE_COLS[name]
            fill_c = hex_to_rgba(hex_c, 0.08)   # ← FIXED: proper rgba string
            fig.add_trace(go.Scatter(
                x=epochs, y=vals,
                mode="lines",
                name=name.split("/")[-1],
                line=dict(color=hex_c, width=2),
                fill="tozeroy",
                fillcolor=fill_c,
                hovertemplate=f"<b>{name}</b><br>Epoch: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            ), row=row, col=plot_col)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK,
        plot_bgcolor=PANEL,
        font=dict(color="white", size=11),
        legend=dict(bgcolor=PANEL, bordercolor=GRID, borderwidth=1,
                    font=dict(size=10)),
        height=420,
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, title_text="Epoch")
    fig.update_yaxes(gridcolor=GRID)
    return fig


def lr_curve_fig(df: pd.DataFrame) -> go.Figure:
    """Learning rate schedule."""
    epochs = df["epoch"].tolist()
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    fig = go.Figure()
    palette = ["#3498db","#e74c3c","#2ecc71"]
    for i, col in enumerate(lr_cols):
        hex_c = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=epochs, y=df[col].tolist(),
            mode="lines", name=col,
            line=dict(color=hex_c, width=2),
            fill="tozeroy",
            fillcolor=hex_to_rgba(hex_c, 0.08),   # ← FIXED
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=PANEL,
        font=dict(color="white"), height=320,
        title="Learning Rate Schedule",
        xaxis_title="Epoch", yaxis_title="LR",
        margin=dict(l=50,r=20,t=50,b=40),
    )
    fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
    return fig


def final_metrics_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of final-epoch metrics."""
    last = df.iloc[-1]
    names = ["Precision","Recall","mAP@50","mAP@50-95"]
    cols  = ["metrics/precision(B)","metrics/recall(B)",
             "metrics/mAP50(B)","metrics/mAP50-95(B)"]
    vals  = [float(last[c]) for c in cols if c in last.index]
    colors= [CLASS_HEX[i] for i in range(len(vals))]
    fig = go.Figure(go.Bar(
        x=names[:len(vals)], y=vals, marker_color=colors,
        text=[f"{v:.3f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=PANEL,
        font=dict(color="white"), height=320,
        title=f"Final Epoch ({int(last['epoch'])}) Metrics",
        yaxis=dict(range=[0,1.1], gridcolor=GRID),
        xaxis=dict(gridcolor=GRID),
        margin=dict(l=40,r=20,t=50,b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    model_source = st.radio("Model Source",
        ["Upload trained model (.pt)", "Use default YOLOv8s"], index=1)
    model = None; model_error = None

    if model_source == "Upload trained model (.pt)":
        mf = st.file_uploader("Upload best.pt / last.pt", type=["pt"])
        if mf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp.write(mf.read()); tmp_path = tmp.name
            with st.spinner("Loading model…"):
                model, model_error = load_model(tmp_path)
    else:
        with st.spinner("Loading YOLOv8s…"):
            model, model_error = load_model("yolov8s.pt")

    if model and not model_error:  st.success("✅ Model ready")
    elif model_error:               st.error(f"❌ {model_error}")

    st.markdown("---")
    conf_thr = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    st.markdown("---")

    st.markdown("### 🎯 Class Legend")
    for cid, name in CLASS_NAMES.items():
        sev = SEVERITY_MAP[name]
        st.markdown(
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{CLASS_HEX[cid]};border-radius:3px;margin-right:6px;'
            f'vertical-align:middle;"></span>**{name}** — '
            f'<span style="color:{SEVERITY_COL[sev]}">{sev}</span>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Training Metrics")
    metrics_zip = st.file_uploader("Upload metrics.zip", type=["zip"])

    st.markdown("---")
    st.caption("YOLOv8s · Road Damage Detection")

# ─────────────────────────────────────────────────────────────────────────────
# Header + tabs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛣️ Road Damage Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">YOLOv8s · Potholes · Longitudinal Cracks · Transverse Cracks · Alligator Cracks</p>',
            unsafe_allow_html=True)

tab_detect, tab_metrics = st.tabs(["🔍 Detection", "📈 Training Metrics"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — DETECTION
# ══════════════════════════════════════════════════════════════════
with tab_detect:
    uploaded_files = st.file_uploader(
        "Upload road image(s)", type=["jpg","jpeg","png","bmp","webp"],
        accept_multiple_files=True, label_visibility="collapsed")

    if not uploaded_files:
        st.info("📤 Upload one or more road images to begin detection.")
    elif not model:
        st.warning("⚠️ Please load a model from the sidebar first.")
    else:
        all_dets = []; results_per_img = []
        prog = st.progress(0, text="Running inference…")
        for i, uf in enumerate(uploaded_files):
            pil = Image.open(uf).convert("RGB")
            ann, dets = run_inference(model, pil, conf_thr)
            results_per_img.append((uf.name, pil, ann, dets))
            all_dets.extend(dets)
            prog.progress((i+1)/len(uploaded_files), text=f"Processed {i+1}/{len(uploaded_files)}")
        prog.empty()

        # Metrics row
        total = len(all_dets)
        avg_c = float(np.mean([d["conf"] for d in all_dets])) if all_dets else 0
        ucls  = len(set(d["label"] for d in all_dets))
        hi    = sum(1 for d in all_dets if SEVERITY_MAP[d["label"]]=="High")

        c1,c2,c3,c4 = st.columns(4)
        for col, lbl, val, suf in [
            (c1,"Total Detections",total,""),
            (c2,"Avg Confidence",f"{avg_c:.2f}",""),
            (c3,"Unique Classes",ucls,"/ 4"),
            (c4,"High Severity",hi,"defects"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                f'<div class="metric-value">{val}'
                f'<small style="font-size:.9rem;color:#a0aec0"> {suf}</small></div></div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📸 Detection Results")

        for fname, orig, ann, dets in results_per_img:
            with st.expander(f"🖼 {fname}  ·  {len(dets)} detection(s)", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original**")
                    st.image(orig, use_container_width=True)
                with c2:
                    st.markdown("**Annotated**")
                    st.image(ann, use_container_width=True)
                if dets:
                    rows = ["| # | Class | Confidence | Severity | BBox |",
                            "|---|-------|-----------|----------|------|"]
                    for j,d in enumerate(dets,1):
                        sev=SEVERITY_MAP[d["label"]]; bb=d["bbox"]
                        rows.append(f"| {j} | {d['label']} | `{d['conf']:.3f}` | {sev} | {bb[0]},{bb[1]},{bb[2]},{bb[3]} |")
                    st.markdown("\n".join(rows))
                else:
                    st.success("✅ No road damage detected.")

        if all_dets:
            st.markdown("---"); st.subheader("📊 Detection Analytics")
            r1,r2 = st.columns(2)
            with r1:
                fig=plot_bar(all_dets)
                if fig: st.pyplot(fig); plt.close(fig)
            with r2:
                fig=plot_pie(all_dets)
                if fig: st.pyplot(fig); plt.close(fig)
            r3,r4 = st.columns(2)
            with r3:
                fig=plot_conf(all_dets)
                if fig: st.pyplot(fig); plt.close(fig)
            with r4:
                fig=plot_area(all_dets)
                if fig: st.pyplot(fig); plt.close(fig)
            r5,r6 = st.columns(2)
            with r5:
                fig=plot_heatmap(all_dets)
                if fig: st.pyplot(fig); plt.close(fig)
            with r6:
                fig=plot_radar(all_dets)
                if fig: st.pyplot(fig); plt.close(fig)

            st.markdown("---"); st.subheader("📋 Class Summary")
            summary=[]
            for cid,cn in CLASS_NAMES.items():
                cd=[d for d in all_dets if d["label"]==cn]
                if cd:
                    cs=[d["conf"] for d in cd]; ar=[d["area"] for d in cd]
                    summary.append({"Class":cn,"Count":len(cd),
                        "Avg Conf":f"{np.mean(cs):.3f}","Max Conf":f"{np.max(cs):.3f}",
                        "Avg Area px²":f"{np.mean(ar):,.0f}","Severity":SEVERITY_MAP[cn]})
            if summary:
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING METRICS
# ══════════════════════════════════════════════════════════════════
with tab_metrics:
    st.subheader("📈 Training Metrics Dashboard")

    # ── Parse zip ──────────────────────────────────────────────────
    df_results   = None
    png_files    = {}   # name → PIL Image

    if metrics_zip:
        zdata = metrics_zip.read()
    else:
        # Check if already extracted in /tmp from session
        csv_path = "/tmp/content/drive/MyDrive/YOLO_RoadDamage/run2/results.csv"
        if os.path.exists(csv_path):
            zdata = None
            df_results = pd.read_csv(csv_path)
            df_results.columns = [c.strip() for c in df_results.columns]
            run_dir = Path(csv_path).parent
            for f in run_dir.glob("*.png"):
                png_files[f.stem] = Image.open(f)
        else:
            zdata = None

    if 'zdata' in dir() and zdata:
        with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if base == "results.csv":
                    with zf.open(name) as f:
                        df_results = pd.read_csv(f)
                        df_results.columns = [c.strip() for c in df_results.columns]
                elif base.endswith(".png"):
                    with zf.open(name) as f:
                        png_files[Path(base).stem] = Image.open(io.BytesIO(f.read())).copy()

    if df_results is None:
        st.info("📤 Upload your **metrics.zip** in the sidebar to see training curves and plots.")
        st.markdown("""
        The zip should contain the YOLOv8 training output folder, including:
        - `results.csv` — epoch-by-epoch losses and metrics
        - `confusion_matrix.png`, `BoxPR_curve.png`, `results.png`, etc.
        """)
    else:
        # ── Summary KPIs ──────────────────────────────────────────
        last = df_results.iloc[-1]
        best_map50_idx = df_results["metrics/mAP50(B)"].idxmax()
        best = df_results.iloc[best_map50_idx]

        k1,k2,k3,k4 = st.columns(4)
        for col, lbl, val in [
            (k1, "Total Epochs",      int(last["epoch"])),
            (k2, "Best mAP@50",       f"{best['metrics/mAP50(B)']:.4f}"),
            (k3, "Best mAP@50-95",    f"{best['metrics/mAP50-95(B)']:.4f}"),
            (k4, "Final Precision",   f"{last['metrics/precision(B)']:.4f}"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                f'<div class="metric-value">{val}</div></div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Training curves ───────────────────────────────────────
        st.plotly_chart(training_curve_fig(df_results), use_container_width=True)

        # ── LR + Final bar side by side ───────────────────────────
        lc1, lc2 = st.columns(2)
        with lc1:
            st.plotly_chart(lr_curve_fig(df_results), use_container_width=True)
        with lc2:
            st.plotly_chart(final_metrics_bar(df_results), use_container_width=True)

        # ── Raw CSV expander ──────────────────────────────────────
        with st.expander("📄 Raw results.csv"):
            st.dataframe(df_results, use_container_width=True)

        # ── PNG plots from zip ────────────────────────────────────
        PNG_ORDER = [
            ("confusion_matrix",            "Confusion Matrix"),
            ("confusion_matrix_normalized", "Confusion Matrix (Normalized)"),
            ("BoxPR_curve",                 "Precision-Recall Curve"),
            ("BoxF1_curve",                 "F1 Curve"),
            ("BoxP_curve",                  "Precision Curve"),
            ("BoxR_curve",                  "Recall Curve"),
            ("results",                     "Results Overview"),
            ("labels",                      "Label Distribution"),
        ]

        available = [(stem, title) for stem, title in PNG_ORDER if stem in png_files]
        if available:
            st.markdown("---")
            st.subheader("🖼 Saved Training Plots")
            for i in range(0, len(available), 2):
                cols = st.columns(2)
                for j, (stem, title) in enumerate(available[i:i+2]):
                    with cols[j]:
                        st.markdown(f"**{title}**")
                        st.image(png_files[stem], use_container_width=True)

        # ── Training sample images ────────────────────────────────
        batch_imgs = [(k, v) for k, v in png_files.items() if "batch" in k]
        if batch_imgs:
            st.markdown("---")
            st.subheader("🏋️ Training Batch Samples")
            for i in range(0, len(batch_imgs), 3):
                cols = st.columns(3)
                for j, (name, img) in enumerate(batch_imgs[i:i+3]):
                    with cols[j]:
                        st.markdown(f"**{name}**")
                        st.image(img, use_container_width=True)
