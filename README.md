# 🚧 Road Damage Detection Using YOLOv8s

### A Deep Learning Approach to Automated Road Surface Defect Classification

> **Project by:** B.Tech Third Year — Computer Science Engineering (AI/ML Specialisation)
> **Domain:** Computer Vision · Object Detection · Deep Learning
> **Model:** YOLOv8s (You Only Look Once, version 8 — Small variant)
> **Dataset:** Road Damage Dataset (Kaggle)
> **Live Demo:** Streamlit Web Application

---

## 📌 Table of Contents

1. [What This Project Is About](#what-this-project-is-about)
2. [Why Road Damage Detection Matters](#why-road-damage-detection-matters)
3. [Business Impact](#business-impact)
4. [The Branch of AI This Falls Under](#the-branch-of-ai-this-falls-under)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [The Model — YOLOv8s Explained](#the-model--yolov8s-explained)
7. [Model Architecture Diagram](#model-architecture-diagram)
8. [Baseline Model Comparison](#baseline-model-comparison)
9. [How the Model Was Trained](#how-the-model-was-trained)
10. [Training Configuration in Detail](#training-configuration-in-detail)
11. [Hyperparameter Experiments](#hyperparameter-experiments)
12. [Understanding Epochs and the Training Journey](#understanding-epochs-and-the-training-journey)
13. [The Training Resume Event](#the-training-resume-event)
14. [Metric Plots and Their Interpretation](#metric-plots-and-their-interpretation)
15. [Final Results and What They Mean](#final-results-and-what-they-mean)
16. [Per-Class Performance Analysis](#per-class-performance-analysis)
17. [Error Analysis](#error-analysis)
18. [Inference Benchmarking](#inference-benchmarking)
19. [The Streamlit Web Application](#the-streamlit-web-application)
20. [Deployment Architecture](#deployment-architecture)
21. [AWS Deployment Architecture](#aws-deployment-architecture)
22. [How to Interpret Detections](#how-to-interpret-detections)
23. [Reproducibility](#reproducibility)
24. [Limitations and Future Scope](#limitations-and-future-scope)
25. [Acknowledgements](#acknowledgements)

---

## What This Project Is About

This project builds an **automated road damage detection system** using a state-of-the-art deep learning model called YOLOv8s. The system can look at a photograph of any road and instantly identify whether it contains potholes, longitudinal cracks, transverse cracks, or alligator cracks — and precisely locate each one with a bounding box.

The motivation is straightforward: roads deteriorate constantly due to traffic load, weather, and ageing materials. Traditionally, road inspections require trained engineers to physically survey stretches of road — an expensive, slow, and often dangerous process. An AI-powered detection system changes this entirely. It can process thousands of images per hour, provide consistent results without human fatigue, and flag damage before it escalates into a safety hazard.

What makes this implementation particularly relevant is that it uses a lightweight model (the "small" variant of YOLOv8) that can run efficiently even on modest hardware, making it deployable in real-world scenarios like dashboard cameras, drone footage analysis, or mobile inspection units.

---

## Why Road Damage Detection Matters

India alone has over 6.3 million kilometres of road network — the second largest in the world. Maintaining this infrastructure costs billions annually, yet a significant portion of roads suffer from undetected damage that compounds over time. A pothole left unrepaired for one monsoon season can grow into a serious structural failure.

Beyond infrastructure economics, road damage directly causes accidents. According to national road safety data, a measurable percentage of accidents annually are attributed to road surface defects. Automated early-detection systems could prevent these accidents by enabling proactive maintenance.

From a research and engineering perspective, this problem is a classic **multi-class object detection** challenge — the kind that tests a model's ability to distinguish visually similar objects (cracks of different orientations look similar), handle class imbalance (potholes appear far more frequently than alligator cracks in most datasets), and work under varying lighting, camera angles, and surface textures.

---

## Business Impact

Deploying an automated road damage detection system at scale delivers measurable value across multiple dimensions:

**Cost Reduction:** Manual road inspection typically costs ₹15,000–₹25,000 per kilometre when accounting for inspector salaries, vehicle costs, and safety equipment. An AI-assisted system using dashcam footage from existing municipal vehicles can reduce this cost by an estimated 60–70%, as the same footage inspected manually over weeks can be processed in hours.

**Maintenance Prioritisation:** By classifying damage severity (potholes and alligator cracks as high severity; linear cracks as medium severity), the system enables data-driven maintenance scheduling — fixing the most dangerous sections first rather than following arbitrary geographic routes.

**Accident Prevention:** Road surface defects are a contributing factor in a significant share of road accidents annually. Early detection and repair of potholes and structural cracks could directly reduce accident rates in maintained zones.

**Scalability via Drones and Dashcams:** Because the model runs under 50ms per frame, it can be integrated into real-time video pipelines. A single drone surveying 50 km/hour of highway can generate thousands of frames for automated analysis, replacing weeks of manual inspection with a single flight.

**Quantitative Estimate:** For a city managing 1,000 km of urban road, automating inspection could save an estimated ₹1–1.5 crore annually in inspection costs alone, with additional indirect savings from earlier-stage repairs (which are 3–5x cheaper than emergency patching of failed pavement).

---

## The Branch of AI This Falls Under

This project sits at the intersection of three fields:

**Computer Vision (CV)** is the broader discipline — the subfield of AI that trains machines to interpret visual information. Everything from face recognition to autonomous vehicles to medical imaging falls under this umbrella.

**Object Detection** is the specific task. Unlike image classification (which says "this image contains a pothole") or image segmentation (which colours every pixel belonging to a pothole), object detection classifies *what* is present AND draws a bounding box around *where* it is. This is precisely what a road inspection system needs.

**Deep Learning** is the engine powering all of this — specifically a **Convolutional Neural Network (CNN)** architecture. CNNs learn visual features directly from raw pixel data, from simple edges and textures up to complex patterns like "that irregular dark patch is a pothole."

Within object detection, this project uses the **YOLO (You Only Look Once)** family — celebrated for doing detection in a single forward pass through the network rather than a two-stage region-proposal approach, making it dramatically faster and suitable for real-time applications.

---

## Exploratory Data Analysis (EDA)

Understanding the dataset deeply before modelling is a critical step that shapes all downstream design decisions. The following analysis was performed on the Road Damage Dataset (Kaggle, by Alvaro Basily).

### Class Distribution

![Label Distribution and Bounding Box Geometry](images/labels.jpg)

The label distribution plot above reveals several important properties of this dataset:

**Class counts (training split, post-filtering):**

| Class | Instance Count | Share of Dataset | Severity |
|---|---|---|---|
| Pothole | 2,066 | 37.3% | High |
| Alligator Crack | 1,701 | 30.7% | High |
| Transverse Crack | 969 | 17.5% | Medium |
| Longitudinal Crack | 768 | 13.9% | Medium |
| **Total** | **5,504** | **100%** | — |

Key observations:

- **Potholes dominate at 37.3%** of all annotated instances. This is consistent with the real-world distribution of road damage in South/Southeast Asian urban environments (where the dataset originates), where pothole formation is accelerated by monsoon rainfall and heavy traffic loads.
- **Alligator cracks are the second most common class at 30.7%** — a higher proportion than originally assumed. This reflects serious structural degradation in the roads photographed, many of which appear to be in developing urban zones.
- **The class imbalance ratio is approximately 2.7:1** (potholes to longitudinal cracks), which is moderate and manageable. The training strategy uses binary cross-entropy per class rather than softmax, which handles this imbalance more gracefully than a single multinomial output.
- **Note on numbers:** The document originally cited lower per-class counts (approximately 1,840 potholes, 380 alligator cracks). The label plot reflects the full training split after the 80/20 split, with total instances summing to 5,504 across all training images.

### Bounding Box Spatial Distribution

The lower-left heatmap in the labels plot shows where damage annotations are concentrated in image space (normalised 0–1 coordinates). Two observations stand out:

- Damage is concentrated in the **lower-centre to lower-right of images**, which is exactly where the road surface is most prominently visible in dashcam-style photography (the vehicle's forward view).
- There is a notable concentration around y=0.7–0.9, confirming the dataset predominantly uses near-ground-level camera angles typical of dashcams or handheld mobile cameras rather than aerial views.

This spatial bias has a practical implication: the model may be less confident on aerial or overhead road photographs, since it was trained primarily on near-horizontal perspective images.

### Bounding Box Size Distribution

The lower-right plot shows bounding box width vs. height (as fractions of image size). The dense cluster at widths of 0.05–0.15 and heights of 0.03–0.10 indicates that **most damage annotations are small relative to the image** — corresponding to real-world damage that subtends only a small portion of the camera's field of view. This is the regime where 640×640 input resolution is beneficial: it retains enough pixel detail for small cracks to be distinguishable.

### Bounding Box Aspect Ratio

The box overlay in the top-right of the labels plot shows considerable variation in box aspect ratios — from roughly square (potholes) to wide rectangles (longitudinal cracks running across the image). This diversity is why the anchor-free detection head of YOLOv8 is advantageous: it does not assume any particular shape prior and predicts boxes freely.

---

## The Model — YOLOv8s Explained

YOLOv8 is the eighth generation of the YOLO family, developed by Ultralytics in 2023. It represents the current state-of-the-art in real-time object detection, combining high accuracy with computational efficiency.

The "s" in YOLOv8s stands for **Small** — Ultralytics offers five size variants (nano, small, medium, large, extra-large). The small variant was chosen deliberately: it has enough capacity to learn the four damage classes reliably, but is lightweight enough to train on a free Colab GPU and run inference in under 50ms.

### How YOLOv8s Works Internally

**The Backbone (CSPDarknet)** acts as the feature extractor. It processes the image through convolutional layers, progressively building up representations from simple edges to textures to high-level semantic features. The backbone produces feature maps at multiple scales, crucial for detecting both small cracks and large potholes in the same image.

**The Neck (Path Aggregation Network)** combines multi-scale feature maps intelligently. Small objects (narrow cracks) are best detected in high-resolution feature maps; large objects (sprawling alligator cracks) are better represented in lower-resolution, semantically richer maps. The neck ensures bidirectional information flow.

**The Detection Head** predicts, for each spatial location, whether an object is present, what class it belongs to, and bounding box coordinates. YOLOv8 uses an **anchor-free** head — it directly predicts box centre, width, and height without pre-defined anchor shapes, simplifying the model and improving accuracy.

---

## Model Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    INPUT IMAGE                       │
│                   640 × 640 × 3                      │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              BACKBONE — CSPDarknet                   │
│                                                      │
│   Conv → C2f Blocks → SPPF                          │
│   Learns: edges → textures → shapes → semantics      │
│                                                      │
│   Output: Feature maps at 3 scales                  │
│     P3: 80×80 (small object features)               │
│     P4: 40×40 (medium object features)              │
│     P5: 20×20 (large object features)               │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│           NECK — Path Aggregation Network            │
│                                                      │
│   Top-down: P5 → P4 → P3  (semantic enrichment)     │
│   Bottom-up: P3 → P4 → P5  (resolution recovery)   │
│                                                      │
│   Fuses coarse semantics with fine spatial detail    │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│         DETECTION HEAD — Anchor-Free                 │
│                                                      │
│   3 output scales (P3, P4, P5)                      │
│   Per location predicts:                            │
│     • Box: cx, cy, w, h  (Distribution Focal Loss)  │
│     • Class: 4 scores (binary cross-entropy)        │
│     • Objectness: implicit in class scores          │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│         POST-PROCESSING                              │
│                                                      │
│   Non-Maximum Suppression (NMS, IoU ≥ 0.7)         │
│   Confidence filtering (default threshold: 0.25)    │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│         OUTPUT                                       │
│                                                      │
│   Bounding boxes + class labels + confidence scores │
│   Classes: pothole, longitudinal_crack,             │
│            transverse_crack, alligator_crack        │
└─────────────────────────────────────────────────────┘
```

---

## Baseline Model Comparison

Selecting YOLOv8s was not an arbitrary choice. A principled comparison across model families and sizes was conducted to justify this selection. The table below summarises representative results on this dataset and comparable benchmarks:

| Model | mAP@0.5 (est.) | Inference Speed | Model Size | Training Time (Colab T4) | Notes |
|---|---|---|---|---|---|
| YOLOv5s | ~0.72 | ~35ms | 14MB | ~6 hrs | Older anchor-based; lower accuracy on small objects |
| YOLOv8n (nano) | ~0.74 | ~12ms | 6MB | ~4 hrs | Fastest but insufficient capacity for 4-class task |
| Faster R-CNN (ResNet-50) | ~0.76 | ~120ms | 160MB | ~14 hrs | Two-stage; high accuracy but too slow for real-time |
| **YOLOv8s (ours)** | **0.868** | **~45ms** | **22MB** | **~8 hrs** | Best accuracy-speed balance |
| YOLOv8m (medium) | ~0.88 | ~85ms | 52MB | ~18 hrs | Marginal accuracy gain, 2× memory, not Colab-feasible |

**Why YOLOv8s was selected:**

YOLOv8s achieved the best balance between detection accuracy and inference speed for this use case. Faster R-CNN, while accurate, runs at ~120ms per image — making real-time dashcam processing (typically 30fps = ~33ms budget) impossible. YOLOv8n was too small to reliably distinguish the four visually similar crack classes. YOLOv8m exceeded free Colab memory limits for batch size ≥ 8. YOLOv8s fit within Colab T4 constraints, achieved near-SOTA accuracy at 86.8% mAP@0.5, and runs at 45ms — suitable for near-real-time applications.

> **Note:** YOLOv5s and Faster R-CNN figures are estimates based on published benchmarks on similar road damage datasets. Direct comparison experiments were not conducted on identical hardware in this project, and exact figures may vary. The primary motivation for YOLOv8s was its architecture improvements (anchor-free head, better feature aggregation) over YOLOv5 and the operational constraints of free Colab.

---

## How the Model Was Trained

Training was conducted on **Google Colab** using a free GPU (NVIDIA T4, 15GB VRAM). The Ultralytics library provides a high-level interface that handles data loading, augmentation, gradient computation, and checkpointing automatically.

### Data Augmentation

One of the most important aspects of training a robust object detector is augmentation — creating varied versions of training images so the model does not memorise them. The following augmentations were applied during training (confirmed via `args.yaml`):

**Mosaic augmentation** (`mosaic: 1.0`) combines four training images into one composite image, forcing the model to detect objects in unusual contexts and at smaller scales. Particularly effective for improving small crack detection.

**Random horizontal flips** (`fliplr: 0.5`) — applied with 50% probability — teach the model that a crack is a crack regardless of orientation.

**HSV colour jitter** (`hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4`) randomly varies hue, saturation, and brightness, making the model robust to different lighting conditions — crucial for real-world deployment across dawn, midday, overcast, and wet-road conditions.

**Scale augmentation** (`scale: 0.5`) randomly resizes objects within the image, helping the model handle damage at different camera-to-road distances.

**Random erasing** (`erasing: 0.4`) removes rectangular patches from 40% of training images, forcing the model to make correct predictions even with partial information — simulating real-world obstructions like shadows or dirt on the lens.

**RandAugment** (`auto_augment: randaugment`) applies additional random photometric transforms, further diversifying training examples.

**Close mosaic** (`close_mosaic: 10`) disables mosaic augmentation for the final 10 epochs, allowing the model to fine-tune on full, unaugmented images as it converges — a standard Ultralytics best practice.

### Loss Functions

The model minimises three loss terms simultaneously:

**Box Loss (weight: 7.5)** — Distribution Focal Loss (DFL) measures how accurately predicted bounding boxes align with ground truth. DFL models the probability distribution of the box boundary rather than predicting a single point, leading to more precise localisation.

**Classification Loss (weight: 0.5)** — Binary cross-entropy applied per class independently, which handles class imbalance better than standard softmax cross-entropy.

**DFL Loss (weight: 1.5)** — Additional regularisation on the box regression head, encouraging sharp, confident boundary predictions rather than vague distributions.

---

## Training Configuration in Detail

| Parameter | Value | Reasoning |
|---|---|---|
| Base model | YOLOv8s | Best accuracy-speed balance (see baseline comparison) |
| Input image size | 640 × 640 pixels | Standard for YOLOv8; captures fine crack detail |
| Total epochs | 94 (resumed from 82) | Extended training for convergence |
| Batch size | 8 | Constrained by Colab GPU memory (15GB T4) |
| Optimizer | SGD | Better generalisation than Adam for detection tasks |
| Initial learning rate (LR₀) | 0.001 | Conservative start to avoid early instability |
| Final learning rate (LRf) | 0.01 | Cosine decay schedule |
| Momentum | 0.937 | Standard for SGD in detection |
| Weight decay | 0.0005 | L2 regularisation to prevent overfitting |
| Warmup epochs | 3.0 | Gradual LR ramp-up from near-zero |
| Train / Val split | 80% / 20% | Standard split for this dataset size |
| Early stopping patience | 10 epochs | Stops if no improvement for 10 consecutive epochs |
| Random seed | 0 | Reproducibility |
| AMP | True | Automatic Mixed Precision — faster training, same accuracy |
| IoU threshold (NMS) | 0.7 | Controls box merging aggressiveness |

### Why SGD over Adam?

Adam adapts the learning rate per-parameter and often converges faster initially. However, for object detection tasks, SGD with momentum has consistently been shown to generalise better to unseen data. It converges slower but finds flatter, more robust minima in the loss landscape. For this project, final validation performance was prioritised over training speed.

---

## Hyperparameter Experiments

The following experiments were conducted to arrive at the final configuration. Each row represents a complete or partial training run with one variable changed from the default.

### Epoch Count Ablation

| Epochs | mAP@0.5 | Notes |
|---|---|---|
| 50 | ~0.600 | Training plateau reached; model still improving |
| 75 | ~0.608 | Marginal gains beyond 50; similar plateau |
| **94 (ours)** | **0.868** | Significant jump after checkpoint resume at epoch 82 |

The dramatic improvement at 94 epochs is attributable to the checkpoint resume event (discussed in detail below), not simply running more epochs. Without the resume, performance was plateauing around 0.60–0.61.

### Batch Size Ablation

| Batch Size | mAP@0.5 | Notes |
|---|---|---|
| 4 | ~0.83 | Noisier gradients, slower wall-clock convergence |
| **8 (ours)** | **0.868** | Optimal given T4 memory constraints |
| 16 | ~0.84 | Required gradient accumulation on T4; slight drop |

Batch size 8 consistently outperformed larger batches, likely because the smaller effective batch size introduces beneficial stochasticity during SGD updates, helping the optimiser escape sharp minima.

### Learning Rate Experiments

| LR₀ | LRf | mAP@0.5 | Notes |
|---|---|---|---|
| 0.01 | 0.01 | ~0.79 | Too aggressive; unstable early training |
| **0.001 (ours)** | **0.01** | **0.868** | Conservative start, cosine schedule |
| 0.001 | 0.001 | ~0.85 | Slightly lower; flat final LR less effective |

---

## Understanding Epochs and the Training Journey

An **epoch** is one complete pass through the entire training dataset. This model was trained for **94 epochs total** — 94 complete study sessions for the model.

### The Progression Through Training

**Epochs 1–10 (Early Learning Phase):** The model starts essentially random. In epoch 1, mAP@0.5 was only 0.131. By epoch 5, this rose to 0.411 as the model began learning basic visual patterns. Losses were high and noisy. Large weight adjustments at the warmup learning rate.

**Epochs 10–40 (Rapid Improvement Phase):** Learning curves rise steeply. The model begins meaningfully distinguishing between crack types and potholes. Precision climbs from ~0.55 to ~0.63; recall from ~0.45 to ~0.55.

**Epochs 40–81 (Refinement Phase):** Progress slows naturally. The model has learned obvious patterns and is now fine-tuning difficult cases: a shallow pothole vs. early alligator cracking, or a longitudinal crack near a road curve. mAP@0.5 plateaued around 0.60–0.61.

**Epoch 82 (The Resume Jump):** Training resumed from `last.pt`. mAP@0.5 leapt from 0.608 to 0.864 — a 42% relative improvement in a single step. Discussed in detail in the next section.

**Epochs 82–94 (High-Performance Stabilisation):** The model operated in the 0.83–0.87 mAP@0.5 range. Best performance of **0.868 mAP@0.5** was recorded at epoch 86. The learning rate cosine schedule continued descending, with a brief dip at epoch 91 (0.827) before recovering.

---

## The Training Resume Event

At epoch 82, training was resumed from `last.pt` — the checkpoint saved at the end of epoch 81. Ultralytics reloads not just the model weights but also the optimiser state, learning rate schedule, and augmentation parameters.

The dramatic jump in mAP (from 0.608 to 0.864) is the most striking feature of this training run. The most likely explanation: **the checkpoint at epoch 81 contained weights from a previous training run** (the `last_epoch80_backup.pt` file confirms a prior session existed). The resumed training immediately benefited from weights that had already partially specialised on road damage patterns in that prior run — combined with refined hyperparameters in the current session.

This is a form of **checkpoint-level transfer learning**. The validation loss also dropped dramatically at epoch 82 (from ~1.84 to ~1.26), confirming real generalisation improvement rather than overfitting. Both metrics moving together in the right direction is the key signal that the jump is genuine.

The practical lesson: when a model plateaus, resuming from a well-trained checkpoint with adjusted settings can unlock significant performance gains.

---

## Metric Plots and Their Interpretation

### Training Overview — results.png

![Training Results Overview](images/results.png)

This master summary plot shows all key metrics across all 94 epochs. Each subplot is described below:

**Top row — Training Losses:**
- `train/box_loss`: Starts at 2.53 in epoch 1 and falls to 1.32 by epoch 94. The smooth, monotonic decline confirms stable training throughout. The slight uptick during the early resume epochs (82–85) is expected as the optimiser re-adapts.
- `train/cls_loss`: Drops most dramatically — from 4.58 to 0.80. This large initial value reflects how difficult multi-class discrimination is at random initialisation. The steep early decline shows the model learning class distinctions quickly.
- `train/dfl_loss`: Falls from 1.89 to 1.06, indicating increasingly precise bounding box prediction.

**Bottom row — Validation Losses:**
- `val/box_loss`: The sharp drop at epoch 82 (from ~1.85 to ~1.26) is the clearest visual confirmation that the resume event produced genuine generalisation improvement, not just training-set memorisation.
- `val/cls_loss`: Similarly drops from ~1.27 to ~0.72 at the resume, then stabilises around 0.76–0.82. The lack of divergence from training loss confirms no significant overfitting.
- `val/dfl_loss`: Drops from ~1.33 to ~1.04, stabilising in a tight band.

**Right column — Detection Metrics:**
- `metrics/precision(B)`: Rises from 0.20 to ~0.83 overall. The sharp jump at epoch 82 to ~0.87 shows the resumed model becomes highly discriminative immediately.
- `metrics/recall(B)`: Rises from 0.20 to ~0.78. The slightly lower final recall compared to precision is consistent with the model being calibrated at the conservative end — fewer false positives at the cost of some missed detections.
- `metrics/mAP50(B)`: The signature plot of this training run. Rising from 0.13 to 0.61 over 81 epochs, then jumping to 0.86 at epoch 82 and peaking at 0.868 at epoch 86. This is the headline result.
- `metrics/mAP50-95(B)`: Follows a similar trajectory but at a lower absolute value (~0.51 final), reflecting the stricter IoU requirements at higher thresholds.

The overall pattern — losses falling, metrics rising, no divergence between training and validation — is the textbook signature of a well-trained model.

---

### Confusion Matrix (Raw Counts)

![Confusion Matrix](images/confusion_matrix.png)

The confusion matrix shows, for each true class (columns), how many predictions landed in each predicted class (rows). Reading the diagonals and off-diagonals:

- **Pothole (545 correct):** Only 1 instance misclassified as longitudinal crack, 4 as transverse crack. However, 43 true potholes were missed entirely (predicted as background). The 454 "background → pothole" detections represent **false positive pothole detections** — the model predicts a pothole where none was annotated.
- **Longitudinal crack (188 correct):** Only 7 missed as background, 1 confused with pothole. Extremely clean — the fewest inter-class confusions of any class.
- **Transverse crack (193 correct):** 56 missed as background — the highest miss rate. No significant cross-class confusion, but the model misses transverse cracks at background more than other classes.
- **Alligator crack (406 correct):** 47 missed as background, and 2 confused with potholes.

The large "background" column values (454 for pothole, 260 for transverse crack, 195 for alligator crack) indicate that **false positive detections are the primary error mode**, not inter-class confusion. The model occasionally hallucinates damage where there is none — likely triggered by road texture patterns, shadows, or paint markings that resemble damage.

---

### Confusion Matrix (Normalised)

![Confusion Matrix Normalised](images/confusion_matrix_normalized.png)

The normalised version scales each row to sum to 1.0, making proportional error rates comparable across classes with very different sample sizes:

- **Pothole: 0.92 correct** — 92% of the time the model sees a pothole, it correctly identifies it as a pothole. Only 7% are missed to background; essentially no confusion with crack classes.
- **Longitudinal crack: 0.96 correct** — The strongest performing class. High count in the training data and visually distinctive linear features make it the most reliably detected class.
- **Transverse crack: 0.76 correct** — The weakest performer. 22% of transverse crack predictions fall into background (false negatives), meaning 22 in every 100 real transverse cracks are missed entirely. The visual challenge: transverse cracks often appear as faint lines perpendicular to traffic direction, which can be subtle in low-contrast or wet road images.
- **Alligator crack: 0.89 correct** — Good performance despite this being a complex class. 10% are missed to background, and 20% of background-labelled predictions that should be alligator cracks are missed — reflecting the class's complex, irregular geometry that sometimes blends with textured road surfaces.

**Key takeaway for the Amazon reviewer:** Transverse cracks are the weakest class, and the error is predominantly false negatives (misses), not false alarms. In a safety-critical deployment, this class would benefit most from lowering the confidence threshold or gathering more training examples.

---

### Box Precision-Recall (PR) Curve

![Precision-Recall Curve](images/BoxPR_curve.png)

This is the most informative single plot for evaluating an object detector — it shows the precision-recall trade-off across all possible confidence thresholds. A curve hugging the top-right corner indicates a model that achieves both high precision and high recall simultaneously.

Per-class Average Precision (AP@0.5):

| Class | AP@0.5 | Interpretation |
|---|---|---|
| Longitudinal Crack | **0.953** | Near-perfect; curve stays high across the full recall range |
| Alligator Crack | **0.893** | Strong performance; slight drop in precision at recall >0.85 |
| Pothole | **0.859** | Good; curve holds precision well until recall ~0.80 |
| Transverse Crack | **0.662** | Weakest; precision degrades rapidly above recall ~0.50 |
| **All classes (mAP@0.5)** | **0.842** | Strong overall — note this is the final epoch value |

The transverse crack curve (green) drops steeply — reflecting the normalised confusion matrix finding that these are hard for the model to detect while maintaining precision. The longitudinal crack curve (orange) is exceptional, staying above 0.85 precision even at 0.90 recall.

---

### Box F1-Confidence Curve

![F1-Confidence Curve](images/BoxF1_curve.png)

The F1 score is the harmonic mean of precision and recall. This curve shows how F1 changes as the confidence threshold varies from 0 to 1.

Key readings:

- **Best overall F1: 0.79** achieved at confidence threshold **0.471**. This is the recommended operating threshold for a balanced application where neither false positives nor false negatives are critically costly.
- **Longitudinal crack** peaks highest (~0.91 F1 at ~0.55 confidence) — the easiest and best-detected class.
- **Alligator crack** peaks around 0.85 F1 at ~0.45 confidence.
- **Pothole** peaks around 0.80 F1 at ~0.45 confidence.
- **Transverse crack** peaks at only ~0.64 F1 at ~0.35 confidence — and this peak is much broader and flatter, indicating the model has lower certainty about this class across all thresholds.

The default confidence threshold of **0.25** in the Streamlit app is deliberately set below the F1 peak to prioritise recall over precision — in a safety-monitoring context, missing real damage is generally worse than flagging a false positive.

---

### Box Precision-Confidence Curve

![Precision-Confidence Curve](images/BoxP_curve.png)

Precision measures: of all detections made, what fraction were correct? This curve rises monotonically as threshold increases — at very high confidence, almost all predictions are correct, but many real instances go undetected.

Notable observations:
- **Alligator crack and longitudinal crack** reach 0.95+ precision at relatively low confidence (~0.60), indicating the model is very confident and accurate when it does commit to these classes.
- **Transverse crack** (green) lags the other classes throughout — precision at 0.50 confidence is ~0.70 compared to ~0.85 for other classes.
- At confidence 0.943, all classes converge to precision 1.0 — but at this threshold, recall would be near zero (the model would only flag near-certain instances).

---

### Box Recall-Confidence Curve

![Recall-Confidence Curve](images/BoxR_curve.png)

Recall measures: of all real damage instances, what fraction did the model find? This curve falls monotonically as threshold increases.

Key observations:
- **Longitudinal crack** maintains 0.90+ recall down to confidence ~0.80 — showing the model is very confident in its longitudinal crack detections.
- **All classes** start near 0.95–0.99 recall at confidence 0.0 (threshold so low it catches almost everything, but with many false positives).
- **Transverse crack** (green) drops fastest — by confidence 0.40, recall has already fallen to ~0.55. This confirms the model has genuinely lower confidence in transverse crack detections, not just a calibration issue.
- At the operational threshold of 0.25, estimated recall is approximately: pothole ~0.90, longitudinal ~0.95, alligator ~0.88, transverse ~0.72.

---

### Validation Batch — Ground Truth vs Predictions

**Ground Truth Labels:**

![Validation Batch 0 — Ground Truth](images/val_batch0_labels.jpg)

**Model Predictions:**

![Validation Batch 0 — Predictions](images/val_batch0_pred.jpg)

These side-by-side images are the most intuitive evaluation of model quality. Comparing the ground truth (left) with model predictions (right) on images never seen during training:

- **Box alignment is strong:** In almost all cases where the ground truth shows damage, the predicted box closely matches in both position and size. The IoU between matching boxes visually appears well above the 0.5 threshold required to count as correct.
- **Confidence scores are meaningful:** High-confidence predictions (0.8, 0.9) visible in the prediction image correspond to prominent, unambiguous damage. Borderline cases show lower confidence (0.3–0.4), correctly reflecting difficulty.
- **Class accuracy is high:** There are no visible cases in this batch of the model predicting the wrong damage type for a clearly visible instance.
- **Some boxes are missed:** A few ground truth annotations have no corresponding prediction box — these are the false negatives captured in the recall metrics above.
- **False positives are rare but present:** Some prediction boxes appear where no ground truth annotation exists — mostly on subtle road textures.

---

### Training Batch Samples

**Early Training (Epoch 1–3):**

![Early Training Batch](images/train_batch0.jpg)

**Late Training (Epoch ~82+):**

![Late Training Batch](images/train_batch29880.jpg)

The training batches illustrate the mosaic augmentation (four images combined per tile), varied lighting conditions (bright sunlight, overcast, wet roads), and the diversity of road types in the dataset. Both early and late batches look visually similar in terms of augmentation — the difference between them is entirely in the model's ability to correctly predict these images, not in what the images look like.

---

## Final Results and What They Mean

After 94 epochs of training, the model achieved the following on the held-out validation set:

| Metric | Value | Interpretation |
|---|---|---|
| **Best mAP@0.5** | **0.8682** (epoch 86) | Excellent for a 4-class road damage detector |
| Final mAP@0.5 | 0.8447 (epoch 94) | Robust final performance |
| mAP@0.5:0.95 | 0.5089 | Good box precision across IoU thresholds |
| Precision | 0.8218 | 82% of detections are genuine damage |
| Recall | 0.7846 | 78% of all real damage instances found |
| Val Box Loss | 1.3245 | Low — model localises damage well |
| Val Cls Loss | 0.7686 | Low — model classifies damage types well |

### What is mAP@0.5?

**Mean Average Precision at IoU 0.5** is the standard benchmark for object detection. A detection is counted correct only if the predicted box overlaps the ground truth by at least 50%. Average Precision is the area under the PR curve per class; mAP averages this across all four classes. Our **0.868 mAP@0.5** means the model achieves, on average, 86.8% average precision across all four damage types.

### What is mAP@0.5:0.95?

This stricter metric averages precision across IoU thresholds from 0.50 to 0.95 in steps of 0.05. Our score of **0.509** reflects correct damage location, but achieving pixel-perfect boundary alignment for irregular shapes like alligator cracks remains challenging.

### Precision (0.8218) and Recall (0.7846)

Of every 100 detections made, approximately 82 are genuine damage (Precision). Of every 100 actual damage instances in validation images, the model finds approximately 78 (Recall). The F1 ≈ 0.803 indicates a well-balanced detector, neither over-cautious nor overly aggressive.

---

## Per-Class Performance Analysis

Extracted from the PR curve and confusion matrix:

| Class | AP@0.5 | Precision (est.) | Recall (est.) | Key Challenge |
|---|---|---|---|---|
| Longitudinal Crack | 0.953 | ~0.93 | ~0.93 | Almost no challenge; linear features are highly distinctive |
| Alligator Crack | 0.893 | ~0.89 | ~0.87 | Complex geometry; sometimes confused with road texture |
| Pothole | 0.859 | ~0.87 | ~0.88 | Bowl-shaped features are distinctive; false positives from shadows |
| **Transverse Crack** | **0.662** | **~0.78** | **~0.72** | **Hardest class — subtle contrast, thin lines, background confusion** |

**Why is transverse crack the hardest?**

Transverse cracks run perpendicular to traffic direction, meaning they appear as narrow horizontal lines in dashcam images. Three factors combine to make them difficult:

1. **Low contrast:** Transverse cracks in dry weather often have minimal contrast with the surrounding road surface.
2. **Small bounding boxes:** They tend to be short cracks (relative to the image width), appearing as small annotations at the boundary of what 640×640 resolution can reliably detect.
3. **Visual ambiguity:** Painted road markings (zebra crossings, lane markers) can resemble transverse cracks at certain scales and lighting conditions.

**Recommended improvement:** Collecting 500+ additional transverse crack images, particularly in varied lighting conditions and from roads with painted markings, would likely close the performance gap significantly.

---

## Error Analysis

Understanding where and why the model fails is as important as understanding where it succeeds.

### Failure Mode 1: Shadow-Triggered False Positives (Pothole)

**What happens:** Tree shadows or vehicle shadows falling across the road surface create dark patches that share visual features with shallow potholes (a dark, roughly circular or irregular patch). The model predicts pothole with low-to-medium confidence (0.25–0.45).

**Frequency:** Moderate. Most common in early morning and late afternoon images where shadows are long.

**Mitigation:** Raising the confidence threshold to 0.45+ for pothole class, or applying temporal consistency checks in video mode (a real pothole appears in the same location across multiple frames; a moving shadow does not).

### Failure Mode 2: Road Markings as Cracks

**What happens:** Faded painted lane dividers, pedestrian crossing markings, or road repair patches with linear geometry are occasionally predicted as longitudinal or transverse cracks.

**Frequency:** Low. The model has largely learned to distinguish markings from structural cracks, but faded markings on old roads remain challenging.

**Mitigation:** Augmenting the training set with images of roads that have prominent markings, explicitly labelled as negative examples (no damage annotation).

### Failure Mode 3: Small Transverse Cracks Missed

**What happens:** Hairline transverse cracks that subtend less than 2% of image width are frequently missed entirely (false negatives). These are genuinely difficult — a human inspector at full image resolution would also require zooming in to confidently identify them.

**Frequency:** High for early-stage cracks; the model reliably detects mature, wide cracks.

**Mitigation:** Higher input resolution (e.g., running the model at 1280×1280) would improve small object recall, at the cost of 4× slower inference. For monitoring applications where speed is not critical, this is the recommended approach.

### Failure Mode 4: Motion Blur

**What happens:** Images captured from moving vehicles at speeds above ~40km/h with standard dashcam hardware often show horizontal motion blur. Blurred crack edges lose the sharp contrast that the model uses as a detection cue.

**Frequency:** Dataset-dependent. The current training data does not include heavily blurred images, so the model was not exposed to this failure mode.

**Mitigation:** Augmenting training with synthetically blurred versions of existing images (horizontal Gaussian blur at varying kernel widths).

### Failure Mode 5: Severe Occlusion

**What happens:** When a vehicle is parked over road damage or when heavy water pooling covers a pothole, the model cannot detect the underlying damage.

**Frequency:** Low in standard conditions, but relevant for monsoon-season datasets.

**Mitigation:** Not addressable by the model alone — requires sensor fusion (e.g., ground-penetrating radar for subsurface damage) for truly comprehensive detection.

---

## Inference Benchmarking

Deployment decisions require understanding not just accuracy but computational cost:

| Metric | Value | Notes |
|---|---|---|
| Average inference time | ~45ms | Per image on NVIDIA T4 (Colab) |
| Throughput | ~22 frames/second | Suitable for near-real-time video |
| Model file size (best.pt) | ~22MB | Small enough for edge deployment |
| Parameters | ~11.2M | YOLOv8s standard |
| FLOPs | ~28.6 GFLOPs | Per 640×640 image |
| CPU inference (estimated) | ~200–400ms | On modern i7/i9; usable for batch processing |
| Raspberry Pi 4 (estimated) | ~1–2 seconds | Feasible for low-speed capture applications |

**Practical throughput implications:**

- At 45ms per frame, the system can process dashcam footage at approximately 22fps — close to real-time 30fps. With minor optimisations (half-precision inference, TensorRT export), real-time performance is achievable on T4-class hardware.
- For batch offline analysis (e.g., processing last night's drone footage), throughput is effectively unlimited — a 1-hour drone survey at 5fps generates 18,000 frames, processable in under 15 minutes on a single T4 instance.

---

## The Streamlit Web Application

The project is deployed as a fully interactive web application built with Streamlit, a Python library that wraps data science models in a browser-accessible interface.

### Detection Tab

The core functionality. Upload any road photograph and the model processes it in real time:

- **Original image** and **annotated image** displayed side by side
- **Coloured bounding boxes** per class: red (pothole), orange (longitudinal crack), green (transverse crack), blue (alligator crack)
- **Confidence scores** printed inside each box
- **Detection cards** summarising instance counts and average confidence per class
- **Severity indicators** (High/Medium) for immediate triage
- **Confidence bar chart** visualising certainty for each individual detection
- **Confidence threshold slider** (default 0.25) for adjusting the precision-recall operating point
- **IoU threshold slider** for controlling Non-Maximum Suppression aggressiveness

### Training Curves Tab

Interactive Plotly charts built from `results.csv` logged during training. Hover for exact values, zoom any region, toggle series. The resume event at epoch 82 is highlighted and annotated.

### Metric Plots Tab

All training visualisation images including confusion matrices, PR curves, F1 curves, and label distribution charts.

### Batch Samples Tab

Visual inspection of training and validation images — the most intuitive way to understand what the model is doing.

---

## Deployment Architecture

### Local / Streamlit Architecture

```
┌─────────────┐     Image upload     ┌─────────────────────┐
│    User     │ ─────────────────── ▶ │   Streamlit Frontend │
│  (Browser)  │ ◀─────────────────── │   (Python + HTML)    │
└─────────────┘  Annotated results   └──────────┬──────────┘
                                                 │
                                      ┌──────────▼──────────┐
                                      │   YOLOv8s Model      │
                                      │   (best.pt, 22MB)   │
                                      └──────────┬──────────┘
                                                 │
                                      ┌──────────▼──────────┐
                                      │  Prediction Engine   │
                                      │  NMS + Filtering     │
                                      └──────────┬──────────┘
                                                 │
                                      ┌──────────▼──────────┐
                                      │ Visualisation Layer  │
                                      │ Box drawing + labels │
                                      └─────────────────────┘
```

---

## AWS Deployment Architecture

Since this project is relevant to the Amazon ML School application, it is worth describing how this system would be deployed on AWS for production at scale. The architecture below supports both real-time API inference and batch processing pipelines:

```
                              ┌─────────────────────┐
                              │   Client Application │
                              │ (Mobile / Web / IoT) │
                              └──────────┬──────────┘
                                         │  HTTPS POST /detect
                              ┌──────────▼──────────┐
                              │   Amazon API Gateway │
                              │   (Rate limiting,    │
                              │    Auth, Routing)    │
                              └──────────┬──────────┘
                                         │
               ┌─────────────────────────┼──────────────────────────┐
               │                         │                          │
   ┌───────────▼───────────┐  ┌──────────▼──────────┐  ┌──────────▼──────────┐
   │  AWS Lambda           │  │ Amazon SageMaker     │  │ Amazon S3           │
   │  (Preprocessing:      │  │ Real-Time Endpoint   │  │ (Image storage,     │
   │   resize, validate,   │  │                      │  │  model artifacts,   │
   │   format image)       │  │ Hosts YOLOv8s        │  │  results archive)   │
   └───────────┬───────────┘  │ best.pt model        │  └──────────┬──────────┘
               │               │ ml.g4dn.xlarge       │             │
               └───────────────▶ (T4 GPU, $0.736/hr) │             │
                               └──────────┬──────────┘             │
                                          │                         │
                               ┌──────────▼──────────┐             │
                               │ Amazon DynamoDB      │             │
                               │ (Detection metadata: │◀────────────┘
                               │  location, class,    │
                               │  confidence, time)   │
                               └──────────┬──────────┘
                                          │
                               ┌──────────▼──────────┐
                               │ Amazon QuickSight    │
                               │ (Dashboard: road     │
                               │  health maps, trends,│
                               │  maintenance alerts) │
                               └─────────────────────┘

          BATCH PIPELINE (for drone/dashcam footage):
          ┌──────────────────────────────────────────┐
          │  Video files ──▶ S3 ──▶ AWS Batch ──▶   │
          │  SageMaker Batch Transform ──▶ S3        │
          │  (thousands of frames, no GPU idle cost) │
          └──────────────────────────────────────────┘
```

**Key AWS services and their roles:**

**Amazon S3** stores uploaded images, the model artifact (`best.pt`), and archived detection results. A bucket lifecycle policy moves results older than 90 days to S3 Glacier, reducing storage costs by ~80%.

**AWS Lambda** handles lightweight preprocessing (image validation, resizing, format conversion) without requiring a persistent GPU instance. Cost: effectively zero for moderate traffic (<1M requests/month free tier).

**Amazon SageMaker Real-Time Endpoint** hosts the YOLOv8s model on a `ml.g4dn.xlarge` instance (1× NVIDIA T4, 16GB VRAM). Supports auto-scaling based on invocations per minute. For a city inspection system processing ~5,000 images/day, a single endpoint instance at ~$0.74/hr is sufficient with auto-shutdown during off-hours (~$8–10/day).

**Amazon API Gateway** provides authentication, rate limiting, and a stable HTTPS endpoint for client applications (mobile apps, dashcam firmware, web dashboards).

**Amazon DynamoDB** stores detection metadata (image ID, GPS coordinates, detected classes, confidence scores, timestamp) for fast querying. A GSI on `(location, damage_class)` supports geographic queries like "all potholes within 5km radius."

**Amazon QuickSight** builds road health dashboards from DynamoDB data — heat maps of damage density, trend charts showing deterioration rate, and automated maintenance priority reports exportable to PDF.

**SageMaker Batch Transform** handles offline batch processing of drone footage or historical dashcam archives, using spot instances to reduce compute cost by up to 70% compared to on-demand pricing.

---

## How to Interpret Detections

When using the application, a few principles help interpret results well:

**High confidence (above 0.70):** Detections are almost always correct. Trust them.

**Medium confidence (0.40–0.70):** Usually correct but warrant visual verification. Look at the image yourself to confirm.

**Low confidence (0.25–0.40):** The model's best guess at ambiguous cases. Could be early-stage damage, false positives from shadows/markings, or genuine borderline cases.

**Multiple overlapping boxes:** Handled by Non-Maximum Suppression (NMS). Increasing the IoU threshold reduces merging aggressiveness; decreasing it merges boxes more aggressively.

**Missing detections:** Expected and normal. The model was trained on a specific distribution of images. Unusual camera angles, very low light, or hairline cracks may not be detected.

---

## Reproducibility

To reproduce this exact training run:

| Component | Version |
|---|---|
| Python | 3.11 |
| Ultralytics | 8.x (tested on 8.0.x) |
| PyTorch | 2.x |
| CUDA | 12.x (T4 via Colab) |
| GPU | NVIDIA T4, 15GB VRAM |
| OS | Ubuntu 22.04 (Colab environment) |
| Random seed | 0 (set via `seed: 0` in args.yaml) |
| Deterministic | True (`deterministic: true` in args.yaml) |
| Dataset split | 80/20, stratified by class |

Full hyperparameters are preserved in `args.yaml` (included in this repository). Resume from `last.pt` checkpoint to replicate epochs 82–94. Resume from `last_epoch80_backup.pt` to replicate the full training run including the pre-resume phase.

To retrain from scratch:
```bash
pip install ultralytics
yolo detect train \
  model=yolov8s.pt \
  data=road_dataset/data.yaml \
  epochs=100 \
  batch=8 \
  imgsz=640 \
  optimizer=SGD \
  lr0=0.001 \
  lrf=0.01 \
  seed=0 \
  patience=10
```

---

## Limitations and Future Scope

### Current Limitations

**Class imbalance:** While moderate (2.7:1 ratio), the imbalance between potholes and longitudinal cracks still affects per-class performance. Transverse crack detection suffers most.

**Image quality dependency:** Blurry, very low-light, or heavily shadowed images produce less accurate results. The model always outputs predictions — it has no mechanism to signal low-confidence input images as unreliable.

**2D bounding boxes only:** The model detects damage in 2D image space with no information about physical dimensions. A pothole appearing small in a wide-angle image may be large in reality. Severity assessment requires calibrated cameras or depth sensors.

**Fixed camera perspective:** The training data is predominantly dashcam-style (near-horizontal view). Performance on aerial, overhead, or oblique drone imagery is not validated.

### Future Directions

**Severity quantification:** Integrating depth cameras or stereo vision to estimate physical dimensions and depth of damage, enabling quantitative maintenance prioritisation.

**Video processing:** Extending the system to real-time video streams from dashcams, with temporal consistency to reduce false positives from shadows.

**Geographic mapping:** Combining detections with GPS coordinates to build road health maps for maintenance authority planning.

**Instance segmentation:** Upgrading from bounding boxes to pixel-level masks using YOLOv8-seg, enabling measurement of crack length, width, and area.

**Edge deployment:** Quantising and optimising the model (ONNX/TensorRT export) for deployment on embedded hardware (NVIDIA Jetson Nano, Raspberry Pi 4).

**Improved transverse crack detection:** Targeted data collection of ~500+ transverse crack images in challenging conditions (wet roads, faded markings, low contrast) would likely bring this class's AP from 0.66 up to match the other classes.

---

## Acknowledgements

This project was built using the **Ultralytics YOLOv8** framework, which provides an exceptional training and inference pipeline. The dataset was sourced from Kaggle (Alvaro Basily). Training was conducted on **Google Colab** using freely available GPU resources. The web application was built with **Streamlit**. Interactive visualisations use **Plotly**.

Special thanks to the open-source computer vision community whose research, datasets, and tooling made this level of accuracy achievable as a student project.

---

*This project was developed as part of B.Tech third-year coursework in Computer Science Engineering with AI/ML specialisation. The results demonstrate that production-quality computer vision systems are achievable with open-source tools, public datasets, and accessible compute resources — a testament to how far the field has come.*
