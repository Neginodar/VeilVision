
# VeilVision 🎭
### Privacy-Preserving Anonymization for Dashcam & CCTV Footage

> Automatically detects and anonymizes **faces** and **license plates** in images and videos to ensure GDPR compliance before sharing or storing footage.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-green)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What It Does

VeilVision is a local, offline privacy anonymization pipeline — similar to what Google Street View uses to blur faces and plates, but fully controllable and customizable.

Given a dashcam or CCTV video, VeilVision:
1. Detects **faces** using YOLOv8-nano (fine-tuned face model) with multi-scale inference
2. Detects **license plates** using a dedicated YOLOv8 plate detector + EasyOCR confirmation
3. **Blurs faces** and **pixelates plates** in every frame
4. Exports a clean anonymized `.mp4` — safe to share publicly

---

## Two Versions

This repository contains two notebooks representing a progression from classical to modern CV:

| Notebook | Detection | Speed | Recall |
|----------|-----------|-------|--------|
| `VeilVision_Classical.ipynb` | Haar Cascades + Morphology | ~150 FPS | Lower |
| `VeilVision_YOLOv8.ipynb` | YOLOv8-nano + ByteTrack + EasyOCR | ~5–18 FPS | Higher |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.12 |
| Detection | YOLOv8-nano (Ultralytics) |
| Tracking | ByteTrack (built into Ultralytics) |
| OCR Confirmation | EasyOCR |
| Computer Vision | OpenCV 4.x |
| Numerical | NumPy |
| Visualization | Matplotlib |
| Demo UI | Gradio |

---

## Pipeline

```
Input Video / Image
        ↓
Frame Extraction
        ↓
┌──────────────────────────────────┐
│   Multi-Scale Detection          │
│   ├── Face Detection (YOLOv8)    │
│   │   ├── Original resolution    │
│   │   └── 1.5x upscale (distant) │
│   └── Plate Detection (YOLOv8)   │
│       └── EasyOCR confirmation   │
└──────────────────────────────────┘
        ↓
Anonymization
  ├── Faces  → Gaussian Blur
  └── Plates → Pixelation
        ↓
Output Anonymized Video (.mp4)
```

---

## Project Structure

```
VeilVision/
├── VeilVision_YOLOv8.ipynb        # Main notebook (deep learning)
├── VeilVision_Classical.ipynb     # Classical CV baseline notebook
├── Videos/                        # Input videos folder
├── output/                        # Anonymized output videos
├── yolov8n-face.pt                # Face detection model (auto-downloaded)
└── license_plate_detector.pt      # Plate detection model (auto-downloaded)
```

---

## Setup

### Install Dependencies

```bash
pip install ultralytics easyocr lapx gradio opencv-python numpy matplotlib
```

### Models

Both models download automatically on first run:
- **Face model** — `arnabdhar/YOLOv8-Face-Detection` (HuggingFace)
- **Plate model** — `Koushim/yolov8-license-plate-detection` (HuggingFace)

---

## Usage

### Single Image

```python
img = cv2.imread("your_image.jpg")
annotated, anonymized, meta = process_image(img, cfg)
show_before_after(img, anonymized)
```

### Single Video

```python
cfg = VeilConfig(
    face_roi_fraction=1.0,
    plate_roi_fraction=1.0,
    enhance_background=False,
    detect_every_n=1,
    resize_width=0,           # 0 = keep original resolution
)

stats, frame_times, previews = process_video(
    "Videos/input.MOV",
    "output/input_out.mp4",
    cfg,
    max_frames=None           # None = full video
)
```

### Batch Processing

```python
VIDEO_FOLDER  = "Videos/"
OUTPUT_FOLDER = "output/"

video_files = sorted([
    f for f in os.listdir(VIDEO_FOLDER)
    if f.endswith(('.mp4', '.avi', '.MOV', '.mkv'))
])[:20]  # process first 20

for filename in video_files:
    input_path  = os.path.join(VIDEO_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"{Path(filename).stem}_out.mp4")
    process_video(input_path, output_path, cfg, max_frames=None)
```

### Gradio Demo (Local + Public URL)

```python
demo.launch(share=True)  # generates a public URL instantly
```

---

## Configuration

All parameters are controlled via `VeilConfig`:

```python
cfg = VeilConfig(
    # Detection
    face_roi_fraction=1.0,     # Fraction of frame to search for faces
    plate_roi_fraction=1.0,    # Fraction of frame to search for plates
    detect_every_n=1,          # Run detection every N frames

    # Anonymization
    face_mode="blur",          # "blur" or "pixelate"
    plate_mode="pixelate",     # "blur" or "pixelate"
    blur_ksize=(51, 51),       # Gaussian blur kernel size
    pixel_size=12,             # Pixelation block size

    # Video
    resize_width=0,            # 0 = keep original resolution

    # Background
    enhance_background=False,  # CLAHE sharpening on non-anonymized areas
)
```

---

## Evaluation Results

Tested on the **INDRA Indian Road Crossing Dataset**
(104 videos, 26,000+ frames, pedestrian POV, Indian street footage):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Precision | 1.0 | No false positives |
| Recall | 0.73 | Catches 3 in 4 faces |
| F1 | 0.845 | Strong overall score |

> **Note:** For privacy tools, **Recall is the critical metric**.
> Missing a face (FN) = GDPR risk.
> Over-blurring (FP) = cosmetic issue only.
> Production deployments should use `conf=0.25` to maximize Recall.

### Detection Quality by Scene Type

| Scene Type | Quality |
|------------|---------|
| Close faces (< 3m) | ✅ Near-perfect |
| Mid-range faces (3–10m) | ✅ Good |
| Distant faces (> 10m) | ⚠️ Challenging |
| Crowded scenes (7–15 faces/frame) | ⚠️ Some missed |

### Performance (CPU, MacBook, 1920×1080)

| Frame Skip | Avg ms/frame | Avg FPS |
|------------|-------------|---------|
| Every frame | ~224ms | ~4.5 FPS |
| Every 2nd | ~110ms | ~9 FPS |
| Every 4th | ~55ms | ~18 FPS |
| Every 8th | ~31ms | ~32 FPS |

---

## Key Technical Features

### Multi-Scale Face Detection
Runs YOLOv8 at both **original resolution** and **1.5× upscale** to catch
distant small faces, then merges results with NMS deduplication.

### ByteTrack Temporal Tracking
Uses ByteTrack (built into Ultralytics) to maintain consistent bounding boxes
across frames, eliminating flickering and improving coverage between detection frames.

### EasyOCR Plate Confirmation
Before blurring, runs OCR on each detected plate region. Only blurs if
the crop contains ≥3 alphanumeric characters — eliminates false positives
like timestamps and road signs.

### Hybrid Classical Pipeline (Classical Notebook)
The classical notebook uses a multi-stage pipeline:
- Bilateral filtering → Canny edge detection → Morphological closing →
  Contour extraction → Aspect ratio filtering → Texture variance check

---

## Known Limitations & Roadmap

| Limitation | Status |
|------------|--------|
| Distant faces in crowds | ✅ Multi-scale inference implemented |
| Flickering boxes | ✅ ByteTrack tracking implemented |
| Plate false positives | ✅ EasyOCR confirmation implemented |
| CPU-only (~5 FPS at 1080p) | 🔲 ONNX Runtime / TensorRT (planned) |
| Not fine-tuned on Indian plates | 🔲 Fine-tune on CCPD/INDRA (planned) |
| No audit logging | 🔲 Production logging (planned) |

---

## GDPR Notes

- **Pixelation** is preferred over blur for plates (harder to reverse-engineer)
- Faces are blurred with a large Gaussian kernel (51×51 default)
- Real deployments should store audit logs of what was anonymized
- This project demonstrates a complete GDPR-aware anonymization pipeline prototype

---

## Dataset

Tested on the [INDRA — Indian Dataset for Road Crossing](https://www.kaggle.com/datasets/siddhi17/road-crossing-dataset):

- 104 videos, 26,000+ frames, 1.79 GB
- Pedestrian head-mounted camera POV
- Real Indian street traffic (Anand, Gujarat, 2019)
- Challenging: crowded scenes, helmeted riders, small distant faces

---

## Requirements

```
ultralytics>=8.0
easyocr>=1.7
lapx>=0.9
gradio>=4.0
opencv-python>=4.6
numpy>=1.23
matplotlib>=3.3
```

---

## License

MIT — free to use, modify and distribute with attribution.

---

## Author

Built as part of an MSc in Artificial Intelligence project
on privacy-preserving computer vision.

---

*VeilVision — because privacy should be automatic.*
```

To save this as a file, copy everything between the triple backticks and save it as `README.md` in your project root. Then on GitHub it will render automatically.
