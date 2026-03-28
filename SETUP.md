# DART Local Setup Guide

> **Machine**: Windows 11, NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)
> **Author**: Khalid Amoura (github.com/Khalidfpv)
> **Date**: March 2026

---

## What is This?

This is a local deployment of **DART (Detect Anything in Real Time)** — a framework that enables real-time open-vocabulary object detection using Meta's SAM3 vision model. You type any object name ("person", "car", "chair") and it detects them live from your webcam or video files.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DART Pipeline                            │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Input    │───>│  ViT-H       │───>│  Encoder-Decoder      │  │
│  │  Frame    │    │  Backbone    │    │  (per-class scoring)  │  │
│  │ (webcam/  │    │  (runs ONCE  │    │  Outputs: boxes,      │  │
│  │  video)   │    │   for ALL    │    │  scores, class labels │  │
│  └──────────┘    │   classes)   │    └───────────────────────┘  │
│                  └──────────────┘                                │
│                                                                 │
│  Text Prompt ──> Text Encoder ──> Class Embeddings (cached)     │
│  "person"        (runs once at startup, milliseconds)           │
│  "car"                                                          │
│  "chair"                                                        │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: SAM3's backbone processes images independently of text prompts. DART exploits this by running the backbone once and sharing features across all classes — reducing cost from O(N) to O(1).

---

## Prerequisites

| Requirement        | Version        | Status  |
|--------------------|----------------|---------|
| Windows 11         | 10.0.26200     | Done    |
| NVIDIA GPU         | RTX 4060 8GB   | Done    |
| NVIDIA Driver      | Latest         | Done    |
| Miniconda          | py313_26.1.1   | Done    |
| Conda env          | dartsam3       | Done    |
| Python             | 3.11.15        | Done    |
| PyTorch            | 2.10.0+cu126   | Done    |
| TensorRT           | 10.16.0.72     | Done    |
| DART               | 0.1.0 (editable)| Done   |
| HuggingFace login  | Authenticated  | Done    |
| SAM3 model access  | Granted        | Done    |
| TRT backbone engine| hf_backbone_fp16.engine | Done |
| TRT enc-dec engine | enc_dec_fp16.engine     | Done |

---

## Installation (Already Completed)

These steps have already been run. Documented here for reproducibility.

### 1. Miniconda
```bash
winget install Anaconda.Miniconda3
# Location: %USERPROFILE%\miniconda3
```

### 2. Conda Environment
```bash
conda create -n dartsam3 python=3.11 -y
conda activate dartsam3
```

### 3. PyTorch + CUDA
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4. TensorRT
```bash
pip install tensorrt
```

### 5. DART
```bash
cd path\to\DART
pip install -e .
```

### 6. HuggingFace Authentication
```bash
python -c "from huggingface_hub import login; login(token='hf_YOUR_TOKEN')"
```

### 7. SSL Fix (Conda-specific)
Conda sets `SSL_CERT_FILE` to a non-existent path. Fixed by copying certifi's certificate:
```bash
mkdir -p %USERPROFILE%\miniconda3\envs\dartsam3\ssl
cp %USERPROFILE%\miniconda3\envs\dartsam3\Lib\site-packages\certifi\cacert.pem \
   %USERPROFILE%\miniconda3\envs\dartsam3\ssl\cacert.pem
```

---

## First-Time Setup (Completed)

SAM3 access was granted on 2026-03-28. TRT engines have been built.

### To rebuild engines (only needed if you change GPU or TensorRT version):
Double-click **`build_trt_engines.bat`** or run manually:
```bash
conda activate dartsam3
cd path\to\DART
set PYTHONIOENCODING=utf-8

python scripts/export_hf_backbone.py --imgsz 1008
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt --output enc_dec.onnx --max-classes 4 --imgsz 1008
python -m sam3.trt.build_engine --onnx enc_dec.onnx --output enc_dec_fp16.engine --fp16 --mixed-precision none
```

---

## Usage

### Live Webcam Detection
```bash
python live_detect.py --source 0 --classes person chair laptop cup bottle
```

### Pre-Recorded Video
```bash
python live_detect.py --source path/to/video.mp4 --classes person car chair
```

### With ByteTrack Object Tracking
```bash
python live_detect.py --source 0 --classes person car --track
```

### Save Annotated Output
```bash
python live_detect.py --source office.mp4 --classes person desk --save output.mp4
```

### All 80 COCO Classes
```bash
python live_detect.py --source 0 --coco
```

### With TensorRT Acceleration (After Engine Build)
```bash
python live_detect.py --source 0 \
    --classes person chair laptop cup \
    --trt hf_backbone_fp16.engine \
    --trt-enc-dec enc_dec_fp16.engine
```

---

## Keyboard Controls

| Key       | Action                          |
|-----------|---------------------------------|
| `q` / ESC | Quit                           |
| `p`       | Pause / Resume                  |
| `s`       | Screenshot (saves as .jpg)      |

---

## File Structure

```
path\to\DART\
├── README.md                  # Original DART readme (upstream)
├── SETUP.md                   # This file — local setup documentation
├── live_detect.py             # Custom live webcam/video detection script
├── setup_and_run.bat          # One-click: build engines + run webcam
├── build_trt_engines.bat      # One-click: build TRT engines only
├── run_webcam.bat             # One-click: webcam detection
├── run_video.bat              # Drag-and-drop: video detection
├── demo_multiclass.py         # DART's image detection demo (upstream)
├── demo_video.py              # DART's video detection demo (upstream)
├── pyproject.toml             # Package config (upstream)
├── sam3/                      # SAM3 model code (upstream)
│   ├── model_builder.py       #   Model loading & checkpoint download
│   ├── model/                 #   Neural network definitions
│   ├── trt/                   #   TensorRT export & engine building
│   ├── tracking.py            #   ByteTrack multi-object tracker
│   ├── video_pipeline.py      #   CUDA-pipelined video processing
│   └── coco_classes.py        #   80 COCO class names
├── scripts/                   # Build & export scripts (upstream)
│   ├── export_hf_backbone.py  #   HF backbone → ONNX → TRT FP16
│   ├── export_student_trt.py  #   Student backbone → TRT
│   ├── block_pruner_search.py #   ViT block pruning search
│   └── distill.py             #   Self-distillation training
└── (generated after engine build)
    ├── hf_backbone_fp16.engine        # TRT backbone engine
    └── enc_dec_fp16.engine            # TRT encoder-decoder engine
```

---

## Expected Performance (RTX 4060 Laptop, 8 GB VRAM)

| Mode                         | FPS (est.) | VRAM   | Notes                    |
|------------------------------|------------|--------|--------------------------|
| Full ViT-H + TRT FP16       | 10–15      | ~4–6 GB| Best accuracy (55.8 AP)  |
| Full ViT-H + torch.compile  | 5–8        | ~4–6 GB| No TRT build needed      |
| Pruned backbone + TRT       | 25–30      | ~3–4 GB| -2.2 AP loss             |
| Distilled student + TRT     | 40–50      | ~2–3 GB| Lighter, 38.7 AP         |

If VRAM is tight, reduce resolution: `--imgsz 768` (must be divisible by 14).

---

## Troubleshooting

### "GatedRepoError: 403 — awaiting review"
Meta hasn't approved your SAM3 access. Check https://huggingface.co/facebook/sam3.
(Resolved 2026-03-28 — access granted.)

### Unicode errors on Windows
Always set `PYTHONIOENCODING=utf-8` before running scripts. The batch files handle this automatically.

### Out of VRAM
- Reduce resolution: `--imgsz 768` or `--imgsz 504`
- Use fewer classes (each class adds minimal VRAM)
- Use a distilled student backbone instead of full ViT-H

### TRT engine rebuild needed
TRT engines are GPU-architecture-specific. If you change GPUs, delete `*.engine` files and rebuild.

---

## References

- **DART Repository**: https://github.com/mkturkcan/DART
- **DART Paper**: https://arxiv.org/abs/2603.11441
- **DART Weights (HuggingFace)**: https://huggingface.co/mehmetkeremturkcan/DART
- **SAM3 Model (Meta)**: https://huggingface.co/facebook/sam3
- **TensorRT**: https://developer.nvidia.com/tensorrt
