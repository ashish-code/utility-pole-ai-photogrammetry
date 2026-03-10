# Model Weights

Large model weight files are **not tracked in this repository** (`.gitignore`).
Download them before running any inference scripts.

---

## YOLO-World — `yolov8x-world.pt`

Open-vocabulary object detection model used for pole and badge detection.

```bash
# Automatic download via ultralytics on first use (recommended):
python - <<'EOF'
from ultralytics import YOLOWorld
YOLOWorld("yolov8x-world.pt")   # downloads to current directory
EOF

# Or with the ultralytics CLI:
yolo detect model=yolov8x-world.pt
```

The model file should be placed in the **repository root** (same directory as
`pyproject.toml`).

---

## EfficientSAM — `efficient_sam_s_cpu.jit` / `efficient_sam_s_gpu.jit`

Lightweight SAM variant used for instance segmentation masks.

```bash
# CPU version (~101 MB)
wget "https://huggingface.co/spaces/SkalskiP/YOLO-World/resolve/main/efficient_sam_s_cpu.jit"

# GPU version (~101 MB)
wget "https://huggingface.co/spaces/SkalskiP/YOLO-World/resolve/main/efficient_sam_s_gpu.jit"
```

Both files should be placed in the **repository root**.  `AI.py` automatically
downloads whichever variant is needed if the file is not present.

---

## Quick setup (all models)

```bash
# From the repo root:
wget "https://huggingface.co/spaces/SkalskiP/YOLO-World/resolve/main/efficient_sam_s_cpu.jit"
python -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8x-world.pt')"
```

---

## Hardware notes

| Variant | VRAM | Speed (RTX 3090) | Speed (CPU) |
|---------|------|-------------------|-------------|
| `efficient_sam_s_gpu.jit` | ~2 GB | ~0.3 s/frame | — |
| `efficient_sam_s_cpu.jit` | none  | — | ~4–8 s/frame |

`AI.py` selects the GPU variant automatically when CUDA is available.
