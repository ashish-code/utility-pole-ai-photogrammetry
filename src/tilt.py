"""
Utility Pole Tilt Estimation
=============================
Estimates the tilt angle (degrees from vertical) of a utility pole in a
still image using:
  1. YOLO-World open-vocabulary detection  → pole bounding box
  2. Efficient-SAM instance segmentation   → pixel-precise pole mask
  3. Medial-axis transform (skimage)        → single-pixel skeleton
  4. Hough line transform                  → dominant axis direction
  5. Angle from vertical                   → final tilt measurement

Output per image
----------------
* ``<stem>_tilt.jpg``   – clean annotated image with angle arc overlay
* ``<stem>_analysis.jpg`` – 4-panel methodology figure (detection, mask,
                              skeleton, Hough candidates)

Author: Ashish Gupta
Date:   2024/07/18
"""

from __future__ import annotations

import os
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no plt.show() blocking
import matplotlib.pyplot as plt
from matplotlib import cm

import skimage
import skimage.morphology
from skimage.transform import hough_line, hough_line_peaks
import supervision as sv

from AI import AI


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Palette: pole = blue, badge = red  (kept consistent with diameter.py)
_PALETTE = sv.ColorPalette.from_hex(["#1616f7", "#f71616"])

# Heuristic thresholds (same as diameter.py)
_MIN_AREA_FRAC = 0.04   # pole must be ≥ 4 % of image area
_MAX_AREA_FRAC = 0.75   # pole must be ≤ 75 % of image area

# Annotation colours (BGR for OpenCV)
_WHITE  = (255, 255, 255)
_BLACK  = (10,  10,  10)
_GREEN  = (50,  220,  50)
_YELLOW = (30,  220, 230)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_pole(image_rgb: np.ndarray, ai: AI) -> sv.Detections | None:
    """Run YOLO-World detection and apply pole heuristics.

    Returns the single best pole detection, or None if no valid pole found.
    """
    results = ai.detection_model.predict(
        source=image_rgb,
        conf=ai.confidence_threshold,
        iou=ai.iou_threshold,
        max_det=ai.max_detections,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(results)
    poles = detections[detections.class_id == 0]

    if len(poles) == 0:
        return None

    h, w = image_rgb.shape[:2]
    image_area = h * w

    # height > width (elongated shape)
    box_w = poles.xyxy[:, 2] - poles.xyxy[:, 0]
    box_h = poles.xyxy[:, 3] - poles.xyxy[:, 1]
    poles = poles[box_h > box_w]

    # area within sensible range
    poles = poles[poles.box_area > _MIN_AREA_FRAC * image_area]
    poles = poles[poles.box_area < _MAX_AREA_FRAC * image_area]

    if len(poles) == 0:
        return None

    # keep largest detection
    biggest = max(poles.box_area)
    poles = poles[poles.box_area >= biggest]
    return poles


def _tilt_from_mask(mask: np.ndarray) -> float:
    """Derive tilt angle (degrees from vertical) from a binary pole mask.

    Uses medial-axis thinning followed by the Hough line transform.
    Convention: 0° = perfectly vertical; positive = leans right;
    negative = leans left.
    """
    skeleton = skimage.morphology.medial_axis(mask).astype(np.uint8) * 255

    # Dense angular sampling around vertical (±45°)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False)
    h, theta, d = hough_line(skeleton, theta=tested_angles)
    peaks = hough_line_peaks(h, theta, d, num_peaks=5)

    if len(peaks[0]) == 0:
        return 0.0

    # theta is the angle of the line's NORMAL from the x-axis.
    # For a near-vertical line the normal is near-horizontal → theta ≈ 0.
    # Tilt from vertical = theta in degrees.
    angle_deg = float(np.median(peaks[1]) * 180.0 / np.pi)
    return round(angle_deg, 2)


def _draw_angle_overlay(
    image_bgr: np.ndarray,
    tilt_deg: float,
    pole_box: np.ndarray,
) -> np.ndarray:
    """Overlay tilt angle arc and text on the annotated image.

    Draws:
      * A short vertical reference line in green
      * A coloured line along the detected pole axis
      * An arc between them labelled with the angle
    """
    out = image_bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in pole_box]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    arm = int(min(x2 - x1, y2 - y1) * 0.55)   # length of reference arms

    # Vertical reference line
    cv2.line(out, (cx, cy - arm), (cx, cy + arm), _GREEN, 3, cv2.LINE_AA)

    # Detected pole axis (rotate vertical by tilt_deg)
    angle_rad = math.radians(tilt_deg)
    dx = int(arm * math.sin(angle_rad))
    dy = int(arm * math.cos(angle_rad))
    cv2.line(out, (cx - dx, cy - dy), (cx + dx, cy + dy), _YELLOW, 4, cv2.LINE_AA)

    # Arc between the two lines
    start_angle = -90          # vertical = -90° in OpenCV convention
    end_angle   = -90 + tilt_deg
    cv2.ellipse(
        out, (cx, cy), (arm // 2, arm // 2),
        0, start_angle, end_angle,
        _YELLOW, 3, cv2.LINE_AA,
    )

    # Text label
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thickness  = 3
    label      = f"Tilt: {tilt_deg:+.1f}\u00b0 from vertical"
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Semi-transparent background rectangle
    pad = 10
    tx  = max(x1, 10)
    ty  = max(y1 - th - 2 * pad, 10)
    bg_roi = out[ty : ty + th + 2 * pad, tx : tx + tw + 2 * pad]
    if bg_roi.size > 0:
        white_rect = np.ones_like(bg_roi) * 255
        out[ty : ty + th + 2 * pad, tx : tx + tw + 2 * pad] = cv2.addWeighted(
            bg_roi, 0.45, white_rect, 0.55, 0
        )

    cv2.putText(
        out, label, (tx + pad, ty + th + pad),
        font, font_scale, _BLACK, thickness, cv2.LINE_AA,
    )
    return out


def _save_analysis_figure(
    image_rgb: np.ndarray,
    pole_mask: np.ndarray,
    tilt_deg: float,
    image_name: str,
    output_path: str,
) -> None:
    """Save a 4-panel methodology figure (non-blocking)."""
    skeleton = skimage.morphology.medial_axis(pole_mask).astype(np.uint8) * 255
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False)
    h, theta, d = hough_line(skeleton, theta=tested_angles)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(
        f"Pole Tilt Analysis — {image_name}\nEstimated tilt: {tilt_deg:+.1f}° from vertical",
        fontsize=13,
    )

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("(1) Input image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pole_mask.view(np.uint8), cmap="bone")
    axes[0, 1].set_title("(2) EfficientSAM pole mask")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(np.invert(skeleton), cmap=cm.gray)
    axes[1, 0].set_title("(3) Medial-axis skeleton")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(image_rgb, cmap=cm.gray)
    axes[1, 1].set_ylim((skeleton.shape[0], 0))
    axes[1, 1].set_axis_off()
    axes[1, 1].set_title(f"(4) Hough line candidates — tilt {tilt_deg:+.1f}°")

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=5)):
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        axes[1, 1].axline(
            (x0, y0), slope=np.tan(angle + np.pi / 2),
            color="red", linestyle="-.", linewidth=1.5,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class Tilt:
    """Estimate the tilt angle of a utility pole from a single image."""

    def __init__(self) -> None:
        self._tilt_angle: float = 0.0

    @property
    def tilt_angle(self) -> float:
        return self._tilt_angle

    def compute_tilt(
        self,
        input_image_path: str,
        output_image_path: str,
        ai: AI,
        ground_truth_tilt: float = 0.0,
    ) -> float:
        """Estimate tilt angle and write annotated output images.

        Parameters
        ----------
        input_image_path:
            Path to the input image file.
        output_image_path:
            Destination for the clean annotated image (``_tilt.jpg``).
            The analysis figure is saved alongside with ``_analysis.jpg``.
        ai:
            Loaded AI object (YOLO-World + EfficientSAM).
        ground_truth_tilt:
            Optional ground-truth tilt in degrees (used only for logging).

        Returns
        -------
        float
            Estimated tilt in degrees from vertical.
        """
        image_bgr  = cv2.imread(input_image_path)
        if image_bgr is None:
            print(f"  [WARN] Cannot read image: {input_image_path}")
            return 0.0

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # ── Detection ──────────────────────────────────────────────────────
        poles = _detect_pole(image_rgb, ai)
        if poles is None:
            print(f"  [WARN] No valid pole detected in {Path(input_image_path).name}")
            cv2.imwrite(output_image_path, image_bgr)
            return 0.0

        # ── Segmentation ───────────────────────────────────────────────────
        poles.mask = ai.inference_with_boxes(
            image=image_rgb,
            xyxy=poles.xyxy,
            model=ai.segmentation_model,
            device=ai.device,
        )
        pole_mask = poles.mask[0]

        # ── Tilt angle ─────────────────────────────────────────────────────
        tilt_deg       = _tilt_from_mask(pole_mask)
        self._tilt_angle = tilt_deg

        # ── Annotated clean output image ───────────────────────────────────
        bb_annotator   = sv.BoundingBoxAnnotator(thickness=5, color=_PALETTE)
        mask_annotator = sv.MaskAnnotator(color=_PALETTE, opacity=0.25)
        annotated      = bb_annotator.annotate(scene=image_bgr.copy(), detections=poles)
        annotated      = mask_annotator.annotate(scene=annotated, detections=poles)
        annotated      = _draw_angle_overlay(annotated, tilt_deg, poles.xyxy[0])
        cv2.imwrite(output_image_path, annotated)

        # ── Methodology figure ─────────────────────────────────────────────
        stem            = Path(output_image_path).stem
        parent          = Path(output_image_path).parent
        analysis_path   = str(parent / f"{stem}_analysis.jpg")
        image_name      = Path(input_image_path).name
        _save_analysis_figure(image_rgb, pole_mask, tilt_deg, image_name, analysis_path)

        return tilt_deg


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def run_tilt_batch(
    input_dir: str,
    output_dir: str,
    ai: AI | None = None,
    results_csv: str = "",
) -> list[dict]:
    """Process all JPEG images in *input_dir* and write annotated results.

    Parameters
    ----------
    input_dir:
        Folder containing input ``.jpg`` images.
    output_dir:
        Folder where annotated images and analysis figures are written.
    ai:
        Pre-loaded AI object.  A new one is instantiated if ``None``.
    results_csv:
        If non-empty, a CSV summary is written to this path with columns
        ``filename, tilt_deg``.

    Returns
    -------
    list[dict]
        One dict per image: ``{"filename": ..., "tilt_deg": ...}``.
    """
    if ai is None:
        ai = AI()

    os.makedirs(output_dir, exist_ok=True)
    images = sorted(
        p for p in os.listdir(input_dir)
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not images:
        print(f"[WARN] No images found in {input_dir}")
        return []

    tilt_obj = Tilt()
    rows: list[dict] = []

    for img_name in images:
        in_path  = os.path.join(input_dir, img_name)
        stem     = Path(img_name).stem
        out_path = os.path.join(output_dir, f"{stem}_tilt.jpg")
        print(f"  Processing: {img_name} ...", end="  ")

        tilt_deg = tilt_obj.compute_tilt(in_path, out_path, ai)
        print(f"tilt = {tilt_deg:+.2f}°")
        rows.append({"filename": img_name, "tilt_deg": tilt_deg})

    if results_csv:
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "tilt_deg"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Results saved → {results_csv}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point (single image)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Utility Pole Tilt Estimation — single image mode"
    )
    parser.add_argument("-i", "--input",  default="data/sample/pole_tilt/leaning_pole_01.jpg")
    parser.add_argument("-o", "--output", default="result/pole_tilt/leaning_pole_01_tilt.jpg")
    parser.add_argument("-t", "--tilt",   type=float, default=0.0,
                        help="Ground-truth tilt in degrees (optional, for logging)")
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    ai   = AI()
    t    = Tilt()
    deg  = t.compute_tilt(args.input, args.output, ai, args.tilt)
    print(f"Pole tilt: {deg:+.2f}° from vertical")
