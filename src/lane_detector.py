# src/lane_detector.py
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np


# ---------------------- ROI utilities ----------------------
def make_trapezoid_roi_mask(shape_hw, top_ratio=0.60, top_width_ratio=0.20, bottom_width_ratio=1.00):
    """
    Build a trapezoid ROI mask for a given (height, width).
    Only pixels inside the trapezoid remain; others are masked out.

    Args:
        shape_hw: Tuple[int, int] -> (height, width)
        top_ratio: float in [0..1], vertical position of trapezoid top (0=top, 1=bottom).
                   Larger value -> lower "horizon".
        top_width_ratio: float in [0..1], relative width of the trapezoid at the top.
        bottom_width_ratio: float in [0..1], relative width at the bottom (usually 1.0).
    Returns:
        mask: np.uint8 mask (H x W), 255 inside ROI, 0 outside.
        top_y: int pixel row where the top edge of the trapezoid lies.
    """
    h, w = shape_hw
    top_y = int(h * top_ratio)
    bottom_y = h - 1

    cx = w // 2
    half_top_w = int((w * top_width_ratio) / 2)
    half_bot_w = int((w * bottom_width_ratio) / 2)

    # Vertices of the trapezoid (clockwise order)
    pts = np.array([
        [cx - half_bot_w, bottom_y],
        [cx + half_bot_w, bottom_y],
        [cx + half_top_w, top_y],
        [cx - half_top_w, top_y],
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, top_y


# ---------------------- core pipeline ----------------------
def detect_lanes_frame(
    frame,
    roi_mask,
    min_angle_deg=20.0,
    canny_low=80,
    canny_high=160,
    hough_rho=1,
    hough_theta=np.pi / 180,
    hough_thresh=50,
    hough_min_len=40,
    hough_max_gap=60
):
    """
    Detect lanes on a single frame and draw them within the ROI only.

    Steps:
      1) Preprocess: grayscale -> Gaussian blur -> Canny.
      2) Apply ROI mask to edges so Hough runs only on the road area.
      3) HoughLinesP to get line segments.
      4) Filter out nearly-horizontal segments by angle.
      5) Draw lines on a separate layer, mask the layer with the same ROI, blend over the original.

    Args:
        frame: BGR frame (H x W x 3)
        roi_mask: np.uint8 mask (H x W) with 255 inside ROI
        min_angle_deg: discard lines with absolute angle < min_angle_deg
        canny_low, canny_high: Canny thresholds
        hough_*: parameters for cv2.HoughLinesP

    Returns:
        output frame with lane lines drawn.
    """
    # 1) Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    # 2) Mask edges with ROI so Hough sees only the road area
    edges_roi = cv2.bitwise_and(edges, roi_mask)

    # 3) Hough on ROI-only edges
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=hough_rho,
        theta=hough_theta,
        threshold=hough_thresh,
        minLineLength=hough_min_len,
        maxLineGap=hough_max_gap
    )

    # 4) Draw results on a separate layer
    line_layer = np.zeros_like(frame)

    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dy, dx = (y2 - y1), (x2 - x1)
            if dx == 0:
                slope_deg = 90.0
            else:
                slope_deg = abs(np.degrees(np.arctan2(dy, dx)))

            # Remove nearly-horizontal segments
            if slope_deg < min_angle_deg:
                continue

            cv2.line(line_layer, (x1, y1), (x2, y2), (0, 220, 255), 3)

    # 5) Mask the line layer with the same ROI to guarantee no drawing above "horizon"
    roi_mask_bgr = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    line_layer = cv2.bitwise_and(line_layer, roi_mask_bgr)

    # Blend with original frame
    out = cv2.addWeighted(frame, 1.0, line_layer, 0.8, 0.0)
    return out


def process_video(input_path: Path, output_path: Path, args, show_progress=True):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare ROI mask once per video (much faster than recomputing per-frame)
    roi_mask, top_y = make_trapezoid_roi_mask(
        (height, width),
        top_ratio=args.roi_top,
        top_width_ratio=args.roi_top_width,
        bottom_width_ratio=args.roi_bottom_width
    )

    # mp4 writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[ERROR] Cannot create writer: {output_path}")
        cap.release()
        return False

    print(
        f"[START] {input_path.name}  | {width}x{height} @ {fps:.1f}fps | frames: {total if total else 'unknown'}"
    )
    print(
        f"[INFO] ROI: top={args.roi_top:.2f}, top_width={args.roi_top_width:.2f}, "
        f"bottom_width={args.roi_bottom_width:.2f} -> top_y={top_y}px\n"
    )

    frame_idx = 0
    ok = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = detect_lanes_frame(
            frame,
            roi_mask=roi_mask,
            min_angle_deg=args.min_angle_deg,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            hough_rho=1,
            hough_theta=np.pi / 180,
            hough_thresh=50,
            hough_min_len=40,
            hough_max_gap=60
        )
        writer.write(out)

        frame_idx += 1
        if show_progress and (frame_idx % 25 == 0 or (total and frame_idx % max(1, total // 100) == 0)):
            if total:
                pct = (frame_idx / total) * 100
                print(f"  - {input_path.name}: {frame_idx}/{total} ({pct:5.1f}%)")
            else:
                print(f"  - {input_path.name}: {frame_idx} frames processed")

    cap.release()
    writer.release()

    if total and frame_idx < max(1, total * 0.2):
        print(f"[WARN] Read too few frames ({frame_idx}/{total}). Output may be invalid.")
        ok = False

    print(f"[DONE] {input_path.name} -> {output_path.name} | written {frame_idx} frames\n")
    return ok


# ---------------------- CLI / entrypoint ----------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Lane detection batch processor (process all videos in a folder)."
    )
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with input videos")
    ap.add_argument("--output_dir", type=str, required=True, help="Folder for processed videos")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # ROI tuning (trapezoid). Defaults are conservative and avoid drawing above mid-frame.
    ap.add_argument("--roi-top", dest="roi_top", type=float, default=0.60,
                    help="Top boundary of ROI as a fraction of frame height (0..1). Larger -> lower horizon.")
    ap.add_argument("--roi-top-width", dest="roi_top_width", type=float, default=0.20,
                    help="Relative trapezoid width at the top (0..1 of frame width).")
    ap.add_argument("--roi-bottom-width", dest="roi_bottom_width", type=float, default=1.00,
                    help="Relative trapezoid width at the bottom (0..1). Usually 1.0.")

    # Basic edge/line filtering knobs (optional, keep sane defaults)
    ap.add_argument("--min-angle-deg", dest="min_angle_deg", type=float, default=20.0,
                    help="Discard line segments with absolute angle smaller than this (in degrees).")
    ap.add_argument("--canny-low", dest="canny_low", type=int, default=80, help="Canny lower threshold.")
    ap.add_argument("--canny-high", dest="canny_high", type=int, default=160, help="Canny upper threshold.")

    return ap.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists():
        print(f"[ERROR] Input dir not found: {in_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common video extensions
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}
    videos = [p for p in sorted(in_dir.iterdir()) if p.suffix in video_exts]

    if not videos:
        print(f"[WARN] No videos found in: {in_dir}")
        sys.exit(0)

    print(f"[INFO] Found {len(videos)} file(s). Output dir: {out_dir}\n")

    processed = 0
    for v in videos:
        out_name = v.stem + "_lanes.mp4"
        out_path = out_dir / out_name

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {v.name} -> exists ({out_name})")
            continue

        ok = process_video(v, out_path, args, show_progress=True)
        processed += int(ok)

    print(f"[SUMMARY] Done. Processed: {processed}/{len(videos)}")


if __name__ == "__main__":
    # Fallback for launching via PyCharm/VSCode “Run” without CLI args
    if len(sys.argv) == 1:
        sys.argv += [
            "--input_dir", "data/input",
            "--output_dir", "data/output",
        ]
        print("[INFO] No CLI args detected. Using defaults: --input_dir data/input --output_dir data/output\n")
    main()
