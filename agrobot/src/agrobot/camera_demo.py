"""Real-time Live Camera Demo & HUD Triage for AgroBot (Phase 9).

Features:
1. Live webcam video streaming at native frame rates.
2. Real-time pathology classification with temperature-scaled confidence.
3. Out-Of-Distribution (OOD) energy-based abstention detection.
4. Rich interactive HUD: Top-3 predictions, confidence meters, disease warning banners.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.infer import AgroBotPredictor, PredictionResult
from agrobot.paths import REPORTS_DIR, RUNS_DIR


def draw_hud(
    frame: np.ndarray,
    result: PredictionResult | None,
    fps: float,
    model_name: str = "Model A (ConvNeXt-Tiny)",
) -> np.ndarray:
    """Render the AgroBot diagnostic HUD onto the video frame."""
    h, w = frame.shape[:2]
    canvas = frame.copy()

    # Semi-transparent top and bottom overlay bars
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 160), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # 1. Top Header: AgroBot Live Triage & FPS
    cv2.putText(
        canvas,
        f"AgroBot Live Diagnostic HUD | {model_name}",
        (20, 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (0, 255, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"FPS: {fps:.1f}",
        (w - 140, 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if result is None:
        cv2.putText(
            canvas,
            "Initializing model pipeline...",
            (20, h - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return canvas

    # 2. Status Banner (Center Top)
    if result.abstain:
        status_color = (0, 180, 255)  # Orange / Yellow warning
        status_text = "ABSTAIN: NON-LEAF / LOW CONFIDENCE"
        sub_text = result.abstain_reason or "Object rejected by open-set energy filter"
    elif "healthy" in result.predicted_class.lower():
        status_color = (50, 220, 50)  # Bright Green
        status_text = "HEALTHY CROP DETECTED"
        sub_text = f"Class: {result.predicted_class}"
    else:
        status_color = (40, 50, 240)  # Red / Amber
        status_text = "PATHOLOGY DETECTED"
        sub_text = f"Disease: {result.predicted_class}"

    cv2.putText(
        canvas,
        status_text,
        (20, 65),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # 3. Bottom Panel: Top-3 Prediction Meters
    y_start = h - 130
    for idx, (cls_name, conf) in enumerate(result.top_k[:3]):
        y_pos = y_start + idx * 38
        clean_name = cls_name.replace("___", " - ").replace("__", " ").replace("_", " ")

        # Class Label
        cv2.putText(
            canvas,
            f"{idx + 1}. {clean_name[:32]}",
            (20, y_pos + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # Confidence Bar
        bar_x = 320
        bar_max_w = w - 440
        bar_w = int(bar_max_w * conf)
        bar_color = status_color if idx == 0 else (140, 140, 140)

        cv2.rectangle(canvas, (bar_x, y_pos), (bar_x + bar_max_w, y_pos + 16), (60, 60, 60), -1)
        if bar_w > 0:
            cv2.rectangle(canvas, (bar_x, y_pos), (bar_x + bar_w, y_pos + 16), bar_color, -1)

        # Confidence Text
        cv2.putText(
            canvas,
            f"{conf:.1%}",
            (bar_x + bar_max_w + 12, y_pos + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Energy Score Info
    cv2.putText(
        canvas,
        f"Free Energy: {result.energy_score:.2f} | Controls: 'q'=Quit, 's'=Save Snapshot",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return canvas


def run_camera_demo(
    camera_id: int = 0,
    checkpoint: Path = RUNS_DIR / "model_a_convnext" / "best_model.pt",
    device: str = "cuda",
) -> None:
    """Launch the real-time webcam demonstration."""
    print(f"Loading AgroBot Predictor from {checkpoint}...")
    predictor = AgroBotPredictor(checkpoint_path=checkpoint, device=device)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return

    print("Camera opened successfully. Press 'q' or ESC in the window to exit.")
    window_name = "AgroBot Real-Time Pathology Triage"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    t_prev = time.time()
    fps_smooth = 0.0
    last_result: PredictionResult | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            fps_current = 1.0 / max(dt, 1e-4)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_current if fps_smooth > 0 else fps_current

            # Predict on current frame
            try:
                last_result = predictor.predict(frame)
            except Exception as e:
                print(f"Prediction error: {e}")

            # Draw HUD
            hud_frame = draw_hud(frame, last_result, fps_smooth)
            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                snap_dir = REPORTS_DIR / "captures"
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_path = snap_dir / f"capture_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), hud_frame)
                print(f"Saved snapshot to: {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="AgroBot Real-Time Camera HUD Demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--checkpoint", type=Path, default=RUNS_DIR / "model_a_convnext" / "best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_camera_demo(camera_id=args.camera, checkpoint=args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
