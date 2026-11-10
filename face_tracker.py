"""
AI Multiple Face Detection & Visual Tracking System
====================================================
Real-time multi-face detection, temporary ID tracking, interactive mouse
target locking, and a futuristic forehead archery-crosshair HUD.

Controls:
  - Left Mouse Click : Select / lock onto any detected face
  - C Key            : Clear current face selection
  - Q / ESC Key      : Quit the application

Dependencies:
  pip install opencv-python mediapipe numpy
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keep terminal output clean from verbose internal OpenCV C++ backend logs
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


# =============================================================================
# CONFIGURATION & COLOR PALETTE (OpenCV uses BGR color order: Blue, Green, Red)
# =============================================================================

# Target marker colors (strictly following design requirements)
COLOR_SELECTED_GREEN = (75, 255, 50)       # Vibrant futuristic green (BGR)
COLOR_SELECTED_CORE = (190, 255, 190)      # Soft bright green center dot
COLOR_OTHER_ORANGE = (0, 150, 255)         # Vibrant futuristic orange (BGR)
COLOR_OTHER_CORE = (130, 215, 255)         # Soft warm orange center dot

# HUD colors
COLOR_HUD_BG = (22, 24, 30)                # Dark slate HUD panel background
COLOR_HUD_ACCENT = (235, 210, 60)          # Subtle futuristic cyan-teal accent
COLOR_STATUS_LOCKED = (75, 255, 50)        # Green for FACE LOCKED
COLOR_STATUS_LOST = (60, 70, 255)          # High-visibility red for FACE LOST
COLOR_STATUS_IDLE = (220, 200, 90)         # Calm cyan for standby / no target
COLOR_TEXT_PRIMARY = (245, 247, 250)       # Crisp off-white text
COLOR_TEXT_MUTED = (165, 175, 190)         # Secondary label text
COLOR_SHADOW = (12, 12, 15)                # Contrast outline for bright scenes

# Official lightweight pre-trained AI model URLs (~227 KB each, cached locally)
MODEL_CACHE_DIR = Path.home() / ".cache" / "ai_face_tracker_models"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


# =============================================================================
# 1. WEBCAM INITIALIZATION & ERROR HANDLING
# =============================================================================

def open_webcam(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
) -> cv2.VideoCapture:
    """
    Safely open the PC webcam, configure resolution, and verify frame capture.
    Raises a RuntimeError with clear troubleshooting instructions on failure.
    """
    # Choose the most reliable OS-specific capture backend first, with fallback
    backends_to_try = [cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform == "darwin":
        backends_to_try = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif sys.platform.startswith("linux"):
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]

    cap: Optional[cv2.VideoCapture] = None
    for backend in backends_to_try:
        candidate = cv2.VideoCapture(camera_index, backend)
        if candidate is not None and candidate.isOpened():
            cap = candidate
            break
        if candidate is not None:
            candidate.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}.\n"
            "Troubleshooting steps:\n"
            "  1. Check that your webcam is plugged in and not used by Zoom/Teams/Browser.\n"
            "  2. Check OS camera privacy settings (allow Python/Terminal camera access).\n"
            "  3. If you have multiple cameras, try: python face_tracker.py --camera 1"
        )

    # Request clean HD resolution (webcam will pick its closest supported mode)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Verify we can actually read a valid non-empty frame (catches busy/locked cameras)
    ret, test_frame = False, None
    for _ in range(10):
        ret, test_frame = cap.read()
        if ret and test_frame is not None and test_frame.size > 0:
            break
        time.sleep(0.05)

    if not ret or test_frame is None or test_frame.size == 0:
        cap.release()
        raise RuntimeError(
            f"Webcam {camera_index} opened, but returned empty video frames.\n"
            "Another application may be locking the camera stream. Close other video apps and retry."
        )

    return cap


# =============================================================================
# 2. AI FACE DETECTION & FOREHEAD CENTER CALCULATION
# =============================================================================

def _ensure_model_file(filename: str, url: str) -> Optional[Path]:
    """Download a lightweight model file once into local cache if missing."""
    try:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        target_path = MODEL_CACHE_DIR / filename
        if target_path.exists() and target_path.stat().st_size > 10_000:
            return target_path

        print(f"[INFO] Downloading lightweight AI face model ({filename}, ~230 KB)...")
        temp_path = target_path.with_suffix(".tmp")
        urllib.request.urlretrieve(url, str(temp_path))
        temp_path.replace(target_path)
        print(f"[INFO] Model cached at: {target_path}")
        return target_path
    except Exception as exc:
        print(f"[WARN] Could not download {filename}: {exc}")
        return None


def init_face_detector(
    preferred_backend: str = "auto",
    min_confidence: float = 0.55,
) -> Dict[str, Any]:
    """
    Initialize the AI face detection backend.

    Supported backends:
      - 'yunet'     : OpenCV DNN FaceDetectorYN (fast, multi-range, 5 landmarks).
      - 'mediapipe' : Google MediaPipe (supports both modern Tasks API >=0.10.30
                      and legacy mp.solutions API <0.10.30).
      - 'haar'      : Offline OpenCV Haar Cascade fallback.
      - 'auto'      : Tries YuNet first for best multi-distance webcam accuracy,
                      then MediaPipe, then Haar offline fallback.
    """
    order = (
        ["yunet", "mediapipe", "haar"]
        if preferred_backend == "auto"
        else [preferred_backend]
    )

    for backend in order:
        # --- Option A: OpenCV YuNet AI Detector ---
        if backend == "yunet" and hasattr(cv2, "FaceDetectorYN"):
            model_path = _ensure_model_file("face_detection_yunet_2023mar.onnx", YUNET_MODEL_URL)
            if model_path is not None:
                try:
                    detector = cv2.FaceDetectorYN.create(
                        str(model_path),
                        "",
                        (640, 480),
                        float(min_confidence),
                        0.3,   # NMS threshold
                        5000,  # Top-K candidates
                    )
                    return {
                        "type": "yunet",
                        "name": "OpenCV YuNet AI",
                        "detector": detector,
                        "min_confidence": min_confidence,
                    }
                except Exception as exc:
                    print(f"[WARN] YuNet initialization failed: {exc}")

        # --- Option B: Google MediaPipe (Modern Tasks API + Legacy Solutions API) ---
        if backend == "mediapipe":
            try:
                import mediapipe as mp

                # Modern MediaPipe (0.10.30+ / 1.x) uses mediapipe.tasks.python.vision
                if hasattr(mp, "tasks"):
                    from mediapipe.tasks import python as mp_python
                    from mediapipe.tasks.python import vision as mp_vision

                    model_path = _ensure_model_file(
                        "blaze_face_short_range.tflite", MEDIAPIPE_MODEL_URL
                    )
                    if model_path is not None:
                        options = mp_vision.FaceDetectorOptions(
                            base_options=mp_python.BaseOptions(
                                model_asset_path=str(model_path)
                            ),
                            running_mode=mp_vision.RunningMode.IMAGE,
                            min_detection_confidence=float(min_confidence),
                        )
                        detector = mp_vision.FaceDetector.create_from_options(options)
                        return {
                            "type": "mediapipe_tasks",
                            "name": "MediaPipe BlazeFace",
                            "detector": detector,
                            "mp_module": mp,
                            "min_confidence": min_confidence,
                        }

                # Legacy MediaPipe (<0.10.30) fallback using mp.solutions
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                    detector = mp.solutions.face_detection.FaceDetection(
                        model_selection=1,  # 1 = full-range multi-face model
                        min_detection_confidence=float(min_confidence),
                    )
                    return {
                        "type": "mediapipe_legacy",
                        "name": "MediaPipe Solutions",
                        "detector": detector,
                        "min_confidence": min_confidence,
                    }
            except Exception as exc:
                print(f"[WARN] MediaPipe initialization failed: {exc}")

        # --- Option C: Built-in Offline Haar Cascade Fallback ---
        if backend == "haar":
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                return {
                    "type": "haar",
                    "name": "OpenCV Haar (Offline)",
                    "detector": cascade,
                    "min_confidence": min_confidence,
                }

    raise RuntimeError(
        f"Failed to initialize face detector backend '{preferred_backend}'. "
        "Please verify opencv-python and/or mediapipe are installed."
    )


def compute_forehead_point(
    bbox: Tuple[float, float, float, float],
    eye_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    Calculate the anatomical center of the forehead from a face region and
    optional eye landmarks.

    When eye landmarks are available, we find the midpoint between the eyes and
    step upward perpendicular to the eye-line so the forehead marker tilts
    naturally when the person tilts their head.
    """
    x, y, w, h = bbox

    if eye_points is not None:
        (e1x, e1y), (e2x, e2y) = eye_points
        # Order eyes left-to-right in screen coordinates
        if e1x <= e2x:
            left_screen, right_screen = (e1x, e1y), (e2x, e2y)
        else:
            left_screen, right_screen = (e2x, e2y), (e1x, e1y)

        dx = right_screen[0] - left_screen[0]
        dy = right_screen[1] - left_screen[1]
        eye_dist = math.hypot(dx, dy)

        if eye_dist > 0.16 * w:
            mid_x = 0.5 * (left_screen[0] + right_screen[0])
            mid_y = 0.5 * (left_screen[1] + right_screen[1])

            # Unit normal pointing straight UP from the eye line toward forehead
            up_x = dy / eye_dist
            up_y = -dx / eye_dist

            # Forehead center sits ~0.56x inter-ocular distance above eye midpoint
            forehead_offset = max(0.56 * eye_dist, 0.21 * h)
            fx = mid_x + up_x * forehead_offset
            fy = mid_y + up_y * forehead_offset

            # Clamp neatly within upper face bounds so extreme angles stay valid
            fx = float(np.clip(fx, x + 0.18 * w, x + 0.82 * w))
            fy = float(np.clip(fy, max(4.0, y + 0.05 * h), y + 0.35 * h))
            return (fx, fy)

    # Fallback when landmarks are unavailable or face is in steep profile
    fx = float(x + 0.50 * w)
    fy = float(max(4.0, y + 0.21 * h))
    return (fx, fy)


def detect_faces(
    frame: np.ndarray,
    detector_bundle: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Detect all visible faces in the current video frame.

    Returns a list of dictionaries, each containing:
      - 'bbox'     : (x, y, w, h) used internally for tracking & click hit-testing
                     (never drawn as a rectangle!)
      - 'center'   : (cx, cy) geometric face center
      - 'forehead' : (fx, fy) exact forehead target marker position
      - 'score'    : detection confidence [0.0 .. 1.0]
    """
    frame_h, frame_w = frame.shape[:2]
    backend_type = detector_bundle["type"]
    detections: List[Dict[str, Any]] = []

    # --- Backend 1: OpenCV YuNet ---
    if backend_type == "yunet":
        detector = detector_bundle["detector"]
        # Downscale very high-res frames slightly for >90 FPS speed, then map back
        max_infer_width = 720
        if frame_w > max_infer_width:
            scale = max_infer_width / float(frame_w)
            infer_w = max_infer_width
            infer_h = max(64, int(round(frame_h * scale)))
            infer_img = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            infer_w, infer_h = frame_w, frame_h
            infer_img = frame

        detector.setInputSize((infer_w, infer_h))
        _, raw_faces = detector.detect(infer_img)

        if raw_faces is not None:
            inv_scale = 1.0 / scale
            for row in raw_faces:
                x = float(row[0] * inv_scale)
                y = float(row[1] * inv_scale)
                w = float(row[2] * inv_scale)
                h = float(row[3] * inv_scale)
                score = float(row[14])

                if w < 20 or h < 20:
                    continue

                eye_a = (float(row[4] * inv_scale), float(row[5] * inv_scale))
                eye_b = (float(row[6] * inv_scale), float(row[7] * inv_scale))
                forehead = compute_forehead_point((x, y, w, h), (eye_a, eye_b))

                detections.append({
                    "bbox": (x, y, w, h),
                    "center": (x + 0.5 * w, y + 0.5 * h),
                    "forehead": forehead,
                    "score": score,
                })

    # --- Backend 2: Modern MediaPipe Tasks API (0.10.30+ / 1.x) ---
    elif backend_type == "mediapipe_tasks":
        mp = detector_bundle["mp_module"]
        detector = detector_bundle["detector"]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        if result and result.detections:
            for det in result.detections:
                bb = det.bounding_box
                # Note: MediaPipe Tasks bounding_box is in pixel coordinates,
                # while keypoints are normalized [0.0 .. 1.0].
                x = float(bb.origin_x)
                y = float(bb.origin_y)
                w = float(bb.width)
                h = float(bb.height)
                score = float(det.categories[0].score) if det.categories else 0.8

                if w < 20 or h < 20:
                    continue

                eyes = None
                if det.keypoints and len(det.keypoints) >= 2:
                    kp0 = (det.keypoints[0].x * frame_w, det.keypoints[0].y * frame_h)
                    kp1 = (det.keypoints[1].x * frame_w, det.keypoints[1].y * frame_h)
                    eyes = (kp0, kp1)

                forehead = compute_forehead_point((x, y, w, h), eyes)
                detections.append({
                    "bbox": (x, y, w, h),
                    "center": (x + 0.5 * w, y + 0.5 * h),
                    "forehead": forehead,
                    "score": score,
                })

    # --- Backend 3: Legacy MediaPipe Solutions API (<0.10.30) ---
    elif backend_type == "mediapipe_legacy":
        detector = detector_bundle["detector"]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb_frame)

        if result and result.detections:
            for det in result.detections:
                rel_box = det.location_data.relative_bounding_box
                x = float(rel_box.xmin * frame_w)
                y = float(rel_box.ymin * frame_h)
                w = float(rel_box.width * frame_w)
                h = float(rel_box.height * frame_h)
                score = float(det.score[0]) if det.score else 0.8

                if w < 20 or h < 20:
                    continue

                kps = det.location_data.relative_keypoints
                eyes = None
                if kps and len(kps) >= 2:
                    eyes = (
                        (kps[0].x * frame_w, kps[0].y * frame_h),
                        (kps[1].x * frame_w, kps[1].y * frame_h),
                    )

                forehead = compute_forehead_point((x, y, w, h), eyes)
                detections.append({
                    "bbox": (x, y, w, h),
                    "center": (x + 0.5 * w, y + 0.5 * h),
                    "forehead": forehead,
                    "score": score,
                })

    # --- Backend 4: Offline Haar Cascade Fallback ---
    elif backend_type == "haar":
        cascade = detector_bundle["detector"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(
            gray, scaleFactor=1.12, minNeighbors=5, minSize=(45, 45)
        )
        for (rx, ry, rw, rh) in rects:
            bbox = (float(rx), float(ry), float(rw), float(rh))
            detections.append({
                "bbox": bbox,
                "center": (rx + 0.5 * rw, ry + 0.5 * rh),
                "forehead": compute_forehead_point(bbox, None),
                "score": 0.75,
            })

    return detections


def close_face_detector(detector_bundle: Optional[Dict[str, Any]]) -> None:
    """Cleanly release detector resources if needed."""
    if not detector_bundle:
        return
    det = detector_bundle.get("detector")
    if hasattr(det, "close"):
        try:
            det.close()
        except Exception:
            pass


# =============================================================================
# 3. IMPROVED VISUAL MULTI-FACE TRACKING (TEMPORARY TRACKING IDs)
# =============================================================================

def create_tracker_state(
    max_disappeared_frames: int = 24,
    visible_grace_frames: int = 2,
) -> Dict[str, Any]:
    """
    Create a fresh state dictionary for our multi-face visual tracker.

    NOTE: Uses purely geometric & motion cues (velocity prediction, IoU overlap,
    and scale consistency) — NEVER facial identity recognition.
    """
    return {
        "next_id": 1,
        "tracks": {},  # track_id -> track dict
        "max_disappeared_frames": max_disappeared_frames,
        "visible_grace_frames": visible_grace_frames,
    }


def _compute_bbox_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union (IoU) between two (x, y, w, h) regions."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    union_area = (aw * ah) + (bw * bh) - inter_area
    return float(inter_area / max(union_area, 1e-6))


def _adaptive_smooth(
    prev_xy: Tuple[float, float],
    new_xy: Tuple[float, float],
    ref_size: float,
) -> Tuple[float, float]:
    """
    Velocity-adaptive exponential smoothing:
      - Filters out tiny 1-2px detector jitter when a person holds still.
      - Responds almost instantly when the person moves rapidly.
    """
    move_dist = math.hypot(new_xy[0] - prev_xy[0], new_xy[1] - prev_xy[1])
    motion_ratio = move_dist / max(ref_size * 0.18, 6.0)
    alpha = float(np.clip(0.35 + 0.55 * min(motion_ratio, 1.0), 0.35, 0.90))
    sx = prev_xy[0] + alpha * (new_xy[0] - prev_xy[0])
    sy = prev_xy[1] + alpha * (new_xy[1] - prev_xy[1])
    return (sx, sy)


def update_face_tracker(
    detections: List[Dict[str, Any]],
    tracker_state: Dict[str, Any],
    dt: float,
) -> List[Dict[str, Any]]:
    """
    Match current-frame face detections to existing temporary tracking IDs.

    Improvements over basic centroid tracking:
      1. Constant-velocity prediction anticipates rapid head movement.
      2. Multi-cue matching cost combines scale-normalized distance, bounding
         box IoU overlap, and face-size ratio consistency (prevents ID swaps
         when foreground and background faces cross paths).
      3. Short-term coasting buffer preserves temporary IDs across brief motion
         blur or blink dropouts, while immediately flagging disappeared tracks.

    Returns the list of currently visible tracks to render on screen.
    """
    tracks: Dict[int, Dict[str, Any]] = tracker_state["tracks"]
    max_disappeared = tracker_state["max_disappeared_frames"]
    visible_grace = tracker_state["visible_grace_frames"]
    dt = float(np.clip(dt, 1.0 / 240.0, 0.25))

    track_ids = list(tracks.keys())

    # Step 1: Predict where each existing track should be in this frame
    predicted_centers: Dict[int, Tuple[float, float]] = {}
    predicted_boxes: Dict[int, Tuple[float, float, float, float]] = {}
    for tid in track_ids:
        tr = tracks[tid]
        vx, vy = tr["velocity"]
        # Damp velocity gradually if track has been coasting unseen
        damping = 0.85 ** tr["missed_frames"]
        px = tr["center"][0] + vx * dt * damping
        py = tr["center"][1] + vy * dt * damping
        predicted_centers[tid] = (px, py)
        bx, by, bw, bh = tr["bbox"]
        predicted_boxes[tid] = (px - 0.5 * bw, py - 0.5 * bh, bw, bh)

    matched_tracks = set()
    matched_dets = set()

    # Step 2: Build pairwise matching cost matrix & assign lowest-cost pairs
    if track_ids and detections:
        candidate_pairs: List[Tuple[float, int, int]] = []

        for t_idx, tid in enumerate(track_ids):
            tr = tracks[tid]
            pred_cx, pred_cy = predicted_centers[tid]
            pred_box = predicted_boxes[tid]
            _, _, tw, th = tr["bbox"]
            ref_scale = max(math.hypot(tw, th), 40.0)

            # Expand gate slightly if the face was briefly occluded for a few frames
            max_gate_dist = ref_scale * (1.25 + 0.06 * min(tr["missed_frames"], 10))

            for d_idx, det in enumerate(detections):
                dcx, dcy = det["center"]
                _, _, dw, dh = det["bbox"]

                dist = math.hypot(dcx - pred_cx, dcy - pred_cy)
                if dist > max_gate_dist:
                    continue

                # Reject implausible sudden size jumps (e.g. foreground vs far background)
                scale_diff = abs(math.log(max(dw, 1.0) / max(tw, 1.0)))
                if scale_diff > 0.65:
                    continue

                iou = _compute_bbox_iou(pred_box, det["bbox"])
                norm_dist = dist / ref_scale

                # Unified lower-is-better matching cost
                cost = (0.60 * norm_dist) + (0.30 * (1.0 - iou)) + (0.10 * scale_diff)
                candidate_pairs.append((cost, tid, d_idx))

        # Sort globally by best match cost so closest/most consistent pairs lock first
        candidate_pairs.sort(key=lambda item: item[0])
        for cost, tid, d_idx in candidate_pairs:
            if tid in matched_tracks or d_idx in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(d_idx)

            tr = tracks[tid]
            det = detections[d_idx]

            # Update smoothed velocity estimate
            inst_vx = (det["center"][0] - tr["center"][0]) / dt
            inst_vy = (det["center"][1] - tr["center"][1]) / dt
            old_vx, old_vy = tr["velocity"]
            tr["velocity"] = (
                0.60 * old_vx + 0.40 * inst_vx,
                0.60 * old_vy + 0.40 * inst_vy,
            )

            # Smooth forehead marker and face width for steady, lag-free UI
            tr["forehead"] = _adaptive_smooth(
                tr["forehead"], det["forehead"], det["bbox"][2]
            )
            bx, by, bw, bh = det["bbox"]
            _, _, prev_w, prev_h = tr["bbox"]
            smooth_w = 0.65 * prev_w + 0.35 * bw
            smooth_h = 0.65 * prev_h + 0.35 * bh

            tr["center"] = det["center"]
            tr["bbox"] = (bx, by, smooth_w, smooth_h)
            tr["score"] = det["score"]
            tr["missed_frames"] = 0
            tr["age"] += 1

    # Step 3: Handle unmatched existing tracks (coasting vs retirement)
    for tid in track_ids:
        if tid not in matched_tracks:
            tr = tracks[tid]
            tr["missed_frames"] += 1
            if tr["missed_frames"] <= visible_grace:
                # Micro-flicker tolerance (1-2 frames): advance gently by velocity
                vx, vy = tr["velocity"]
                tr["center"] = (tr["center"][0] + vx * dt, tr["center"][1] + vy * dt)
                tr["forehead"] = (
                    tr["forehead"][0] + vx * dt,
                    tr["forehead"][1] + vy * dt,
                )
            if tr["missed_frames"] > max_disappeared:
                del tracks[tid]

    # Step 4: Register brand-new temporary tracking IDs for new faces
    # (Sorted left-to-right on screen for intuitive initial ID numbering)
    unmatched_dets = [
        detections[d_idx] for d_idx in range(len(detections)) if d_idx not in matched_dets
    ]
    unmatched_dets.sort(key=lambda d: d["center"][0])

    for det in unmatched_dets:
        new_id = tracker_state["next_id"]
        tracker_state["next_id"] += 1
        tracks[new_id] = {
            "id": new_id,
            "bbox": det["bbox"],
            "center": det["center"],
            "forehead": det["forehead"],
            "velocity": (0.0, 0.0),
            "score": det["score"],
            "missed_frames": 0,
            "age": 1,
        }

    # Return only currently visible tracks (sorted by ID for clean consistency)
    visible_tracks = [
        tr for tr in tracks.values() if tr["missed_frames"] <= visible_grace
    ]
    visible_tracks.sort(key=lambda t: t["id"])
    return visible_tracks


# =============================================================================
# 4. MOUSE SELECTION LOGIC
# =============================================================================

def select_face_at_point(
    click_x: int,
    click_y: int,
    visible_tracks: List[Dict[str, Any]],
) -> Optional[int]:
    """
    Determine which visible face the user clicked on.

    Accepts clicks EITHER directly on the forehead crosshair marker OR anywhere
    on that person's face region. If multiple faces overlap, selects the one
    whose forehead/face center is closest to the mouse click.
    """
    best_id: Optional[int] = None
    best_distance = float("inf")

    for tr in visible_tracks:
        fx, fy = tr["forehead"]
        bx, by, bw, bh = tr["bbox"]
        marker_radius = max(18.0, bw * 0.22)

        # Check 1: Direct click on or near the circular forehead target marker
        dist_to_forehead = math.hypot(click_x - fx, click_y - fy)
        hit_forehead = dist_to_forehead <= marker_radius * 1.45

        # Check 2: Click anywhere on the visible face area (slight padding for moving faces)
        pad_x, pad_y = 0.10 * bw, 0.10 * bh
        hit_face_region = (
            (bx - pad_x) <= click_x <= (bx + bw + pad_x)
            and (by - pad_y) <= click_y <= (by + bh + pad_y)
        )

        if hit_forehead or hit_face_region:
            # Prefer direct forehead hits, otherwise closest face center
            face_cx, face_cy = tr["center"]
            score_dist = min(dist_to_forehead, math.hypot(click_x - face_cx, click_y - face_cy))
            if score_dist < best_distance:
                best_distance = score_dist
                best_id = tr["id"]

    return best_id


def mouse_callback(event: int, x: int, y: int, flags: int, app_state: Dict[str, Any]) -> None:
    """
    OpenCV mouse event handler. Records left-click coordinates so the main loop
    can immediately lock onto the clicked face.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        app_state["pending_click"] = (int(x), int(y))


# =============================================================================
# 5. FUTURISTIC ARCHERY TARGET / CROSSHAIR & HUD UI RENDERING
# =============================================================================

def draw_forehead_crosshair(
    frame: np.ndarray,
    track: Dict[str, Any],
    is_selected: bool,
    anim_phase: float = 0.0,
) -> None:
    """
    Draw a clean, futuristic circular Archery Target / Crosshair marker near
    the center of the face's forehead.

    Design strictly includes:
      - NO rectangular bounding box around the face
      - Outer circle
      - Inner concentric circle + bullseye center point
      - Four small directional crosshair lines (N, S, E, W)
      - Green for selected face, Orange for other detected faces
      - Compact tracking ID callout next to the marker
    """
    fx, fy = track["forehead"]
    cx, cy = int(round(fx)), int(round(fy))
    face_w = track["bbox"][2]

    # Keep the target marker compact and proportional to face size
    outer_r = int(np.clip(face_w * 0.16, 13, 24))
    inner_r = max(5, int(round(outer_r * 0.44)))
    tick_inner = inner_r + 3
    tick_outer = outer_r + max(5, int(round(outer_r * 0.42)))

    primary_color = COLOR_SELECTED_GREEN if is_selected else COLOR_OTHER_ORANGE
    core_color = COLOR_SELECTED_CORE if is_selected else COLOR_OTHER_CORE

    # 1. Soft blended contrast halo (computed on a tiny local ROI for maximum FPS)
    pad = tick_outer + 8
    rx1, ry1 = max(0, cx - pad), max(0, cy - pad)
    rx2, ry2 = min(frame.shape[1], cx + pad), min(frame.shape[0], cy + pad)
    if rx2 > rx1 and ry2 > ry1:
        roi = frame[ry1:ry2, rx1:rx2]
        halo_roi = roi.copy()
        lcx, lcy = cx - rx1, cy - ry1
        cv2.circle(halo_roi, (lcx, lcy), outer_r, COLOR_SHADOW, 3, cv2.LINE_AA)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            p1 = (lcx + dx * tick_inner, lcy + dy * tick_inner)
            p2 = (lcx + dx * tick_outer, lcy + dy * tick_outer)
            cv2.line(halo_roi, p1, p2, COLOR_SHADOW, 3, cv2.LINE_AA)
        cv2.addWeighted(halo_roi, 0.55, roi, 0.45, 0, roi)

    # 2. Outer target circle
    cv2.circle(frame, (cx, cy), outer_r, primary_color, 2, cv2.LINE_AA)

    # 3. Inner concentric target circle + tiny center bullseye dot
    cv2.circle(frame, (cx, cy), inner_r, primary_color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, core_color, -1, cv2.LINE_AA)

    # 4. Four small directional crosshair lines (North, South, West, East)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        p1 = (cx + dx * tick_inner, cy + dy * tick_inner)
        p2 = (cx + dx * tick_outer, cy + dy * tick_outer)
        cv2.line(frame, p1, p2, primary_color, 2, cv2.LINE_AA)

    # 5. Subtle rotating circular lock arcs for the actively locked face
    if is_selected:
        arc_r = outer_r + 5
        base_deg = (anim_phase * 90.0) % 360.0
        for offset in (0.0, 180.0):
            start_a = base_deg + offset
            cv2.ellipse(
                frame,
                (cx, cy),
                (arc_r, arc_r),
                0.0,
                start_a,
                start_a + 48.0,
                primary_color,
                1,
                cv2.LINE_AA,
            )

    # 6. Minimal futuristic tracking ID callout beside the forehead marker
    frame_h, frame_w = frame.shape[:2]
    label = f"ID #{track['id']}" if not is_selected else f"ID #{track['id']} LOCK"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.42
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Place callout to the upper-right of the forehead reticle (flip left near screen edge)
    leader_dx = tick_outer + 5
    leader_dy = -max(6, outer_r // 2)
    if cx + leader_dx + tw + 16 > frame_w:
        tx = max(8, cx - leader_dx - tw - 6)
        anchor_end = (cx - tick_outer - 2, cy + leader_dy)
        anchor_start = (cx - outer_r, cy - 3)
    else:
        tx = cx + leader_dx + 5
        anchor_end = (cx + leader_dx, cy + leader_dy)
        anchor_start = (cx + int(outer_r * 0.75), cy - int(outer_r * 0.55))

    ty = max(th + 58, min(frame_h - 36, cy + leader_dy + th // 2))

    # Diagonal micro-leader line + translucent local ROI micro-pill for crisp legibility
    cv2.line(frame, anchor_start, anchor_end, primary_color, 1, cv2.LINE_AA)
    px1, py1 = max(0, tx - 4), max(0, ty - th - 3)
    px2, py2 = min(frame_w, tx + tw + 4), min(frame_h, ty + 4)
    if px2 > px1 and py2 > py1:
        pill_roi = frame[py1:py2, px1:px2]
        bg_patch = np.full_like(pill_roi, COLOR_HUD_BG)
        cv2.addWeighted(bg_patch, 0.62, pill_roi, 0.38, 0, pill_roi)
    cv2.putText(frame, label, (tx, ty), font, font_scale, primary_color, thickness, cv2.LINE_AA)


def draw_hud_interface(
    frame: np.ndarray,
    detected_count: int,
    selected_id: Optional[int],
    tracking_status: str,
    fps: float,
    detector_name: str,
) -> None:
    """
    Render the modern, clean futuristic HUD overlay displaying:
      - FACE LOCKED / FACE LOST / STANDBY status
      - Number of detected faces
      - Real-time FPS counter
      - Selected face indicator
      - Quick keyboard & mouse control guide
    """
    frame_h, frame_w = frame.shape[:2]
    top_bar_h = 52
    bottom_bar_h = 30

    # Semi-transparent top & bottom glassmorphic bars (local slice blend for max FPS)
    top_slice = frame[0:top_bar_h, 0:frame_w]
    cv2.addWeighted(
        np.full_like(top_slice, COLOR_HUD_BG), 0.76, top_slice, 0.24, 0, top_slice
    )
    bot_slice = frame[frame_h - bottom_bar_h : frame_h, 0:frame_w]
    cv2.addWeighted(
        np.full_like(bot_slice, COLOR_HUD_BG), 0.76, bot_slice, 0.24, 0, bot_slice
    )

    # Crisp accent lines bordering the HUD panels
    if tracking_status == "FACE LOCKED":
        status_color = COLOR_STATUS_LOCKED
    elif tracking_status == "FACE LOST":
        status_color = COLOR_STATUS_LOST
    else:
        status_color = COLOR_STATUS_IDLE

    cv2.line(frame, (0, top_bar_h), (frame_w, top_bar_h), status_color, 1, cv2.LINE_AA)
    cv2.line(
        frame,
        (0, frame_h - bottom_bar_h),
        (frame_w, frame_h - bottom_bar_h),
        (65, 75, 90),
        1,
        cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_DUPLEX

    # --- Left Section: FPS Counter & Detected Faces Count ---
    left_metrics = f"FPS: {fps:4.1f}   |   FACES: {detected_count}"
    cv2.putText(
        frame, left_metrics, (16, 32), font, 0.56, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA
    )

    # --- Center Section: Prominent Status Badge (FACE LOCKED / FACE LOST / STANDBY) ---
    if tracking_status == "FACE LOCKED":
        badge_text = "FACE LOCKED"
    elif tracking_status == "FACE LOST":
        badge_text = "FACE LOST"
    else:
        badge_text = "STANDBY - CLICK FACE"

    (bw, bh), _ = cv2.getTextSize(badge_text, font, 0.62, 1)
    pill_pad_x = 22
    pill_w = bw + pill_pad_x * 2 + 14
    pill_x1 = max(240, (frame_w - pill_w) // 2)
    pill_x2 = pill_x1 + pill_w
    pill_y1, pill_y2 = 9, top_bar_h - 9

    # Subtle tinted status badge background + crisp outline
    pill_slice = frame[pill_y1:pill_y2, pill_x1:pill_x2]
    cv2.addWeighted(
        np.full_like(pill_slice, status_color), 0.18, pill_slice, 0.82, 0, pill_slice
    )
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), status_color, 1, cv2.LINE_AA)

    # Status indicator LED dot + status text
    dot_x = pill_x1 + 14
    dot_y = (pill_y1 + pill_y2) // 2
    cv2.circle(frame, (dot_x, dot_y), 5, status_color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        badge_text,
        (dot_x + 12, pill_y2 - 9),
        font,
        0.62,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # --- Right Section: Selected Face Indicator ---
    if selected_id is None:
        selected_label = "TARGET: NONE"
        sel_color = COLOR_TEXT_MUTED
    elif tracking_status == "FACE LOCKED":
        selected_label = f"TARGET: ID #{selected_id}"
        sel_color = COLOR_SELECTED_GREEN
    else:
        selected_label = f"TARGET: ID #{selected_id} (LOST)"
        sel_color = COLOR_STATUS_LOST

    (sw, _), _ = cv2.getTextSize(selected_label, font, 0.56, 1)
    sel_x = max(pill_x2 + 14, frame_w - sw - 18)
    cv2.putText(
        frame, selected_label, (sel_x, 32), font, 0.56, sel_color, 1, cv2.LINE_AA
    )

    # --- Bottom Bar: Clean Controls & Active AI Engine Indicator ---
    controls_text = "[CLICK FACE] Lock Target    [C] Clear Selection    [Q] Quit"
    cv2.putText(
        frame,
        controls_text,
        (16, frame_h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    engine_tag = f"AI: {detector_name}"
    (ew, _), _ = cv2.getTextSize(engine_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    if frame_w - ew - 16 > 470:
        cv2.putText(
            frame,
            engine_tag,
            (frame_w - ew - 16, frame_h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            COLOR_HUD_ACCENT,
            1,
            cv2.LINE_AA,
        )


# =============================================================================
# 6. FRAME PROCESSING PIPELINE & MAIN REAL-TIME APPLICATION LOOP
# =============================================================================

def determine_tracking_status(
    selected_id: Optional[int],
    visible_tracks: List[Dict[str, Any]],
) -> str:
    """
    Compute the current selection state:
      - 'IDLE'        : No face is currently selected (initial state or after 'C').
      - 'FACE LOCKED' : A face is selected and currently visible in frame.
      - 'FACE LOST'   : A face was selected, and it has disappeared from view.
    """
    if selected_id is None:
        return "IDLE"
    for tr in visible_tracks:
        if tr["id"] == selected_id:
            return "FACE LOCKED"
    return "FACE LOST"


def render_scene(
    frame: np.ndarray,
    visible_tracks: List[Dict[str, Any]],
    selected_id: Optional[int],
    fps: float,
    detector_name: str,
    anim_phase: float = 0.0,
) -> Tuple[np.ndarray, str]:
    """
    Draw all forehead crosshair markers (orange first, selected green on top)
    and the futuristic HUD overlay onto a video frame.
    """
    tracking_status = determine_tracking_status(selected_id, visible_tracks)

    # Draw non-selected faces first so the locked green marker always renders cleanly on top
    selected_track: Optional[Dict[str, Any]] = None
    for tr in visible_tracks:
        if tr["id"] == selected_id:
            selected_track = tr
        else:
            draw_forehead_crosshair(frame, tr, is_selected=False, anim_phase=anim_phase)

    if selected_track is not None:
        draw_forehead_crosshair(frame, selected_track, is_selected=True, anim_phase=anim_phase)

    draw_hud_interface(
        frame=frame,
        detected_count=len(visible_tracks),
        selected_id=selected_id,
        tracking_status=tracking_status,
        fps=fps,
        detector_name=detector_name,
    )
    return frame, tracking_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Multiple Face Detection & Futuristic Visual Tracking System"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["auto", "yunet", "mediapipe", "haar"],
        default="auto",
        help="Face detector backend (default: auto)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Preferred webcam width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Preferred webcam height in pixels (default: 720)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror flip of the webcam feed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    window_title = "AI Multiple Face Detection & Visual Tracking System"

    cap: Optional[cv2.VideoCapture] = None
    detector_bundle: Optional[Dict[str, Any]] = None

    try:
        print("[INFO] Initializing AI Face Detector...")
        detector_bundle = init_face_detector(preferred_backend=args.detector)
        print(f"[INFO] Active Detector: {detector_bundle['name']}")

        print(f"[INFO] Opening webcam (index {args.camera})...")
        cap = open_webcam(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
        )

        tracker_state = create_tracker_state()
        app_state: Dict[str, Any] = {
            "selected_id": None,
            "pending_click": None,
        }

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_title, mouse_callback, app_state)

        print("[READY] Application running!")
        print("  - Left-click any face to LOCK target (Green crosshair)")
        print("  - Press 'C' to CLEAR current selection")
        print("  - Press 'Q' or 'ESC' to QUIT")

        prev_time = time.perf_counter()
        smoothed_fps = 30.0
        start_time = prev_time
        consecutive_failures = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures > 20:
                    print("[ERROR] Lost connection to webcam stream. Exiting safely.")
                    break
                time.sleep(0.02)
                continue
            consecutive_failures = 0

            # Mirror the webcam horizontally for intuitive mouse-clicking
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(1e-4, now - prev_time)
            prev_time = now
            instant_fps = 1.0 / dt
            smoothed_fps = 0.88 * smoothed_fps + 0.12 * instant_fps

            # 1. Detect all visible faces and compute forehead positions
            detections = detect_faces(frame, detector_bundle)

            # 2. Update temporary visual tracking IDs
            visible_tracks = update_face_tracker(detections, tracker_state, dt)

            # 3. Process manual mouse selection if the user clicked
            if app_state["pending_click"] is not None:
                click_x, click_y = app_state["pending_click"]
                app_state["pending_click"] = None
                clicked_id = select_face_at_point(click_x, click_y, visible_tracks)
                if clicked_id is not None:
                    app_state["selected_id"] = clicked_id

            # 4. Draw futuristic forehead crosshairs & HUD
            anim_phase = now - start_time
            render_scene(
                frame=frame,
                visible_tracks=visible_tracks,
                selected_id=app_state["selected_id"],
                fps=smoothed_fps,
                detector_name=detector_bundle["name"],
                anim_phase=anim_phase,
            )

            cv2.imshow(window_title, frame)

            # 5. Keyboard controls ('C' to clear, 'Q' / ESC to quit)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("[INFO] Quit requested. Closing application...")
                break
            if key in (ord("c"), ord("C")):
                app_state["selected_id"] = None

            # Exit gracefully if user clicked the window's [X] close button
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break

    except RuntimeError as cam_err:
        print(f"\n[WEBCAM / SETUP ERROR]\n{cam_err}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Closing safely...")
    finally:
        if cap is not None:
            cap.release()
        close_face_detector(detector_bundle)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
