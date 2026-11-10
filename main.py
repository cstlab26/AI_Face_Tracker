
import cv2
import numpy as np
import time
import math

# ==========================================
# AI FACE TRACKING - FUTURISTIC DASHBOARD
# ==========================================

CAM_W = 960
CAM_H = 540
PANEL_W = 280

MAX_MISSED = 12
MAX_DISTANCE = 130

# Face detectors
base = cv2.data.haarcascades

face_detector = cv2.CascadeClassifier(
    base + "haarcascade_frontalface_default.xml"
)

profile_detector = cv2.CascadeClassifier(
    base + "haarcascade_profileface.xml"
)

if face_detector.empty() or profile_detector.empty():
    raise RuntimeError("Face detector could not load!")

# Camera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

if not cap.isOpened():
    raise RuntimeError("Webcam not found!")

# Tracking data
tracks = {}
next_id = 1
selected_id = None
current_faces = []

fps = 0
last_time = time.time()


# ==========================================
# FACE DETECTION
# ==========================================

def detect_faces(gray):

    detected = []

    # Front face
    front = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(55, 55)
    )

    detected.extend(list(front))

    # Side profile - normal
    profiles = profile_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(55, 55)
    )

    detected.extend(list(profiles))

    # Side profile - mirrored
    flipped = cv2.flip(gray, 1)

    profiles_flip = profile_detector.detectMultiScale(
        flipped,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(55, 55)
    )

    for x, y, w, h in profiles_flip:
        detected.append(
            (CAM_W - x - w, y, w, h)
        )

    # Remove duplicate detections
    clean = []

    for box in detected:

        x, y, w, h = map(int, box)

        duplicate = False

        for old in clean:

            ox, oy, ow, oh = old

            ix1 = max(x, ox)
            iy1 = max(y, oy)
            ix2 = min(x + w, ox + ow)
            iy2 = min(y + h, oy + oh)

            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            union = w * h + ow * oh - intersection

            if union > 0 and intersection / union > 0.35:
                duplicate = True
                break

        if not duplicate:
            clean.append((x, y, w, h))

    return clean


# ==========================================
# TRACKING SYSTEM
# ==========================================

def update_tracks(detections):

    global tracks, next_id

    candidates = []

    for index, (x, y, w, h) in enumerate(detections):

        cx = x + w // 2
        cy = y + h // 2

        for track_id, old in tracks.items():

            ox, oy = old["center"]

            distance = math.hypot(cx - ox, cy - oy)

            limit = max(
                MAX_DISTANCE,
                max(w, h) * 1.3
            )

            if distance < limit:
                candidates.append(
                    (distance, index, track_id)
                )

    candidates.sort()

    matched_detections = set()
    matched_tracks = set()
    updated = {}

    # Match existing IDs
    for distance, index, track_id in candidates:

        if index in matched_detections:
            continue

        if track_id in matched_tracks:
            continue

        x, y, w, h = detections[index]

        cx = x + w // 2
        cy = y + h // 2

        old = tracks[track_id]

        # Smooth movement
        old_cx, old_cy = old["center"]

        smooth_x = int(old_cx * 0.35 + cx * 0.65)
        smooth_y = int(old_cy * 0.35 + cy * 0.65)

        updated[track_id] = {
            "box": (x, y, w, h),
            "center": (smooth_x, smooth_y),
            "missed": 0
        }

        matched_detections.add(index)
        matched_tracks.add(track_id)

    # New faces
    for index, (x, y, w, h) in enumerate(detections):

        if index in matched_detections:
            continue

        updated[next_id] = {
            "box": (x, y, w, h),
            "center": (x + w // 2, y + h // 2),
            "missed": 0
        }

        next_id += 1

    # Temporarily retain lost faces
    for track_id, old in tracks.items():

        if track_id not in matched_tracks:

            old["missed"] += 1

            if old["missed"] <= MAX_MISSED:
                updated[track_id] = old

    tracks = updated


# ==========================================
# FUTURISTIC TARGET MARKER
# ==========================================

def draw_target(frame, x, y, color, selected):

    # Outer circular ring
    cv2.circle(frame, (x, y), 19, color, 2)

    # Inner circular ring
    cv2.circle(frame, (x, y), 9, color, 1)

    # Center point
    cv2.circle(frame, (x, y), 3, color, -1)

    # Crosshair lines
    cv2.line(frame, (x - 29, y), (x - 13, y), color, 2)
    cv2.line(frame, (x + 13, y), (x + 29, y), color, 2)

    cv2.line(frame, (x, y - 29), (x, y - 13), color, 2)
    cv2.line(frame, (x, y + 13), (x, y + 29), color, 2)

    if selected:
        cv2.circle(frame, (x, y), 25, color, 1)


# ==========================================
# MOUSE SELECTION
# ==========================================

def mouse_click(event, x, y, flags, param):

    global selected_id

    if event == cv2.EVENT_LBUTTONDOWN:

        # Ignore dashboard clicks
        if x >= CAM_W:
            return

        for face in current_faces:

            x1, y1, x2, y2, face_id = face

            if x1 <= x <= x2 and y1 <= y <= y2:

                selected_id = face_id

                print("Selected Face:", face_id)
                break


# ==========================================
# DASHBOARD
# ==========================================

def draw_dashboard(frame, people, status):

    panel = np.zeros((CAM_H, PANEL_W, 3), dtype=np.uint8)

    # Header
    cv2.putText(
        panel, "AI FACE TRACKER",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 180), 2
    )

    cv2.line(
        panel, (15, 55), (PANEL_W - 15, 55),
        (0, 130, 100), 1
    )

    # System status
    cv2.putText(
        panel, "SYSTEM STATUS",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (150, 150, 150), 1
    )

    color = (0, 255, 0) if status == "FACE LOCKED" else (0, 180, 255)

    cv2.putText(
        panel, status,
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, color, 2
    )

    # Face count
    cv2.putText(
        panel, "DETECTED FACES",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (150, 150, 150), 1
    )

    cv2.putText(
        panel, str(people),
        (20, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3, (0, 255, 180), 2
    )

    # Selected ID
    cv2.putText(
        panel, "SELECTED FACE ID",
        (20, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (150, 150, 150), 1
    )

    selected_text = str(selected_id) if selected_id is not None else "NONE"

    cv2.putText(
        panel, selected_text,
        (20, 325),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 255, 0), 2
    )

    # FPS
    cv2.putText(
        panel, f"FPS: {fps:.1f}",
        (20, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 1
    )

    cv2.line(
        panel, (15, 425), (PANEL_W - 15, 425),
        (0, 130, 100), 1
    )

    cv2.putText(
        panel, "CONTROLS",
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 180), 1
    )

    cv2.putText(
        panel, "CLICK: SELECT FACE",
        (20, 490),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (220, 220, 220), 1
    )

    cv2.putText(
        panel, "C: CLEAR  |  Q: QUIT",
        (20, 515),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (220, 220, 220), 1
    )

    return np.hstack((frame, panel))


# ==========================================
# MAIN WINDOW
# ==========================================

window = "AI FACE TRACKING SYSTEM"

cv2.namedWindow(window)
cv2.setMouseCallback(window, mouse_click)

print("AI Face Tracking Started!")
print("Click a face to select.")
print("C = Clear | Q = Quit")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame error!")
        break

    frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (CAM_W, CAM_H))

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    detections = detect_faces(gray)

    update_tracks(detections)

    current_faces = []

    # Draw markers only, never draw face rectangles
    for face_id, data in tracks.items():

        if data["missed"] > 0:
            continue

        x, y, w, h = data["box"]

        # Internal selection area only
        current_faces.append(
            (x, y, x + w, y + h, face_id)
        )

        # Approximate forehead position
        cx = data["center"][0]
        cy = y + int(h * 0.24)

        selected = face_id == selected_id

        if selected:
            color = (0, 255, 0)
        else:
            color = (0, 165, 255)

        draw_target(
            frame, cx, cy, color, selected
        )

        # Small ID label
        cv2.putText(
            frame,
            f"ID {face_id}",
            (cx + 32, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # Status
    visible_ids = [p[4] for p in current_faces]

    if selected_id is None:
        status = "NO FACE SELECTED"

    elif selected_id in visible_ids:
        status = "FACE LOCKED"

    else:
        status = "FACE LOST"

    # FPS calculation
    now = time.time()
    delta = now - last_time

    if delta > 0:
        fps = 1 / delta

    last_time = now

    # Dashboard
    output = draw_dashboard(
        frame,
        len(current_faces),
        status
    )

    cv2.imshow(window, output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        break

    elif key == ord('c'):
        selected_id = None
        print("Selection cleared!")


# ==========================================
# RELEASE RESOURCES
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("System Closed!")
