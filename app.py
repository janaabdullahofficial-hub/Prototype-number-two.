import os
import cv2
import time
import math
import uuid
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="بصير | Baseer - VisionAid",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(0, 180, 180, 0.08), transparent 30%),
            radial-gradient(circle at 90% 80%, rgba(255, 70, 70, 0.06), transparent 30%),
            #070b11;
        color: #e8eef5;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .baseer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 24px;
        margin-bottom: 20px;
        border: 1px solid #1d2b38;
        border-radius: 14px;
        background: linear-gradient(135deg, #0d151e, #091018);
        box-shadow: 0 10px 35px rgba(0,0,0,.25);
    }

    .baseer-title {
        font-size: 30px;
        font-weight: 800;
        color: #f4f8fb;
        margin: 0;
    }

    .baseer-title span {
        color: #19d3c5;
    }

    .baseer-subtitle {
        color: #81909f;
        font-size: 14px;
        margin-top: 5px;
    }

    .visionaid {
        text-align: right;
    }

    .visionaid-name {
        color: #19d3c5;
        font-size: 19px;
        font-weight: 800;
    }

    .visionaid-org {
        color: #748392;
        font-size: 12px;
        margin-top: 3px;
    }

    /* Section */
    .section-title {
        font-size: 18px;
        font-weight: 800;
        color: #dfe9f2;
        margin: 4px 0 12px 0;
    }

    /* KPI */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 15px;
    }

    .kpi-card {
        background: #0d151e;
        border: 1px solid #1b2a37;
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 85px;
    }

    .kpi-label {
        color: #7f8e9c;
        font-size: 12px;
        font-weight: 500;
    }

    .kpi-value {
        color: #edf5fa;
        font-size: 27px;
        font-weight: 800;
        margin-top: 5px;
    }

    .kpi-accent {
        color: #19d3c5;
    }

    .kpi-alert {
        color: #ff5555;
    }

    /* Video */
    .video-container {
        border: 1px solid #1b2a37;
        border-radius: 14px;
        overflow: hidden;
        background: #030609;
    }

    /* Alert */
    .triage-log {
        background: #080e15;
        border: 1px solid #182632;
        border-radius: 14px;
        padding: 12px;
        min-height: 500px;
        max-height: 720px;
        overflow-y: auto;
    }

    .alert-card {
        border-radius: 10px;
        padding: 12px 13px;
        margin-bottom: 10px;
        background: #0e161e;
        border: 1px solid #263541;
    }

    .alert-card.critical {
        border-left: 4px solid #ff4242;
    }

    .alert-card.high {
        border-left: 4px solid #ff9d32;
    }

    .alert-card.low {
        border-left: 4px solid #3c9eff;
    }

    .alert-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .alert-condition {
        color: #eef5f9;
        font-weight: 800;
        font-size: 14px;
    }

    .alert-time {
        color: #697b8a;
        font-size: 10px;
    }

    .alert-meta {
        color: #8c9ba8;
        font-size: 11px;
        margin-top: 5px;
    }

    .priority-critical {
        color: #ff5252;
        font-weight: 800;
    }

    .priority-high {
        color: #ffab40;
        font-weight: 800;
    }

    .priority-low {
        color: #4ca6ff;
        font-weight: 800;
    }

    .empty-log {
        color: #526371;
        text-align: center;
        padding: 90px 20px;
        font-size: 14px;
    }

    /* Status */
    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .status-live {
        color: #49e0bd;
        background: rgba(73, 224, 189, .08);
        border: 1px solid rgba(73, 224, 189, .2);
    }

    .status-idle {
        color: #91a0ad;
        background: rgba(145,160,173,.08);
        border: 1px solid rgba(145,160,173,.15);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #080d13;
        border-right: 1px solid #182531;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.3rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #263845;
        background: #101b24;
        color: #dbe7ef;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #19d3c5;
        color: #19d3c5;
    }

    /* Responsive */
    @media (max-width: 900px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .baseer-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
        }

        .visionaid {
            text-align: left;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CLINICAL TAXONOMY
# ============================================================

TAXONOMY = {
    "heatstroke_exhaustion": {
        "ar": "إجهاد حراري / ضربة شمس",
        "priority": "Critical",
        "priority_ar": "حرج",
        "color_class": "critical",
        "action": "إرسال فريق إسعاف فوراً",
    },
    "sudden_fall": {
        "ar": "سقوط مفاجئ",
        "priority": "Critical",
        "priority_ar": "حرج",
        "color_class": "critical",
        "action": "إرسال فريق إسعاف فوراً",
    },
    "slow_fall": {
        "ar": "سقوط تدريجي",
        "priority": "Critical",
        "priority_ar": "حرج",
        "color_class": "critical",
        "action": "إرسال فريق إسعاف وتقييم الحالة",
    },
    "severe_choking_on_ground": {
        "ar": "اختناق شديد على الأرض",
        "priority": "Critical",
        "priority_ar": "حرج",
        "color_class": "critical",
        "action": "إرسال فريق طوارئ فوراً",
    },
    "seizure_convulsion": {
        "ar": "تشنج / نوبة صرع",
        "priority": "Critical",
        "priority_ar": "حرج",
        "color_class": "critical",
        "action": "إرسال فريق طوارئ فوراً",
    },
    "severe_gait_limping": {
        "ar": "عرج شديد أثناء المشي",
        "priority": "High",
        "priority_ar": "مرتفع",
        "color_class": "high",
        "action": "إرسال فريق دعم طبي",
    },
    "stooped_walking_resting": {
        "ar": "انحناء / توقف أثناء المشي",
        "priority": "High",
        "priority_ar": "مرتفع",
        "color_class": "high",
        "action": "إرسال فريق للمراقبة والتقييم",
    },
    "running_sprinting": {
        "ar": "جري / ركض سريع",
        "priority": "Low",
        "priority_ar": "منخفض",
        "color_class": "low",
        "action": "مراقبة الحالة",
    },
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "running": False,
    "alerts": [],
    "frames": 0,
    "tracks": 0,
    "fps": 0.0,
    "alert_count": 0,
    "current_frame": None,
    "video_path": None,
    "video_signature": None,
    "cap": None,
    "bg_subtractor": None,
    "last_time": time.time(),
    "simulation_frame": 0,
    "simulation_phase": "Normal Walk",
    "alert_cooldowns": {},
    "track_history": {},
    "last_detection_time": 0.0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def reset_runtime():
    """Reset runtime state without leaving stale OpenCV handles."""
    cap = st.session_state.get("cap")

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    st.session_state.cap = None
    st.session_state.bg_subtractor = None
    st.session_state.running = False
    st.session_state.alerts = []
    st.session_state.frames = 0
    st.session_state.tracks = 0
    st.session_state.fps = 0.0
    st.session_state.alert_count = 0
    st.session_state.current_frame = None
    st.session_state.simulation_frame = 0
    st.session_state.simulation_phase = "Normal Walk"
    st.session_state.alert_cooldowns = {}
    st.session_state.track_history = {}
    st.session_state.last_detection_time = 0.0


def create_video_capture(uploaded_file):
    """
    Safely persist Streamlit's UploadedFile to disk before
    handing it to OpenCV.
    """
    if uploaded_file is None:
        return None, None

    file_bytes = uploaded_file.getvalue()

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
    }:
        suffix = ".mp4"

    signature = (
        uploaded_file.name,
        len(file_bytes),
        hash(file_bytes[:100000]),
    )

    # Reuse existing capture if the exact same upload is still selected.
    if (
        st.session_state.video_signature == signature
        and st.session_state.cap is not None
    ):
        return st.session_state.cap, st.session_state.video_path

    # Close previous capture.
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass

    # Remove previous temporary video.
    old_path = st.session_state.video_path
    if old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except Exception:
            pass

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix="baseer_",
    )

    temp.write(file_bytes)
    temp.flush()
    temp.close()

    cap = cv2.VideoCapture(temp.name)

    if not cap.isOpened():
        try:
            os.remove(temp.name)
        except Exception:
            pass
        return None, None

    st.session_state.video_path = temp.name
    st.session_state.video_signature = signature
    st.session_state.cap = cap
    st.session_state.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=32,
        detectShadows=False,
    )

    return cap, temp.name


def add_alert(condition_key, frame_idx, zone, confidence=0.85):
    """Add a deduplicated alert to the live triage log."""

    now = time.time()

    # Avoid flooding the log with the same event every frame.
    cooldown = 2.5

    previous = st.session_state.alert_cooldowns.get(condition_key, 0)

    if now - previous < cooldown:
        return

    st.session_state.alert_cooldowns[condition_key] = now

    data = TAXONOMY[condition_key]

    alert = {
        "id": uuid.uuid4().hex[:10],
        "condition": condition_key,
        "ar": data["ar"],
        "priority": data["priority"],
        "priority_ar": data["priority_ar"],
        "action": data["action"],
        "zone": zone,
        "confidence": float(confidence),
        "frame": int(frame_idx),
        "timestamp": time.strftime("%H:%M:%S"),
        "dispatched": False,
    }

    st.session_state.alerts.insert(0, alert)

    # Prevent an unlimited in-memory log.
    st.session_state.alerts = st.session_state.alerts[:50]

    st.session_state.alert_count += 1


def calculate_fps():
    now = time.time()
    previous = st.session_state.last_time

    delta = now - previous

    if delta > 0:
        instant_fps = 1.0 / delta

        if st.session_state.fps <= 0:
            st.session_state.fps = instant_fps
        else:
            # Exponential smoothing.
            st.session_state.fps = (
                0.85 * st.session_state.fps
                + 0.15 * instant_fps
            )

    st.session_state.last_time = now


# ============================================================
# VIDEO PIPELINE
# ============================================================

def detect_people_and_events(frame, sensitivity, zone):
    """
    Lightweight MOG2-based movement detector.

    This is intentionally a simple demo/triage pipeline:
    it does NOT claim to clinically diagnose a medical emergency.
    """

    if st.session_state.bg_subtractor is None:
        st.session_state.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=32,
            detectShadows=False,
        )

    fg = st.session_state.bg_subtractor.apply(frame)

    # Sensitivity controls morphology/area threshold.
    threshold = int(np.clip(2600 - sensitivity * 20, 700, 2200))

    _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < threshold:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 18 or h < 25:
            continue

        if h > frame.shape[0] * 0.95:
            continue

        aspect_ratio = w / max(h, 1)

        cx = x + w / 2
        cy = y + h / 2

        detections.append(
            {
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": aspect_ratio,
                "cx": cx,
                "cy": cy,
            }
        )

    # Largest moving regions first.
    detections.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    detections = detections[:8]

    # Basic pseudo-tracking using nearest-center matching.
    previous_tracks = st.session_state.track_history

    new_tracks = {}
    tracks_count = 0

    for detection in detections:
        cx = detection["cx"]
        cy = detection["cy"]

        best_id = None
        best_distance = float("inf")

        for track_id, previous in previous_tracks.items():
            distance = math.hypot(
                cx - previous["cx"],
                cy - previous["cy"],
            )

            if distance < best_distance and distance < 100:
                best_distance = distance
                best_id = track_id

        if best_id is None:
            best_id = len(previous_tracks) + tracks_count + 1

        previous = previous_tracks.get(best_id)

        speed = 0.0
        vertical_drop = 0.0

        if previous is not None:
            speed = math.hypot(
                cx - previous["cx"],
                cy - previous["cy"],
            )

            vertical_drop = cy - previous["cy"]

        detection["track_id"] = best_id
        detection["speed"] = speed
        detection["vertical_drop"] = vertical_drop

        new_tracks[best_id] = {
            "cx": cx,
            "cy": cy,
            "w": detection["bbox"][2],
            "h": detection["bbox"][3],
            "aspect_ratio": detection["aspect_ratio"],
            "frame": st.session_state.frames,
        }

        tracks_count += 1

    st.session_state.track_history = new_tracks
    st.session_state.tracks = tracks_count

    # --------------------------------------------------------
    # Event heuristics
    # --------------------------------------------------------

    for detection in detections:
        x, y, w, h = detection["bbox"]

        aspect = detection["aspect_ratio"]
        speed = detection["speed"]
        drop = detection["vertical_drop"]

        # Sudden fall:
        # significant downward movement + horizontalized body.
        if drop > 25 and (
            aspect > 1.15
            or h < w * 1.15
        ):
            add_alert(
                "sudden_fall",
                st.session_state.frames,
                zone,
                confidence=0.88,
            )

        # Slow fall:
        elif drop > 10 and aspect > 0.95:
            add_alert(
                "slow_fall",
                st.session_state.frames,
                zone,
                confidence=0.76,
            )

        # Stooped/resting posture:
        elif aspect > 0.65 and aspect < 1.20 and h < frame.shape[0] * 0.35:
            add_alert(
                "stooped_walking_resting",
                st.session_state.frames,
                zone,
                confidence=0.70,
            )

        # Severe gait / limping approximation:
        elif speed > 7 and aspect < 0.55:
            add_alert(
                "severe_gait_limping",
                st.session_state.frames,
                zone,
                confidence=0.68,
            )

        # Running:
        elif speed > 15:
            add_alert(
                "running_sprinting",
                st.session_state.frames,
                zone,
                confidence=0.67,
            )

        # Very wide body lying on ground:
        if aspect > 1.5 and h < frame.shape[0] * 0.18:
            add_alert(
                "severe_choking_on_ground",
                st.session_state.frames,
                zone,
                confidence=0.64,
            )

        # Draw detection.
        color = (35, 211, 197)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2,
        )

        label = f"ID {detection['track_id']}"

        cv2.putText(
            frame,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# SIMULATION PIPELINE
# ============================================================

def get_simulation_phase(frame_index):
    """
    Deterministic 300-frame cycle.

    0-59     Normal Walk
    60-109   Heatstroke / Limping
    110-159  Stooping
    160-189  Sudden Fall
    190-299  Immobilized
    """

    cycle = frame_index % 300

    if cycle < 60:
        return "Normal Walk"

    if cycle < 110:
        return "Heatstroke / Limping"

    if cycle < 160:
        return "Stooping"

    if cycle < 190:
        return "Sudden Fall"

    return "Immobilized"


def draw_simulated_person(frame, frame_index):
    """
    Draw a synthetic human using OpenCV primitives.
    """

    h, w = frame.shape[:2]

    cycle = frame_index % 300
    phase = get_simulation_phase(frame_index)

    # Horizontal walking motion.
    center_x = int(
        w * 0.5
        + math.sin(frame_index * 0.045) * w * 0.22
    )

    ground_y = int(h * 0.78)

    # --------------------------------------------------------
    # NORMAL WALK
    # --------------------------------------------------------

    if cycle < 60:

        head = (center_x, ground_y - 160)
        shoulder = (center_x, ground_y - 125)
        hip = (center_x, ground_y - 70)

        leg_offset = int(math.sin(frame_index * 0.35) * 24)
        arm_offset = int(math.sin(frame_index * 0.35) * 18)

        cv2.ellipse(
            frame,
            head,
            (22, 22),
            0,
            0,
            360,
            (180, 220, 230),
            -1,
        )

        cv2.line(
            frame,
            shoulder,
            hip,
            (180, 220, 230),
            10,
        )

        cv2.line(
            frame,
            shoulder,
            (
                center_x - 35 - arm_offset,
                ground_y - 60,
            ),
            (180, 220, 230),
            8,
        )

        cv2.line(
            frame,
            shoulder,
            (
                center_x + 35 + arm_offset,
                ground_y - 60,
            ),
            (180, 220, 230),
            8,
        )

        cv2.line(
            frame,
            hip,
            (
                center_x - 22 + leg_offset,
                ground_y,
            ),
            (180, 220, 230),
            9,
        )

        cv2.line(
            frame,
            hip,
            (
                center_x + 22 - leg_offset,
                ground_y,
            ),
            (180, 220, 230),
            9,
        )

    # --------------------------------------------------------
    # HEATSTROKE / LIMPING
    # --------------------------------------------------------

    elif cycle < 110:

        head = (center_x, ground_y - 155)
        shoulder = (center_x, ground_y - 120)
        hip = (center_x + 7, ground_y - 65)

        cv2.ellipse(
            frame,
            head,
            (22, 22),
            0,
            0,
            360,
            (180, 220, 230),
            -1,
        )

        cv2.line(
            frame,
            shoulder,
            hip,
            (180, 220, 230),
            10,
        )

        # One hand near head = heat exhaustion cue.
        cv2.line(
            frame,
            shoulder,
            (
                center_x + 48,
                ground_y - 155,
            ),
            (180, 220, 230),
            8,
        )

        cv2.line(
            frame,
            hip,
            (
                center_x - 18,
                ground_y,
            ),
            (180, 220, 230),
            9,
        )

        cv2.line(
            frame,
            hip,
            (
                center_x + 55,
                ground_y - 12,
            ),
            (180, 220, 230),
            9,
        )

        add_alert(
            "heatstroke_exhaustion",
            frame_index,
            st.session_state.camera_zone,
            confidence=0.96,
        )

        add_alert(
            "severe_gait_limping",
            frame_index,
            st.session_state.camera_zone,
            confidence=0.91,
        )

    # --------------------------------------------------------
    # STOOPING
    # --------------------------------------------------------

    elif cycle < 160:

        head = (
            center_x + 55,
            ground_y - 100,
        )

        shoulder = (
            center_x + 30,
            ground_y - 75,
        )

        hip = (
            center_x - 25,
            ground_y - 55,
        )

        cv2.ellipse(
            frame,
            head,
            (22, 22),
            0,
            0,
            360,
            (180, 220, 230),
            -1,
        )

        cv2.line(
            frame,
            shoulder,
            hip,
            (180, 220, 230),
            10,
        )

        cv2.line(
            frame,
            shoulder,
            (
                center_x + 10,
                ground_y - 25,
            ),
            (180, 220, 230),
            8,
        )

        cv2.line(
            frame,
            hip,
            (
                center_x - 50,
                ground_y,
            ),
            (180, 220, 230),
            9,
        )

        cv2.line(
            frame,
            hip,
            (
                center_x + 20,
                ground_y,
            ),
            (180, 220, 230),
            9,
        )

        add_alert(
            "stooped_walking_resting",
            frame_index,
            st.session_state.camera_zone,
            confidence=0.94,
        )

    # --------------------------------------------------------
    # SUDDEN FALL
    # --------------------------------------------------------

    elif cycle < 190:

        fall_progress = (cycle - 160) / 30.0

        fall_progress = np.clip(
            fall_progress,
            0.0,
            1.0,
        )

        x = int(
            center_x
            + fall_progress * 20
        )

        y = int(
            ground_y - 120
            + fall_progress * 95
        )

        cv2.ellipse(
            frame,
            (x + 85, y),
            (22, 22),
            0,
            0,
            360,
            (180, 220, 230),
            -1,
        )

        cv2.line(
            frame,
            (x + 65, y),
            (x - 35, y + 15),
            (180, 220, 230),
            11,
        )

        cv2.line(
            frame,
            (x + 10, y + 10),
            (x - 20, y + 45),
            (180, 220, 230),
            9,
        )

        cv2.line(
            frame,
            (x + 10, y + 10),
            (x + 55, y + 45),
            (180, 220, 230),
            9,
        )

        if cycle >= 166:
            add_alert(
                "sudden_fall",
                frame_index,
                st.session_state.camera_zone,
                confidence=0.99,
            )

    # --------------------------------------------------------
    # IMMOBILIZED
    # --------------------------------------------------------

    else:

        x = center_x

        y = ground_y - 15

        cv2.ellipse(
            frame,
            (x + 95, y),
            (21, 21),
            0,
            0,
            360,
            (180, 220, 230),
            -1,
        )

        cv2.line(
            frame,
            (x + 75, y),
            (x - 70, y),
            (180, 220, 230),
            12,
        )

        cv2.line(
            frame,
            (x - 15, y),
            (x - 80, y + 30),
            (180, 220, 230),
            9,
        )

        cv2.line(
            frame,
            (x + 25, y),
            (x + 85, y + 35),
            (180, 220, 230),
            9,
        )

        # Trigger choking/ground event only periodically.
        if cycle in {200, 240, 280}:
            add_alert(
                "severe_choking_on_ground",
                frame_index,
                st.session_state.camera_zone,
                confidence=0.87,
            )

    return frame, phase


def generate_simulation_frame(frame_index, zone):
    h, w = 720, 1280

    frame = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    # Dark command-center background.
    frame[:] = (7, 12, 18)

    # Subtle floor.
    cv2.rectangle(
        frame,
        (0, int(h * 0.78)),
        (w, h),
        (10, 17, 24),
        -1,
    )

    # Ground line.
    cv2.line(
        frame,
        (0, int(h * 0.78)),
        (w, int(h * 0.78)),
        (70, 95, 110),
        2,
    )

    frame, phase = draw_simulated_person(
        frame,
        frame_index,
    )

    # Simulation overlay.
    cv2.putText(
        frame,
        "BASEER // SIMULATION",
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (80, 215, 200),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"ZONE: {zone}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (140, 160, 175),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"PHASE: {phase}",
        (30, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (210, 220, 230),
        2,
        cv2.LINE_AA,
    )

    return frame, phase


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:800;
            color:#19d3c5;
            margin-bottom:3px;
        ">
            بصير
        </div>
        <div style="
            color:#71808d;
            font-size:12px;
            margin-bottom:20px;
        ">
            AI Early Emergency Triage
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### مصدر التغذية")

    feed_source = st.radio(
        "Feed Source",
        [
            "Simulation Mode",
            "Video Upload",
        ],
        key="feed_source",
    )

    st.markdown("### المنطقة")

    camera_zone = st.selectbox(
        "Camera Zone",
        [
            "ممشى المشاعر",
            "ساحة الحرم",
            "المسعى",
            "مداخل المسجد الحرام",
            "ممرات المشاة",
        ],
        key="camera_zone",
    )

    # Make zone accessible to simulation.
    st.session_state.camera_zone = camera_zone

    st.markdown("### إعدادات الكشف")

    sensitivity = st.slider(
        "Sensitivity",
        min_value=20,
        max_value=100,
        value=65,
        step=5,
        key="sensitivity",
    )

    max_frames = st.slider(
        "Max Frames to Process",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100,
        key="max_frames",
    )

    uploaded_video = None

    if feed_source == "Video Upload":

        uploaded_video = st.file_uploader(
            "Upload Video",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
                "webm",
                "m4v",
            ],
            key="video_uploader",
        )

    st.markdown("---")

    start_col, reset_col = st.columns(2)

    with start_col:
        start_clicked = st.button(
            "▶ Start",
            use_container_width=True,
            key="start_button",
        )

    with reset_col:
        reset_clicked = st.button(
            "↻ Reset",
            use_container_width=True,
            key="reset_button",
        )

    if reset_clicked:
        reset_runtime()
        st.rerun()

    if start_clicked:

        if feed_source == "Video Upload":

            if uploaded_video is None:
                st.warning("Please upload a video first.")
            else:
                cap, _ = create_video_capture(uploaded_video)

                if cap is None:
                    st.error(
                        "Unable to open the uploaded video. "
                        "Please use a supported video format."
                    )
                else:
                    st.session_state.running = True
                    st.session_state.last_time = time.time()

        else:
            st.session_state.running = True
            st.session_state.last_time = time.time()

        st.rerun()

    st.markdown("---")

    status_class = (
        "status-live"
        if st.session_state.running
        else "status-idle"
    )

    status_text = (
        "● SYSTEM ONLINE"
        if st.session_state.running
        else "● SYSTEM IDLE"
    )

    st.markdown(
        f"""
        <div class="status-pill {status_class}">
            {status_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Baseer is a prototype decision-support and "
        "early-triage system. It is not a clinical diagnostic device."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="baseer-header">

        <div>
            <div class="baseer-title">
                <span>بصير</span> — Baseer
            </div>

            <div class="baseer-subtitle">
                AI Early Emergency Triage Command Center
            </div>
        </div>

        <div class="visionaid">
            <div class="visionaid-name">
                VisionAid
            </div>

            <div class="visionaid-org">
                معهد خادم الحرمين الشريفين لأبحاث الحج والعمرة
                <br>
                Umm Al-Qura University
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN LAYOUT
# ============================================================

left_col, right_col = st.columns(
    [1.35, 1],
    gap="large",
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_col:

    st.markdown(
        '<div class="section-title">مراقبة العمليات المباشرة</div>',
        unsafe_allow_html=True,
    )

    # KPI placeholders.
    kpi_placeholder = st.empty()

    video_placeholder = st.empty()

    # Initial render.
    def render_kpis():
        st.session_state.frames = int(
            st.session_state.frames
        )

        st.session_state.tracks = int(
            st.session_state.tracks
        )

        st.session_state.alert_count = int(
            st.session_state.alert_count
        )

        kpi_placeholder.markdown(
            f"""
            <div class="kpi-grid">

                <div class="kpi-card">
                    <div class="kpi-label">Frames</div>
                    <div class="kpi-value">
                        {st.session_state.frames:,}
                    </div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-label">Tracks</div>
                    <div class="kpi-value kpi-accent">
                        {st.session_state.tracks:,}
                    </div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-label">FPS</div>
                    <div class="kpi-value">
                        {st.session_state.fps:.1f}
                    </div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-label">Alerts</div>
                    <div class="kpi-value kpi-alert">
                        {st.session_state.alert_count:,}
                    </div>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    render_kpis()

    if st.session_state.current_frame is not None:

        video_placeholder.image(
            st.session_state.current_frame,
            channels="BGR",
            use_container_width=True,
        )

    else:

        # Empty command-center screen.
        blank = np.zeros(
            (600, 1000, 3),
            dtype=np.uint8,
        )

        blank[:] = (5, 9, 14)

        cv2.putText(
            blank,
            "BASEER",
            (400, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (80, 110, 125),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            blank,
            "Press START to begin monitoring",
            (330, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (65, 90, 105),
            2,
            cv2.LINE_AA,
        )

        video_placeholder.image(
            blank,
            channels="BGR",
            use_container_width=True,
        )


# ============================================================
# RIGHT COLUMN — TRIAGE LOG
# ============================================================

with right_col:

    st.markdown(
        '<div class="section-title">سجل الفرز الطبي المباشر</div>',
        unsafe_allow_html=True,
    )

    triage_placeholder = st.empty()

    def render_triage_log():

        alerts = st.session_state.alerts

        if not alerts:

            triage_placeholder.markdown(
                """
                <div class="triage-log">
                    <div class="empty-log">
                        لا توجد تنبيهات حالياً<br><br>
                        <span style="font-size:12px;">
                            ستظهر التنبيهات هنا عند اكتشاف حالة تستدعي التقييم.
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            return

        cards = []

        for idx, alert in enumerate(alerts):

            priority_class = (
                "priority-critical"
                if alert["priority"] == "Critical"
                else "priority-high"
                if alert["priority"] == "High"
                else "priority-low"
            )

            card_class = alert["priority"].lower()

            dispatch_text = (
                "✓ تم الإرسال"
                if alert["dispatched"]
                else "إرسال فريق"
            )

            cards.append(
                f"""
                <div class="alert-card {card_class}">

                    <div class="alert-top">

                        <div class="alert-condition">
                            {alert["ar"]}
                        </div>

                        <div class="{priority_class}">
                            {alert["priority_ar"]}
                        </div>

                    </div>

                    <div class="alert-meta">
                        {alert["timestamp"]}
                        &nbsp; | &nbsp;
                        {alert["zone"]}
                        &nbsp; | &nbsp;
                        Frame {alert["frame"]}
                    </div>

                    <div class="alert-meta">
                        Confidence:
                        {alert["confidence"] * 100:.0f}%
                        &nbsp; • &nbsp;
                        الإجراء:
                        {alert["action"]}
                    </div>

                </div>
                """
            )

            # IMPORTANT:
            # enumerate() + alert id + frame index guarantee
            # unique Streamlit widget keys.
            #
            # Buttons are rendered below the HTML card.
            #
            # The key contains:
            # dispatch + list index + unique alert id + frame.
            button_key = (
                f"dispatch_"
                f"{idx}_"
                f"{alert['id']}_"
                f"{alert['frame']}"
            )

            # Store button metadata for rendering below.
            cards.append(
                f"""
                <div style="
                    margin-top:-7px;
                    margin-bottom:10px;
                    color:#607381;
                    font-size:10px;
                ">
                    Dispatch ID: {alert["id"]}
                </div>
                """
            )

        triage_placeholder.markdown(
            f"""
            <div class="triage-log">
                {''.join(cards)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render dispatch buttons separately.
        for idx, alert in enumerate(alerts):

            button_key = (
                f"dispatch_"
                f"{idx}_"
                f"{alert['id']}_"
                f"{alert['frame']}"
            )

            if st.button(
                (
                    "✓ تم إرسال الفريق"
                    if alert["dispatched"]
                    else "🚑 إرسال فريق"
                ),
                key=button_key,
                use_container_width=True,
            ):

                st.session_state.alerts[idx]["dispatched"] = True
                st.rerun()

    render_triage_log()


# ============================================================
# LIVE PROCESSING
# ============================================================
#
# Streamlit reruns the script after an interaction.
# To create a browser-friendly live loop, we use a fragment
# when available. Otherwise the application still functions
# through the Start/Reset controls and Streamlit reruns.
# ============================================================

def process_one_frame():
    """
    Process exactly one frame per invocation.

    This prevents an infinite Python loop from blocking
    Streamlit's UI.
    """

    if not st.session_state.running:
        return

    # Stop when frame limit has been reached.
    if st.session_state.frames >= st.session_state.max_frames:
        st.session_state.running = False
        return

    # --------------------------------------------------------
    # SIMULATION
    # --------------------------------------------------------

    if st.session_state.feed_source == "Simulation Mode":

        frame, phase = generate_simulation_frame(
            st.session_state.simulation_frame,
            st.session_state.camera_zone,
        )

        st.session_state.simulation_phase = phase
        st.session_state.simulation_frame += 1

        st.session_state.frames += 1

        # Simulation contains one synthetic tracked person.
        st.session_state.tracks = 1

        calculate_fps()

        st.session_state.current_frame = frame

        return

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = st.session_state.cap

    if cap is None:

        st.session_state.running = False
        return

    ok, frame = cap.read()

    if not ok:

        # End of video.
        cap.release()
        st.session_state.cap = None
        st.session_state.running = False

        return

    # Optional resize for performance.
    max_width = 1100

    if frame.shape[1] > max_width:

        scale = max_width / frame.shape[1]

        frame = cv2.resize(
            frame,
            (
                int(frame.shape[1] * scale),
                int(frame.shape[0] * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    processed = detect_people_and_events(
        frame,
        st.session_state.sensitivity,
        st.session_state.camera_zone,
    )

    st.session_state.frames += 1

    calculate_fps()

    st.session_state.current_frame = processed


# ============================================================
# PROCESS CURRENT FRAME
# ============================================================

process_one_frame()


# ============================================================
# UPDATE UI AFTER PROCESSING
# ============================================================

if st.session_state.current_frame is not None:

    video_placeholder.image(
        st.session_state.current_frame,
        channels="BGR",
        use_container_width=True,
    )

render_kpis()

# Re-render triage log after processing so newly-created alerts
# immediately appear.
render_triage_log()


# ============================================================
# AUTO REFRESH
# ============================================================
#
# Streamlit has historically provided st.rerun(), but continuous
# reruns need a delay. This implementation uses a lightweight
# HTML/JS refresh only while the system is running.
#
# The meta refresh is intentionally short enough to give the
# simulation a live-command-center appearance while avoiding
# an infinite Python-side loop.
# ============================================================

if st.session_state.running:

    st.markdown(
        """
        <script>
        setTimeout(function() {
            window.parent.location.reload();
        }, 180);
        </script>
        """,
        unsafe_allow_html=True,
    )
