"""
Baseer (بصير) — AI Medical Emergency & Triage Command Center
================================================================
Single-file Streamlit application for crowd surveillance, pre-collapse /
anomaly movement detection, clinical triage mapping, and interactive
emergency dispatch. Includes a deterministic offline simulation mode so
the demo runs with zero external dependencies (no video upload required).

Run:  streamlit run app.py
"""

import math
import os
import tempfile
import time
import uuid
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np
import streamlit as st

# ============================================================================
# 1. PAGE CONFIG (must be the first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="Baseer | بصير — AI Medical Emergency & Triage Command Center",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 2. THEME / CSS — Command Center dark-slate + neon cyan look
# ============================================================================
CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #0B132B 0%, #0F172A 100%);
    color: #E2E8F0;
}
section[data-testid="stSidebar"] {
    background: #0B1226;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * { color: #CBD5E1; }

.header-banner {
    padding: 18px 26px;
    border-radius: 14px;
    background: linear-gradient(90deg, rgba(56,189,248,0.14), rgba(11,19,43,0.35));
    border: 1px solid rgba(56,189,248,0.35);
    margin-bottom: 10px;
}
.header-title { font-size: 34px; font-weight: 800; color: #38BDF8; letter-spacing: 1px; }
.header-title .ar { color: #F1F5F9; font-weight: 700; margin-left: 10px; }
.header-sub { color: #94A3B8; font-size: 14px; margin-top: 6px; }

div[data-testid="stMetric"] {
    background: #111C3B;
    border-radius: 10px;
    padding: 10px 6px;
    border: 1px solid #1E293B;
}
div[data-testid="stMetricValue"] { color: #38BDF8; }

.triage-card {
    background: #111C3B;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.45);
}
.card-top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.cond-name { font-weight: 700; font-size: 15px; color: #F1F5F9; }
.cond-ar { color: #94A3B8; font-size: 13px; margin-top: 3px; }
.card-meta { color: #64748B; font-size: 12px; margin-top: 7px; }

.priority-badge {
    padding: 2px 11px; border-radius: 20px; font-size: 11px;
    font-weight: 800; letter-spacing: 0.6px;
}
.dispatch-badge {
    background: #14321F; color: #4ADE80; border: 1px solid #22C55E;
    padding: 2px 11px; border-radius: 20px; font-size: 11px; font-weight: 700;
}

.stButton>button {
    background: linear-gradient(90deg, #0EA5E9, #38BDF8);
    color: #04121F; font-weight: 700; border: none; border-radius: 8px;
}
.stButton>button:hover { filter: brightness(1.12); }

hr { border-color: #1E293B; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# 3. CLINICAL TAXONOMY  (Binary: Normal=0 / Abnormal=1) + Triage mapping
# ============================================================================
TAXONOMY = {
    "Physical Violence & Assaults": {
        "boxing_fighting": {"ar": "شجار / ملاكمة", "priority": "High",
                             "action": "Dispatch security + medical unit; separate & restrain safely."},
        "kicking_assault": {"ar": "اعتداء بالركل", "priority": "High",
                             "action": "Dispatch security + medical unit immediately; assess for trauma."},
    },
    "Falls & Abnormal Locomotion": {
        "sudden_fall": {"ar": "سقوط مفاجئ", "priority": "Critical",
                         "action": "Immediate EMS dispatch; check consciousness, airway & spine precautions."},
        "slow_fall": {"ar": "سقوط تدريجي", "priority": "Critical",
                      "action": "Immediate EMS dispatch; assess for syncope / cardiac event."},
        "fall_and_recovery": {"ar": "سقوط مع تعافٍ", "priority": "Medium",
                               "action": "Send first-aid team to verify condition & vitals."},
        "severe_gait_limping": {"ar": "عرج شديد", "priority": "High",
                                 "action": "Dispatch mobility support & medical assessment for limb injury."},
        "irregular_limping": {"ar": "عرج غير منتظم", "priority": "Medium",
                               "action": "Monitor closely; send first-aid team if pattern persists."},
        "crawling_on_floor": {"ar": "زحف على الأرض", "priority": "Critical",
                               "action": "Immediate EMS dispatch; possible collapse or severe injury."},
        "crawling_exhausted": {"ar": "زحف من الإرهاق", "priority": "High",
                                "action": "Dispatch hydration & medical support team."},
        "stooped_walking_resting": {"ar": "مشي منحنٍ / استراحة قسرية", "priority": "High",
                                     "action": "Send welfare-check team; monitor for pre-collapse signs."},
        "arm_injury": {"ar": "إصابة بالذراع", "priority": "Medium",
                        "action": "Send first-aid team for limb assessment & immobilization."},
    },
    "Medical & Respiratory Distress": {
        "heatstroke_exhaustion": {"ar": "ضربة شمس / إجهاد حراري", "priority": "Critical",
                                   "action": "Immediate active cooling + EMS dispatch; IV fluids on arrival."},
        "severe_choking_on_ground": {"ar": "اختناق شديد على الأرض", "priority": "Critical",
                                      "action": "Immediate EMS + airway management team; prepare for CPR."},
        "choking_cough": {"ar": "سعال / اختناق خفيف", "priority": "High",
                           "action": "Dispatch medical team for airway check & observation."},
        "seizure_convulsion": {"ar": "نوبة تشنجية", "priority": "Critical",
                                "action": "Immediate EMS; protect from injury, do not restrain, time the seizure."},
        "rapid_breathing": {"ar": "تسارع في التنفس", "priority": "Medium",
                             "action": "Monitor vitals; send medical team for respiratory assessment."},
    },
    "Fast Movement & Dynamic Activities": {
        "running_sprinting": {"ar": "جري سريع", "priority": "Low",
                               "action": "Monitor for crowd-surge / stampede risk; log event."},
        "jogging": {"ar": "هرولة", "priority": "Normal", "action": "No action required."},
        "jumping": {"ar": "قفز", "priority": "Low", "action": "Monitor local crowd density."},
        "dancing": {"ar": "رقص", "priority": "Normal", "action": "No action required."},
        "situps_exercise": {"ar": "تمارين رياضية", "priority": "Normal", "action": "No action required."},
    },
    "Object Interaction & Environmental Events": {
        "bag_throwing_airborne": {"ar": "رمي حقيبة / جسم", "priority": "Medium",
                                   "action": "Security review; check for unattended objects."},
        "flying_papers": {"ar": "تطاير أوراق", "priority": "Low",
                           "action": "Log event; no dispatch needed."},
    },
}

CODE_INFO = {}
for _cat, _items in TAXONOMY.items():
    for _code, _info in _items.items():
        CODE_INFO[_code] = {**_info, "category": _cat}

PRIORITY_COLORS = {
    "Critical": "#EF4444",
    "High": "#F97316",
    "Medium": "#F59E0B",
    "Low": "#38BDF8",
    "Normal": "#22C55E",
}
PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Normal": 4}

CAMERA_ZONES = [
    "Zone A — Main Gate / البوابة الرئيسية",
    "Zone B — Mataf Area / منطقة المطاف",
    "Zone C — Jamarat Bridge / جسر الجمرات",
    "Zone D — Central Station / المحطة المركزية",
    "Zone E — Outer Courtyard / الساحة الخارجية",
]


def hex_to_bgr(hexcolor: str):
    hexcolor = hexcolor.lstrip("#")
    r, g, b = tuple(int(hexcolor[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def build_alert(cond, conf, zone, frame_idx, start_time, fps, track_id=None):
    info = CODE_INFO[cond]
    ts = start_time + timedelta(seconds=frame_idx / max(fps, 1))
    return {
        "id": uuid.uuid4().hex,
        "code": cond,
        "category": info["category"],
        "ar": info["ar"],
        "priority": info["priority"],
        "action": info["action"],
        "confidence": conf,
        "zone": zone,
        "frame": frame_idx,
        "timestamp": ts.strftime("%H:%M:%S"),
        "track_id": track_id,
    }


# ============================================================================
# 4. VISION PIPELINE — background subtraction + centroid tracking + kinematics
# ============================================================================
class Track:
    def __init__(self, track_id, centroid, bbox, frame_idx):
        self.id = track_id
        self.centroids = deque(maxlen=20)
        self.bboxes = deque(maxlen=20)
        self.centroids.append(centroid)
        self.bboxes.append(bbox)
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.missed = 0
        self.last_alert = {}


class CentroidTracker:
    """Lightweight greedy centroid tracker with track-age persistence."""

    def __init__(self, max_missed=15, max_distance=90):
        self.tracks = {}
        self.next_id = 0
        self.max_missed = max_missed
        self.max_distance = max_distance
        self.pair_alerts = {}  # (id1, id2, condition) -> frame_idx, for altercation cooldown

    def _register(self, cx, cy, bbox, frame_idx):
        t = Track(self.next_id, (cx, cy), bbox, frame_idx)
        self.tracks[self.next_id] = t
        self.next_id += 1

    def update(self, detections, frame_idx):
        if not self.tracks:
            for cx, cy, bbox in detections:
                self._register(cx, cy, bbox, frame_idx)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroids[-1] for tid in track_ids]

        if detections:
            D = np.zeros((len(track_centroids), len(detections)))
            for i, tc in enumerate(track_centroids):
                for j, (cx, cy, bbox) in enumerate(detections):
                    D[i, j] = math.hypot(tc[0] - cx, tc[1] - cy)

            assigned_rows, assigned_cols = set(), set()
            flat_idx = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
            for r, c in flat_idx:
                r, c = int(r), int(c)
                if r in assigned_rows or c in assigned_cols:
                    continue
                if D[r, c] > self.max_distance:
                    continue
                tid = track_ids[r]
                cx, cy, bbox = detections[c]
                self.tracks[tid].centroids.append((cx, cy))
                self.tracks[tid].bboxes.append(bbox)
                self.tracks[tid].last_frame = frame_idx
                self.tracks[tid].missed = 0
                assigned_rows.add(r)
                assigned_cols.add(c)

            for j, (cx, cy, bbox) in enumerate(detections):
                if j not in assigned_cols:
                    self._register(cx, cy, bbox, frame_idx)
            for i, tid in enumerate(track_ids):
                if i not in assigned_rows:
                    self.tracks[tid].missed += 1
        else:
            for tid in track_ids:
                self.tracks[tid].missed += 1

        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]

        return self.tracks


def get_features(track: Track):
    """Scale/perspective-normalized kinematic features for a track."""
    if len(track.bboxes) < 2:
        return None
    bboxes = list(track.bboxes)
    centroids = list(track.centroids)

    disps = []
    for i in range(1, len(centroids)):
        dx = centroids[i][0] - centroids[i - 1][0]
        dy = centroids[i][1] - centroids[i - 1][1]
        d = math.hypot(dx, dy)
        hh = bboxes[i][3] if bboxes[i][3] > 0 else 1
        disps.append(d / hh)  # normalize displacement by current person height

    avg_disp = float(np.mean(disps)) if disps else 0.0
    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    x_std = float(np.std(xs))
    y_std = float(np.std(ys))
    heights = [b[3] for b in bboxes]
    height_ratio = heights[-1] / max(heights[0], 1)
    aspect = bboxes[-1][2] / max(bboxes[-1][3], 1)
    height_drop_rate = (heights[0] - heights[-1]) / max(len(heights), 1)

    return dict(avg_disp=avg_disp, x_std=x_std, y_std=y_std,
                height_ratio=height_ratio, aspect=aspect,
                height_drop_rate=height_drop_rate)


def classify_track(feats):
    """Heuristic kinematic classifier mapping motion features -> taxonomy code."""
    ar, hr = feats["aspect"], feats["height_ratio"]
    disp, xstd, ystd, drop = feats["avg_disp"], feats["x_std"], feats["y_std"], feats["height_drop_rate"]

    # --- Lying-down states (low height ratio, wide aspect) ---
    if hr < 0.55 and ar > 1.25:
        if drop > 1.2:
            return "sudden_fall", round(min(0.95, 0.75 + drop * 0.05), 2)
        elif drop > 0.3:
            return "slow_fall", round(min(0.90, 0.70 + drop * 0.10), 2)
        elif disp > 0.15:
            return "crawling_on_floor", round(min(0.90, 0.65 + disp), 2)
        elif ystd > 4 and xstd < 2:
            return "seizure_convulsion", round(min(0.92, 0.70 + ystd * 0.02), 2)
        else:
            return "severe_choking_on_ground", 0.78

    # --- Stooped / partial crouch states ---
    if 0.55 <= hr < 0.78 and ar > 0.9:
        if disp < 0.05:
            return "stooped_walking_resting", 0.72
        elif disp < 0.12:
            return "crawling_exhausted", 0.74
        else:
            return "severe_gait_limping", 0.70

    # --- Upright dynamic states ---
    if hr >= 0.78:
        if disp > 0.55:
            return "running_sprinting", round(min(0.90, 0.60 + disp * 0.30), 2)
        if xstd > 10 and disp < 0.20:
            return "heatstroke_exhaustion", round(min(0.88, 0.60 + xstd * 0.02), 2)
        if xstd > 6 and disp > 0.20:
            return "irregular_limping", 0.68
        if disp > 0.30:
            return "jogging", 0.60

    return None, 0.0


def process_uploaded_video(file_bytes, zone, min_area, cooldown_s, min_track_age, max_frames):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps_src <= 1:
        fps_src = 25.0

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tracker = CentroidTracker(max_missed=15, max_distance=90)

    alerts, frames, frame_meta = [], [], []
    prev_small = []
    last_object_alert = {}
    start_time = datetime.now()
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        new_w = 480
        new_h = int(new_w * frame.shape[0] / frame.shape[1])
        frame = cv2.resize(frame, (new_w, new_h))

        # --- Background subtraction + anti-clutter morphology ---
        fgmask = bg_sub.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections, small_candidates = [], []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append((x + w // 2, y + h // 2, (x, y, w, h)))
            elif min_area * 0.10 <= area < min_area * 0.5:
                x, y, w, h = cv2.boundingRect(c)
                small_candidates.append((x + w // 2, y + h // 2, area, w / max(h, 1)))

        tracks = tracker.update(detections, frame_idx)

        # --- Per-track classification ---
        for tid, track in tracks.items():
            age = frame_idx - track.first_frame
            x, y, w, h = track.bboxes[-1]
            color = (148, 163, 184)
            if age >= min_track_age:
                feats = get_features(track)
                if feats:
                    cond, conf = classify_track(feats)
                    if cond:
                        info = CODE_INFO[cond]
                        color = hex_to_bgr(PRIORITY_COLORS[info["priority"]])
                        last = track.last_alert.get(cond, -9999)
                        if frame_idx - last >= cooldown_s * fps_src:
                            alerts.append(build_alert(cond, conf, zone, frame_idx, start_time, fps_src, tid))
                            track.last_alert[cond] = frame_idx
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID{tid}", (x, max(y - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # --- Pairwise altercation detection (Physical Violence & Assaults) ---
        ids = list(tracks.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                t1, t2 = tracks[ids[i]], tracks[ids[j]]
                if len(t1.centroids) < 2 or len(t2.centroids) < 2:
                    continue
                c1, c2 = t1.centroids[-1], t2.centroids[-1]
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                h_avg = (t1.bboxes[-1][3] + t2.bboxes[-1][3]) / 2
                if dist < h_avg * 0.6:
                    f1, f2 = get_features(t1), get_features(t2)
                    if f1 and f2 and (f1["avg_disp"] + f2["avg_disp"]) > 0.5:
                        cond = "kicking_assault" if abs(c1[1] - c2[1]) > 15 else "boxing_fighting"
                        key = (ids[i], ids[j], cond)
                        if frame_idx - tracker.pair_alerts.get(key, -9999) >= cooldown_s * fps_src:
                            conf = round(min(0.90, 0.60 + (f1["avg_disp"] + f2["avg_disp"]) * 0.20), 2)
                            alerts.append(build_alert(cond, conf, zone, frame_idx, start_time, fps_src))
                            tracker.pair_alerts[key] = frame_idx

        # --- Small-object environmental events (thrown bag / flying papers) ---
        matched_prev = set()
        for (cx, cy, area, aspect) in small_candidates:
            best_idx, best_d = None, 999
            for idx2, (pcx, pcy, _, _) in enumerate(prev_small):
                d = math.hypot(cx - pcx, cy - pcy)
                if d < best_d:
                    best_d, best_idx = d, idx2
            if best_idx is not None and best_d < 60 and best_idx not in matched_prev:
                matched_prev.add(best_idx)
                if best_d > 18:
                    cond = "bag_throwing_airborne" if 0.6 <= aspect <= 1.6 else "flying_papers"
                    if frame_idx - last_object_alert.get(cond, -9999) >= cooldown_s * fps_src:
                        conf = round(min(0.85, 0.50 + best_d * 0.01), 2)
                        alerts.append(build_alert(cond, conf, zone, frame_idx, start_time, fps_src))
                        last_object_alert[cond] = frame_idx
        prev_small = small_candidates

        active_tracks = len(tracks)
        elapsed = max(time.time() - t0, 1e-6)
        proc_fps = (frame_idx + 1) / elapsed
        cv2.putText(frame, f"F{frame_idx} | Tracks:{active_tracks} | {proc_fps:.1f} FPS",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_meta.append({
            "frame": frame_idx,
            "time": start_time + timedelta(seconds=frame_idx / fps_src),
            "active_tracks": active_tracks,
            "fps": round(proc_fps, 1),
        })
        frame_idx += 1

    cap.release()
    try:
        os.unlink(tmp.name)
    except Exception:
        pass

    return alerts, frames, frame_meta


# ============================================================================
# 5. DETERMINISTIC SIMULATION MODE — Normal walk -> Heatstroke -> Stoop ->
#    Sudden fall -> Immobilized on ground
# ============================================================================
STAGE_CONDITION = {
    "walk": None,
    "heatstroke": "heatstroke_exhaustion",
    "stoop": "stooped_walking_resting",
    "fall": "sudden_fall",
    "immobile": "severe_choking_on_ground",
}


def sim_stage(frame_idx, total):
    p = frame_idx / total
    if p < 0.20:
        return "walk"
    if p < 0.45:
        return "heatstroke"
    if p < 0.60:
        return "stoop"
    if p < 0.72:
        return "fall"
    return "immobile"


def make_background(w, h, frame_idx):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, 4):
        shade = int(20 + 15 * math.sin(y / 40.0))
        bg[y:y + 4, :] = (15 + shade // 3, 20 + shade // 2, 35 + shade)
    cv2.rectangle(bg, (0, int(h * 0.78)), (w, h), (30, 35, 55), -1)
    cv2.putText(bg, "BASEER SIMULATION FEED", (14, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (56, 189, 248), 1, cv2.LINE_AA)
    return bg


def draw_humanoid(canvas, frame_idx, total):
    h_img, w_img = canvas.shape[:2]
    stage = sim_stage(frame_idx, total)
    t = frame_idx
    progress = frame_idx / total
    x = int(60 + progress * (w_img - 160))
    ground_y = int(h_img * 0.78)
    color = (232, 232, 238)

    if stage == "walk":
        bw, bh, lean, wobble = 34, 92, 0, 6 * math.sin(t * 0.5)
    elif stage == "heatstroke":
        bw, bh, lean, wobble = 36, 84, 10 * math.sin(t * 0.6), 20 * math.sin(t * 0.9)
    elif stage == "stoop":
        bw, bh, lean, wobble = 40, 58, 30, 4 * math.sin(t * 0.4)
    elif stage == "fall":
        bw, bh, lean, wobble = 60, 30, 75, 0
    else:  # immobile
        bw, bh, lean, wobble = 62, 20, 85, 0

    cx = x + int(wobble)
    cy = ground_y - bh // 2
    cv2.ellipse(canvas, (cx, cy), (bw // 2, bh // 2), lean, 0, 360, color, -1)
    head_y = max(ground_y - bh - 10, 12) if stage in ("fall", "immobile") else ground_y - bh - 14
    cv2.circle(canvas, (cx, head_y), 13, color, -1)

    lean_r = abs(lean)
    width = int(bw + lean_r * 0.9)
    height = int(max(bh + 20 - lean_r * 0.5, 20))
    x0, y0 = cx - width // 2, ground_y - height
    bbox = (x0, y0, width, height)
    return bbox, stage


def run_simulation(sim_len, zone, cooldown_s):
    alerts, frames, frame_meta = [], [], []
    last_alert_frame = {}
    fps_est = 25
    w, h = 480, 300
    start_time = datetime.now()

    for i in range(sim_len):
        canvas = make_background(w, h, i)
        bbox, stage = draw_humanoid(canvas, i, sim_len)
        cond = STAGE_CONDITION[stage]

        box_color = (34, 197, 94)
        if cond:
            info = CODE_INFO[cond]
            box_color = hex_to_bgr(PRIORITY_COLORS[info["priority"]])

        x0, y0, ww, hh = bbox
        cv2.rectangle(canvas, (x0, y0), (x0 + ww, y0 + hh), box_color, 2)
        label = stage.upper() if not cond else cond
        cv2.putText(canvas, label, (x0, max(y0 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"F{i:03d}", (w - 74, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

        if cond:
            last = last_alert_frame.get(cond, -9999)
            if i - last >= cooldown_s * fps_est:
                conf = round(min(0.97, max(0.75, 0.80 + 0.15 * abs(math.sin(i * 0.13)))), 2)
                alerts.append(build_alert(cond, conf, zone, i, start_time, fps_est))
                last_alert_frame[cond] = i

        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        frame_meta.append({
            "frame": i,
            "time": start_time + timedelta(seconds=i / fps_est),
            "active_tracks": 1,
            "fps": fps_est,
        })

    return alerts, frames, frame_meta


# ============================================================================
# 6. SESSION STATE
# ============================================================================
def init_state():
    defaults = {
        "alerts": [],
        "frames": [],
        "frame_meta": [],
        "dispatch_status": {},
        "processed": False,
        "run_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ============================================================================
# 7. SIDEBAR — CONTROL PANEL
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel / لوحة التحكم")
    mode = st.radio(
        "Input Source / مصدر الفيديو",
        ["🧪 Simulation Mode", "📤 Upload Video"],
        key="mode_radio",
    )
    zone = st.selectbox("Camera Zone / منطقة الكاميرا", CAMERA_ZONES, key="zone_select")

    st.markdown("### 🎚️ Detection Sensitivity")
    min_area = st.slider("Min Contour Area (px)", 200, 5000, 800, step=100, key="min_area_slider")
    cooldown_s = st.slider("Alert Cooldown (seconds)", 1, 15, 5, key="cooldown_slider")
    min_track_age = st.slider("Min Track Age (frames)", 3, 30, 8, key="min_age_slider")

    st.markdown("---")
    uploaded_file = None
    max_frames = 300
    sim_len = 200
    if mode == "📤 Upload Video":
        uploaded_file = st.file_uploader(
            "Upload Surveillance Clip / رفع مقطع المراقبة",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )
        max_frames = st.slider("Max Frames to Process", 60, 600, 300, step=20, key="max_frames_slider")
    else:
        sim_len = st.slider("Simulation Length (frames)", 120, 300, 200, step=10, key="sim_len_slider")

    st.markdown("---")
    run_clicked = st.button("▶️ Start Analysis / بدء التحليل", key="run_button", use_container_width=True)
    reset_clicked = st.button("🔄 Reset Session / إعادة تعيين", key="reset_button", use_container_width=True)

if reset_clicked:
    for k in ["alerts", "frames", "frame_meta", "dispatch_status", "processed", "run_id"]:
        st.session_state.pop(k, None)
    init_state()
    st.rerun()

# ============================================================================
# 8. HEADER
# ============================================================================
st.markdown(
    """
    <div class="header-banner">
      <div class="header-title">🩺 BASEER <span class="ar">بصير</span></div>
      <div class="header-sub">AI Medical Emergency &amp; Triage Command Center —
      مركز قيادة الطوارئ الطبية بالذكاء الاصطناعي</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "⚠️ Illustrative heuristic vision system for crowd-safety triage demonstration. "
    "Not a certified medical device. / نظام توضيحي غير معتمد كجهاز طبي."
)

# ============================================================================
# 9. TRIGGER PROCESSING
# ============================================================================
if run_clicked:
    if mode == "🧪 Simulation Mode":
        with st.spinner("Running deterministic simulation... / تشغيل المحاكاة الحتمية"):
            alerts, frames, frame_meta = run_simulation(sim_len, zone, cooldown_s)
        st.session_state.alerts = alerts
        st.session_state.frames = frames
        st.session_state.frame_meta = frame_meta
        st.session_state.processed = True
        st.session_state.dispatch_status = {}
        st.session_state.run_id = uuid.uuid4().hex
    else:
        if uploaded_file is None:
            st.warning("Please upload a video file first. / يرجى رفع ملف فيديو أولاً")
        else:
            try:
                with st.spinner("Processing video feed... / جارٍ تحليل بث الفيديو"):
                    alerts, frames, frame_meta = process_uploaded_video(
                        uploaded_file.read(), zone, min_area, cooldown_s, min_track_age, max_frames
                    )
                st.session_state.alerts = alerts
                st.session_state.frames = frames
                st.session_state.frame_meta = frame_meta
                st.session_state.processed = True
                st.session_state.dispatch_status = {}
                st.session_state.run_id = uuid.uuid4().hex
            except Exception as e:
                st.error(f"Video processing failed: {e}")

# ============================================================================
# 10. TOP KPI ROW
# ============================================================================
kpi = st.columns(4)
total_alerts = len(st.session_state.alerts)
critical_count = sum(1 for a in st.session_state.alerts if a["priority"] == "Critical")
dispatched_count = len(st.session_state.dispatch_status)
active_now = st.session_state.frame_meta[-1]["active_tracks"] if st.session_state.frame_meta else 0
kpi[0].metric("Total Alerts / إجمالي التنبيهات", total_alerts)
kpi[1].metric("🔴 Critical / حرجة", critical_count)
kpi[2].metric("🚑 Dispatched / تم التوجيه", dispatched_count)
kpi[3].metric("👥 Active Tracks / مسارات نشطة", active_now)

st.markdown("---")

# ============================================================================
# 11. MAIN LAYOUT
# ============================================================================
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.markdown("### 🎥 Video Feed / بث الفيديو")
    if st.session_state.processed and st.session_state.frames:
        n = len(st.session_state.frames)
        idx = st.slider(
            "Frame Scrubber / شريط استعراض الإطارات",
            0, n - 1, 0, key="frame_scrubber_slider"
        )
        st.image(st.session_state.frames[idx], use_container_width=True)

        meta = st.session_state.frame_meta[idx]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Frame", meta["frame"])
        m2.metric("Time", meta["time"].strftime("%H:%M:%S"))
        m3.metric("Active Tracks", meta["active_tracks"])
        m4.metric("Proc. FPS", meta["fps"])
        m5.metric("Total Alerts", len(st.session_state.alerts))
    else:
        st.info(
            "Configure the input source in the sidebar and click ▶️ Start Analysis. "
            "/ اضبط مصدر الفيديو من اللوحة الجانبية ثم اضغط بدء التحليل."
        )

with col_right:
    st.markdown("### 🩺 Triage Feed / لوحة الفرز")

    if not st.session_state.alerts:
        st.info("No alerts yet. / لا توجد تنبيهات حتى الآن")
    else:
        priorities = list(PRIORITY_ORDER.keys())
        selected_priorities = st.multiselect(
            "Filter by Priority / تصفية حسب الأولوية",
            priorities, default=priorities, key="priority_filter",
        )

        sorted_alerts = sorted(st.session_state.alerts, key=lambda a: -a["frame"])

        for idx, alert in enumerate(sorted_alerts):
            if alert["priority"] not in selected_priorities:
                continue

            pcolor = PRIORITY_COLORS[alert["priority"]]
            status = st.session_state.dispatch_status.get(alert["id"])
            badge_html = (
                f'<span class="priority-badge" '
                f'style="background:{pcolor}22;color:{pcolor};border:1px solid {pcolor};">'
                f'{alert["priority"].upper()}</span>'
            )
            status_html = f'<span class="dispatch-badge">🚑 {status}</span>' if status else ""

            st.markdown(
                f"""
                <div class="triage-card" style="border-left:4px solid {pcolor};">
                  <div class="card-top">
                    <span class="cond-name">{alert['code'].replace('_', ' ').title()}</span>
                    {badge_html} {status_html}
                  </div>
                  <div class="cond-ar">{alert['ar']}</div>
                  <div class="card-meta">📍 {alert['zone']} &nbsp;|&nbsp; 🕒 {alert['timestamp']}
                  &nbsp;|&nbsp; 🎞️ Frame {alert['frame']} &nbsp;|&nbsp; 🎯 {int(alert['confidence']*100)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"Details — {alert['category']}"):
                st.write(f"**Condition Code:** `{alert['code']}`")
                st.write(f"**Confidence:** {alert['confidence'] * 100:.0f}%")
                st.write(f"**Camera Zone:** {alert['zone']}")
                st.write(f"**Recommended Clinical Action:** {alert['action']}")
                if alert.get("track_id") is not None:
                    st.write(f"**Track ID:** {alert['track_id']}")

                btn_key = f"dsp_{alert['id']}_{idx}"
                if st.session_state.dispatch_status.get(alert["id"]):
                    st.success(f"Unit dispatched — {st.session_state.dispatch_status[alert['id']]}")
                else:
                    if st.button("🚑 Dispatch Unit / توجيه فرقة", key=btn_key, use_container_width=True):
                        st.session_state.dispatch_status[alert["id"]] = "En Route"
                        st.rerun()

st.markdown("---")
st.caption(
    "Baseer / بصير — Vision pipeline: MOG2 background subtraction, morphological anti-clutter filtering, "
    "height-normalized centroid tracking with cooldown-debounced clinical triage classification."
)
