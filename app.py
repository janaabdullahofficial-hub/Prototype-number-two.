"""
نظام بصير للإنذار والفرز الإسعافي المبكر
Baseer – AI Early Behavior & Emergency Triage Command Center
جامعة أم القرى | معهد خادم الحرمين الشريفين لأبحاث الحج والعمرة
تقديم فريق: VisionAid
"""

import math
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

# ============================================================================
# إعدادات الواجهة وهوية مشروع بصير
# ============================================================================

st.set_page_config(
    page_title="بصير | نظام الإنذار والفرز الإسعافي المبكر",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&family=JetBrains+Mono:wght@500;700&display=swap');
    * { font-family: 'Tajawal', -apple-system, sans-serif; }
    code, .mono { font-family: 'JetBrains Mono', monospace !important; }

    .block-container { padding-top: 1rem; max-width: 1440px; }
    
    .header-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 100%);
        padding: 1.1rem 1.4rem;
        border-radius: 12px;
        border: 1px solid #3A506B;
        margin-bottom: 1rem;
    }
    .system-title {
        font-size: 1.85rem;
        font-weight: 900;
        background: linear-gradient(90deg, #48CAE4, #00B4D8, #90E0EF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .system-sub { color: #94A3B8; font-size: 0.88rem; margin: 0.2rem 0 0 0; }
    
    .live-badge {
        background: #DC2626;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .kpi-card {
        background: #0D1B2A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.5rem 0.4rem;
        text-align: center;
    }
    .kpi-num { font-size: 1.25rem; font-weight: 700; color: #38BDF8; font-family: 'JetBrains Mono', monospace; }
    .kpi-title { font-size: 0.72rem; color: #64748B; font-weight: 700; }

    .alert-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .triage-badge {
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 800;
        color: white;
        font-family: 'JetBrains Mono', monospace;
    }
    .card-ar { font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-top: 0.35rem; }
    .card-en { font-size: 0.82rem; color: #94A3B8; margin-bottom: 0.25rem; }
    .card-meta { color: #64748B; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; }
    .category-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.7rem;
        margin-bottom: 0.25rem;
    }
    .eta-box {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10B981;
        color: #10B981;
        padding: 6px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# محرك الفرز الطبي والتصنيف السريري المعتمد
# ============================================================================

TAXONOMY_RULES = {
    "heatstroke_exhaustion": {
        "category": "الحالات الطارئة والصحية",
        "ar": "ضربة شمس حادة / إجهاد حراري وبوادر هبوط",
        "en": "heatstroke_exhaustion",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "☀️",
        "action": "توجيه فرقة إسعافية فورية مع معدات التبريد والإرواء",
    },
    "sudden_fall": {
        "category": "السقوط والإصابات الحركية",
        "ar": "سقوط مفاجئ وفقدان فوري للتوازن",
        "en": "sudden_fall",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🚨",
        "action": "توجيه فرقة الإنعاش القلبي والتدخل السريع",
    },
    "slow_fall": {
        "category": "السقوط والإصابات الحركية",
        "ar": "سقوط بطيء وتدريجي (هبوط إعياء حاد)",
        "en": "slow_fall",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "⬇️",
        "action": "فحص العلامات الحيوية ونقل المصاب لمنطقة مظللة",
    },
    "severe_gait_limping": {
        "category": "السقوط والإصابات الحركية",
        "ar": "عرج شديد ومطرد (اختلال توازن حركي)",
        "en": "severe_gait_limping",
        "priority": "High",
        "color": "#F97316",
        "icon": "🚶",
        "action": "توجيه كرسي إسعافي متحرك ومسعف راجل للتقييم",
    },
    "stooped_walking_resting": {
        "category": "السقوط والإصابات الحركية",
        "ar": "مشي بظهر منحنٍ واستناد للراحة عند الرصيف",
        "en": "stooped_walking_resting",
        "priority": "High",
        "color": "#F97316",
        "icon": "🧍",
        "action": "نقل المصاب إلى مظلة رعاية وتفقد الضغط والسكر",
    },
    "seizure_convulsion": {
        "category": "الحالات الطارئة والصحية",
        "ar": "تشنج عصبي نشط ونوبة صرع",
        "en": "seizure_convulsion",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "⚡",
        "action": "حماية رأس المصاب وتأمين المحيط لمنع التدافع",
    },
    "severe_choking_on_ground": {
        "category": "الحالات الطارئة والصحية",
        "ar": "استلقاء أرضي ممتد مع ضائقة تنفسية",
        "en": "severe_choking_on_ground",
        "priority": "Critical",
        "color": "#DC2626",
        "icon": "🫁",
        "action": "تأمين مجرى التنفس والتدخل الإسعافي الفوري",
    },
}

PRIORITY_COLOR = {"Critical": "#DC2626", "High": "#F97316", "Medium": "#F59E0B"}

LOCATIONS = [
    "ممشى المشاعر – ممر رقم 12 (Pilgrim Corridor 12)",
    "ساحة الحرم المركزية – بوابة الملك فهد (King Fahd Gate)",
    "محطة قطار الحرمين – الصالة 2 (Train Station Hub)",
    "المستشفى الميداني – محيط جسر الجمرات (Jamarat Bridge)",
]

# ============================================================================
# هياكل البيانات والحالة
# ============================================================================

@dataclass
class Alert:
    id: str
    unique_key: str
    frame_idx: int
    video_time_s: float
    wall_clock: str
    location: str
    condition_key: str
    confidence: float
    dispatched: bool = False


class Track:
    def __init__(self, track_id, centroid, bbox, frame_idx):
        self.id = track_id
        self.history = deque(maxlen=40)
        self.age = 0
        self.update(centroid, bbox, frame_idx)

    def update(self, centroid, bbox, frame_idx):
        self.centroid = centroid
        self.bbox = bbox
        self.last_seen = frame_idx
        self.age += 1
        self.history.append({"c": centroid, "b": bbox, "f": frame_idx})


if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# ============================================================================
# رأس الصفحة (Header)
# ============================================================================

st.markdown(
    """
    <div class="header-box">
        <div>
            <div class="system-title">🚑 نظام بصير | AI Anomaly Detection & Triage</div>
            <div class="system-sub">نظام إنذار مبكر يكتشف المؤشرات السلوكية والجسدية قبل حدوث السقوط أو فقدان الوعي – فريق VisionAid</div>
        </div>
        <div class="live-badge">● LIVE COMMAND CENTER</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# القائمة الجانبية (Sidebar Controls)
# ============================================================================

with st.sidebar:
    st.markdown("### 🎛️ غرفة العمليات والمراقبة")
    st.caption("Baseer AI · Operations & Triage Hub")

    feed_mode = st.radio(
        "مصدر البث (Feed Source)",
        ["Simulation", "Upload Video"],
        format_func=lambda x: "وضع المحاكاة التفاعلي (Simulation Mode)" if x == "Simulation" else "رفع فيديو مراقبة (Upload Video)",
    )

    uploaded_vid = None
    if feed_mode == "Upload Video":
        uploaded_vid = st.file_uploader("اختر مقطع الكاميرا (.mp4)", type=["mp4", "avi", "mov"])

    st.markdown("---")
    selected_zone = st.selectbox("نطاق الكاميرا والموقع (Zone)", LOCATIONS)
    sens = st.slider("حساسية الرصد والاستجابة (Sensitivity)", 20, 100, 65)

    st.markdown("---")
    play_speed = st.slider("معدل العرض التفاعلي (FPS)", 8, 35, 20)
    max_f = st.slider("إجمالي الإطارات للفحص (Max Frames)", 60, 600, 240, step=20)

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ تشغيل الرصد", use_container_width=True, type="primary")
    reset_btn = col2.button("⟲ إعادة ضبط", use_container_width=True)

    if reset_btn:
        st.session_state.alerts = []
        st.session_state.metrics = {"frame": 0, "tracks": 0, "fps": 0.0, "time": 0.0}
        st.session_state.last_frame = None
        st.rerun()

# ============================================================================
# تخطيط لوحة التحكم (Main Workspace)
# ============================================================================

col_cam, col_triage = st.columns([1.35, 1])

with col_cam:
    st.markdown("##### 📹 البث التحليلي المباشر (Analytical Feed)")
    cam_holder = st.empty()
    kpi_holder = st.empty()

with col_triage:
    st.markdown("##### 🚨 سجل الفرز والتوجيه الميداني (Live Triage Log)")
    triage_holder = st.container()


def render_kpis(m):
    kpi_holder.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card"><div class="kpi-num">{m['frame']}</div><div class="kpi-title">الإطار (Frame)</div></div>
            <div class="kpi-card"><div class="kpi-num">{m['time']:.1f}s</div><div class="kpi-title">الزمن (Time)</div></div>
            <div class="kpi-card"><div class="kpi-num">{m['tracks']}</div><div class="kpi-title">الأشخاص (Active)</div></div>
            <div class="kpi-card"><div class="kpi-num">{m['fps']:.1f}</div><div class="kpi-title">المعالجة (FPS)</div></div>
            <div class="kpi-card"><div class="kpi-num" style="color:#EF4444">{len(st.session_state.alerts)}</div><div class="kpi-title">البلاغات (Alerts)</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_triage():
    triage_holder.empty()
    with triage_holder:
        if not st.session_state.alerts:
            st.info("لا توجد بلاغات إسعافية حرجة حتى الآن. النظام يعمل ويراقب المؤشرات الحركية...")
            return

        for idx, a in enumerate(reversed(st.session_state.alerts)):
            info = TAXONOMY_RULES.get(a.condition_key, TAXONOMY_RULES["sudden_fall"])
            b_color = PRIORITY_COLOR[info["priority"]]
            st.markdown(
                f"""
                <div class="alert-card" style="border-left: 6px solid {b_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="triage-badge" style="background:{b_color}">{info['priority']} PRIORITY</span>
                        <span class="card-meta">#{a.id} · {a.wall_clock} · t={a.video_time_s:.1f}s</span>
                    </div>
                    <div style="margin-top:0.3rem;"><span class="category-tag">📂 {info['category']}</span></div>
                    <div class="card-ar">{info['icon']} {info['ar']}</div>
                    <div class="card-en"><b>Class:</b> <code>{info['en']}</code> (الثقة: {a.confidence*100:.0f}%)</div>
                    <div class="card-meta">📍 {a.location}</div>
                    <div style="margin-top:0.4rem; font-size:0.8rem; color:#CBD5E1;"><b>الإجراء الموصى به:</b> {info['action']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            b1, b2 = st.columns([1.3, 1])
            with b1:
                if not a.dispatched:
                    if st.button("🚑 توجيه فرقة التدخل السريع", key=f"btn_dsp_{a.unique_key}_{idx}", type="primary"):
                        a.dispatched = True
                        st.rerun()
                else:
                    st.button("✅ تم توجيه الفرقة بنجاح", key=f"btn_done_{a.unique_key}_{idx}", disabled=True)
            with b2:
                if a.dispatched:
                    st.markdown(f'<div class="eta-box">🚨 الفرقة في الطريق (وصول: دقيقة ونصف)</div>', unsafe_allow_html=True)
            st.write("")


# ============================================================================
# محرك المعالجة السحابي (Cloud-Optimized Stream Loop)
# ============================================================================

def run_pipeline():
    st.session_state.alerts = []
    is_sim = (feed_mode == "Simulation")
    w, h = 640, 400
    fps_src = 25.0
    
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=45, detectShadows=False)
    tracks = {}
    next_id = 1
    global_cd = {}
    
    cap = None
    tfile_path = None
    
    if not is_sim:
        if uploaded_vid is None:
            st.warning("الرجاء رفع ملف فيديو أولاً.")
            return
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile_path = tfile.name
        tfile.close()
        cap = cv2.VideoCapture(tfile_path)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        v_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_f
        total_frames = min(max_f, v_total)
    else:
        total_frames = max_f

    prog = st.progress(0.0, text="جاري فحص وتتبع حركة الحشود بالذكاء الاصطناعي...")
    start_t = time.time()

    for frame_idx in range(1, total_frames + 1):
        if is_sim:
            # سيناريو تدرج الحالات المبكرة (إجهاد حراري -> ترنح -> انحناء -> سقوط)
            phases = [
                ("Normal Walk", "normal_walk", 40),
                ("Severe Heatstroke Symptoms", "heatstroke_exhaustion", 50),
                ("Pre-Collapse Stoop", "stooped_walking_resting", 45),
                ("Sudden Fall Event", "sudden_fall", 45),
                ("Immobilized on Ground", "severe_choking_on_ground", 60),
            ]
            total_cycle = sum(p[2] for p in phases)
            curr_t = frame_idx % total_cycle
            accum, curr_cond, prog_phase = 0, "normal_walk", 0.0
            for _, cond, dur in phases:
                if accum <= curr_t < accum + dur:
                    curr_cond = cond
                    prog_phase = (curr_t - accum) / dur
                    break
                accum += dur

            canvas = np.full((h, w, 3), (15, 23, 42), dtype=np.uint8)
            ground_y = h - 70
            cv2.line(canvas, (0, ground_y), (w, ground_y), (51, 65, 85), 3)

            if curr_cond == "normal_walk":
                bw, bh = 48, 140
                cx, cy = int(w * 0.2 + prog_phase * w * 0.3), int(ground_y - bh / 2)
            elif curr_cond == "heatstroke_exhaustion":
                bw, bh = 54, 130
                cx, cy = int(w * 0.5 + math.sin(prog_phase * 20) * 16), int(ground_y - bh / 2)
            elif curr_cond == "stooped_walking_resting":
                bw, bh = int(55 + prog_phase * 20), int(120 - prog_phase * 40)
                cx, cy = int(w * 0.55), int(ground_y - bh / 2)
            elif curr_cond == "sudden_fall":
                fall_t = min(1.0, prog_phase / 0.4)
                bw, bh = int(55 + fall_t * 85), int(120 - fall_t * 90)
                cx, cy = int(w * 0.58), int(ground_y - bh / 2)
            else:
                bw, bh = 140, 32
                cx, cy = int(w * 0.58), int(ground_y - 18)

            cv2.ellipse(canvas, (cx, cy), (max(int(bw / 2), 6), max(int(bh / 2), 6)), 0, 0, 360, (56, 189, 248), -1)
            if bh > 40:
                cv2.circle(canvas, (cx, cy - int(bh / 2) + 12), 14, (125, 211, 252), -1)

            evts = []
            if curr_cond in TAXONOMY_RULES:
                last_f = global_cd.get(curr_cond, -9999)
                if frame_idx - last_f > 45:
                    global_cd[curr_cond] = frame_idx
                    evts.append((curr_cond, 0.94))

            b_color = (40, 40, 235) if curr_cond in TAXONOMY_RULES else (40, 200, 100)
            tag = f"Abnormal: {curr_cond}" if curr_cond in TAXONOMY_RULES else "ID 1 - Normal"
            cv2.rectangle(canvas, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), b_color, 2)
            cv2.putText(canvas, tag, (int(cx - bw / 2), max(int(cy - bh / 2) - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, b_color, 2, cv2.LINE_AA)
            frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            active_count = 1
        else:
            ok, raw = cap.read()
            if not ok or raw is None:
                break
            raw = cv2.resize(raw, (w, h))
            fgmask = bg.apply(raw)
            _, fgmask = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            evts = []
            active_count = 0
            for c in contours:
                if cv2.contourArea(c) > 2400:
                    x, y, bw, bh = cv2.boundingRect(c)
                    active_count += 1
                    asp = bw / max(bh, 1)
                    cond = "sudden_fall" if asp > 1.05 else "severe_gait_limping"
                    
                    last_f = global_cd.get(cond, -9999)
                    if frame_idx - last_f > 75:
                        global_cd[cond] = frame_idx
                        evts.append((cond, 0.89))
                    
                    cv2.rectangle(raw, (x, y), (x + bw, y + bh), (40, 40, 235), 2)
                    cv2.putText(raw, f"Abnormal: {cond}", (x, max(y - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 235), 2, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        for cond, conf in evts:
            seq_num = len(st.session_state.alerts) + 1
            st.session_state.alerts.append(
                Alert(
                    id=f"EMS-{seq_num:03d}",
                    unique_key=f"{seq_num}_{frame_idx}_{int(time.time()*1000)}",
                    frame_idx=frame_idx,
                    video_time_s=frame_idx / fps_src,
                    wall_clock=datetime.now().strftime("%H:%M:%S"),
                    location=selected_zone,
                    condition_key=cond,
                    confidence=conf,
                )
            )

        st.session_state.last_frame = frame_rgb
        elapsed = max(time.time() - start_t, 1e-6)
        st.session_state.metrics = {
            "frame": frame_idx,
            "tracks": active_count,
            "fps": frame_idx / elapsed,
            "time": frame_idx / fps_src,
        }

        # تحديث فوري وسلس للبث ومؤشرات الأداء
        cam_holder.image(frame_rgb, use_container_width=True)
        render_kpis(st.session_state.metrics)
        prog.progress(frame_idx / total_frames, text=f"تحليل الإطارات الذكي... {frame_idx}/{total_frames}")
        time.sleep(1.0 / play_speed)

    if cap:
        cap.release()
    if tfile_path and os.path.exists(tfile_path):
        try:
            os.remove(tfile_path)
        except Exception:
            pass

    prog.empty()
    render_triage()


if start_btn:
    run_pipeline()

if st.session_state.last_frame is not None:
    cam_holder.image(st.session_state.last_frame, use_container_width=True)
else:
    placeholder = np.full((400, 640, 3), (15, 23, 42), dtype=np.uint8)
    cv2.putText(placeholder, "BASEER AI COMMAND CENTER", (120, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (72, 202, 228), 2, cv2.LINE_AA)
    cv2.putText(placeholder, "اضغط بدء الرصد للتشغيل الميداني", (160, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (144, 224, 239), 1, cv2.LINE_AA)
    cam_holder.image(placeholder, use_container_width=True)

render_kpis(st.session_state.metrics)
render_triage()
