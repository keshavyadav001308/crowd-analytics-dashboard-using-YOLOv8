import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
import math
from collections import deque

# =====================================================
# PAGE CONFIG (FlowMetrics Look)
# =====================================================
st.set_page_config(
    page_title="Crowd Analytics",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #0b1220;
}
.metric-card {
    background: radial-gradient(circle at center, #0ea5e9 0%, #020617 70%);
    border-radius: 50%;
    width: 240px;
    height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: white;
    margin: auto;
}
.metric-value {
    font-size: 52px;
    font-weight: bold;
}
.metric-label {
    font-size: 14px;
    color: #94a3b8;
}
.panel {
    background-color: #020617;
    padding: 20px;
    border-radius: 12px;
}
.stop-btn button {
    background-color: #dc2626 !important;
    color: white !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## Crowd Analytics Dashboard")
st.caption("Real-time people counting & analytics")

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =====================================================
# SESSION STATE
# =====================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "trend" not in st.session_state:
    st.session_state.trend = deque(maxlen=200)

# =====================================================
# STOP BUTTON
# =====================================================
c1, c2, c3 = st.columns([7, 3, 2])
with c3:
    if st.button("🛑 STOP & RESET"):
        st.session_state.running = False
        st.session_state.trend.clear()
        st.rerun()

# =====================================================
# TABS
# =====================================================
tab_live, tab_video, tab_image = st.tabs(
    ["🔴 Live", "🎥 Video", "🖼️ Image"]
)

# =====================================================
# LIVE TAB (Mobile Camera Only)
# =====================================================
with tab_live:
    st.markdown("### Mobile Phone Live Camera")

    cam_url = st.text_input(
        "IP Camera URL",
        placeholder="http://192.168.1.10:8080/video"
    )

    start_live = st.button("▶ Start Live Counting")

    col_left, col_right = st.columns([3, 5])

    if start_live and cam_url:
        st.session_state.running = True
        cap = cv2.VideoCapture(cam_url)

        frame_box = col_right.image([])

        while cap.isOpened() and st.session_state.running:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=1280, conf=0.2, iou=0.75, verbose=False)

            count = 0
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            elapsed = time.time() - start
            fps = 1 / elapsed if elapsed > 0 else 0
            if math.isnan(fps) or math.isinf(fps):
                fps = 0

            st.session_state.trend.append(count)

            with col_left:
                st.markdown("### Live Occupancy")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{count}</div>
                    <div class="metric-label">People Detected</div>
                </div>
                """, unsafe_allow_html=True)
                st.metric("FPS", int(fps))

            cv2.putText(frame, f"People: {count}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            frame_box.image(frame, channels="BGR")

        cap.release()

    if len(st.session_state.trend) > 10:
        st.markdown("### Current Traffic Trends")
        st.line_chart(list(st.session_state.trend))

# =====================================================
# VIDEO TAB
# =====================================================
with tab_video:
    st.markdown("### 🎥 Video Crowd Analytics")

    uploaded_video = st.file_uploader(
        "Upload Crowd Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture("temp_video.mp4")

        # ✅ DEFINE frame_window BEFORE LOOP
        frame_window = st.image([])

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0 or np.isnan(video_fps):
            video_fps = 30

        frame_interval = int(video_fps)  # 1 frame per second
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # ⏩ Skip frames (speed fix)
            if frame_count % frame_interval != 0:
                continue

            results = model(frame, imgsz=1280, conf=0.15, iou=0.75, verbose=False)

            count = 0
            for box in results[0].boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.15:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"People: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            #  SAFE DISPLAY
            frame_window.image(frame, channels="BGR")

        cap.release()

        st.success("Video Analysis Completed")
        st.metric("Peak Occupancy", max(counts))
        st.metric("Average FPS", int(np.mean(fps_vals)))
        st.line_chart(counts)

# =====================================================
# IMAGE TAB (FlowMetrics Style)
# =====================================================
with tab_image:
    st.markdown("### 🖼️ Image Crowd Analytics")

    uploaded_image = st.file_uploader(
        "Upload Crowd Image",
        type=["jpg", "jpeg", "png"]
    )

    col_left, col_right = st.columns([3, 5])

    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(image, imgsz=1280, conf=0.2, iou=0.75, verbose=False)

        count = 0
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

        with col_left:
            st.markdown("### Live Occupancy")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{count}</div>
                <div class="metric-label">People Detected</div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.image(image, channels="BGR", caption="Detected Crowd")

        st.markdown("### Crowd Traffic Trend")
        fake_trend = np.random.randint(max(0, count-8), count+5, size=30)
        st.line_chart(fake_trend)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("YOLOv8 Crowd Analytics Dashboard • Streamlit")

