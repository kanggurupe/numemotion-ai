import cv2
import os
import pandas as pd
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="NumeMotion-AI", layout="wide")

# Header Utama
st.title("🏋️‍♂️ NumeMotion-AI: Multi-Person Fitness & Assessment Dashboard")
st.caption("Integrasi Computer Vision & Spatial Numeracy untuk Pembelajaran PJOK Transformatif Sekolah Dasar")

# Sidebar Konfigurasi
st.sidebar.header("⚙️ Pengaturan Latihan")
exercise_type = st.sidebar.selectbox(
    "Pilih Jenis Gerakan:", 
    [
        "Squat Tracker (Kekuatan Otot Tungkai)", 
        "Push-up Tracker (Kekuatan Otot Lengan)", 
        "Jumping Jack (Koordinasi & Daya Tahan)", 
        "Sit and Reach (Kelentukan)", 
        "Shuttle Run (Kelincahan)"
    ]
)

st.sidebar.subheader("👥 Input Nama Kelompok (Maksimal 3 Orang)")
name_1 = st.sidebar.text_input("Siswa 1 (Kiri):", value="Siswa A", key="s_name1")
name_2 = st.sidebar.text_input("Siswa 2 (Tengah):", value="Siswa B", key="s_name2")
name_3 = st.sidebar.text_input("Siswa 3 (Kanan):", value="Siswa C", key="s_name3")

# Load Model YOLOv8 Pose
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n-pose.pt')

model = load_yolo()

# Fungsi Hitung Sudut
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return int(angle)

# Fungsi Menggambar Skeleton (Garis Hijau & Biru + Titik Sendi)
def draw_skeleton(frame, keypoints):
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Wajah
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Tangan/Lengan
        (5, 11), (6, 12), (11, 12),               # Badan/Pinggul
        (11, 13), (13, 15), (12, 14), (14, 16)    # Kaki/Tungkai
    ]
    
    for p1_idx, p2_idx in SKELETON_CONNECTIONS:
        if p1_idx < len(keypoints) and p2_idx < len(keypoints):
            pt1 = (int(keypoints[p1_idx][0]), int(keypoints[p1_idx][1]))
            pt2 = (int(keypoints[p2_idx][0]), int(keypoints[p2_idx][1]))
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                color = (0, 255, 0) if p1_idx < 11 else (255, 255, 0)
                cv2.line(frame, pt1, pt2, color, 2)
                
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

# Processor Video WebRTC untuk pemrosesan per frame secara real-time
class MotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.counters = [0, 0, 0]
        self.stages = [None, None, None]
        self.student_names = [name_1, name_2, name_3]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)

        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                persons = result.keypoints.xy.cpu().numpy()
                valid_persons = [p for p in persons if len(p) > 0 and p[0][0] > 0]
                valid_persons = sorted(valid_persons, key=lambda x: x[0][0])[:3]

                for idx, keypoints in enumerate(valid_persons):
                    if idx >= 3: break
                    
                    self.student_names = [name_1, name_2, name_3]
                    current_name = self.student_names[idx] if idx < len(self.student_names) and self.student_names[idx].strip() else f"Siswa {idx+1}"

                    # Gambar Kerangka Pose
                    draw_skeleton(img, keypoints)

                    # Logika Gerakan
                    if "Squat" in exercise_type and len(keypoints) > 16:
                        hip, knee, ankle = keypoints[12], keypoints[14], keypoints[16]
                        if hip[0] > 0 and knee[0] > 0 and ankle[0] > 0:
                            current_angle = calculate_angle(hip, knee, ankle)
                            if current_angle > 160: self.stages[idx] = "BERDIRI"
                            if current_angle < 100 and self.stages[idx] == 'BERDIRI':
                                self.stages[idx] = "SQUAT"
                                self.counters[idx] += 1

                    elif "Push-up" in exercise_type and len(keypoints) > 10:
                        shoulder, elbow, wrist = keypoints[6], keypoints[8], keypoints[10]
                        if shoulder[0] > 0 and elbow[0] > 0 and wrist[0] > 0:
                            current_angle = calculate_angle(shoulder, elbow, wrist)
                            if current_angle > 160: self.stages[idx] = "LURUS"
                            if current_angle < 90 and self.stages[idx] == 'LURUS':
                                self.stages[idx] = "TURUN"
                                self.counters[idx] += 1

                    elif "Jumping Jack" in exercise_type and len(keypoints) > 10:
                        hip, shoulder, elbow = keypoints[12], keypoints[6], keypoints[8]
                        if hip[0] > 0 and shoulder[0] > 0 and elbow[0] > 0:
                            current_angle = calculate_angle(hip, shoulder, elbow)
                            if current_angle < 40: self.stages[idx] = "RAPAT"
                            if current_angle > 130 and self.stages[idx] == "RAPAT":
                                self.stages[idx] = "LOMPAT"
                                self.counters[idx] += 1

                    # Tampilkan Nama & Repetisi di video
                    nose_x, nose_y = int(keypoints[0][0]), int(keypoints[0][1])
                    if nose_x > 0 and nose_y > 0:
                        label = f"{current_name}: {self.counters[idx]} Reps"
                        cv2.putText(img, label, (nose_x - 50, max(20, nose_y - 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout Tampilan Web
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Stream Kamera WebRTC")
    # Konfigurasi STUN Server Google agar koneksi WebRTC Cloud stabil
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_ctx = webrtc_streamer(
        key="numemotion-pose",
        video_processor_factory=MotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📊 Skor Real-Time")
    m1, m2, m3 = st.columns(3)
    
    reps = [0, 0, 0]
    if webrtc_ctx.video_processor:
        reps = webrtc_ctx.video_processor.counters
        
    m1.metric(f"👤 {name_1 if name_1.strip() else 'Siswa 1'}", f"{reps[0]} Reps")
    m2.metric(f"👤 {name_2 if name_2.strip() else 'Siswa 2'}", f"{reps[1]} Reps")
    m3.metric(f"👤 {name_3 if name_3.strip() else 'Siswa 3'}", f"{reps[2]} Reps")

    st.divider()
    st.subheader("📝 Self-Assessment Terstruktur")
    
    active_students = [(name_1, "ref_1"), (name_2, "ref_2"), (name_3, "ref_3")]
    ref_results = []
    
    for idx, (s_name, prefix) in enumerate(active_students):
        if s_name and s_name.strip():
            st.markdown(f"**👤 Evaluasi: {s_name}**")
            p1 = st.selectbox(
                f"1. Perasaan {s_name}:", 
                ["😃 Senang & Bugar", "😐 Biasa Saja", "😫 Sangat Lelah"], 
                key=f"{prefix}_p1"
            )
            p2 = st.selectbox(
                f"2. Kesulitan Gerakan:", 
                ["✅ Mudah (Sudut Tepat)", "⚠️ Agak Sulit", "❌ Sulit Mencapai Sudut"], 
                key=f"{prefix}_p2"
            )
            p3 = st.text_input(f"3. Catatan Kendala:", placeholder="Contoh: Lutut agak kaku...", key=f"{prefix}_p3")
            ref_combined = f"[Perasaan: {p1}] | [Kesesuaian: {p2}] | [Catatan: {p3}]"
            ref_results.append(ref_combined)
        else:
            ref_results.append("")

    if st.button("💾 Simpan Data ke CSV"):
        file_path = "data_kebugaran.csv"
        data_list = []
        for name, score, ref in zip([name_1, name_2, name_3], reps, ref_results):
            if name and name.strip():
                data_list.append({
                    "Nama Siswa": name,
                    "Jenis Gerakan": exercise_type,
                    "Jumlah Repetisi": score,
                    "Catatan Self-Assessment": ref
                })
        if data_list:
            df = pd.DataFrame(data_list)
            if not os.path.exists(file_path):
                df.to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, mode='a', header=False, index=False)
            st.success("Data berhasil disimpan!")
