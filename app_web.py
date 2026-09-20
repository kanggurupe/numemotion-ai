import cv2
import os
import time
import pandas as pd
import numpy as np
import streamlit as st
from ultralytics import YOLO

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

duration = st.sidebar.slider("Durasi Timer Tantangan (Detik):", min_value=10, max_value=120, value=60, step=10)
run_app = st.sidebar.checkbox("Nyalakan Kamera", value=False)
start_timer = st.sidebar.button("▶️ Mulai Tantangan Timer (1 Menit)")

# Layout Tampilan Web
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⏱️ Timer & Skor Latihan")
    timer_display = st.empty()
    
    st.divider()
    st.subheader("📊 Hasil Repetisi Real-Time")
    m1, m2, m3 = st.columns(3)
    score1_ui = m1.empty()
    score2_ui = m2.empty()
    score3_ui = m3.empty()
    
    st.divider()
    st.subheader("📈 Visualisasi Rentang Gerak (Sudut Real-Time)")
    chart_placeholder = st.empty()
    
    st.divider()
    st.subheader("📝 Self-Assessment Terstruktur")
    st.caption("Isi evaluasi diri setelah timer selesai:")

    # Multi-Pertanyaan Self-Assessment
    active_students = [
        (name_1, "ref_1"), 
        (name_2, "ref_2"), 
        (name_3, "ref_3")
    ]
    
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
                f"2. Tingkat Kesulitan Gerakan:", 
                ["✅ Mudah (Sudut Tepat)", "⚠️ Agak Sulit", "❌ Sulit Mencapai Sudut"], 
                key=f"{prefix}_p2"
            )
            p3 = st.text_input(f"3. Kendala/Pegal Tubuh:", placeholder="Contoh: Lutut agak kaku...", key=f"{prefix}_p3")
            ref_combined = f"[Perasaan: {p1}] | [Kesesuaian: {p2}] | [Catatan: {p3}]"
            ref_results.append(ref_combined)
        else:
            ref_results.append("")

    save_btn = st.button("💾 Simpan Semua Data ke Database (CSV)")

with col1:
    st_frame = st.empty()

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
    # Pasangan sendi YOLOv8 (Skeleton Connection Lines)
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Wajah
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Tangan/Lengan
        (5, 11), (6, 12), (11, 12),               # Badan/Pinggul
        (11, 13), (13, 15), (12, 14), (14, 16)    # Kaki/Tungkai
    ]
    
    # Gambar Garis Skeleton
    for p1_idx, p2_idx in SKELETON_CONNECTIONS:
        if p1_idx < len(keypoints) and p2_idx < len(keypoints):
            pt1 = (int(keypoints[p1_idx][0]), int(keypoints[p1_idx][1]))
            pt2 = (int(keypoints[p2_idx][0]), int(keypoints[p2_idx][1]))
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                color = (0, 255, 0) if p1_idx < 11 else (255, 255, 0) # Hijau untuk atas, Biru/Kuning untuk bawah
                cv2.line(frame, pt1, pt2, color, 2)
                
    # Gambar Titik Sendi (Keypoints)
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

# Fungsi Simpan Database Berkelompok
def save_multi_data(names, exercise, scores, reflections):
    file_path = "data_kebugaran.csv"
    data_list = []
    for name, score, ref in zip(names, scores, reflections):
        if name and name.strip() != "":
            data_list.append({
                "Nama Siswa": name,
                "Jenis Gerakan": exercise,
                "Jumlah Repetisi/Nilai": score,
                "Catatan Refleksi Self-Assessment": ref
            })
    if data_list:
        df = pd.DataFrame(data_list)
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)

if run_app:
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(0)

    # Inisialisasi Data
    counters = [0, 0, 0]
    stages = [None, None, None]
    student_names = [name_1, name_2, name_3]
    angle_history = []

    start_time = time.time() if start_timer else None

    while cap.isOpened() and run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Kamera tidak terdeteksi.")
            break

        # Logika Hitung Mundur Timer
        remaining_time = duration
        if start_time:
            elapsed = int(time.time() - start_time)
            remaining_time = max(0, duration - elapsed)
            timer_display.header(f"⏳ Sisa Waktu: **{remaining_time} Detik**")
            if remaining_time == 0:
                st.sidebar.warning("⏰ Waktu Habis! Silakan isi Self-Assessment lalu simpan data.")

        results = model(frame, verbose=False)

        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                persons = result.keypoints.xy.cpu().numpy()
                
                # Filter orang terdeteksi
                valid_persons = []
                for p in persons:
                    if len(p) > 0 and p[0][0] > 0:
                        valid_persons.append(p)
                
                # Urutkan posisi dari Kiri (X terkecil) ke Kanan (X terbesar)
                valid_persons = sorted(valid_persons, key=lambda x: x[0][0])[:3]

                # Deteksi hingga maksimal 3 orang
                for idx, keypoints in enumerate(valid_persons):
                    if idx >= 3: break
                    
                    current_name = student_names[idx] if idx < len(student_names) and student_names[idx].strip() else f"Siswa {idx+1}"
                    current_angle = 0

                    # 1. Gambar Kerangka Pose (Garis Hijau-Biru)
                    draw_skeleton(frame, keypoints)

                    # 2. SQUAT TRACKER
                    if "Squat" in exercise_type and len(keypoints) > 16:
                        hip, knee, ankle = keypoints[12], keypoints[14], keypoints[16]
                        if hip[0] > 0 and knee[0] > 0 and ankle[0] > 0:
                            current_angle = calculate_angle(hip, knee, ankle)
                            if current_angle > 160: stages[idx] = "BERDIRI"
                            if current_angle < 100 and stages[idx] == 'BERDIRI' and remaining_time > 0:
                                stages[idx] = "SQUAT"
                                counters[idx] += 1

                    # 3. PUSH-UP TRACKER
                    elif "Push-up" in exercise_type and len(keypoints) > 10:
                        shoulder, elbow, wrist = keypoints[6], keypoints[8], keypoints[10]
                        if shoulder[0] > 0 and elbow[0] > 0 and wrist[0] > 0:
                            current_angle = calculate_angle(shoulder, elbow, wrist)
                            if current_angle > 160: stages[idx] = "LURUS"
                            if current_angle < 90 and stages[idx] == 'LURUS' and remaining_time > 0:
                                stages[idx] = "TURUN"
                                counters[idx] += 1

                    # 4. JUMPING JACK
                    elif "Jumping Jack" in exercise_type and len(keypoints) > 10:
                        hip, shoulder, elbow = keypoints[12], keypoints[6], keypoints[8]
                        if hip[0] > 0 and shoulder[0] > 0 and elbow[0] > 0:
                            current_angle = calculate_angle(hip, shoulder, elbow)
                            if current_angle < 40: stages[idx] = "RAPAT"
                            if current_angle > 130 and stages[idx] == "RAPAT" and remaining_time > 0:
                                stages[idx] = "LOMPAT"
                                counters[idx] += 1

                    # Simpan data grafik dari siswa utama (Siswa 1)
                    if idx == 0 and current_angle > 0:
                        angle_history.append(current_angle)
                        if len(angle_history) > 30: angle_history.pop(0)

                    # Gambarkan Nama & Skor di Atas Kepala
                    nose_x, nose_y = int(keypoints[0][0]), int(keypoints[0][1])
                    if nose_x > 0 and nose_y > 0:
                        label = f"{current_name}: {counters[idx]} Reps"
                        cv2.putText(frame, label, (nose_x - 50, max(20, nose_y - 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update UI Skor & Grafik
        score1_ui.metric(f"👤 {name_1 if name_1.strip() else 'Siswa 1'}", f"{counters[0]} Reps")
        score2_ui.metric(f"👤 {name_2 if name_2.strip() else 'Siswa 2'}", f"{counters[1]} Reps")
        score3_ui.metric(f"👤 {name_3 if name_3.strip() else 'Siswa 3'}", f"{counters[2]} Reps")

        if angle_history:
            chart_placeholder.line_chart(angle_history)

        # Tampilkan Video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

# Logika Simpan Data & Unduh
if save_btn:
    student_names = [name_1, name_2, name_3]
    save_multi_data(student_names, exercise_type, counters, ref_results)
    st.sidebar.success("Semua data siswa & refleksi berhasil disimpan ke 'data_kebugaran.csv'!")
    
    if os.path.exists("data_kebugaran.csv"):
        with open("data_kebugaran.csv", "rb") as file:
            st.sidebar.download_button(
                label="📥 Unduh File Excel/CSV",
                data=file,
                file_name="data_kebugaran_siswa.csv",
                mime="text/csv"
            )