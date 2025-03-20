import streamlit as st
import cv2
import numpy as np
import tempfile

# Path model (sudah disiapkan dalam folder model)
prototxt_path = "model/deploy.prototxt"
model_path = "model/mobilenet_iter_73000.caffemodel"

# Load model MobileNet SSD
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Label kelas dari MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Konfigurasi layout halaman
st.set_page_config(layout="wide")
st.title("🎥 Video Person Detection")

# Sidebar untuk upload file
st.sidebar.header("📂 Upload Video")
uploaded_file = st.sidebar.file_uploader("Pilih file video", type=["mp4", "avi", "mov"])

# Tombol untuk menjalankan deteksi
process_video = st.sidebar.button("🔍 Proses Video")

# Jika video sudah diupload dan tombol diklik
if uploaded_file and process_video:
    # Simpan video sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    # Buat dua kolom (kiri: video, kanan: hasil analisis)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Video Asli")
        st.video(video_path)

    with col2:
        st.subheader("📊 Hasil Deteksi")

        # Buka video dengan OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Ambil FPS video
        frame_count = 0
        skip_frames = 5  # Hanya proses setiap 5 frame sekali
        person_detected = False
        screenshot_time = None
        screenshot_path = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue  # Lewati frame jika bukan kelipatan skip_frames

            # Resize frame ke 640x360 untuk mempercepat deteksi
            frame = cv2.resize(frame, (640, 360))

            # Konversi frame menjadi blob untuk deteksi objek
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # Looping setiap objek yang terdeteksi
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Ambil objek dengan confidence > 50%
                    class_id = int(detections[0, 0, i, 1])
                    if CLASSES[class_id] == "person":  # Jika objek adalah "person"
                        person_detected = True
                        screenshot_time = frame_count / fps  # Hitung waktu dalam detik

                        # Simpan screenshot hanya satu kali
                        screenshot_path = "screenshot.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        break  # Stop loop jika ada orang

            if person_detected:
                break  # Stop proses jika sudah ada orang terdeteksi

        cap.release()

        # Format waktu menjadi menit:detik
        if screenshot_time is not None:
            minutes = int(screenshot_time // 60)
            seconds = int(screenshot_time % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
        else:
            time_str = "N/A"

        # Menampilkan hasil deteksi
        if person_detected:
            st.success(f"✅ Ya, video ini ada orang.")
            st.info(f"📸 Screenshot pertama diambil pada menit {time_str}.")
            st.image(screenshot_path, caption=f"Deteksi pertama pada {time_str}", use_column_width=True)
        else:
            st.error("❌ Tidak ada orang yang terdeteksi dalam video.")
