import streamlit as st
import face_recognition
import cv2
import numpy as np
import sqlite3
import json
from PIL import Image

# --- 1. SETUP DATABASE ---
def init_db():
    conn = sqlite3.connect('forensic_lab.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS suspects 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, encoding TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 2. FUNGSI PEMBANTU ---
def get_all_suspects():
    conn = sqlite3.connect('forensic_lab.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM suspects")
    data = cursor.fetchall()
    conn.close()
    
    known_names = []
    known_encodings = []
    for name, encoding_str in data:
        known_names.append(name)
        known_encodings.append(np.array(json.loads(encoding_str)))
    return known_names, known_encodings

# --- 3. UI STREAMLIT ---
st.set_page_config(page_title="Forensic Face ID", layout="wide")
st.title("🕵️‍♂️ Forensic Face Identification System")
st.markdown("---")

# Menu pilihan di Sidebar
menu = st.sidebar.selectbox("Pilih Mod", ["🏠 Utama", "📝 Pendaftaran Suspek", "🔍 Imbasan Forensik"])

if menu == "🏠 Utama":
    st.subheader("Selamat Datang ke Makmal Forensik Digital")
    st.write("Sistem ini menggunakan kecerdasan buatan untuk mengenal pasti identiti berdasarkan koordinat biometrik wajah.")
    st.info("Sila pilih mod 'Pendaftaran' untuk memasukkan data atau 'Imbasan' untuk memulakan pengecaman.")

elif menu == "📝 Pendaftaran Suspek":
    st.subheader("Daftar Profil Suspek Baru")
    new_name = st.text_input("Nama Suspek")
    upload_img = st.file_uploader("Muat naik gambar wajah", type=['jpg', 'jpeg', 'png'])
    
    if st.button("Simpan ke Pangkalan Data"):
        if new_name and upload_img:
            image = face_recognition.load_image_file(upload_img)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                encoding_str = json.dumps(encodings[0].tolist())
                conn = sqlite3.connect('forensic_lab.db')
                cursor = conn.cursor()
                cursor.execute("INSERT INTO suspects (name, encoding) VALUES (?, ?)", (new_name, encoding_str))
                conn.commit()
                conn.close()
                st.success(f"Rekod {new_name} berjaya disimpan!")
            else:
                st.error("Wajah tidak dikesan dalam gambar. Sila guna gambar yang lebih jelas.")

elif menu == "🔍 Imbasan Forensik":
    st.subheader("Live Identification Feed")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        run_cam = st.checkbox("Aktifkan Kamera Unit")
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        
        # Ambil data sedia ada
        known_names, known_encodings = get_all_suspects()

        while run_cam:
            ret, frame = cap.read()
            if not ret: break
            
            # Kecilkan imej untuk kelajuan (optional)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Cari wajah
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                color = (0, 0, 255) # Merah (Tak dikenali)

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                    color = (0, 255, 0) # Hijau (Ditemui)

                # Skala balik koordinat (sebab tadi dikecilkan 0.25)
                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Papar info di kolum sebelah jika dikesan
                if name != "Unknown":
                    with col2:
                        st.success(f"IDENTITI DIKESAN: {name}")
                        st.metric("Status", "Dikehendaki" if name else "Tiada")

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()