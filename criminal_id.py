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

# --- 3. UI STREAMLIT (iPad Optimized) ---
st.set_page_config(page_title="Forensic ID", layout="wide", initial_sidebar_state="expanded")

# CSS untuk jadikan butang sidebar nampak lebih macam aplikasi iPad
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #f0f2f6;
        border: none;
        text-align: left;
        padding-left: 20px;
        margin-bottom: 10px;
    }
    .main-title {
        font-size: 35px;
        font-weight: bold;
        color: #1E1E1E;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar dengan Ikon iPad-style
with st.sidebar:
    st.markdown("### 📱 Navigation")
    if st.button("🏠 Home"):
        st.session_state.menu = "Home"
    if st.button("📝 Register Suspect"):
        st.session_state.menu = "Register"
    if st.button("🔍 Forensic Scan"):
        st.session_state.menu = "Scan"
    
    st.markdown("---")
    st.caption(f"System Developer: {st.session_state.get('user_name', 'Hafiz')}")

# Default Menu
if 'menu' not in st.session_state:
    st.session_state.menu = "Home"

# --- MOD 🏠 HOME ---
if st.session_state.menu == "Home":
    st.markdown("<div class='main-title'>Forensic Intelligence System</div>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/fingerprint.png", width=100)
    st.subheader("Selamat Datang ke Makmal Forensik Digital")
    st.write("Sistem ini menggunakan koordinat biometrik wajah untuk pengecaman identiti secara real-time.")
    st.info("Gunakan menu di sebelah kiri untuk memulakan pendaftaran atau imbasan.")

# --- MOD 📝 REGISTER ---
elif st.session_state.menu == "Register":
    st.markdown("<div class='main-title'>📝 Register New Profile</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        new_name = st.text_input("Nama Penuh Suspek", placeholder="Masukkan nama...")
        upload_img = st.file_uploader("Pilih gambar wajah", type=['jpg', 'jpeg', 'png'])
        
        if st.button("💾 Save to Database"):
            if new_name and upload_img:
                image = face_recognition.load_image_file(upload_img)
                encodings = face_recognition.face_encodings(image, model="hog")
                if len(encodings) > 0:
                    encoding_str = json.dumps(encodings[0].tolist())
                    conn = sqlite3.connect('forensic_lab.db')
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO suspects (name, encoding) VALUES (?, ?)", (new_name, encoding_str))
                    conn.commit()
                    conn.close()
                    st.success(f"Rekod {new_name} telah berjaya disimpan.")
                else:
                    st.error("Wajah tidak dikesan. Sila gunakan gambar yang lebih terang.")

# --- MOD 🔍 SCAN ---
elif st.session_state.menu == "Scan":
    st.markdown("<div class='main-title'>🔍 Live Forensic Scan</div>", unsafe_allow_html=True)
    
    known_names, known_encodings = get_all_suspects()
    
    if not known_names:
        st.warning("Database kosong. Sila daftar suspek dahulu.")
    else:
        # Interface kamera yang sangat mesra iPad
        img_file = st.camera_input("Halakan kamera ke arah subjek")

        if img_file:
            image = Image.open(img_file)
            frame = np.array(image.convert('RGB'))
            
            # Pengecaman Biometrik
            face_locations = face_recognition.face_locations(frame, model="hog")
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            if not face_locations:
                st.info("Tiada wajah dikesan dalam bingkai.")

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                if True in matches:
                    idx = matches.index(True)
                    name = known_names[idx]
                    st.markdown(f"<h2 style='color:green; text-align:center;'>✅ IDENTITI DIKESAN: {name}</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color:red; text-align:center;'>⚠️ TIADA REKOD DITEMUI</h2>", unsafe_allow_html=True)
                
                # Draw box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 5)
            
            st.image(frame, use_container_width=True)