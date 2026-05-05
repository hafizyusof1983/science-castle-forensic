import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import sqlite3
import json
from PIL import Image

# --- 1. INITIALIZE MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- 2. DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('forensic_lab.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS suspects 
                       (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, encoding TEXT)''')
    conn.commit()
    conn.close()

init_db()

def get_all_suspects():
    conn = sqlite3.connect('forensic_lab.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM suspects")
    data = cursor.fetchall()
    conn.close()
    
    names, encodings = [], []
    for name, enc_str in data:
        names.append(name)
        encodings.append(np.array(json.loads(enc_str)))
    return names, encodings

# --- 3. BIOMETRIC ENGINE ---
def get_face_signature(image_np):
    """Menukar 468 titik wajah kepada vektor matematik (biometric fingerprint)"""
    results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        # Extract x, y, z coordinates for all 468 points
        signature = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return signature
    return None

def compare_faces(current_sig, known_sigs, threshold=0.18):
    """Membandingkan jarak Euclidean antara wajah semasa dengan pangkalan data"""
    if not known_sigs:
        return None, 0
    
    distances = [np.linalg.norm(current_sig - k_sig) for k_sig in known_sigs]
    min_dist = min(distances)
    if min_dist < threshold:
        return distances.index(min_dist), min_dist
    return None, min_dist

# --- 4. UI STYLING ---
st.set_page_config(page_title="Forensic ID", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: 800; color: #111827; text-align: center; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; font-weight: 700; }
    .status-card { padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 10px; }
    .bg-blue { background: linear-gradient(135deg, #3B82F6, #2563EB); }
    </style>
    """, unsafe_allow_html=True)

# --- 5. NAVIGATION ---
if 'menu' not in st.session_state:
    st.session_state.menu = "Home"

with st.sidebar:
    st.title("🕵️ Forensic Menu")
    if st.button("🏠 Dashboard"): st.session_state.menu = "Home"
    if st.button("📝 Daftar Suspek"): st.session_state.menu = "Register"
    if st.button("🔍 Imbasan Lab"): st.session_state.menu = "Scan"

st.markdown("<div class='main-title'>Forensic Intelligence Lab</div>", unsafe_allow_html=True)

# --- 6. MODULES ---
known_names, known_sigs = get_all_suspects()

if st.session_state.menu == "Home":
    st.markdown(f"<div class='status-card bg-blue'><b>Database Profiles</b><br><h2>{len(known_names)}</h2></div>", unsafe_allow_html=True)
    st.write("Sistem pengecaman jenayah menggunakan analisis biometrik 468-titik wajah.")

elif st.session_state.menu == "Register":
    st.subheader("Pendaftaran Rekod Jenayah")
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("Nama Penuh Suspek")
        file = st.file_uploader("Muat Naik Foto Evidence", type=['jpg', 'png'])
        if st.form_submit_button("Simpan Biometrik"):
            if name and file:
                img = Image.open(file)
                img_np = np.array(img.convert('RGB'))
                sig = get_face_signature(img_np)
                if sig is not None:
                    conn = sqlite3.connect('forensic_lab.db')
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO suspects (name, encoding) VALUES (?, ?)", 
                                 (name, json.dumps(sig.tolist())))
                    conn.commit()
                    conn.close()
                    st.success(f"Rekod {name} berjaya disimpan!")
                else:
                    st.error("Wajah tidak dikesan. Sila gunakan gambar yang lebih jelas.")

elif st.session_state.menu == "Scan":
    st.subheader("Imbasan Forensik Masa Nyata")
    camera_img = st.camera_input("Imbas Wajah")
    if camera_img:
        img = Image.open(camera_img)
        img_np = np.array(img.convert('RGB'))
        current_sig = get_face_signature(img_np)
        
        if current_sig is not None:
            idx, dist = compare_faces(current_sig, known_sigs)
            if idx is not None:
                st.error(f"🚨 PADANAN DIJUMPAI: {known_names[idx]}")
                st.write(f"Tahap Keyakinan: {(1 - dist)*100:.2f}%")
            else:
                st.success("✅ Tiada rekod jenayah ditemui.")
        else:
            st.warning("Sila pastikan wajah berada di tengah kamera.")