import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import sqlite3
import json
from PIL import Image
from deepface import DeepFace # Alternatif dlib yang sangat stabil di cloud
import os

# --- 1. DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('forensic_lab.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS suspects 
                       (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, encoding TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 2. HELPER FUNCTIONS ---
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
        known_encodings.append(json.loads(encoding_str))
    return known_names, known_encodings

# --- 3. UI CONFIG & STYLING (Kekal seperti asal) ---
st.set_page_config(page_title="Forensic ID", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #111827; }
    .stButton>button {
        display: block; width: 100% !important; border-radius: 12px;
        height: 3.5em; background-color: white !important;
        color: #111827 !important; font-weight: 700 !important;
    }
    .main-title { font-size: 40px; font-weight: 800; color: #111827; text-align: center; }
    .mini-card { padding: 15px; border-radius: 15px; text-align: center; color: white !important; margin-bottom: 10px; }
    .bg-status { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
    .bg-profiles { background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); }
    .bg-mode { background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%); }
    </style>
    """, unsafe_allow_html=True)

# --- NAVIGATION LOGIC ---
if 'menu' not in st.session_state: st.session_state.menu = "Home"

with st.sidebar:
    st.markdown("## 🕵️ Menu")
    if st.button("🏠 Home Dashboard"): st.session_state.menu = "Home"; st.rerun()
    if st.button("📝 Register Suspect"): st.session_state.menu = "Register"; st.rerun()
    if st.button("🔍 Forensic Scan"): st.session_state.menu = "Scan"; st.rerun()

st.markdown("<div class='main-title'>Forensic Lab</div>", unsafe_allow_html=True)

# Status Cards
known_names, known_encodings = get_all_suspects()
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1: st.markdown("<div class='mini-card bg-status'><b>Status</b><div>Active</div></div>", unsafe_allow_html=True)
with m_col2: st.markdown(f"<div class='mini-card bg-profiles'><b>DB Profiles</b><div>{len(known_names)}</div></div>", unsafe_allow_html=True)
with m_col3: st.markdown("<div class='mini-card bg-mode'><b>Mode</b><div>MediaPipe + DeepFace</div></div>", unsafe_allow_html=True)

# --- MODULES ---
if st.session_state.menu == "Home":
    st.subheader("Project Overview")
    st.info("Sistem ini menggunakan MediaPipe untuk pengesanan wajah pantas dan DeepFace (VGG-Face) untuk pengecaman identiti jenayah.")

elif st.session_state.menu == "Register":
    st.markdown("### 📝 New Registration")
    with st.form("reg_form"):
        new_name = st.text_input("Suspect Full Name")
        upload_img = st.file_uploader("Upload Evidence Photo", type=['jpg', 'png'])
        if st.form_submit_button("💾 Save Record"):
            if new_name and upload_img:
                # Simpan sementara untuk DeepFace
                with open("temp.jpg", "wb") as f: f.write(upload_img.getbuffer())
                try:
                    # Ganti face_encodings dengan representasi DeepFace
                    embeddings = DeepFace.represent(img_path="temp.jpg", model_name="VGG-Face")[0]["embedding"]
                    encoding_str = json.dumps(embeddings)
                    
                    conn = sqlite3.connect('forensic_lab.db')
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO suspects (name, encoding) VALUES (?, ?)", (new_name, encoding_str))
                    conn.commit()
                    conn.close()
                    st.success(f"✅ {new_name} registered!")
                except:
                    st.error("Ralat: Wajah tidak dikesan dengan jelas.")
                os.remove("temp.jpg")

elif st.session_state.menu == "Scan":
    st.markdown("### 🔍 Live Forensic Scan")
    img_file = st.camera_input("Scan Subject Face")
    if img_file:
        img = Image.open(img_file)
        img_np = np.array(img.convert('RGB'))
        img_path = "scan_temp.jpg"
        cv2.imwrite(img_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        try:
            # 1. Detect & Represent Face
            current_emb = DeepFace.represent(img_path=img_path, model_name="VGG-Face")[0]["embedding"]
            
            found_name = "Unknown"
            max_similarity = -1
            
            # 2. Manual Matching (Cosine Similarity)
            for name, db_emb in zip(known_names, known_encodings):
                # Kira dot product untuk similarity
                sim = np.dot(current_emb, db_emb) / (np.linalg.norm(current_emb) * np.linalg.norm(db_emb))
                if sim > 0.65 and sim > max_similarity: # Threshold 0.65
                    max_similarity = sim
                    found_name = name
            
            if found_name != "Unknown":
                st.error(f"🚨 TARGET IDENTIFIED: {found_name} (Match: {max_similarity:.2%})")
            else:
                st.warning("⚠️ No Match Found in Database")
                
        except:
            st.error("No face detected. Please adjust position.")
        
        if os.path.exists(img_path): os.remove(img_path)