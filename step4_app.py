import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import subprocess
import sys

st.set_page_config(page_title="Wine Quality Analyzer", page_icon="🍷", layout="wide")

st.markdown("""
<style>
.main-title { font-size:2.4rem; font-weight:700; color:#7F0000; text-align:center; }
.sub-title  { text-align:center; color:#888; margin-bottom:2rem; }
.result-high   { background:#E8F5E9; border:2px solid #2E7D32; padding:1.5rem; border-radius:12px; text-align:center; }
.result-medium { background:#FFF8E1; border:2px solid #C8A84B; padding:1.5rem; border-radius:12px; text-align:center; }
.result-low    { background:#FFEBEE; border:2px solid #B71C1C; padding:1.5rem; border-radius:12px; text-align:center; }
.rec-box { background:#F8F4FF; border:1.5px solid #9C6FD6; border-radius:12px; padding:1.2rem 1.5rem; margin-top:1rem; }
.rec-box-white { background:#F0F8FF; border:1.5px solid #4A90D9; border-radius:12px; padding:1.2rem 1.5rem; margin-top:1rem; }
.rec-title { font-size:1rem; font-weight:700; margin-bottom:0.5rem; }
.rec-item { padding:6px 0; border-bottom:1px solid #eee; font-size:0.9rem; }
.rec-item:last-child { border-bottom:none; }
.tip-box { background:#FFFDE7; border:1.5px solid #F9A825; border-radius:10px; padding:1rem 1.2rem; margin-top:1rem; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ---- Auto Training ----
def auto_train():
    if not os.path.exists("model_red.pkl"):
        st.warning("⏳ Model belum ada, sedang training otomatis... mohon tunggu 1-2 menit")
        with st.spinner("Downloading dataset..."):
            subprocess.run([sys.executable, "step1_download_data.py"])
        with st.spinner("Training model Red Wine..."):
            subprocess.run([sys.executable, "step3_train_model.py"])
        st.success("✅ Training selesai! Silakan refresh halaman.")
        st.stop()

auto_train()

# ---- Data rekomendasi produk wine ----
RED_WINE_PRODUCTS = {
    "High": [
        {"name": "Château Margaux",     "origin": "Perancis",        "desc": "Mewah, kompleks, aroma buah gelap & kayu oak"},
        {"name": "Opus One",            "origin": "California, USA",  "desc": "Full-bodied, fruity, salah satu red wine terbaik dunia"},
        {"name": "Penfolds Grange",     "origin": "Australia",        "desc": "Kuat, tannic, cocok untuk aging jangka panjang"},
        {"name": "Sassicaia",           "origin": "Italia",           "desc": "Elegan, mineral, Super Tuscan terbaik"},
        {"name": "Caymus Cabernet",     "origin": "Napa Valley, USA", "desc": "Lembut, kaya, cocok untuk pemula wine premium"},
    ],
    "Medium": [
        {"name": "Jacob's Creek Shiraz","origin": "Australia",        "desc": "Populer, mudah didapat, rasa buah yang menyenangkan"},
        {"name": "Malbec Catena",       "origin": "Argentina",        "desc": "Buah gelap, lembut, harga terjangkau"},
        {"name": "Chianti Classico",    "origin": "Italia",           "desc": "Asam, earthy, cocok diminum bersama pasta"},
        {"name": "Yellow Tail Cabernet","origin": "Australia",        "desc": "Ramah di lidah, cocok untuk sehari-hari"},
        {"name": "Concha y Toro",       "origin": "Chile",            "desc": "Buah segar, ringan, sangat populer di Asia"},
    ],
    "Low": [
        {"name": "Belum memenuhi standar produk komersial", "origin": "-", "desc": "Parameter kimia perlu diperbaiki sebelum layak dijual"},
    ],
}

WHITE_WINE_PRODUCTS = {
    "High": [
        {"name": "Puligny-Montrachet",          "origin": "Perancis",        "desc": "Mewah, mineral, salah satu white wine terbaik dunia"},
        {"name": "Cloudy Bay Sauvignon Blanc",  "origin": "New Zealand",     "desc": "Segar, citrusy, sangat populer & mudah dinikmati"},
        {"name": "Riesling Dr. Loosen",         "origin": "Jerman",          "desc": "Manis-asam sempurna, aromatik, cocok makanan pedas"},
        {"name": "Gaja Gaia & Rey Chardonnay",  "origin": "Italia",          "desc": "Kompleks, creamy, premium Italia"},
        {"name": "Leeuwin Estate Art Series",   "origin": "Australia",       "desc": "Buttery, vanilla, chardonnay kelas dunia"},
    ],
    "Medium": [
        {"name": "Kim Crawford Sauvignon Blanc",  "origin": "New Zealand",     "desc": "Ringan, fruity, sangat mudah diminum"},
        {"name": "Santa Margherita Pinot Grigio", "origin": "Italia",          "desc": "Kering, segar, pasangan sempurna seafood"},
        {"name": "Kendall-Jackson Chardonnay",    "origin": "California, USA", "desc": "Creamy, vanilla, populer di restoran"},
        {"name": "Babich Sauvignon Blanc",        "origin": "New Zealand",     "desc": "Herbal, segar, harga terjangkau"},
        {"name": "Wolf Blass Riesling",           "origin": "Australia",       "desc": "Bunga, citrus, cocok untuk pemula"},
    ],
    "Low": [
        {"name": "Belum memenuhi standar produk komersial", "origin": "-", "desc": "Parameter kimia perlu diperbaiki sebelum layak dijual"},
    ],
}

TIPS = {
    "red": {
        "High":   "Wine kamu memiliki profil kimia premium! Alcohol tinggi dan volatile acidity rendah adalah kunci kualitas red wine terbaik. 🏆",
        "Medium": "Wine kamu cukup baik untuk konsumsi sehari-hari. Coba tingkatkan kadar alcohol dan turunkan volatile acidity untuk hasil lebih baik. 👍",
        "Low":    "Wine kamu perlu perbaikan. Volatile acidity terlalu tinggi atau alcohol terlalu rendah adalah penyebab utama kualitas rendah. ⚠️",
    },
    "white": {
        "High":   "Profil kimia white wine kamu sangat baik! Keseimbangan antara keasaman, gula, dan alcohol sangat ideal. 🏆",
        "Medium": "White wine kamu layak konsumsi. Perhatikan kadar Free SO₂ dan density untuk meningkatkan kualitas. 👍",
        "Low":    "White wine kamu perlu perbaikan. Cek kadar volatile acidity dan pastikan Free SO₂ dalam batas optimal. ⚠️",
    },
}

# ---- Load model ----
@st.cache_resource
def load_model(wine_type):
    if not os.path.exists(f"model_{wine_type}.pkl"):
        return None, None, None
    model    = joblib.load(f"model_{wine_type}.pkl")
    scaler   = joblib.load(f"scaler_{wine_type}.pkl")
    features = joblib.load(f"features_{wine_type}.pkl")
    return model, scaler, features

# ---- Header ----
st.markdown('<h1 class="main-title">🍷 Wine Quality Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Prediksi kualitas wine berdasarkan parameter kimia + rekomendasi produk serupa</p>', unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.header("⚙️ Pengaturan")
wine_type = st.sidebar.radio("Jenis Wine", ["Red Wine 🍷", "White Wine 🥂"])
wt = "red" if "Red" in wine_type else "white"
color_accent = "#7F0000" if wt == "red" else "#4A90D9"

model, scaler, features = load_model(wt)

if model is None:
    st.error("⚠️ Model belum ditraining! Jalankan dulu: python step3_train_model.py")
    st.stop()

st.sidebar.success(f"Model siap ✅")

# ---- Input parameter ----
st.subheader("📊 Masukkan Parameter Kimia Wine")

col1, col2, col3 = st.columns(3)
with col1:
    fixed_acidity    = st.slider("Fixed Acidity (g/L)",    4.0, 16.0, 7.5,  0.1)
    volatile_acidity = st.slider("Volatile Acidity (g/L)", 0.1,  1.5, 0.45, 0.01)
    citric_acid      = st.slider("Citric Acid (g/L)",      0.0,  1.0, 0.3,  0.01)
    residual_sugar   = st.slider("Residual Sugar (g/L)",   1.0, 15.0, 2.5,  0.1)
with col2:
    chlorides  = st.slider("Chlorides (g/L)",   0.01, 0.2,   0.08, 0.001)
    free_so2   = st.slider("Free SO₂ (mg/L)",   1.0,  72.0,  30.0, 1.0)
    total_so2  = st.slider("Total SO₂ (mg/L)",  6.0,  200.0, 80.0, 1.0)
with col3:
    density   = st.slider("Density (g/cm³)", 0.990, 1.003, 0.996, 0.0001, format="%.4f")
    ph        = st.slider("pH",              2.8,   4.0,   3.3,  0.01)
    sulphates = st.slider("Sulphates (g/L)", 0.3,   2.0,   0.65, 0.01)
    alcohol   = st.slider("Alcohol (%)",     8.0,   15.0,  10.5, 0.1)

st.markdown("---")

# ---- Tombol analisa ----
if st.button("🔬 Analisa Kualitas Wine", use_container_width=True, type="primary"):
    input_data   = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                   chlorides, free_so2, total_so2, density, ph, sulphates, alcohol]],
                                  columns=features)
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]
    proba_dict   = dict(zip(model.classes_, proba))

    st.subheader("📋 Hasil Analisa")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        css   = {"High":"result-high","Medium":"result-medium","Low":"result-low"}[prediction]
        emoji = {"High":"🌟","Medium":"👍","Low":"⚠️"}[prediction]
        desc  = {"High":"Kualitas Tinggi","Medium":"Kualitas Sedang","Low":"Kualitas Rendah"}[prediction]
        st.markdown(f'''
        <div class="{css}">
            <div style="font-size:3rem">{emoji}</div>
            <div style="font-size:1.8rem;font-weight:700">{prediction}</div>
            <div style="font-size:1rem;margin-top:0.3rem">{desc}</div>
        </div>''', unsafe_allow_html=True)

    with col_b:
        st.markdown("**Probabilitas setiap kelas:**")
        for label in ["Low", "Medium", "High"]:
            p = proba_dict.get(label, 0)
            st.markdown(f"`{label}` {p:.1%}")
            st.progress(float(p))
        feat_series = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(5, 2.5))
        feat_series.plot(kind="barh", ax=ax, color=color_accent, edgecolor="white")
        ax.invert_yaxis()
        ax.set_title("Top 5 Faktor Berpengaruh")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🍾 Rekomendasi Produk Wine Serupa")

    products  = RED_WINE_PRODUCTS[prediction] if wt == "red" else WHITE_WINE_PRODUCTS[prediction]
    box_class = "rec-box" if wt == "red" else "rec-box-white"
    wine_icon = "🍷" if wt == "red" else "🥂"

    rec_html = f'<div class="{box_class}"><div class="rec-title">{wine_icon} Produk dengan karakteristik kimia serupa:</div>'
    for p in products:
        if p["origin"] != "-":
            rec_html += f'''<div class="rec-item">
                <strong>{p["name"]}</strong> &nbsp;
                <span style="color:#888;font-size:0.82rem">({p["origin"]})</span><br>
                <span style="color:#555">{p["desc"]}</span>
            </div>'''
        else:
            rec_html += f'<div class="rec-item" style="color:#B71C1C"><strong>{p["name"]}</strong><br><span>{p["desc"]}</span></div>'
    rec_html += "</div>"
    st.markdown(rec_html, unsafe_allow_html=True)

    tip = TIPS[wt][prediction]
    st.markdown(f'<div class="tip-box">💡 <strong>Tips Sommelier AI:</strong> {tip}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#aaa;font-size:0.8rem'>Wine Quality Analyzer — Python + scikit-learn + Streamlit</p>", unsafe_allow_html=True)
