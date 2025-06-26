import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load('model_linear.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Prediksi Nilai Akhir Siswa (G3)")

# Input dari user
Medu = st.slider("Tingkat Pendidikan Ibu (Medu)", 0, 4, 2)
Fedu = st.slider("Tingkat Pendidikan Ayah (Fedu)", 0, 4, 2)
studytime = st.slider("Jam Belajar (studytime)", 1, 4, 2)
failures = st.slider("Jumlah Kegagalan Sebelumnya", 0, 3, 0)
absences = st.slider("Jumlah Ketidakhadiran", 0, 93, 4)
G1 = st.slider("Nilai G1", 0, 20, 10)
G2 = st.slider("Nilai G2", 0, 20, 10)

# Buat array fitur
features = np.array([[Medu, Fedu, studytime, failures, absences, G1, G2]])

# Scaling
features_scaled = scaler.transform(features)

# Prediksi
prediction = model.predict(features_scaled)[0]

st.subheader("Hasil Prediksi G3:")
st.success(f"Nilai Akhir Diprediksi: {round(prediction, 2)}")
