import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("Prediksi Energi Tingkat Dasar – Sumur Potensial 1D")
st.write("Model machine learning dilatih langsung di Streamlit Cloud agar kompatibel dan bebas error.")

# ---------- Generate dataset ----------
def generate_data(n=5000):
    h = 6.626e-34  # konstanta Planck
    m = np.random.uniform(9e-31, 9e-30, n)  # massa partikel
    L = np.random.uniform(1e-10, 1e-8, n)   # panjang sumur
    E1 = (h**2) / (8 * m * L**2)
    X = np.column_stack([m, L])
    return X, E1

X, y = generate_data()

# ---------- Train model ----------
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

st.success("Model berhasil dilatih di Streamlit Cloud! 🚀")

# ---------- Input pengguna ----------
st.subheader("Masukkan Parameter Fisika")

m_user = st.number_input(
    "Massa partikel (kg)",
    min_value=1e-31,
    max_value=1e-28,
    value=9.11e-31,
    format="%.2e"
)

L_user = st.number_input(
    "Lebar sumur potensial (m)",
    min_value=1e-11,
    max_value=1e-8,
    value=1e-9,
    format="%.2e"
)

if st.button("Prediksi Energi Tingkat Dasar"):
    input_data = np.array([[m_user, L_user]])
    predicted_energy = model.predict(input_data)[0]

    st.write("### Hasil Prediksi Energi (Joule):")
    st.code(f"{predicted_energy:.4e}")
