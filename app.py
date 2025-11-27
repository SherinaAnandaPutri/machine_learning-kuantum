import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Simulasi Fisika Kuantum 1D",
    layout="wide"
)

# ====================
# LOAD MODEL
# ====================
@st.cache_resource
def load_model():
    return joblib.load("rf_infinite_well_model.joblib")

model = load_model()

# ====================
# UI HEADER
# ====================
st.title("🔮 Simulasi Fisika Kuantum: Sumur Potensial Tak Hingga 1D")
st.markdown("Model machine learning + visualisasi bentuk fungsi gelombang kuantum")

st.divider()

# ====================
# USER INPUT
# ====================
col1, col2 = st.columns(2)

with col1:
    posisi = st.slider("Posisi x dalam sumur (0 - 1)", 0.01, 0.99, 0.5, 0.01)

with col2:
    n = st.slider("Tingkat Energi (n)", 1, 5, 1)

# ====================
# PREDIKSI DENGAN MODEL ML
# ====================
predicted_energy = model.predict(np.array([[posisi]]))[0]

st.metric("Prediksi Energi Model ML", f"{predicted_energy:.4f}")

# ====================
# FUNGSI GELOMBANG ANALITIK
# ====================
x = np.linspace(0, 1, 400)
psi = np.sqrt(2) * np.sin(n * np.pi * x)

# ====================
# PLOT 1: Fungsi Gelombang
# ====================
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=x,
    y=psi,
    mode="lines",
    line=dict(width=3),
    name="Wave Function"
))
fig1.update_layout(
    title="Fungsi Gelombang Partikel dalam Sumur",
    xaxis_title="Posisi x",
    yaxis_title="Ψ(x)",
    template="simple_white",
    width=900,
    height=400
)

st.plotly_chart(fig1, use_container_width=True)

# ====================
# PLOT 2: DENSITAS PROBABILITAS
# ====================
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=x,
    y=psi**2,
    mode="lines",
    line=dict(width=3),
))
fig2.update_layout(
    title="Densitas Probabilitas |Ψ(x)|²",
    xaxis_title="Posisi x",
    yaxis_title="|Ψ(x)|²",
    template="plotly_white",
    width=900,
    height=400
)

st.plotly_chart(fig2, use_container_width=True)

# ====================
# PLOT 3: ANIMASI 3D SUMUR KUANTUM
# ====================
X, T = np.meshgrid(x, np.linspace(0, 2*np.pi, 200))
psi_t = np.sqrt(2)*np.sin(n*np.pi*X)*np.cos(5*T)

fig3 = go.Figure(
    data=[go.Surface(
        x=X,
        y=T,
        z=psi_t,
        colorscale="Viridis"
    )]
)

fig3.update_layout(
    title="Animasi Kuantum 3D: Evolusi Fungsi Gelombang",
    scene=dict(
        xaxis_title="x",
        yaxis_title="t",
        zaxis_title="Ψ"
    ),
    width=900,
    height=600
)

st.plotly_chart(fig3, use_container_width=True)

st.success("Simulasi selesai dirender 🎉")
