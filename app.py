# app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import time

# ---------------------------
# UI - page config
# ---------------------------
st.set_page_config(
    page_title="ML Quantum: Prediksi Energi E₀ (1D Infinite Well)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:16px">
      <div style="flex:1">
        <h1 style="margin-bottom:0">🔬 ML Quantum — Prediksi Energi Tingkat Dasar (E₀)</h1>
        <p style="margin-top:6px;color:#555">
           Aplikasi interaktif: generate data fisika, latih model ML, visualisasikan hasil, dan simulasikan fungsi gelombang.
           Untuk skripsi — Pendidikan Fisika • Universitas Sriwijaya.
        </p>
      </div>
      <div style="text-align:right;color:#888">
        <small>by: mahasiswa — demo</small>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Physical helper functions
# ---------------------------
h = 6.62607015e-34  # Planck constant (J s)
hbar = h / (2 * np.pi)
m_e = 9.10938356e-31  # electron mass (kg)

def analytic_E0_joule(a_m, m_kg):
    """Analytic ground-state energy E1 = h^2 / (8 m a^2) (Joule)"""
    return h**2 / (8.0 * m_kg * a_m**2)

# ---------------------------
# Dataset + training (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def generate_dataset(N=2000, a_min_nm=0.2, a_max_nm=5.0, m_min_me=0.1, m_max_me=5.0):
    """Generate dataset using analytic infinite-well formula.
       Returns features in (a_nm, m_me) and targets in Joule.
    """
    np.random.seed(42)
    a_nm = np.random.uniform(a_min_nm, a_max_nm, N)
    m_me = np.random.uniform(m_min_me, m_max_me, N)
    a_m = a_nm * 1e-9
    m_kg = m_me * m_e
    E_j = analytic_E0_joule(a_m, m_kg)
    # small noise to simulate measurement error
    noise = np.random.normal(scale=1e-6 * np.mean(E_j), size=E_j.shape)
    E_j_noisy = E_j + noise
    df = pd.DataFrame({"a_nm": a_nm, "m_me": m_me, "E_j": E_j_noisy})
    return df

@st.cache_data(show_spinner=False)
def train_model(df, test_size=0.2, random_state=42, n_estimators=200):
    X = df[["a_nm", "m_me"]].values
    y = df["E_j"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    ))
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # return model and evaluation on test
    return pipeline, X_test, y_test, y_pred, rmse, r2

# Sidebar controls
st.sidebar.header("Kontrol Dataset & Model")
N = st.sidebar.slider("Ukuran dataset (N)", 500, 8000, 2000, step=500)
n_estimators = st.sidebar.slider("n_estimators (RandomForest)", 50, 600, 200, step=50)
test_size = st.sidebar.slider("Test size (%)", 5, 40, 20, step=5) / 100.0

# Generate dataset & train (show spinner)
with st.spinner("Mengenerate dataset & melatih model ML — tunggu sebentar..."):
    df = generate_dataset(N=N)
    model, X_test, y_test, y_pred, rmse, r2 = train_model(df, test_size=test_size, n_estimators=n_estimators)

st.success("Model siap — pelatihan selesai ✅")

# ---------------------------
# Layout: left (controls + sim), right (plots)
# ---------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("🎛️ Input Parameter (User)")
    st.markdown("Masukkan parameter fisika untuk prediksi E₀. Satuan: nm untuk `a`, massa dalam `m_e`.")
    a_input_nm = st.number_input("Lebar sumur `a` (nm)", min_value=0.10, max_value=20.0, value=1.0, step=0.01, format="%.3f")
    m_input_me = st.number_input("Massa partikel `m` (dalam m_e)", min_value=0.01, max_value=10.0, value=1.0, step=0.01, format="%.3f")
    # conversion
    a_input_m = a_input_nm * 1e-9
    m_input_kg = m_input_me * m_e

    if st.button("🔮 Prediksi E₀ (ML + Analitik)"):
        # ML prediction (model expects (a_nm, m_me))
        x_in = np.array([[a_input_nm, m_input_me]])
        pred_ml_joule = model.predict(x_in)[0]
        pred_analytic_joule = analytic_E0_joule(a_input_m, m_input_kg)
        # display nicely (J and eV)
        eV_to_J = 1.602176634e-19
        st.metric("Prediksi ML — E₀ (J)", f"{pred_ml_joule:.4e}")
        st.metric("Analitik — E₀ (J)", f"{pred_analytic_joule:.4e}")
        st.write(f"Prediksi ML — E₀ (eV): **{pred_ml_joule / eV_to_J:.6f} eV**")
        st.write(f"Analitik — E₀ (eV): **{pred_analytic_joule / eV_to_J:.6f} eV**")

    st.markdown("---")
    st.subheader("🔭 Simulasi Fungsi Gelombang Dasar (Ground State)")
    st.markdown("Menampilkan |ψ(x,t)|² (probability density) dan ψ(x,t) (real part) untuk mode dasar pada interval [0, a].")

    # wavefunction visualization controls
    x_points = st.slider("Resolusi x (points)", 200, 2000, 500, step=100)
    animate = st.checkbox("Animasi fase waktu (simulate time evolution)", value=True)
    t_speed = st.slider("Kecepatan animasi (fps)", 1, 30, 8)

    # compute wavefunction arrays
    a_plot_m = a_input_m
    x = np.linspace(0, a_plot_m, x_points)
    # ground-state wavefunction (infinite well) psi(x,t) = sqrt(2/a) * sin(pi x / a) * exp(-i E t / hbar)
    psi0_x = np.sqrt(2.0 / a_plot_m) * np.sin(np.pi * x / a_plot_m)
    E0_j = analytic_E0_joule(a_plot_m, m_input_kg)
    # choose small times for animation
    times = np.linspace(0, 2*np.pi*hbar/E0_j, 40) if (E0_j>0 and animate) else [0.0]

    # Prepare static plot (probability density)
    fig_psi = go.Figure()
    fig_psi.add_trace(go.Line(x=x * 1e9, y=psi0_x**2, name="|ψ(x)|²", line=dict(width=3)))
    fig_psi.update_layout(
        title=f"Probabilitas |ψ(x)|² pada a={a_input_nm:.3f} nm, ground state",
        xaxis_title="x (nm)",
        yaxis_title="|ψ|² (1/nm)",
        template="plotly_white",
        height=360
    )
    st.plotly_chart(fig_psi, use_container_width=True)

    # Animated real part if requested
    if animate and E0_j > 0:
        st.markdown("Animasi real(ψ(x,t)) (fase waktu)")
        frames = []
        # time sampling
        tvals = np.linspace(0, 1.0, 30)
        # Using a simple phase factor with angular frequency w = E0/hbar
        w = E0_j / hbar
        # create frames
        fig_anim = go.Figure(
            data=[go.Scatter(x=x*1e9, y=(psi0_x * np.cos(0)) , mode='lines', line=dict(width=2))],
            layout=go.Layout(
                xaxis=dict(title="x (nm)"),
                yaxis=dict(title="Re(ψ)"),
                title="Re(ψ(x,t)) — Ground state (animated)",
                updatemenus=[dict(type="buttons",
                                  buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None, {"frame": {"duration": int(1000/t_speed), "redraw": True},
                                                             "fromcurrent": True, "transition": {"duration": 0}}])])],
                template="plotly_white",
                height=360
            ),
            frames=[go.Frame(data=[go.Scatter(x=x*1e9, y=(psi0_x * np.cos(w * t)))]) for t in tvals]
        )
        st.plotly_chart(fig_anim, use_container_width=True)

with right_col:
    st.subheader("📊 Evaluasi Model & Visualisasi Dataset")
    # Show small sample
    st.markdown("Contoh sampel dataset (a [nm], m [m_e], E [J])")
    st.dataframe(df.sample(6).reset_index(drop=True), use_container_width=True)

    # 1) Scatter Predicted vs True (test set)
    st.markdown("**Prediksi (Test set): ML vs True analytic**")
    scatter_df = pd.DataFrame({
        "E_true": y_test,
        "E_pred": y_pred,
        "error": y_test - y_pred
    })
    # Convert to eV for readability
    scatter_df_ev = scatter_df.copy()
    scatter_df_ev[["E_true","E_pred","error"]] = scatter_df_ev[["E_true","E_pred","error"]] / 1.602176634e-19

    fig_scatter = px.scatter(
        scatter_df_ev.sample(min(1000, len(scatter_df_ev)), random_state=1),
        x="E_true", y="E_pred",
        title=f"Prediksi vs True (samples) — RMSE={rmse:.3e} J, R²={r2:.4f}",
        labels={"E_true":"True E (eV)","E_pred":"Predicted E (eV)"},
        trendline="ols",
        opacity=0.7
    )
    fig_scatter.add_shape(type="line",
                          x0=scatter_df_ev["E_true"].min(), x1=scatter_df_ev["E_true"].max(),
                          y0=scatter_df_ev["E_true"].min(), y1=scatter_df_ev["E_true"].max(),
                          line=dict(dash="dash", color="black"))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 2) Residual histogram
    st.markdown("**Distribusi Residual (E_true - E_pred)**")
    fig_res = px.histogram(scatter_df_ev, x="error", nbins=60, title="Residual distribution (eV)")
    st.plotly_chart(fig_res, use_container_width=True)

    # 3) Heatmap: analytic E over grid and ML predicted E over same grid (compare)
    st.markdown("**Heatmap Perbandingan: Analitik vs Prediksi ML**")
    a_grid = np.linspace(df["a_nm"].min(), df["a_nm"].max(), 80)
    m_grid = np.linspace(df["m_me"].min(), df["m_me"].max(), 80)
    A, M = np.meshgrid(a_grid, m_grid)
    grid_points = np.column_stack([A.ravel(), M.ravel()])
    # analytic in Joule
    A_m = A.ravel() * 1e-9
    M_kg = M.ravel() * m_e
    E_analytic_grid = analytic_E0_joule(A_m, M_kg)
    # ML predicted
    E_ml_grid = model.predict(grid_points)
    # reshape to 2D
    E_analytic_2d = E_analytic_grid.reshape(A.shape) / 1.602176634e-19  # eV
    E_ml_2d = E_ml_grid.reshape(A.shape) / 1.602176634e-19  # eV
    diff_2d = (E_ml_2d - E_analytic_2d)

    # show analytic heatmap
    fig_heat = go.Figure()
    fig_heat.add_trace(go.Heatmap(
        z=E_analytic_2d,
        x=a_grid, y=m_grid,
        colorbar=dict(title="E (eV)"),
        zmid=np.median(E_analytic_2d),
    ))
    fig_heat.update_layout(title="Analytic E₀ (eV) over (a_nm, m_me)", xaxis_title="a (nm)", yaxis_title="m (m_e)", height=420, template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # show ML heatmap
    fig_heat_ml = go.Figure()
    fig_heat_ml.add_trace(go.Heatmap(
        z=E_ml_2d,
        x=a_grid, y=m_grid,
        colorbar=dict(title="E (eV)"),
        zmid=np.median(E_ml_2d),
    ))
    fig_heat_ml.update_layout(title="ML predicted E₀ (eV) over (a_nm, m_me)", xaxis_title="a (nm)", yaxis_title="m (m_e)", height=420, template="plotly_white")
    st.plotly_chart(fig_heat_ml, use_container_width=True)

    # difference heatmap
    fig_diff = go.Figure()
    fig_diff.add_trace(go.Heatmap(
        z=diff_2d,
        x=a_grid, y=m_grid,
        colorbar=dict(title="ΔE (eV)"),
        colorscale="RdBu",
        zmid=0
    ))
    fig_diff.update_layout(title="Perbedaan (ML - Analytic) [eV]", xaxis_title="a (nm)", yaxis_title="m (m_e)", height=420, template="plotly_white")
    st.plotly_chart(fig_diff, use_container_width=True)

# ---------------------------
# Footer / notes
# ---------------------------
st.markdown("---")
with st.expander("Catatan teknis & saran untuk skripsi"):
    st.write(
        """
        - Model dilatih pada data analitik (infinite square well). Untuk finite well, Anda perlu solver numerik (root finding).
        - Jika ingin menyimpan model hasil tuning, training harus dilakukan dalam environment yang sama untuk menghindari masalah pickle/joblib.
        - Untuk bab hasil & pembahasan: sertakan RMSE, R², plot Predicted vs True, heatmap perbedaan, dan contoh kasus (mis. a=1nm, m=m_e).
        """
    )

st.markdown("© ML Quantum — Demo untuk skripsi (Universitas Sriwijaya)")
