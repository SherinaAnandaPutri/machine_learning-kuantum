# Prediksi Energi Tingkat Dasar Partikel dalam Sumur Potensial 1D Menggunakan Machine Learning

Repositori ini berisi kode Streamlit untuk memprediksi energi tingkat dasar (ground state energy) partikel dalam sumur potensial satu dimensi (1D). Pendekatan yang digunakan adalah *machine learning* dengan model Random Forest Regressor yang dilatih secara otomatis di Streamlit Cloud.

## 🎯 Fitur Utama
- Generasi dataset fisika kuantum otomatis
- Model ML dilatih langsung di cloud (tidak perlu upload file joblib)
- Input interaktif massa partikel & panjang sumur
- Output energi tingkat dasar dalam satuan Joule
- Bebas error perbedaan versi Python/NumPy/Scikit-learn

## 🚀 Cara Menjalankan di Streamlit Cloud
1. Fork folder ini ke GitHub Anda
2. Masuk ke: https://share.streamlit.io/
3. Pilih “Deploy an app”
4. Hubungkan ke repositori GitHub ini
5. Pilih file utama: `app.py`
6. Deploy dan jalankan

## 📌 Teknologi
- Python
- Streamlit
- NumPy
- scikit-learn

## 📚 Rumus Fisika
Energi tingkat dasar sumur potensial 1D:

\[
E_1 = \frac{h^2}{8mL^2}
\]

Model ML belajar mendekati rumus ini berdasarkan dataset simulasi.

---

Dibuat untuk keperluan penelitian skripsi mahasiswa Pendidikan Fisika Universitas Sriwijaya.
