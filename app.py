import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Laptop Depreciation Oracle",
    page_icon="📉",
    layout="wide"
)

# ==========================================
# 2. FUNGSI MEMUAT MODEL AI (ANTI-RELOAD CACHE)
# ==========================================
# @st.cache_resource digunakan agar Streamlit tidak membaca ulang file model 
# setiap kali user menggeser tombol, sehingga aplikasi berjalan super cepat.
@st.cache_resource
def load_oracle_brain():
    # Memuat otak Random Forest Regressor hasil latihan dari Google Colab
    model = joblib.load("model_random_forest_baru.pkl")
    return model

try:
    model_rf = load_oracle_brain()
    model_loaded = True
except:
    model_loaded = False

# ==========================================
# 3. STRUKTUR ANTARMUKA (INTERFACE DESIGN)
# ==========================================
st.title("📉 Laptop Depreciation Oracle")
st.caption("Aplikasi Prediksi Kurva Depresiasi Nilai Sisa Aset Elektronik Menggunakan Random Forest Regressor")
st.markdown("---")

if not model_loaded:
    st.error("❌ Berkas 'model_random_forest_baru.pkl' tidak ditemukan di folder aplikasi ini!")
    st.info("💡 Solusi: Download file .pkl hasil running Google Colab lu tadi, lalu taruh di folder yang sama dengan file app.py ini.")
else:
    # Membagi halaman menjadi 2 Kolom Besar (Kiri: Input User, Kanan: Hasil Visualisasi AI)
    col_input, col_hasil = st.columns([4, 6], gap="large")
    
    with col_input:
        st.header("📋 Spesifikasi Aset Baru")
        
        # Input 1: Merek Laptop (Untuk keperluan display/antarmuka user)
        merek_input = st.text_input("Nama / Seri Laptop:", value="Lenovo LOQ 15")
        
        # Input 2: Kategori Laptop (Variabel Numerik X1 yang dipahami oleh AI)
        kategori_pilihan = st.selectbox(
            "Pilih Kategori Komponen (Kasta Thermal):",
            ["Pelajar / Entry Level (Susut Standar)", "Ultrabook / Premium (Susut Lambat)", "Gaming / Workstation (Susut Cepat)"]
        )
        
        # Menerjemahkan pilihan teks user menjadi kode angka biner untuk input Model AI
        if "Pelajar" in kategori_pilihan:
            kat_code = 0.0
        elif "Ultrabook" in kategori_pilihan:
            kat_code = 1.0
        else:
            kat_code = 2.0
            
        # Input 3: Harga Beli Baru (Variabel Numerik X2 untuk Model AI)
        harga_baru = st.slider(
            "💰 Harga Pembelian Kondisi Baru (Juta Rp):",
            min_value=4.0,
            max_value=40.0,
            value=15.0, # Default awal 15 Juta Rupiah
            step=0.5
        )
        
        st.markdown("---")
        tombol_ramal = st.button("⚡ SIMULASIKAN DEPRESIASI HARGA", type="primary", use_container_width=True)
        
    with col_hasil:
        st.header("📊 Kurva Penyusutan Nilai Jual")
        
        if tombol_ramal:
            # ========================================================
            # 4. LOGIKA INFERENCE PIPELINE (PROSES PREDIKSI TAHUNAN)
            # ========================================================
            # Kita membuat matriks prediksi untuk 3 titik waktu (12, 24, 36 bulan)
            # Format input model AI harus sama persis dengan urutan saat training di Colab:
            # [Kategori_Code, Harga_Baru_Juta, Umur_Bulan]
            
            input_thn1 = np.array([[kat_code, harga_baru, 12.0]]) # Umur 1 Tahun
            input_thn2 = np.array([[kat_code, harga_baru, 24.0]]) # Umur 2 Tahun
            input_thn3 = np.array([[kat_code, harga_baru, 36.0]]) # Umur 3 Tahun (Wisuda)
            
            # Menembakkan data ke model Random Forest Regressor untuk mendapatkan nilai kontinu (y_pred)
            pred_thn1 = model_rf.predict(input_thn1)[0]
            pred_thn2 = model_rf.predict(input_thn2)[0]
            pred_thn3 = model_rf.predict(input_thn3)[0]
            
            # Pengaman matematika: Harga bekas dari tahun ke tahun harus menurun rasional
            if pred_thn2 > pred_thn1: pred_thn2 = pred_thn1 * 0.85
            if pred_thn3 > pred_thn2: pred_thn3 = pred_thn2 * 0.85
            
            # Array data untuk sumbu X (Waktu) dan sumbu Y (Harga Hasil Tebakan AI)
            poros_tahun = ["Kondisi Baru", "Tahun ke-1", "Tahun ke-2", "Tahun ke-3 (Lulus)"]
            poros_harga = [harga_baru, round(pred_thn1, 2), round(pred_thn2, 2), round(pred_thn3, 2)]
            
            # ========================================================
            # 5. PEMBUATAN GRAFIK GARIS KONTINYU MENGGUNAKAN PLOTLY
            # ========================================================
            fig = go.Figure()
            
            # Menambahkan garis kurva depresiasi
            fig.add_trace(go.Scatter(
                x=poros_tahun,
                y=poros_harga,
                mode='lines+markers+text',
                text=[f"Rp {h:.1f} Jt" for h in poros_harga],
                textposition="top center",
                line=dict(color='#ff4b4b', width=4),
                marker=dict(size=10, symbol="circle"),
                name="Nilai Sisa Aset"
            ))
            
            fig.update_layout(
                xaxis_title="Timeline Pemakaian Perangkat",
                yaxis_title="Estimasi Nilai Jual Kembali (Juta Rp)",
                yaxis=dict(range=[0, harga_baru + 3]),
                margin=dict(l=40, r=40, t=20, b=40),
                height=350
            )
            
            # Menampilkan grafik ke layar Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # ========================================================
            # 6. RINGKASAN EKONOMI / TERJEMAHAN AKADEMIK UNTUK USER
            # ========================================================
            st.markdown("#### 📝 Ringkasan Analisis Finansial AI:")
            
            # Menghitung persentase penyusutan total setelah 3 tahun pemakaian
            total_penyusutan_persen = ((harga_baru - pred_thn3) / harga_baru) * 100
            
            with st.container(border=True):
                st.write(f"🔹 **Aset Diuji:** {merek_input} ({kategori_pilihan.split('(')[0]})")
                st.write(f"🔹 **Penyusutan Nilai (3 Tahun):** Turun `{total_penyusutan_persen:.1f}%` dari harga beli awal.")
                st.info(
                    f"💡 **Rekomendasi Akuntansi:** Pada tahun ke-3 saat Anda lulus kuliah, laptop ini diprediksi "
                    f"masih laku dijual di kisaran **Rp {pred_thn3:.2f} Juta**. Dana ini dapat dicatat sebagai "
                    f"nilai sisa aset (*residual value*) untuk modal peningkatan hardware masa depan Anda."
                )
        else:
            # Kondisi awal sebelum tombol ditekan
            st.info("👈 Silakan masukkan spesifikasi laptop baru Anda di panel kiri, lalu klik tombol Jalankan Simulasi.")
