import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Laptop Depreciation Oracle",
    page_icon="📉",
    layout="wide"
)

@st.cache_resource
def load_oracle_brain():
    # Streamlit Cloud otomatis membaca file pkl yang berada di satu root folder GitHub
    return joblib.load("model_random_forest_baru.pkl")

try:
    model_rf = load_oracle_brain()
    model_loaded = True
except:
    model_loaded = False

st.title("📉 Laptop Depreciation Oracle")
st.caption("Aplikasi Prediksi Kurva Depresiasi Nilai Sisa Aset Elektronik Menggunakan Random Forest Regressor")
st.markdown("---")

if not model_loaded:
    st.error("❌ Berkas 'model_random_forest_baru.pkl' tidak ditemukan di repositori GitHub Anda!")
else:
    col_input, col_hasil = st.columns([4, 6], gap="large")
    
    with col_input:
        st.header("📋 Spesifikasi Aset Baru")
        merek_input = st.text_input("Nama / Seri Laptop:", value="Lenovo LOQ 15")
        
        kategori_pilihan = st.selectbox(
            "Pilih Kategori Komponen (Kasta Thermal):",
            ["Pelajar / Entry Level (Susut Standar)", "Ultrabook / Premium (Susut Lambat)", "Gaming / Workstation (Susut Cepat)"]
        )
        
        if "Pelajar" in kategori_pilihan:
            kat_code = 0.0
        elif "Ultrabook" in kategori_pilihan:
            kat_code = 1.0
        else:
            kat_code = 2.0
            
        harga_baru = st.slider(
            "💰 Harga Pembelian Kondisi Baru (Juta Rp):",
            min_value=4.0, max_value=40.0, value=15.0, step=0.5
        )
        
        st.markdown("---")
        tombol_ramal = st.button("⚡ SIMULASIKAN DEPRESIASI HARGA", type="primary", use_container_width=True)
        
    with col_hasil:
        st.header("📊 Kurva Penyusutan Nilai Jual")
        
        if tombol_ramal:
            # PROSES INFERENCE MULTI-TAHAP KONTINU (12, 24, 36 BULAN)
            input_thn1 = np.array([[kat_code, harga_baru, 12.0]])
            input_thn2 = np.array([[kat_code, harga_baru, 24.0]])
            input_thn3 = np.array([[kat_code, harga_baru, 36.0]])
            
            pred_thn1 = model_rf.predict(input_thn1)[0]
            pred_thn2 = model_rf.predict(input_thn2)[0]
            pred_thn3 = model_rf.predict(input_thn3)[0]
            
            # Pengaman tren matematika keuangan
            if pred_thn2 > pred_thn1: pred_thn2 = pred_thn1 * 0.85
            if pred_thn3 > pred_thn2: pred_thn3 = pred_thn2 * 0.85
            
            poros_tahun = ["Kondisi Baru", "Tahun ke-1", "Tahun ke-2", "Tahun ke-3 (Lulus)"]
            poros_harga = [harga_baru, round(pred_thn1, 2), round(pred_thn2, 2), round(pred_thn3, 2)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=poros_tahun, y=poros_harga, mode='lines+markers+text',
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
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📝 Ringkasan Analisis Finansial AI:")
            total_penyusutan_persen = ((harga_baru - pred_thn3) / harga_baru) * 100
            
            with st.container(border=True):
                st.write(f"🔹 **Aset Diuji:** {merek_input} ({kategori_pilihan.split('(')[0]})")
                st.write(f"🔹 **Penyusutan Nilai (3 Tahun):** Turun `{total_penyusutan_persen:.1f}%` dari harga beli awal.")
                st.info(f"💡 **Rekomendasi Akuntansi:** Pada tahun ke-3 saat Anda lulus kuliah, laptop ini diprediksi masih laku dijual di kisaran **Rp {pred_thn3:.2f} Juta**.")
        else:
            st.info("👈 Silakan masukkan spesifikasi laptop baru Anda di panel kiri, lalu klik tombol Jalankan Simulasi.")
