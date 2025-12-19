import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ตั้งค่าหน้ากระดาษ
st.set_page_config(page_title="Waste Prediction System", layout="wide")

# ปรับแต่ง CSS
st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        .stMetric {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .title-text {
            font-size: 40px;
            font-weight: 800;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- ส่วนของการเตรียมข้อมูล (Data Preparation) ---
@st.cache_data # ใช้ cache เพื่อให้โหลดข้อมูลครั้งเดียว
def load_data():
    file_path = 'sustainable_waste_management_dataset_2024.csv'
    df = pd.read_csv(file_path)
    return df

try:
    df = load_data()
    X = df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]
    y = df['waste_kg']

    # สร้างและฝึกโมเดล
    model = LinearRegression()
    model.fit(X, y)

    # --- ส่วนของ Sidebar (Input) ---
    st.sidebar.header("📊 ตั้งค่าปัจจัยการทำนาย")
    st.sidebar.markdown("ปรับค่าด้านล่างเพื่อดูผลลัพธ์แบบ Real-time")
    
    with st.sidebar:
        population = st.slider('จำนวนประชากร', 1000, 50000, 17990)
        recyclable_kg = st.slider('ขยะรีไซเคิล (kg)', 1000, 10000, 5000)
        organic_kg = st.slider('ขยะอินทรีย์ (kg)', 1000, 10000, 5000)
        collection_cap = st.slider('ความสามารถในการเก็บ (kg)', 1000, 10000, 5000)
        overflow = st.slider('ปริมาณขยะล้น (kg)', 100, 2000, 500)
        temp_c = st.slider('อุณหภูมิ (°C)', -10, 40, 25)
        rain_mm = st.slider('ปริมาณฝน (mm)', 0, 500, 100)

    # --- ส่วนหน้าจอหลัก (Main Display) ---
    st.markdown('<p class="title-text">ระบบพยากรณ์ปริมาณขยะอัจฉริยะ</p>', unsafe_allow_html=True)
    
    # คำนวณผลการทำนาย
    input_data = np.array([[population, recyclable_kg, organic_kg, collection_cap, overflow, temp_c, rain_mm]])
    prediction = model.predict(input_data)[0]

    # แสดงผลลัพธ์แบบ Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="ปริมาณขยะที่คาดการณ์", value=f"{prediction:,.2f} kg")
    with col2:
        st.metric(label="ประชากรเป้าหมาย", value=f"{population:,.0f} คน")
    with col3:
        st.metric(label="ขยะรีไซเคิล", value=f"{recyclable_kg:,.0f} kg")

    st.write("---")

    # --- ส่วนของกราฟ (Visualization) ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📈 กราฟเปรียบเทียบผลการทำนาย")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.regplot(x=y, y=model.predict(X), scatter_kws={'alpha':0.3, 'color':'#3498db'}, line_kws={'color':'#e74c3c'}, ax=ax)
        # จุดแดงแสดงตำแหน่งปัจจุบันที่เลือกจาก Slider
        ax.scatter(prediction, prediction, color='yellow', s=200, edgecolors='black', label='Current Prediction', zorder=5)
        ax.set_xlabel('Actual Waste (kg)')
        ax.set_ylabel('Predicted Waste (kg)')
        ax.legend()
        st.pyplot(fig)

    with col_right:
        st.subheader("📋 ข้อมูลปัจจุบัน")
        # แสดงตารางข้อมูลเปรียบเทียบค่าที่ Input เข้าไป
        input_df = pd.DataFrame({
            'ปัจจัย': ['ประชากร', 'รีไซเคิล', 'ขยะอินทรีย์', 'ความจุการเก็บ', 'ขยะล้น', 'อุณหภูมิ', 'ฝน'],
            'ค่าที่เลือก': [population, recyclable_kg, organic_kg, collection_cap, overflow, temp_c, rain_mm]
        })
        st.table(input_df)

    st.markdown('<p style="text-align:center; color:gray; padding-top:50px;">Developed by ไอไก่วิว - © 2024 | Data Driven Insights</p>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"ไม่สามารถโหลดไฟล์ข้อมูลได้: {e}")
    st.info("กรุณาตรวจสอบว่าไฟล์ 'sustainable_waste_management_dataset_2024.csv' อยู่ในโฟลเดอร์เดียวกับโค้ด")