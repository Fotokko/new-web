import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Waste Prediction", layout="wide")

# 2. ปรับแต่ง CSS สำหรับโหมดมืด (Dark Mode) ให้ชัดเจน
st.markdown("""
    <style>
        [data-testid="stMetricValue"] { color: #FFD700 !important; font-size: 35px !important; }
        [data-testid="stMetricLabel"] { color: #FFFFFF !important; font-size: 18px !important; }
        div[data-testid="stMetric"] {
            background-color: #1E293B !important;
            border: 2px solid #FF6347 !important;
            border-radius: 10px;
            padding: 10px !important;
        }
        .title-text {
            font-size: 36px;
            font-weight: bold;
            color: #FF6347;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">🚮 ระบบพยากรณ์ปริมาณขยะตามข้อมูลจริง</p>', unsafe_allow_html=True)

# 3. โหลดข้อมูล (จากตัวอย่างที่คุณส่งมา)
@st.cache_data
def load_data():
    file_path = 'sustainable_waste_management_dataset_2024.csv'
    try:
        df = pd.read_csv(file_path)
        # ใช้เฉพาะ Feature ที่สำคัญตาม Dataset
        features = ['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']
        X = df[features]
        y = df['waste_kg']
        model = LinearRegression().fit(X, y)
        return model, df, y
    except: return None, None, None

model, df, y_data = load_data()

if model is not None:
    # 4. Sidebar ปรับ Range ให้ใกล้เคียง Dataset จริง
    st.sidebar.header("⚙️ ปรับค่าตามสถานการณ์จริง")
    with st.sidebar:
        # อ้างอิงจากข้อมูล: ประชากรประมาณ 17,900
        pop = st.slider('👥 จำนวนประชากร', 17000, 20000, 17950)
        # ขยะรีไซเคิล: ประมาณ 2,600 - 7,000
        recy = st.slider('♻️ ขยะรีไซเคิล (kg)', 1000, 10000, 5000)
        # ขยะอินทรีย์: ประมาณ 6,000 - 11,000
        org = st.slider('🍎 ขยะอินทรีย์ (kg)', 5000, 15000, 8500)
        # ความจุการจัดเก็บ: ประมาณ 18,000 - 22,000
        cap = st.slider('🚛 ความจุการเก็บ (kg)', 15000, 25000, 21000)
        # ขยะล้น: 0 หรือ 1 ตาม Dataset
        over = st.select_slider('⚠️ สถานะขยะล้น (0=ปกติ, 1=ล้น)', options=[0, 1], value=0)
        # อุณหภูมิ: ประมาณ 22 - 26 องศา
        temp = st.slider('🌡️ อุณหภูมิ (°C)', 20, 35, 24)
        # ปริมาณฝน: ประมาณ 0 - 20 มม.
        rain = st.slider('🌧️ ปริมาณฝน (mm)', 0, 50, 5)

    # 5. คำนวณผลลัพธ์
    input_val = np.array([[pop, recy, org, cap, over, temp, rain]])
    prediction = model.predict(input_val)[0]

    # 6. แสดงผล Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("ทำนายขยะรวม", f"{prediction:,.2f} kg")
    c2.metric("ประชากรในพื้นที่", f"{pop:,} คน")
    c3.metric("อุณหภูมิวันนี้", f"{temp} °C")

    # 7. กราฟ (Dynamic Scale ที่แคบลงเพื่อให้เห็นความต่างชัดขึ้น)
    st.write("---")
    st.subheader("📊 การเปรียบเทียบกับข้อมูลในฐานข้อมูล")
    
    # กำหนดขอบเขตขวานขวานให้พอดีกับข้อมูลจริง (ประมาณ 15k - 25k)
    min_chart = 15000
    max_chart = 25000
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 5))

    # ข้อมูลเดิมจาก Dataset
    ax.scatter(y_data, model.predict(df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]), 
               alpha=0.5, color='#475569', label='History Data (Riverside)')

    # เส้นแบ่ง Trend
    ax.plot([min_chart, max_chart], [min_chart, max_chart], '--', color='white', alpha=0.3)

    # จุดที่ผู้ใช้เลือก (สีแดง)
    ax.scatter(prediction, prediction, color='#FF6347', s=400, edgecolor='white', 
               linewidth=3, label='Your Setting', zorder=10)

    # เส้นประนำสายตา
    ax.axhline(prediction, color='#FF6347', linestyle=':', alpha=0.6)
    ax.axvline(prediction, color='#FF6347', linestyle=':', alpha=0.6)

    ax.set_xlim(min_chart, max_chart)
    ax.set_ylim(min_chart, max_chart)
    ax.set_xlabel('Actual Waste in Dataset (kg)')
    ax.set_ylabel('Predicted Waste (kg)')
    ax.legend()
    ax.grid(True, alpha=0.1)

    st.pyplot(fig)
    
    st.markdown('<p style="text-align:center; color:gray;">ระบบปรับแต่งช่วงข้อมูล (Range) ให้อัตโนมัติตามสถิติ Riverside Area</p>', unsafe_allow_html=True)

else:
    st.error("กรุณาอัปโหลดไฟล์ dataset เพื่อเริ่มใช้งาน")