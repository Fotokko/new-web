import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. ตั้งค่าหน้าเว็บให้เป็นแบบกว้าง
st.set_page_config(page_title="Waste Prediction Pro", layout="wide")

# 2. ปรับแต่ง CSS ให้ Metric ชัดเจนและสวยงาม
st.markdown("""
    <style>
        /* ปรับแต่ง Metric Card */
        [data-testid="stMetricValue"] {
            color: #FFD700 !important; /* สีทอง */
            font-size: 38px !important;
            font-weight: bold;
        }
        [data-testid="stMetricLabel"] {
            color: #FFFFFF !important;
            font-size: 18px !important;
        }
        div[data-testid="stMetric"] {
            background-color: #1E293B !important;
            border: 2px solid #FF6347 !important;
            border-radius: 12px;
            padding: 15px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        /* หัวข้อหลัก */
        .main-title {
            font-size: 45px;
            font-weight: 800;
            color: #FF6347;
            text-align: center;
            margin-bottom: 5px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚮 Waste Prediction System (Riverside)</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#94A3B8;">ระบบวิเคราะห์และพยากรณ์ขยะแบบ Real-time ตามข้อมูลจริง</p>', unsafe_allow_html=True)

# 3. ฟังก์ชันโหลดข้อมูลและฝึกโมเดล
@st.cache_data
def load_and_train():
    file_path = 'sustainable_waste_management_dataset_2024.csv'
    try:
        df = pd.read_csv(file_path)
        # ระบุ Features ตาม Dataset จริง
        features = ['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']
        X = df[features]
        y = df['waste_kg']
        model = LinearRegression().fit(X, y)
        return model, df, y
    except Exception as e:
        return None, None, None

model, df, y_data = load_and_train()

if model is not None:
    # 4. Sidebar: ปรับช่วง (Range) ให้ใกล้เคียงข้อมูล Riverside จริงๆ
    st.sidebar.header("⚙️ ปรับแต่งสถานการณ์")
    with st.sidebar:
        st.write("---")
        pop = st.slider('👥 ประชากร (คน)', 17000, 20000, 17950)
        recy = st.slider('♻️ ขยะรีไซเคิล (kg)', 1000, 15000, 5000)
        org = st.slider('🍎 ขยะอินทรีย์ (kg)', 5000, 20000, 8500)
        cap = st.slider('🚛 ความจุการเก็บ (kg)', 15000, 30000, 21000)
        # ใช้ Select Slider สำหรับสถานะ 0 หรือ 1 เพื่อให้ใช้งานง่าย
        over = st.select_slider('⚠️ ขยะล้น (0=ปกติ, 1=ล้น)', options=[0, 1], value=0)
        temp = st.slider('🌡️ อุณหภูมิ (°C)', 15, 40, 24)
        rain = st.slider('🌧️ ปริมาณฝน (mm)', 0, 100, 5)

    # 5. คำนวณผลการทำนาย
    input_features = np.array([[pop, recy, org, cap, over, temp, rain]])
    prediction = model.predict(input_features)[0]

    # 6. แสดงผลลัพธ์ผ่าน Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("ปริมาณขยะทำนาย", f"{prediction:,.2f} kg")
    col2.metric("ประชากรเป้าหมาย", f"{pop:,} คน")
    col3.metric("สภาพอากาศ", f"{temp} °C")

    # 7. กราฟวิเคราะห์แบบ Dynamic Scaling (ขยับแกนตามจุด)
    st.write("---")
    st.subheader("📊 การวิเคราะห์ตำแหน่งข้อมูล (Dynamic Chart)")

    # คำนวณขอบเขตแกน XY ให้ครอบคลุมทุกจุดเสมอ (ไม่มีหลุดขอบ)
    history_preds = model.predict(df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']])
    
    # หาค่า Min/Max จากทั้งข้อมูลเก่าและค่าใหม่ที่เพิ่งทำนาย
    total_min = min(y_data.min(), prediction, history_preds.min())
    total_max = max(y_data.max(), prediction, history_preds.max())
    
    # เผื่อพื้นที่ว่าง (Padding) 15% เพื่อความสวยงาม
    margin = (total_max - total_min) * 0.15
    ax_limit_min = total_min - margin
    ax_limit_max = total_max + margin

    # ตั้งค่าสไตล์กราฟ
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0E1117') # สีพื้นหลังเดียวกับ Streamlit Dark Mode
    ax.set_facecolor('#1E293B')

    # วาดข้อมูลประวัติศาสตร์ (Past Data)
    ax.scatter(y_data, history_preds, alpha=0.4, color='#64748B', label='ข้อมูลในอดีต (Riverside)')

    # วาดเส้นทแยงมุม Baseline (Perfect Prediction Line)
    ax.plot([ax_limit_min, ax_limit_max], [ax_limit_min, ax_limit_max], '--', color='white', alpha=0.2)

    # วาดจุดปัจจุบันที่ได้จาก Slider (สีส้มแดงสด ขนาดใหญ่)
    ax.scatter(prediction, prediction, color='#FF6347', s=500, edgecolor='white', 
               linewidth=3, label='ค่าที่คุณปรับแต่งตอนนี้', zorder=10)

    # เส้นประนำสายตาชี้ไปที่แกนเลข
    ax.axhline(prediction, color='#FF6347', linestyle=':', alpha=0.5)
    ax.axvline(prediction, color='#FF6347', linestyle=':', alpha=0.5)

    # ตั้งค่า Limit แกน XY ให้ขยับตามจุดทำนายเสมอ
    ax.set_xlim(ax_limit_min, ax_limit_max)
    ax.set_ylim(ax_limit_min, ax_limit_max)

    # ชื่อแกนและรายละเอียด
    ax.set_xlabel('Actual Waste in History (kg)', fontsize=12, color='#94A3B8')
    ax.set_ylabel('Predicted Waste (kg)', fontsize=12, color='#94A3B8')
    ax.legend(facecolor='#1E293B', edgecolor='white')
    ax.grid(True, linestyle='--', alpha=0.1)

    st.pyplot(fig)

    # 8. ส่วนสรุปด้านล่าง
    st.info(f"💡 วิเคราะห์: หากมีประชากร {pop:,} คน และมีปริมาณฝน {rain} mm ระบบคาดการณ์ว่าขยะจะอยู่ที่ประมาณ {prediction:,.2f} kg")
    st.markdown('<p style="text-align:center; color:#475569; font-size:14px; margin-top:30px;">Developed by ไอไก่วิว | © 2024 Data Science Project</p>', unsafe_allow_html=True)

else:
    st.error("❌ ไม่พบไฟล์ 'sustainable_waste_management_dataset_2024.csv' กรุณาตรวจสอบตำแหน่งไฟล์")