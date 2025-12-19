import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Waste Prediction", layout="wide")

# 2. ปรับปรุง CSS เพื่อให้ Metric และ Slider อ่านง่ายในทุกโหมดสี
st.markdown("""
    <style>
        /* บังคับให้ตัวเลขใน Metric เป็นสีเหลืองทองและพื้นหลังเข้มเพื่อให้ตัดกับหน้าจอ */
        [data-testid="stMetricValue"] {
            color: #FFD700 !important;
            font-size: 35px !important;
        }
        [data-testid="stMetricLabel"] {
            color: #FFFFFF !important;
            font-size: 18px !important;
        }
        div[data-testid="stMetric"] {
            background-color: #1E293B !important;
            border: 2px solid #FF6347 !important;
            border-radius: 10px;
            padding: 10px !important;
        }
        .title-text {
            font-size: 40px;
            font-weight: bold;
            color: #FF6347;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">🚮 Waste Prediction System (Real-time)</p>', unsafe_allow_html=True)

# 3. โหลดข้อมูล
@st.cache_data
def load_data():
    file_path = 'sustainable_waste_management_dataset_2024.csv'
    try:
        df = pd.read_csv(file_path)
        features = ['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']
        X = df[features]
        y = df['waste_kg']
        model = LinearRegression().fit(X, y)
        return model, df, y
    except: return None, None, None

model, df, y_data = load_data()

if model is not None:
    # 4. Sidebar ปรับปรุงสัญลักษณ์ให้น่าใช้
    st.sidebar.header("⚙️ Adjust Factors")
    with st.sidebar:
        pop = st.slider('Population (👥)', 1000, 150000, 74765)
        recy = st.slider('Recyclable (♻️)', 0, 100000, 50000)
        org = st.slider('Organic (🍎)', 0, 100000, 41667)
        cap = st.slider('Capacity (🚛)', 0, 50000, 5000)
        over = st.slider('Overflow (⚠️)', 0, 20000, 500)
        temp = st.slider('Temperature (🌡️)', -10, 50, 25)
        rain = st.slider('Rain (🌧️)', 0, 1000, 100)

    # 5. คำนวณผลลัพธ์
    input_val = np.array([[pop, recy, org, cap, over, temp, rain]])
    prediction = model.predict(input_val)[0]

    # 6. แสดงผล Metrics (เน้นความชัดเจน)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Waste", f"{prediction:,.2f} kg")
    c2.metric("Population Size", f"{pop:,} People")
    c3.metric("Temp", f"{temp} °C")

    # 7. กราฟ (แก้ปัญหาภาษาต่างดาวและ Scale)
    st.write("---")
    st.subheader("📊 Visual Analytics")
    
    # คำนวณขอบเขตให้ขยับตาม Slider
    max_val = max(y_data.max(), prediction) * 1.2
    
    # ใช้สไตล์ Dark สำหรับกราฟเพื่อให้เข้ากับหน้าจอคุณ
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 5))

    # วาดข้อมูลเก่า
    ax.scatter(y_data, model.predict(df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]), 
               alpha=0.2, color='#475569', label='Past Data')

    # เส้นแบ่ง 45 องศา
    ax.plot([0, max_val], [0, max_val], '--', color='white', alpha=0.3)

    # จุดทำนายปัจจุบัน (สีแดงขอบขาว)
    ax.scatter(prediction, prediction, color='#FF6347', s=350, edgecolor='white', 
               linewidth=2, label='Current Prediction', zorder=10)

    # เส้นประนำสายตา
    ax.axhline(prediction, color='#FF6347', linestyle=':', alpha=0.4)
    ax.axvline(prediction, color='#FF6347', linestyle=':', alpha=0.4)

    # ตั้งชื่อแกนเป็นภาษาอังกฤษเพื่อเลี่ยงปัญหา Font ภาษาไทย
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('Actual Waste (kg)', fontweight='bold')
    ax.set_ylabel('Predicted Waste (kg)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.1)

    st.pyplot(fig)
    
    st.info("💡 Tip: ค่าพยากรณ์ของคุณตอนนี้สูงกว่าข้อมูลในอดีต (กราฟจึงขยายตามอัตโนมัติ)")

else:
    st.error("ไม่พบไฟล์ .csv กรุณาอัปโหลดไฟล์ในโฟลเดอร์เดียวกัน")