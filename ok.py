import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Waste Prediction", layout="wide")

# 2. ปรับปรุง CSS (เน้นความชัดเจนของ Font และ Border)
st.markdown("""
    <style>
        /* พื้นหลังหน้าเว็บเป็นสีสว่างเพื่อให้ตัวหนังสือสีเข้มเด่นขึ้น */
        .main { background-color: #F8F9FA; }
        
        /* การ์ดแสดงผล (Metric) */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 2px solid #1E3A8A !important; /* ขอบสีน้ำเงินเข้ม ชัดเจน */
            border-radius: 15px;
            padding: 15px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* ปรับสี Label ของ Metric ให้เป็นสีดำเข้ม */
        [data-testid="stMetricLabel"] {
            color: #111827 !important;
            font-size: 18px !important;
            font-weight: bold !important;
        }

        /* หัวข้อหลัก */
        .title-text {
            font-size: 42px;
            font-weight: 900;
            color: #1E3A8A; /* สีน้ำเงินเข้ม */
            text-align: center;
            border-bottom: 4px solid #FF6347; /* ขีดเส้นใต้สีส้มแดง */
            padding-bottom: 10px;
            margin-bottom: 25px;
        }

        /* ปรับแต่ง Sidebar ให้ตัวหนังสือชัด */
        .css-1d391kg { background-color: #FFFFFF; }
    </style>
""", unsafe_allow_html=True)

# 3. หัวข้อ
st.markdown('<p class="title-text">🚮 ระบบพยากรณ์ปริมาณขยะ (Real-time)</p>', unsafe_allow_html=True)

# 4. โหลดข้อมูล
@st.cache_data
def load_data():
    file_path = 'sustainable_waste_management_dataset_2024.csv'
    try:
        df = pd.read_csv(file_path)
        X = df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]
        y = df['waste_kg']
        model = LinearRegression().fit(X, y)
        return model, df, y
    except: return None, None, None

model, df, y_data = load_data()

if model:
    # 5. Sidebar (ใช้โทนสีเข้มเพื่อให้เห็นชัด)
    st.sidebar.header("🎨 ปรับค่าปัจจัย")
    with st.sidebar:
        pop = st.slider('👥 ประชากร', 1000, 100000, 17990)
        recy = st.slider('♻️ ขยะรีไซเคิล', 0, 50000, 5000)
        org = st.slider('🍎 ขยะอินทรีย์', 0, 50000, 5000)
        cap = st.slider('🚛 ความจุการเก็บ', 0, 50000, 5000)
        over = st.slider('⚠️ ขยะล้น', 0, 10000, 500)
        temp = st.slider('🌡️ อุณหภูมิ', -10, 50, 25)
        rain = st.slider('🌧️ ปริมาณฝน', 0, 1000, 100)

    # 6. คำนวณ
    input_val = np.array([[pop, recy, org, cap, over, temp, rain]])
    prediction = model.predict(input_val)[0]

    # 7. ผลลัพธ์ (แก้ปัญหา Font กลืนกับ Border)
    c1, c2, c3 = st.columns(3)
    c1.metric("ปริมาณขยะที่ทำนาย", f"{prediction:,.2f} kg")
    c2.metric("สถานะประชากร", f"{pop:,} คน")
    c3.metric("สภาพอากาศ", f"{temp} °C")

    # 8. กราฟ (เน้นเส้นขอบและสีตัดกัน)
    st.write("---")
    st.subheader("📊 วิเคราะห์แผนภูมิขยะ")
    
    # คำนวณ Scale ให้ขยับตาม (Dynamic Scaling)
    max_val = max(y_data.max(), prediction) * 1.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#F8F9FA') # พื้นหลังนอกกราฟ
    ax.set_facecolor('#FFFFFF')      # พื้นหลังในกราฟ

    # ข้อมูลเดิม (สีเทาจาง)
    ax.scatter(y_data, model.predict(df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]), 
               alpha=0.2, color='#94A3B8', label='ข้อมูลในอดีต')

    # เส้นแบ่ง 45 องศา
    ax.plot([0, max_val], [0, max_val], '--', color='#64748B', lw=1)

    # จุดทำนายปัจจุบัน (สีแดงขอบดำ - เด่นที่สุด)
    ax.scatter(prediction, prediction, color='#EF4444', s=400, edgecolor='black', 
               linewidth=3, label='จุดที่คุณเลือก', zorder=10)

    # เส้นประนำสายตา (สีน้ำเงินเข้ม)
    ax.axhline(prediction, color='#1E3A8A', linestyle=':', alpha=0.5)
    ax.axvline(prediction, color='#1E3A8A', linestyle=':', alpha=0.5)

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('ค่าจริง (kg)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ค่าพยากรณ์ (kg)', fontsize=12, fontweight='bold')
    ax.legend(prop={'weight':'bold'})
    ax.grid(True, linestyle='--', alpha=0.3)

    st.pyplot(fig)

    st.markdown(f'<p style="text-align:center; color:#475569; font-weight:bold;">Developed by ไอไก่วิว • อัปเดตล่าสุดปี 2024</p>', unsafe_allow_html=True)
else:
    st.error("ไม่พบไฟล์ข้อมูล CSV ในโฟลเดอร์")