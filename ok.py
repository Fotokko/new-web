import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. ตั้งค่าหน้าเว็บให้ดูเป็นมืออาชีพ
st.set_page_config(page_title="ระบบทำนายปริมาณขยะ", layout="wide")

# 2. ปรับแต่ง CSS ให้สวยงามและอ่านง่าย
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        .stMetric {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #FF6347;
        }
        .title-text {
            font-size: 45px;
            font-weight: 800;
            color: #2E4053;
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle-text {
            font-size: 20px;
            color: #5D6D7E;
            text-align: center;
            margin-bottom: 30px;
        }
        div[data-testid="stSidebarUserContent"] {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# 3. ส่วนหัวของเว็บ
st.markdown('<p class="title-text">🚮 ระบบพยากรณ์ปริมาณขยะอัจฉริยะ</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Made by ไอไก่วิว • วิเคราะห์แม่นยำด้วย Machine Learning</p>', unsafe_allow_html=True)

# 4. โหลดข้อมูลและฝึกโมเดล (ทำแค่ครั้งเดียวเพื่อความเร็ว)
@st.cache_data
def train_model():
    file_path = 'sustainable_waste_management_dataset_2024.csv'
    try:
        df = pd.read_csv(file_path)
        X = df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]
        y = df['waste_kg']
        model = LinearRegression()
        model.fit(X, y)
        return model, df, y
    except:
        return None, None, None

model, df, y_data = train_model()

if model is not None:
    # 5. ส่วน Sidebar สำหรับรับค่า (ทำให้ใช้งานง่าย)
    st.sidebar.header("⚙️ ปรับแต่งปัจจัย")
    st.sidebar.write("ลาก Slider เพื่อทำนายผลทันที")
    
    with st.sidebar:
        population = st.slider('👥 จำนวนประชากร', 1000, 100000, 17990)
        recyclable_kg = st.slider('♻️ ขยะรีไซเคิล (kg)', 0, 50000, 5000)
        organic_kg = st.slider('🍎 ขยะอินทรีย์ (kg)', 0, 50000, 5000)
        collection_cap = st.slider('🚛 ความจุการเก็บ (kg)', 0, 50000, 5000)
        overflow = st.slider('⚠️ ปริมาณขยะล้น (kg)', 0, 10000, 500)
        temp_c = st.slider('🌡️ อุณหภูมิ (°C)', -10, 50, 25)
        rain_mm = st.slider('🌧️ ปริมาณฝน (mm)', 0, 1000, 100)

    # 6. คำนวณผลการทำนาย
    input_data = np.array([[population, recyclable_kg, organic_kg, collection_cap, overflow, temp_c, rain_mm]])
    prediction = model.predict(input_data)[0]

    # 7. แสดงผลลัพธ์แบบการ์ด (Metrics)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="📊 ผลการทำนายปริมาณขยะ", value=f"{prediction:,.2f} kg")
    with c2:
        st.metric(label="👥 ประชากรที่ระบุ", value=f"{population:,} คน")
    with c3:
        st.metric(label="🌡️ อุณหภูมิเฉลี่ย", value=f"{temp_c} °C")

    st.write("---")

    # 8. ส่วนของกราฟแบบ Real-time และขยับตามค่า (Dynamic Scaling)
    col_graph, col_info = st.columns([2, 1])

    with col_graph:
        st.subheader("📈 กราฟวิเคราะห์แนวโน้ม")
        
        # ตั้งค่าขอบเขตกราฟให้ขยับตามค่าที่ทำนายเสมอ (สำคัญ!)
        max_limit = max(y_data.max(), prediction) * 1.2
        min_limit = min(y_data.min(), prediction) * 0.8
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # วาดข้อมูลเก่า (สีเทา)
        ax.scatter(y_data, model.predict(df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]), 
                   alpha=0.3, color='#BDC3C7', label='ข้อมูลในอดีต')
        
        # วาดเส้น Baseline
        ax.plot([min_limit, max_limit], [min_limit, max_limit], '--', color='#E74C3C', alpha=0.5)
        
        # วาดจุดปัจจุบันที่ได้จากการลาก Slider (สีส้มขนาดใหญ่)
        ax.scatter(prediction, prediction, color='#FF6347', s=300, edgecolor='white', 
                   linewidth=2, label='ผลการทำนายของคุณ', zorder=5)
        
        # เส้นประนำสายตา
        ax.axhline(prediction, color='#FF6347', linestyle=':', alpha=0.5)
        ax.axvline(prediction, color='#FF6347', linestyle=':', alpha=0.5)

        # ปรับ Scale แกน x และ y ให้ขยับตามค่าทำนาย
        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)
        
        ax.set_xlabel('ค่าจริง (จากฐานข้อมูล)')
        ax.set_ylabel('ค่าที่ทำนายได้')
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        st.pyplot(fig)

    with col_info:
        st.subheader("💡 คำแนะนำ")
        if prediction > 300000:
            st.warning("⚠️ ปริมาณขยะสูงมาก! ควรเพิ่มรอบการจัดเก็บหรือขยายจุดรีไซเคิล")
        elif prediction > 150000:
            st.info("ℹ️ ปริมาณขยะอยู่ในเกณฑ์ปกติ แต่ควรเฝ้าระวังช่วงฝนตกหนัก")
        else:
            st.success("✅ ปริมาณขยะอยู่ในเกณฑ์ที่จัดการได้ดี")
            
        st.write("---")
        st.write("**สรุปค่าที่เลือก:**")
        st.write(f"- ประชากร: {population:,}")
        st.write(f"- ขยะอินทรีย์: {organic_kg:,} kg")
        st.write(f"- ขยะล้น: {overflow:,} kg")

    # 9. ฟุตเตอร์
    st.markdown(f'<p style="text-align:center; color:#95A5A6; font-size:12px; margin-top:50px;">© 2024 พัฒนาโดย ไอไก่วิว | ระบบอัปเดตอัตโนมัติแบบ Real-time</p>', unsafe_allow_html=True)

else:
    st.error("❌ ไม่พบไฟล์ข้อมูล! กรุณาตรวจสอบว่ามีไฟล์ 'sustainable_waste_management_dataset_2024.csv' อยู่ในโฟลเดอร์เดียวกัน")