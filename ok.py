import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ระบบคำนวณขยะ โดย วิว", layout="wide")

# 2. เอฟเฟกต์ถังขยะร่วงหล่น
st.markdown("""
<script>
const canvas = window.parent.document.createElement('canvas');
canvas.id = 'trash-rain';
canvas.style.position = 'fixed';
canvas.style.top = '0'; canvas.style.left = '0';
canvas.style.width = '100vw'; canvas.style.height = '100vh';
canvas.style.pointerEvents = 'none'; canvas.style.zIndex = '0';
window.parent.document.body.appendChild(canvas);
const ctx = canvas.getContext('2d');
canvas.width = window.parent.innerWidth;
canvas.height = window.parent.innerHeight;
const trashIcons = ['🗑️', '♻️', '📦', '🍎', '🧴', '🦴'];
const particles = [];
for (let i = 0; i < 20; i++) {
    particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 20 + 10,
        speed: Math.random() * 2 + 0.5,
        text: trashIcons[Math.floor(Math.random() * trashIcons.length)]
    });
}
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '24px serif';
    particles.forEach(p => {
        ctx.fillText(p.text, p.x, p.y);
        p.y += p.speed;
        if (p.y > canvas.height) { p.y = -30; p.x = Math.random() * canvas.width; }
    });
    requestAnimationFrame(draw);
}
draw();
</script>
""", unsafe_allow_html=True)

# 3. ปรับแต่ง CSS (Title ใหญ่ขึ้น และ Sidebar สีเขียวเข้ม)
st.markdown("""
    <style>
        /* พื้นหลังหน้าเว็บ */
        .stApp { background-color: #98FB98; } 
        
        /* ขยาย Title ให้ใหญ่สะใจ */
        .ai-title {
            font-size: 120px; /* ใหญ่ขึ้นเป็น 120px */
            font-weight: 1000;
            color: #000000;
            text-align: center;
            line-height: 1.1;
            margin-top: -50px;
            margin-bottom: 10px;
            text-shadow: 4px 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        /* ปรับแต่ง Sidebar (แผงควบคุม) ให้เป็นสีเขียวเข้ม */
        [data-testid="stSidebar"] {
            background-color: #004d00 !important; /* เขียวเข้มมาก */
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
            color: #FFFFFF !important; /* ตัวหนังสือใน Sidebar เป็นสีขาว */
            font-weight: bold;
        }

        /* ตกแต่ง Metric */
        [data-testid="stMetricValue"] {
            color: #FFD700 !important; 
            font-size: 45px !important;
            font-weight: bold;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(0, 50, 0, 0.8) !important; /* พื้นหลัง Metric เขียวเข้มโปร่งแสง */
            border: 3px solid #FFD700 !important;
            border-radius: 20px;
            padding: 20px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ส่วนหัว
st.markdown('<p class="ai-title">ระบบคำนวณขยะ<br>โดย วิว</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#1B4D3E; font-size:24px; font-weight:bold;">วิเคราะห์ข้อมูลอัจฉริยะ แม่นยำ และรวดเร็ว</p>', unsafe_allow_html=True)

# 4. โหลดข้อมูล
@st.cache_data
def load_and_train():
    file_path = 'sustainable_waste_management_dataset_2024.csv'
    try:
        df = pd.read_csv(file_path)
        features = ['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']
        X = df[features]
        y = df['waste_kg']
        model = LinearRegression().fit(X, y)
        return model, df, y
    except: return None, None, None

model, df, y_data = load_and_train()

if model is not None:
    # 5. Sidebar (แผงควบคุม)
    st.sidebar.markdown("<h2 style='color:white; text-align:center;'>🛠️ แผงควบคุม</h2>", unsafe_allow_html=True)
    with st.sidebar:
        pop = st.slider('👥 จำนวนประชากร', 17000, 20000, 17950)
        recy = st.slider('♻️ ขยะรีไซเคิล (kg)', 1000, 15000, 5000)
        org = st.slider('🍎 ขยะอินทรีย์ (kg)', 5000, 20000, 8500)
        cap = st.slider('🚛 ความจุการเก็บ (kg)', 15000, 30000, 21000)
        over = st.select_slider('⚠️ ขยะล้น (0/1)', options=[0, 1], value=0)
        temp = st.slider('🌡️ อุณหภูมิ (°C)', 15, 40, 24)
        rain = st.slider('🌧️ ปริมาณฝน (mm)', 0, 100, 5)

    # 6. คำนวณ
    input_val = np.array([[pop, recy, org, cap, over, temp, rain]])
    prediction = model.predict(input_val)[0]

    # 7. Metric Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("พยากรณ์ขยะ", f"{prediction:,.2f} kg")
    col2.metric("ประชากร", f"{pop:,} คน")
    col3.metric("อากาศ", f"{temp} °C")

    # 8. กราฟ
    st.write("---")
    history_preds = model.predict(df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']])
    all_vals = np.concatenate([y_data, history_preds, [prediction]])
    t_min, t_max = all_vals.min(), all_vals.max()
    margin = (t_max - t_min) * 0.15
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#002200') # กราฟพื้นหลังเขียวมืด
    ax.set_facecolor('#003300')
    ax.scatter(y_data, history_preds, alpha=0.3, color='#90EE90', label='History Data')
    ax.plot([t_min - margin, t_max + margin], [t_min - margin, t_max + margin], '--', color='white', alpha=0.1)
    ax.scatter(prediction, prediction, color='#FF4500', s=600, edgecolor='white', linewidth=3, zorder=10, label='AI Prediction')
    ax.axhline(prediction, color='#FF4500', linestyle=':', alpha=0.5)
    ax.axvline(prediction, color='#FF4500', linestyle=':', alpha=0.5)
    ax.set_xlim(t_min - margin, t_max + margin)
    ax.set_ylim(t_min - margin, t_max + margin)
    ax.legend()
    st.pyplot(fig)

    st.markdown('<p style="text-align:center; color:#004d00; font-weight:bold;">© 2024 AI View - Smart Waste Solution</p>', unsafe_allow_html=True)
else:
    st.error("ไฟล์ข้อมูลไม่ถูกต้อง")