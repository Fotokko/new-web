import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI วิว - ระบบคำนวณขยะ", layout="wide")

# 2. เอฟเฟกต์ถังขยะร่วงหล่น (Trash Rain Effect)
st.markdown("""
<script>
const canvas = window.parent.document.createElement('canvas');
canvas.id = 'trash-rain';
canvas.style.position = 'fixed';
canvas.style.top = '0';
canvas.style.left = '0';
canvas.style.width = '100vw';
canvas.style.height = '100vh';
canvas.style.pointerEvents = 'none';
canvas.style.zIndex = '0';
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
        if (p.y > canvas.height) {
            p.y = -30;
            p.x = Math.random() * canvas.width;
        }
    });
    requestAnimationFrame(draw);
}
draw();
</script>
""", unsafe_allow_html=True)

# 3. ปรับแต่ง CSS (ขยาย Title ให้ใหญ่ขึ้นมาก)
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; }
        
        /* แก้ไขตรงนี้: ขยายขนาด Title และเพิ่ม Effect */
        .ai-title {
            font-size: 80px; /* ขนาดใหญ่พิเศษ */
            font-weight: 900;
            color: #FF6347;
            text-align: center;
            margin-top: -20px;
            margin-bottom: 0px;
            text-transform: uppercase;
            letter-spacing: 5px;
            background: -webkit-linear-gradient(#FF6347, #FFD700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 0 15px rgba(255, 99, 71, 0.5));
        }
        
        .ai-subtitle {
            font-size: 26px; /* ขยาย Subtitle ด้วย */
            color: #94A3B8;
            text-align: center;
            margin-bottom: 40px;
            font-style: italic;
        }

        [data-testid="stMetricValue"] {
            color: #FFD700 !important;
            font-size: 40px !important;
            font-weight: bold;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(30, 41, 59, 0.7) !important;
            border: 2px solid #FF6347 !important;
            border-radius: 15px;
            padding: 20px !important;
            backdrop-filter: blur(5px);
        }
    </style>
""", unsafe_allow_html=True)

# ส่วนหัวของเว็บไซต์ (Title ที่ใหญ่ขึ้น)
st.markdown('<p class="ai-title">🤖 AI วิว</p>', unsafe_allow_html=True)
st.markdown('<p class="ai-subtitle">ระบบพยากรณ์ปริมาณขยะอัจฉริยะ</p>', unsafe_allow_html=True)

# 4. โหลดข้อมูลและสร้างโมเดล (ใช้ข้อมูลจากไฟล์ CSV ของคุณ)
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
    # 5. Sidebar
    st.sidebar.header("🛠️ แผงควบคุม AI วิว")
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

    # 7. Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("AI คาดการณ์ขยะ", f"{prediction:,.2f} kg")
    col2.metric("ประชากรในพื้นที่", f"{pop:,} คน")
    col3.metric("อุณหภูมิเฉลี่ย", f"{temp} °C")

    # 8. กราฟ Dynamic Scaling
    st.write("---")
    st.subheader("📈 วิเคราะห์แนวโน้มข้อมูล")

    history_preds = model.predict(df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']])
    all_vals = np.concatenate([y_data, history_preds, [prediction]])
    t_min, t_max = all_vals.min(), all_vals.max()
    margin = (t_max - t_min) * 0.15
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1E293B')
    ax.scatter(y_data, history_preds, alpha=0.3, color='#94A3B8', label='ประวัติข้อมูล')
    ax.plot([t_min - margin, t_max + margin], [t_min - margin, t_max + margin], '--', color='white', alpha=0.1)
    ax.scatter(prediction, prediction, color='#FF6347', s=500, edgecolor='white', linewidth=3, zorder=10, label='จุดที่ AI ทำนาย')
    ax.axhline(prediction, color='#FF6347', linestyle=':', alpha=0.5)
    ax.axvline(prediction, color='#FF6347', linestyle=':', alpha=0.5)
    ax.set_xlim(t_min - margin, t_max + margin)
    ax.set_ylim(t_min - margin, t_max + margin)
    ax.set_xlabel('Actual Value')
    ax.set_ylabel('AI Predicted Value')
    ax.legend()
    st.pyplot(fig)

    st.markdown('<p style="text-align:center; color:#475569; margin-top:50px;">© 2024 AI วิว - Smart Waste Prediction</p>', unsafe_allow_html=True)
else:
    st.error("ไม่พบข้อมูล CSV")