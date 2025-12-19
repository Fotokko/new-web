import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ระบบคำนวณขยะ โดย วิว", layout="wide")

# 2. Trash Rain Effect
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

# 3. CSS ฉบับแก้ไขการมองเห็น (Visibility Fix)
st.markdown("""
    <style>
        /* พื้นหลังหน้าเว็บ */
        .stApp { background-color: #98FB98; } 
        
        /* Title ใหญ่ยักษ์ สีดำเข้ม */
        .ai-title {
            font-size: 110px;
            font-weight: 1000;
            color: #000000 !important;
            text-align: center;
            line-height: 1.0;
            margin-top: -40px;
            margin-bottom: 10px;
            text-shadow: 2px 2px 0px #FFFFFF; /* ใส่ขอบขาวให้ตัวหนังสือดำ */
        }
        
        /* Subtitle สีดำเข้ม */
        .ai-subtitle {
            font-size: 28px;
            color: #000000 !important;
            text-align: center;
            font-weight: bold;
            margin-bottom: 40px;
        }

        /* ปรับแต่ง Sidebar เขียวเข้ม ตัวหนังสือขาวหนา */
        [data-testid="stSidebar"] {
            background-color: #004d00 !important;
        }
        [data-testid="stSidebar"] .stMarkdown, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] p {
            color: #FFFFFF !important;
            font-size: 18px !important;
            font-weight: 900 !important;
        }

        /* Metric Cards: พื้นหลังดำทึบ ตัวเลขทอง (ให้อ่านง่ายที่สุด) */
        [data-testid="stMetricValue"] {
            color: #FFD700 !important; 
            font-size: 45px !important;
            font-weight: 900 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #FFFFFF !important;
            font-size: 20px !important;
            font-weight: bold !important;
        }
        div[data-testid="stMetric"] {
            background-color: #000000 !important; /* เปลี่ยนเป็นดำทึบ */
            border: 4px solid #006400 !important;
            border-radius: 20px;
            padding: 20px !important;
        }

        /* เส้นคั่น */
        hr { border: 2px solid #000000 !important; }
        
        /* หัวข้อกราฟ */
        h3 { color: #000000 !important; font-weight: 900 !important; font-size: 30px !important; }
    </style>
""", unsafe_allow_html=True)

# ส่วนหัว
st.markdown('<p class="ai-title">ระบบคำนวณขยะ<br>โดย วิว</p>', unsafe_allow_html=True)
st.markdown('<p class="ai-subtitle">วิเคราะห์ข้อมูลแม่นยำ 100%</p>', unsafe_allow_html=True)

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
    # 5. Sidebar
    st.sidebar.markdown("<h1 style='color:white; text-align:center;'>⚙️ SETTINGS</h1>", unsafe_allow_html=True)
    with st.sidebar:
        pop = st.slider('👥 ประชากร', 17000, 20000, 17950)
        recy = st.slider('♻️ รีไซเคิล (kg)', 1000, 15000, 5000)
        org = st.slider('🍎 อินทรีย์ (kg)', 5000, 20000, 8500)
        cap = st.slider('🚛 ความจุรถ (kg)', 15000, 30000, 21000)
        over = st.select_slider('⚠️ ขยะล้น', options=[0, 1], value=0)
        temp = st.slider('🌡️ อุณหภูมิ', 15, 40, 24)
        rain = st.slider('🌧️ ฝน (mm)', 0, 100, 5)

    # 6. คำนวณ
    input_val = np.array([[pop, recy, org, cap, over, temp, rain]])
    prediction = model.predict(input_val)[0]

    # 7. Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("พยากรณ์ขยะรวม", f"{prediction:,.2f} kg")
    col2.metric("จำนวนคน", f"{pop:,} คน")
    col3.metric("อุณหภูมิวันนี้", f"{temp} °C")

    # 8. กราฟ (เน้นเส้นหนาและตัวเลขชัด)
    st.write("---")
    st.subheader("📈 แผนภูมิวิเคราะห์ตำแหน่งข้อมูล")
    
    history_preds = model.predict(df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']])
    all_vals = np.concatenate([y_data, history_preds, [prediction]])
    t_min, t_max = all_vals.min(), all_vals.max()
    margin = (t_max - t_min) * 0.15
    
    plt.style.use('default') # ใช้สีสว่างสำหรับกราฟเพื่อให้เข้ากับเว็บ
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#98FB98')
    ax.set_facecolor('#FFFFFF')

    # จุดข้อมูลเก่า (สีเขียวขี้ม้า)
    ax.scatter(y_data, history_preds, alpha=0.4, color='#556B2F', s=50, label='ข้อมูลเดิม')
    
    # เส้นทแยงมุม
    ax.plot([t_min - margin, t_max + margin], [t_min - margin, t_max + margin], '--', color='red', lw=2)

    # จุดพยากรณ์ (สีส้มแดง ขอบดำ หนาๆ)
    ax.scatter(prediction, prediction, color='#FF4500', s=700, edgecolor='black', linewidth=4, zorder=10, label='จุดที่ทำนาย')
    
    ax.set_xlim(t_min - margin, t_max + margin)
    ax.set_ylim(t_min - margin, t_max + margin)
    ax.set_xlabel('Actual Value', fontsize=12, fontweight='bold', color='black')
    ax.set_ylabel('Predicted Value', fontsize=12, fontweight='bold', color='black')
    ax.legend(prop={'weight':'bold'})
    ax.grid(True, linestyle='-', alpha=0.2)

    st.pyplot(fig)

    st.markdown('<p style="text-align:center; color:#000000; font-weight:900; font-size:20px;">© 2024 AI View - Smart Waste Solution</p>', unsafe_allow_html=True)
else:
    st.error("ไม่พบข้อมูล CSV")