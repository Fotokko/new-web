import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# เพิ่ม CSS สำหรับการตกแต่งพื้นหลังและองค์ประกอบต่างๆ
st.markdown("""
    <style>
        body {
            background-color: #FF6347;  /* สีแดงสด (Tomato) */
            font-family: 'Arial', sans-serif;
            color: white;  /* ให้ตัวหนังสือเป็นสีขาว */
        }
        .title {
            font-size: 50px;
            font-weight: bold;
            color: #FFFFFF;  /* สีขาว */
            text-align: center;
            margin-top: 30px;
        }
        .subtitle {
            font-size: 32px;
            color: #FFFACD;  /* สีครีมอ่อน (Lemon Chiffon) */
            text-align: center;
            margin-top: 10px;
        }
        .header {
            font-size: 24px;
            font-weight: bold;
            color: #FFD700;  /* สีทอง (Gold) */
        }
        .description {
            font-size: 18px;
            color: #FFFFFF;  /* สีขาว */
        }
        .container {
            background-color: #FFFFFF;  /* พื้นหลังของกล่องข้อมูลเป็นสีขาว */
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin-top: 20px;
        }
        .content {
            text-align: center;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            color: #808080;  /* สีเทาอ่อน */
        }
        .slider {
            margin-top: 20px;
        }
        .call-to-action {
            font-size: 18px;
            font-weight: bold;
            color: #32CD32;  /* สีเขียว */
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ชื่อหัวข้อ
st.markdown('<p class="title">Made by ไอไก่วิว</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ระบบปริมาณทํานายขยะ</p>', unsafe_allow_html=True)

# ข้อมูลตัวอย่าง
file_path = 'sustainable_waste_management_dataset_2024.csv' 
df = pd.read_csv(file_path)

# เตรียมข้อมูลสำหรับการฝึกโมเดล
X = df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]  
y = df['waste_kg']

# สร้างโมเดล Linear Regression
model = LinearRegression()
model.fit(X, y)

# การตั้งค่าของ Slider
st.markdown('<p class="header">ปรับค่าปัจจัยต่างๆ และทำนายปริมาณขยะ</p>', unsafe_allow_html=True)

population = st.slider('เลือกจำนวนประชากร (population)', min_value=1000, max_value=50000, value=17990, step=100)
recyclable_kg = st.slider('เลือกจำนวนขยะรีไซเคิล (recyclable_kg)', min_value=1000, max_value=10000, value=5000, step=100)
organic_kg = st.slider('เลือกจำนวนขยะอินทรีย์ (organic_kg)', min_value=1000, max_value=10000, value=5000, step=100)
collection_capacity_kg = st.slider('เลือกความสามารถในการเก็บขยะ (collection_capacity_kg)', min_value=1000, max_value=10000, value=5000, step=100)
overflow = st.slider('เลือกปริมาณขยะที่ล้น (overflow)', min_value=100, max_value=2000, value=500, step=50)
temp_c = st.slider('เลือกอุณหภูมิ (temp_c)', min_value=-10, max_value=40, value=25, step=1)
rain_mm = st.slider('เลือกปริมาณฝน (rain_mm)', min_value=0, max_value=500, value=100, step=10)

# นำค่าที่ได้จาก Slider ไปใช้ในการทำนาย
input_data = np.array([[population, recyclable_kg, organic_kg, collection_capacity_kg, overflow, temp_c, rain_mm]])

# ทำนายผลลัพธ์
prediction = model.predict(input_data)

# แสดงผลการทำนาย
st.markdown('<p class="call-to-action">ผลลัพธ์การทำนายขยะ:</p>', unsafe_allow_html=True)
st.write(f"การทำนายจำนวนขยะ (waste_kg): {prediction[0]:.2f} กิโลกรัม")

# สร้างกราฟแสดงผล
plt.figure(figsize=(10, 6))
plt.scatter(y, model.predict(X), color='blue', label="Data points")
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual waste_kg')
plt.ylabel('Predicted waste_kg')
plt.title('Predicted vs Actual waste_kg')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# ฟุตเตอร์ (footer)
st.markdown('<p class="footer">Developed by ไอไก่วิว - © 2024</p>', unsafe_allow_html=True)
