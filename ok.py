import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# สร้าง CSS สำหรับการตกแต่งหน้าเว็บ
st.markdown("""
    <style>
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #2F4F4F;
        }
        .subtitle {
            font-size: 32px;
            color: #4B0082;
        }
        .main-container {
            background-color: #F5F5F5;
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            font-size: 24px;
            font-weight: bold;
            color: #FF6347;
        }
        .description {
            font-size: 18px;
            color: #4682B4;
        }
    </style>
    """, unsafe_allow_html=True)

# ชื่อหัวข้อ
st.markdown('<p class="title">Made by ไอไก่วิว</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ระบบปริมาณทํานายขยะ</p>', unsafe_allow_html=True)

# ข้อมูลตัวอย่าง
file_path = 'sustainable_waste_management_dataset_2024.csv' 
df = pd.read_csv(file_path)

# แสดงข้อมูล
st.markdown('<p class="header">Dataset Preview:</p>', unsafe_allow_html=True)
st.write(df.head())  # แสดง 5 แถวแรก

# เตรียมข้อมูลสำหรับการฝึกโมเดล
X = df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]  
y = df['waste_kg']

# แบ่งข้อมูลเป็นชุดฝึกและทดสอบ
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Linear Regression
model = LinearRegression()
model.fit(X_train, Y_train)

# ทำนายข้อมูล
Y_pred = model.predict(X_test)

# คำนวณ MSE และ R squared
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# แสดงผล
st.markdown('<p class="description">Model Evaluation:</p>', unsafe_allow_html=True)
st.write(f"MSE: {mse}")
st.write(f"R squared: {r2}")

# สร้างกราฟ
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual waste_kg (Y_test)')
plt.ylabel('Predicted waste_kg (Y_pred)')
plt.title('Predicted vs. Actual waste_kg')
plt.legend()
plt.grid(True)
st.pyplot(plt)

