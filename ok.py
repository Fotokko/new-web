import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'sustainable_waste_management_dataset_2024.csv' 

df = pd.read_csv(file_path)

st.title("ระบบปริมาณทํานายขยะ")

st.write("Dataset Preview:")
st.write(df.head()) 

X = df[['population', 'recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'overflow', 'temp_c', 'rain_mm']]  
y = df['waste_kg']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

st.write(f"MSE: {mse}")
st.write(f"R squared: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual waste_kg (Y_test)')
plt.ylabel('Predicted waste_kg (Y_pred)')
plt.title('Predicted vs. Actual waste_kg')
plt.legend()
plt.grid(True)
st.pyplot(plt)

