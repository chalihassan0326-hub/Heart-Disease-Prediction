# Heart-Disease-Prediction
A machine learning based web application that predicts the risk of heart disease using patient health parameters.
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Sample Dataset
data = {
    'age': [45, 50, 35, 60, 40, 55, 30, 65],
    'bp': [130, 140, 120, 150, 125, 145, 118, 160],
    'cholesterol': [220, 250, 180, 280, 190, 260, 170, 300],
    'disease': [1, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['age', 'bp', 'cholesterol']]
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.title("Heart Disease Prediction System")

age = st.number_input("Age", 1, 100, 30)
bp = st.number_input("Blood Pressure", 50, 250, 120)
cholesterol = st.number_input("Cholesterol", 100, 500, 200)

if st.button("Predict"):
    prediction = model.predict([[age, bp, cholesterol]])

    if prediction[0] == 1:
        st.error("Heart Disease Risk Detected")
    else:
        st.success("No Significant Heart Disease Risk")