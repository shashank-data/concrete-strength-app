import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load models
pipeline = joblib.load("num_pipeline.pkl")
model = joblib.load("ConcreteStrengthPrediction.pkl")

# UI Title
st.title("🏗️ Concrete Strength Predictor")
st.write("Enter the mix ingredients to estimate compressive strength (MPa).")

# Sidebar / Form for inputs
cement = st.number_input("Cement (kg/m³)", value=250.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", value=0.0)
flyash = st.number_input("Fly Ash (kg/m³)", value=0.0)
water = st.number_input("Water (kg/m³)", value=185.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", value=0.0)
coarseaggregate = st.number_input("Coarse Aggregate (kg/m³)", value=1100.0)
fineaggregate = st.number_input("Fine Aggregate (kg/m³)", value=780.0)
age = st.slider("Age (Days)", min_value=1, max_value=365, value=28)

if st.button("Predict Strength"):
    # Feature Engineering
    input_df = pd.DataFrame([{
        'cement': cement, 'slag': slag, 'flyash': flyash, 'water': water,
        'superplasticizer': superplasticizer, 'coarseaggregate': coarseaggregate,
        'fineaggregate': fineaggregate, 'age': age
    }])
    
    input_df['w/c'] = input_df['water'] / input_df['cement']
    input_df['total_binder'] = input_df['cement'] + input_df['slag'] + input_df['flyash']
    input_df['log(age)'] = np.log(input_df['age'])
    
    # Column ordering
    cols = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 
            'coarseaggregate', 'fineaggregate', 'age', 'w/c', 'total_binder', 'log(age)']
    
    prepared_data = pipeline.transform(input_df[cols])
    prediction = model.predict(prepared_data)[0]
    
    st.success(f"**Predicted Concrete Strength:** {prediction:.2f} MPa")
