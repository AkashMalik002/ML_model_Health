import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Fetal Health Prediction", layout="wide")
st.title("Fetal Health Classification")

# Load Resources
@st.cache_resource
def load_resources(model_name):
    model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model_name = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])

try:
    model, scaler = load_resources(model_name)
except:
    st.error("Please run the training script first.")
    st.stop()

# Inputs
st.subheader("Cardiotocogram Features")
col1, col2 = st.columns(2)

with col1:
    baseline_value = st.slider("Baseline Fetal Heart Rate (bpm)", 100, 160, 120)
    accelerations = st.number_input("Accelerations (per sec)", 0.0, 0.02, 0.003, format="%.3f")
    uterine_contractions = st.number_input("Uterine Contractions (per sec)", 0.0, 0.02, 0.004, format="%.3f")

with col2:
    prolongued_decelerations = st.number_input("Prolongued Decelerations", 0.0, 0.005, 0.0)
    abnormal_short_term_variability = st.slider("Abnormal Short Term Variability (%)", 0, 100, 40)

if st.button("Predict Health Status"):
    # Create array of 21 zeros (mean placeholder)
    input_data = np.zeros((1, 21))
    
    # Fill specific indices based on dataset structure
    input_data[0][0] = baseline_value           # Index 0
    input_data[0][1] = accelerations            # Index 1
    input_data[0][3] = uterine_contractions     # Index 3
    input_data[0][6] = prolongued_decelerations # Index 6
    input_data[0][7] = abnormal_short_term_variability # Index 7
    
    # Scale
    if model_name in ["Logistic Regression", "KNN"]:
        input_data = scaler.transform(input_data)
        
    pred = model.predict(input_data)[0]
    
    # Map back to labels (Model outputs 0,1,2 -> We map to Normal, Suspect, Pathological)
    status_map = {0: "Normal", 1: "Suspect", 2: "Pathological"}
    
    result = status_map[pred]
    
    if result == "Normal":
        st.success(f"Prediction: {result}")
    elif result == "Suspect":
        st.warning(f"Prediction: {result}")
    else:
        st.error(f"Prediction: {result}")