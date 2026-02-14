import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title="Fetal Health Classification", layout="wide")
st.title("Fetal Health Classification App")

# 1. Load Data
@st.cache_data
def load_data():
    # Ensure fetal_health.csv is in the same folder
    df = pd.read_csv('fetal_health.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'fetal_health.csv' not found. Please upload it to your GitHub repository.")
    st.stop()

# 2. Sidebar - Model Selection
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose Classifier", 
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# 3. Load Models & Scaler
@st.cache_resource
def load_resources(name):
    try:
        model = joblib.load(f"model/{name.replace(' ', '_')}.pkl")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_resources(model_name)

if model is None:
    st.error(f"Error: Model file for {model_name} not found. Did you run train_models.py and push the 'model/' folder?")
    st.stop()

# --- TAB SELECTION ---
tab1, tab2 = st.tabs(["🏥 Predict Health Status", "📊 Model Performance"])

# ==========================================
# TAB 1: Prediction Interface
# ==========================================
with tab1:
    st.subheader(f"Predict using {model_name}")
    st.write("Adjust the sliders below to simulate a patient's CTG data.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Define inputs and save to variables
        baseline_value = st.slider("Baseline Fetal Heart Rate (bpm)", 100, 160, 120)
        accelerations = st.number_input("Accelerations (per sec)", 0.0, 0.02, 0.003, format="%.3f")
        uterine_contractions = st.number_input("Uterine Contractions (per sec)", 0.0, 0.02, 0.004, format="%.3f")
    with col2:
        prolongued_decelerations = st.number_input("Prolongued Decelerations", 0.0, 0.005, 0.0, format="%.3f")
        abnormal_short_term_variability = st.slider("Abnormal Short Term Variability (%)", 0, 100, 40)

    if st.button("Predict"):
        # Prepare input
        input_data = np.zeros((1, 21)) # Create array with 21 features (filled with 0s)
        
        # Fill important features (Variables must match definitions above)
        input_data[0][0] = baseline_value
        input_data[0][1] = accelerations
        input_data[0][3] = uterine_contractions  # <--- THIS WAS THE ERROR LINE
        input_data[0][6] = prolongued_decelerations
        input_data[0][7] = abnormal_short_term_variability
        
        # Scale if needed
        if model_name in ["Logistic Regression", "KNN"]:
            input_data = scaler.transform(input_data)
            
        prediction = model.predict(input_data)[0]
        
        status_map = {0: "Normal", 1: "Suspect", 2: "Pathological"}
        result = status_map.get(prediction, "Unknown")
        
        # Display Result
        if result == "Normal":
            st.success(f"### Prediction: {result}")
        elif result == "Suspect":
            st.warning(f"### Prediction: {result}")
        else:
            st.error(f"### Prediction: {result}")

# ==========================================
# TAB 2: Model Performance (Metrics & Plots)
# ==========================================
with tab2:
    st.subheader(f"Performance Metrics for {model_name}")
    
    with st.spinner("Calculating metrics..."):
        # Re-create the Test Split (Must match the random_state in train_models.py)
        X = df.drop('fetal_health', axis=1)
        y = df['fetal_health'] - 1  # Adjust to 0, 1, 2
        
        # IMPORTANT: Use same random_state=42 as training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale X_test if needed
        if model_name in ["Logistic Regression", "KNN"]:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test
            
        # Get Predictions
        y_pred = model.predict(X_test_processed)
        
        # 1. Accuracy Score
        acc = accuracy_score(y_test, y_pred)
        st.metric("Test Accuracy", f"{acc:.2%}")
        
        # 2. Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["Normal", "Suspect", "Pathological"], 
                    yticklabels=["Normal", "Suspect", "Pathological"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # 3. Classification Report
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, target_names=["Normal", "Suspect", "Pathological"], output_dict=True)
        # Convert to DataFrame and Transpose for better readability
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
