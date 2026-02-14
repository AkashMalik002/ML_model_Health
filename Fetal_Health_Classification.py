import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os

# 1. Load Dataset
df = pd.read_csv('fetal_health.csv')

# 2. Prepare Data
# Target is 'fetal_health' (1.0, 2.0, 3.0). We convert to 0, 1, 2 for XGBoost compatibility
X = df.drop('fetal_health', axis=1)
y = df['fetal_health'] - 1  # Shift classes to 0, 1, 2

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss')
}

if not os.path.exists('model'):
    os.makedirs('model')

print(f"{'Model':<25} | {'Accuracy':<10} | {'AUC':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'MCC':<10}")
print("-" * 105)

for name, model in models.items():
    # Train
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
    
    # Calculate Metrics (Weighted for Multi-class)
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs, multi_class='ovr')
    except:
        auc = 0.0
    
    prec = precision_score(y_test, preds, average='weighted')
    rec = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    mcc = matthews_corrcoef(y_test, preds)
    
    print(f"{name:<25} | {acc:.4f}     | {auc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}     | {mcc:.4f}")
    
    # Save
    joblib.dump(model, f'model/{name.replace(" ", "_")}.pkl')

joblib.dump(scaler, 'model/scaler.pkl')
print("\nModels saved.")