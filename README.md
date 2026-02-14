# Fetal Health Classification Project

## a. Problem Statement

The goal of this project is to classify the health status of a fetus into one of three categories: **Normal**, **Suspect**, or **Pathological** based on Cardiotocogram (CTG) data, which includes fetal heart rate and uterine contractions. This Machine Learning solution helps medical professionals more efficiently identify high‑risk pregnancies and intervene early to reduce fetal mortality and morbidity.

## b. Dataset Description

- **Dataset name:** Fetal Health Classification  
- **Source:** Kaggle / UCI Machine Learning Repository  
- **Features:** 21 diagnostic features, including:
  - Baseline Fetal Heart Rate (FHR)
  - Accelerations and decelerations
  - Uterine contractions
  - Short‑term and long‑term variability percentages  
- **Instances:** 2,126 records  
- **Target variable:** `fetal_health` (multi‑class)
  - 1.0 = Normal  
  - 2.0 = Suspect  
  - 3.0 = Pathological  

## c. Models Used

### 1. Performance Comparison

Below is the evaluation of all 6 models implemented on the dataset.

| Model                | Accuracy | AUC    | Precision | Recall  | F1      | MCC    |
|----------------------|---------:|-------:|----------:|--------:|--------:|-------:|
| Logistic Regression  | 0.8803   | 0.9586 | 0.8817    | 0.8803  | 0.8810  | 0.6718 |
| Decision Tree        | 0.9296   | 0.9272 | 0.9334    | 0.9296  | 0.9309  | 0.8127 |
| KNN                  | 0.9155   | 0.9438 | 0.9121    | 0.9155  | 0.9128  | 0.7575 |
| Naive Bayes          | 0.8028   | 0.8994 | 0.8698    | 0.8028  | 0.8205  | 0.5951 |
| Random Forest        | 0.9531   | 0.9863 | 0.9524    | 0.9531  | 0.9526  | 0.8689 |
| XGBoost              | 0.9554   | 0.9875 | 0.9554    | 0.9554  | 0.9554  | 0.8773 |

### 2. Observations

Below are the observations regarding the performance of each model on this dataset.

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Provides a solid baseline with decent accuracy but lags behind ensemble methods because it assumes a linear relationship between CTG features and health classes, which may not always hold. |
| Decision Tree        | Performs well and is easy to interpret, but shows slight overfitting compared to Random Forest, as single trees are sensitive to small variations in the data. |
| KNN                  | Performance is heavily dependent on scaling (e.g., StandardScaler). It captures local clusters of “Suspect” and “Pathological” cases well but is slower at prediction time than tree‑based models. |
| Naive Bayes          | Shows the lowest performance overall, likely because it assumes feature independ
