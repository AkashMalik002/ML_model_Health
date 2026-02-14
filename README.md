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

Below are the observations for each model, based on the performance metrics above.

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Works as a strong baseline and gives reliable accuracy, but it clearly falls behind the ensemble models. This is expected because it assumes a mostly linear relationship between CTG features and fetal health, which does not fully capture the complexity in the data. |
| Decision Tree        | Delivers good performance and is easy to understand, which makes it attractive for interpretability. However, compared to Random Forest, it is more prone to overfitting and reacts more strongly to small changes in the training data. |
| KNN                  | Performs reasonably well when the features are properly scaled and is able to pick up local patterns for Suspect and Pathological cases. The downside is that predictions can be slower and it is more sensitive to how the neighbors and distance metrics are chosen. |
| Naive Bayes          | Shows the weakest overall performance in this setup. A likely reason is its strong independence assumption between features, which does not hold well for physiological signals where many variables are naturally correlated. |
| Random Forest (Ensemble) | Emerges as one of the best models, combining high accuracy with better generalization than a single Decision Tree. By aggregating many trees, it reduces variance and handles the class imbalance (especially fewer Pathological cases) more robustly. |
| XGBoost (Ensemble)   | Achieves the best scores across most metrics, making it the top performer in this project. Its gradient boosting strategy allows it to continuously correct previous mistakes, which helps it separate the more confusing Suspect cases from Normal ones very effectively. |
