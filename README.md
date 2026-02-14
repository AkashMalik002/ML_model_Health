Fetal Health Classification Project
a. Problem Statement
The primary objective of this project is to classify the health status of a fetus into one of three categories: Normal, Suspect, or Pathological. This classification is based on data derived from Cardiotocogram (CTG) exams, which measure fetal heart rate and uterine contractions. By automating this diagnosis using Machine Learning, medical professionals can more efficiently identify high-risk pregnancies and intervene early to prevent fetal mortality and morbidity.

b. Dataset Description
Dataset Name: Fetal Health Classification

Source: Kaggle / UCI Machine Learning Repository

Dataset URL: Fetal Health Classification on Kaggle

Features: The dataset contains 21 diagnostic features (satisfying the >12 requirement), including:

Baseline Fetal Heart Rate (FHR)

Accelerations & Decelerations

Uterine Contractions

Short-term and Long-term variability percentages

Instances: The dataset contains 2,126 records (satisfying the >500 requirement).

Target Variable: fetal_health (Multi-class):

1.0 = Normal

2.0 = Suspect

3.0 = Pathological
------------------------------
c. Models Used
1. Performance Comparison
Below is the evaluation of all 6 models implemented on the dataset.

Model                     | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC       
---------------------------------------------------------------------------------------------------------
Logistic Regression       | 0.8803     | 0.9586     | 0.8817     | 0.8803     | 0.8810     | 0.6718
Decision Tree             | 0.9296     | 0.9272     | 0.9334     | 0.9296     | 0.9309     | 0.8127
KNN                       | 0.9155     | 0.9438     | 0.9121     | 0.9155     | 0.9128     | 0.7575
Naive Bayes               | 0.8028     | 0.8994     | 0.8698     | 0.8028     | 0.8205     | 0.5951
Random Forest             | 0.9531     | 0.9863     | 0.9524     | 0.9531     | 0.9526     | 0.8689
XGBoost                   | 0.9554     | 0.9875     | 0.9554     | 0.9554     | 0.9554     | 0.8773

2. Observations
Below are the observations regarding the performance of each model on this specific dataset.

ML Model Name		        | Observation about model performance
----------------------------------------------------------------------------------------------------------------------------------------------------------
Logistic Regression	    | Provides a solid baseline with decent accuracy. However, it struggles slightly compared to ensemble methods because it assumes a 
			                    linear relationship between the CTG features and the health classes, which may not always hold true.
Decision Tree		        | Performs well and is easy to interpret. However, it showed signs of slight overfitting (high training accuracy vs test accuracy) 
			                    compared to the Random Forest, as single trees are sensitive to small variations in the data.
kNN			                | Performance was heavily dependent on the scaling of data (StandardScaler). It captured local clusters of "Suspect" and 
			                    "Pathological" cases well but is computationally slower during the prediction phase compared to tree-based models.
Naive Bayes		          | Generally showed the lowest performance among the group. This is likely because Naive Bayes assumes all features (like 
			                    accelerations and uterine contractions) are independent, but in physiological data, these features are often highly correlated.
Random Forest (Ensemble)| One of the top performers. By averaging multiple decision trees, it successfully reduced the variance and overfitting seen in the 
			                    single Decision Tree model. It handled the imbalanced nature of the classes (fewer Pathological cases) very well.
XGBoost (Ensemble)	    | The Best Performer. It achieved the highest Accuracy and F1 Score. Its gradient boosting approach allowed it to iteratively 


			  correct errors made by previous trees, making it extremely effective at distinguishing the difficult "Suspect" cases from "Normal" ones.
===================================================================================================================================================================
