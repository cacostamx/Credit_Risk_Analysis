# Credit Risk Analysis

## Overview

The purpose of this analysis is to employ different machine learning techniques to train and evaluate models with unbalanced classes using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, in order to **predict credit risk**.

First, we use oversampling and undersampling algorithms. Then use a combinatorial approach of over- and undersampling. Finally, we'll use ensemble algorithms to reduce bias.

We compared the performance of all these models to assess the best one at predicting credit risk.

### Resources

- Data source: LendingClub dataset from '/Resources/LoanStats_2019Q1.csv'.

- Software use to perform the analysis: Jupyter Notebook v6.4.5 with Scikit-learn v0.24.2


## Results

### Preprocessing data

The dataset included the target variable which was "loan_status" which was encoded to include two results: high_risk and low_risk, that were assigned values of 0 and 1 respectively.

![Initial DataFrame](/Resources/dataframe.png)

Nine feature labels had string values so they were encoded with pandas to dummy variables with the pd.get_dummies method.

```python:
# Convert string values to numerical
dummyCols = ['home_ownership', 'verification_status', 'issue_d',
       'pymnt_plan', 'initial_list_status', 'next_pymnt_d', 'application_type',
       'hardship_flag', 'debt_settlement_flag']
binary_df = pd.get_dummies(df, columns=dummyCols)
```
Our target data ***'y'*** was assigned the array of the loan_status column.

For ***'X'***, all the remaining 96 feature columns were assigned.

![X features](/Resources/dataframeX.png)

### Balancing the data

Our target variable was clearly unbalanced, with 68,470 low_risk credits versus 347 high_risk ones.  This created also a  y_train sample imbalance. 

![Y_train sample](/Resources/ytrainImbalance.png)

#### Oversampling

These algorithms try to balance the sample by increasing the number of the minimun class to that of the maximum class.

- **Random Oversample.** With RandomOverSampler alogrithm and then aplying Logistic Regression Model we obtained a balanced accuracy score of **0.6464** with the following confusion matrix and imbalanced classification report.

|Confusion Matrix                                |Imbalanced Classification                         |
|:-----------------------------------------------|:------------------------------------------------:|
|![Confusion Matrix](/Resources/cmRandomOver.png)|![Report RandomOver](/Resources/repRandomOver.png)|

- **SMOTE Oversample.** With SMOTE alogrithm and then aplying Logistic Regression Model we obtained a balanced accuracy score of **0.6586** with the following confusion matrix and imbalanced classification report.

|Confusion Matrix                           |Imbalanced Classification                    |
|:------------------------------------------|:-------------------------------------------:|
|![Confusion Matrix](/Resources/cmSMOTE.png)|![Imbalanced Report](/Resources/repSMOTE.png)|

### Undersampling

These algorithms try to balance the sample by decreasing the number of the maximum class to that of the minimun class.

- **Cluster Centroids.** With ClusterCentroids alogrithm and then aplying Logistic Regression Model we obtained a balanced accuracy score of **0.5442** with the following confusion matrix and imbalanced classification report.

|Confusion Matrix                           |Imbalanced Classification                    |
|:------------------------------------------|:-------------------------------------------:|
|![Confusion Matrix](/Resources/cmSMOTE.png)|![Imbalanced Report](/Resources/repSMOTE.png)|

### Combination of over and under sampling

- **SMOTEEN.** With SMOTEENN alogrithm and then aplying Logistic Regression Model we obtained a balanced accuracy score of **0.6480** with the following confusion matrix and imbalanced classification report.

|Confusion Matrix                              |Imbalanced Classification                       |
|:---------------------------------------------|:----------------------------------------------:|
|![Confusion Matrix](/Resources/cmSMOTEENN.png)|![Imbalanced Report](/Resources/repSMOTEENN.png)|

#### Ensemble Learners

These algorithms try to elaborate weak models and combining them to create a more accurate and robust prediction engine.

- **Random Forest Classifier.** With RandomForestClassifier alogrithm and then aplying Logistic Regression Model we obtained a balanced accuracy score of **0.6830** with the following confusion matrix and imbalanced classification report.

|Confusion Matrix                               |Imbalanced Classification                        |
|:----------------------------------------------|:-----------------------------------------------:|
|![Confusion Matrix](/Resources/cmRndForest.png)|![Imbalanced Report](/Resources/repRndForest.png)|

In this case, the feature importance showed the following features as the most relevant with a weight over 5%.

![Relevant Features](/Resources/featRndForest.png)

- **Easy Ensemble Classifier.** With EasyEnsembleClassifier alogrithm and then aplying Logistic Regression Model we obtained a balanced accuracy score of **0.9317** with the following confusion matrix and imbalanced classification report.

|Confusion Matrix                             |Imbalanced Classification                      |
|:--------------------------------------------|:---------------------------------------------:|
|![Confusion Matrix](/Resources/cmEasyEns.png)|![Imbalanced Report](/Resources/repEasyEns.png)|

## Summary

- The following table summarizes the results for all models, regarding prediction and detection of high_risk credits:

|Model                 |Type                     |Precision|Sensitivity|F1-score|
|:---------------------|:-----------------------:|:-------:|:---------:|:------:|
|RandomOverSampler     |Logistic Over Sampler    |     0.01|       0.71|    0.02|
|SMOTE                 |Logistic Over Sampler    |     0.01|       0.63|    0.02|
|ClusterCentroids      |Logistic Under Sampler   |     0.01|       0.69|    0.01|
|SMOTEENN              |Logistic Combined Sampler|     0.01|       0.72|    0.02|
|RandomForestClassifier|Ensemble Decision Tree   |     0.88|       0.37|    0.52|
|EasyEnsembleClassifier|Boosting Adaptive        |     0.09|       0.92|    0.16|

Precision = TruePositive / (TruePositive + FalsePositive)
Sensitivity = TruePositive / (TruePositive + FalseNegative)

- From these results we can see that regarding precision, meaning to determine true high_risk loans, all models perform bad except for the Random Forest that gives a score of 0.88.  However, in this case, it is very important to have a high sensitivity (recall) rate, because we don't want to have false negatives, or true hig_risk loans predicted as low_risk.  In this case, the best one is Easy Ensemble.

- Al models have very low combined F1 score, with the Random Forest having the highest with 0.52.

- It is clear that the simple Logistic Regression model is not good for this kind of prediction. Ensemble models do provide a better accuracy level for our purposes.

- I would recommend using (a)  the Easy Ensemble because it offers the best protection for having high_risk loans detected as low_risk, but in the downside a lot of low_risk loans will be labeled as high_risk; and (b) using in second instance the RandomForestClassifier on all the predicted as high_risk to obtain a better subset of those true high_risk.

- Also, as seen on the features importance rank for the Random Forest we could enhance the models by getting rid of those low ranked features.