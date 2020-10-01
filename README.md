# Mortgage-Default-Detection

## Introduction
The total outstanding value of US mortgages loans was $10.5 trillion by the end 2019 and it is still growing. Therefore, mortgage risk is one of the most important types of risk we should be carefully monitering. The default on mortgage was also the direct cause to the 2008 financial crisis. This project aims to investigate features that are crucial in assessing the default probability of loan applicants and build classification models to successfully detect high-risk applications.

## Table of Contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [ File Descriptions ](#File_Description)
2. [ Technologies Used ](#Technologies_Used)    
3. [ Executive Summary ](#Executive_Summary)
</details>

## File_Description
* <strong>[ Data ](https://www.kaggle.com/c/home-credit-default-risk/data)</strong>: Home Credit's Mortgage Loan Dataset (2.6GB)
   * <strong>1.application.csv</strong>: This is application_train.csv from kaggle, containing over 300K application data and 120+ features
   * <strong>2.bureau.csv</strong>: Applicant's previous credits record
   * <strong>3.bureau_balance.csv</strong>: Monthly balances of previous credits in Credit Bureau
   * <strong>4.POS_CASH_balance.csv</strong>: Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit 
   * <strong>5.credit_card_balance.csv</strong>: Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
   * <strong>6.installments_payments.csv</strong>: Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
   * <strong>7.previous_application.csv</strong>: All previous applications for Home Credit loans of clients who have loans in our sample.
* <strong>Notebooks</strong>:
   * <strong>[1.Logistic Regression Baseline and Simple EDA ](https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/1.%20Logistic%20Regression%20BaseLine.ipynb)</strong>
   * <strong>[2.Random Forest, More EDA and Feature Engineering](https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/2.%20Random%20Forest%20and%20Further%20EDA.ipynb)</strong>
   * <strong>[3.LightGBM](https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/3.%20Light%20GBM.ipynb)</strong>
   * <strong>[4.XGBoost](https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/1.%20Logistic%20Regression%20BaseLine.ipynb)</strong>
   * <strong>[5.Model Stacking](https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/1.%20Logistic%20Regression%20BaseLine.ipynb)</strong>
* <strong>[ Models ](https://github.com/garyjxgong/Morgage-Default-Detection/tree/master/Models)</strong>: Trained models for this project stored by joblib.
* <strong>[ Images ](https://github.com/garyjxgong/Morgage-Default-Detection/tree/master/Images)</strong>: Images used in summary.

## Technologies_Used
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>Matplotlib</strong>
* <strong>Seaborn</strong>
* <strong>Scikit-Learn</strong>
* <strong>LightGBM</strong>
* <strong>XGBoost-Learn</strong>
* <strong>joblib</strong>
* <strong>gc</strong>

## Executive_Summary

### Unbalanced Data
As this dataset is about mortgage loans, we should expect this dataset to be highly unbalance in our target label <strong>(1: Default, 0: Repaid)</strong>. Below is a count plot to visualize target composition created by seaborn.
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/unbalanced_label.png" width=600>
</p>

### Logistic Regression Baseline
After dealing with obvious anomalies, imputing missing value and one-hot-encoding categorical varibles, we build a simple Logistic Regression to serve as a baseline for this project. Below is a set of plot demonstrating the model in terms of learning convergence, scalability and performance.
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/lr_base.png" width=1200>
</p>

#### Confusion Matrix of Out-of-Sample Prediction
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/lr_base_cm.png" width=450>
</p>

### Correlation Analysis
By examing the correlation between features and target, we can get a basic idea on which features may be more important in making predition. However, correlation does not mean causation and we should also examine correlations between features to minimize multicollinearity. Another useful to assess the ability of features in distinguishing target is by using kernel density plot. Below are the kde plots for two features which have top correlation with the target.
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/kde_ext_source_3.png" width=600>
</p>
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/kde_day_birth.png" width=600>
</p>

### Feature Engineering

#### Domain Knowledge
Based on my research on morgage risk, I have inclued 6 new features.
* credit to income ratio: the higher this ratio, the heavier the applicants in debt
* annuity to income ratio: repaid amount compare to income
* credit term: amount that the applicants repaid in percentage
* price to credit: whether applicant is borrowing for goods exceed their repay ability
* employment length to age ratio: higher value means applicants started to work earlier
* average household income: total income divided by number of family number
Again I use kde plot to examine features.
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/kde_credit_term.png" width=600>
</p>
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/kde_credit_income_percent.png" width=600>
</p>

### Random Forest
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/rf.png" width=1200>
</p>

#### Feature Importances from Random Forest Model
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/rf_fi.png" width=600>
</p>

#### Confusion Matrix of Out-of-Sample Prediction
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/rf_cm.png" width=450>
</p>


### LightGBM
The lightGBM model is fitted on dataset including all 7 tables.
<br>
Training Size is <strong>246008 x 696</strong>.
<br>
Training Size is <strong>61503 x 696</strong>.
<br>
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/lgb.png" width=600>
</p>

#### Feature Importance from LightGBM Model
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/lgb_fi.png" width=600>
</p>

#### Confusion Matrix of Out-of-Sample Prediction
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/lgb_cm.png" width=450>
</p>

### XGBoost
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/xgb.png" width=600>
</p>

#### Feature Importance from XGBoost Model
<p align="center">
  <img src="https://github.com/garyjxgong/Morgage-Default-Detection/blob/master/Images/xgb_fi.png" width=600>
</p>

### Conclusion
Feature engineering is diffinitely the most powerful tool in tackling complex data science problems. We can see that the ROC score improved dramaticly after adding features into the models. Among all ensemble classifiers we have tried, LightGBM seems to stands out with a 0.78 ROC.
