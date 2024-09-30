**Business Validation Prediction using Yelp Dataset**.
This project aims to predict whether a business will open or remain closed based on data from the Yelp academic dataset. The target variable is is_open, which indicates the current status of the business.

**Project Overview**
The objective is to build machine learning models that predict the business's operational status (is_open) using various business-related features provided by Yelp. The project involves data preprocessing, feature engineering, model training, and evaluation using different classification algorithms.

**Dataset**
The data is derived from the Yelp academic dataset, particularly focusing on:

business.json: Information about businesses, including attributes like name, location, categories, etc.
checkin.json: Data about check-ins made by customers at the businesses.
**Libraries Used**
eda
Data Manipulation: pandas, numpy
Visualization: matplotlib, seaborn
Machine Learning: scikit-learn, xgboost
Model Interpretability: lime, shap
Dimensionality Reduction: TSNE
Models Applied
Decision Tree Classifier
Random Forest Classifier
AdaBoost Classifier
XGBoost Classifier
