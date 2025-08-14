# House Price Prediction

Predicting house prices using advanced ensemble machine learning techniques.

## Project Overview
This project aims to predict house prices using feature engineering, encoding, scaling, and a stacking ensemble model. The approach leverages multiple base models with a meta-model for better accuracy.

## Dataset
- Used the [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from Kaggle.
- Dataset includes numerical, categorical, and ordinal features.

## Data Preprocessing
- **Target Encoding**: Applied to categorical features to handle high-cardinality variables.
- **Robust Scaling**: Applied to numeric features to reduce the effect of outliers.
- **Feature Selection**: Used **RFM (Relevance, Frequency, Mutual information)** technique to select the most important features.

## Model Architecture
- **Stacking Ensemble Model**:
  - **Base Models**:
    - CatBoost Regressor
    - XGBoost Regressor
    - LightGBM Regressor
    - Ridge Regression
    - Lasso Regression
  - **Meta Model**:
    - ElasticNetCV (Linear model combining L1 and L2 regularization)
    
- **Why Stacking?** Combining multiple models reduces bias and variance, improving predictive performance.



# Run the notebook
jupyter notebook HousePrice_Prediction.ipynb
