# Disease Prediction System

This project implements a machine learning-based disease prediction system using symptoms as input. The model predicts the most likely disease based on selected symptoms and provides additional information such as disease description and precautions.

---

## 🚀 Project Overview

- Predict diseases based on user-selected symptoms using a Random Forest Classifier.
- Provides a user-friendly interface via Streamlit for easy interaction.
## 📊 Model Description & Performance
### Model: Random Forest Classifier

- Evaluation:

5-fold Cross-Validation Accuracy Scores: [0.5497, 0.5468, 0.5322, 0.5365, 0.5249]

- Mean Cross-Validation Accuracy: 53.80%

- Final Test Accuracy: 68.98%

The model demonstrates moderate predictive performance. The difference between cross-validation and test accuracy suggests some overfitting or data variance.

## Models Compared

This project supports multiple machine learning models for disease prediction:

- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Each model is trained with 5-fold cross-validation, and accuracy scores are shown in the app to help compare performance.

### Current Observations

| Model              | CV Mean Accuracy |
|--------------------|------------------|
| Random Forest      | 0.5590           |
| Gradient Boosting  | 0.5208     |
| XGBoost            |  0.5541     |
| LightGBM           |  0.5553          |


## How to Improve Further

- Perform hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Try neural networks for complex patterns
- Add more data or features
- Handle class imbalance with advanced techniques
