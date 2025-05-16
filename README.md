# Disease Prediction System

This project implements a machine learning-based disease prediction system using symptoms as input. The model predicts the most likely disease based on selected symptoms and provides additional information such as disease description and precautions.

---

## 🚀 Project Overview

- Predict diseases based on user-selected symptoms using a Random Forest Classifier.
- Provides a user-friendly interface via Streamlit for easy interaction.
## 📊 Model Description & Performance
# Model: Random Forest Classifier

# Evaluation:

5-fold Cross-Validation Accuracy Scores: [0.5497, 0.5468, 0.5322, 0.5365, 0.5249]

# Mean Cross-Validation Accuracy: 53.80%

# Final Test Accuracy: 68.98%

The model demonstrates moderate predictive performance. The difference between cross-validation and test accuracy suggests some overfitting or data variance.

## 💡 Potential Improvements
To improve model accuracy and robustness:

Experiment with other algorithms such as Gradient Boosting, XGBoost, LightGBM, or Neural Networks.

Perform hyperparameter tuning (Grid Search, Randomized Search) to optimize model parameters.

Apply feature engineering to add new relevant features or transform existing ones.

Address class imbalance with alternative oversampling/undersampling methods or class-weighted algorithms.

Expand dataset size and diversity to enhance generalization.

Use interpretability tools like SHAP or LIME to better understand feature impact.

Combine multiple models via ensemble methods for improved predictions.

