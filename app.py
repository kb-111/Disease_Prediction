
# import streamlit as st
# from model import predict_disease, get_all_symptoms

# st.set_page_config(page_title="Disease Predictor", layout="wide")

# st.title("🩺 Disease Prediction from Symptoms")

# st.markdown("Select symptoms from the list below. You can choose multiple symptoms to predict the most likely disease.")

# # Get all available symptoms from model.py
# symptoms_list = get_all_symptoms()

# # Allow multi-select input
# selected_symptoms = st.multiselect(
#     "Choose your symptoms:",
#     options=symptoms_list
# )

# # Predict button
# if st.button("Predict Disease"):
#     if selected_symptoms:
#         input_symptoms = ",".join(selected_symptoms)
#         prediction = predict_disease(input_symptoms)
#         st.success(f"🧾 Predicted Disease: **{prediction}**")
#     else:
#         st.warning("Please select at least one symptom.")


import streamlit as st
from model import (
    get_all_symptoms,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
    train_lightgbm,
    predict_disease
)

st.set_page_config(page_title="Disease Predictor", layout="wide")

st.title("🩺 Disease Prediction from Symptoms")

st.markdown("Select symptoms from the list and choose the model to predict the disease.")

# Get symptom list
symptoms_list = get_all_symptoms()

# User input: symptoms and model choice
selected_symptoms = st.multiselect("Choose your symptoms:", symptoms_list)
model_choice = st.selectbox("Choose Model", ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"])

if st.button("Train & Predict"):

    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Train the selected model and get accuracy
        if model_choice == "Random Forest":
            model, accuracy = train_random_forest()
        elif model_choice == "Gradient Boosting":
            model, accuracy = train_gradient_boosting()
        elif model_choice == "XGBoost":
            model, accuracy = train_xgboost()
        else:
            model, accuracy = train_lightgbm()

        input_symptoms_str = ",".join(selected_symptoms)
        prediction = predict_disease(input_symptoms_str, model)

        st.success(f"🧾 Predicted Disease: **{prediction}**")
        st.info(f"Model: **{model_choice}** | Accuracy (CV mean): **{accuracy:.4f}**")
