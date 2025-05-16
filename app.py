
import streamlit as st
from model import predict_disease, get_all_symptoms

st.set_page_config(page_title="Disease Predictor", layout="wide")

st.title("🩺 Disease Prediction from Symptoms")

st.markdown("Select symptoms from the list below. You can choose multiple symptoms to predict the most likely disease.")

# Get all available symptoms from model.py
symptoms_list = get_all_symptoms()

# Allow multi-select input
selected_symptoms = st.multiselect(
    "Choose your symptoms:",
    options=symptoms_list
)

# Predict button
if st.button("Predict Disease"):
    if selected_symptoms:
        input_symptoms = ",".join(selected_symptoms)
        prediction = predict_disease(input_symptoms)
        st.success(f"🧾 Predicted Disease: **{prediction}**")
    else:
        st.warning("Please select at least one symptom.")


