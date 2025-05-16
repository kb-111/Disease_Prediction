
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import kagglehub

# Download latest version
path = kagglehub.dataset_download("itachi9604/disease-symptom-description-dataset")

print("Path to dataset files:", path)
# Load dataset
data = pd.read_csv('improved_disease_dataset.csv')
df = pd.DataFrame(data)

# Encode target
encoder = LabelEncoder()
df["disease"] = encoder.fit_transform(df["disease"])

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Handle categorical
if 'gender' in X.columns:
    le = LabelEncoder()
    X['gender'] = le.fit_transform(X['gender'])

# Handle missing values
X = X.fillna(0)

# Resample
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Train model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Symptom mapping
symptoms = X.columns.values
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

# Prediction function
def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)

    for symptom in input_symptoms:
        symptom = symptom.strip()
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)
    prediction = encoder.classes_[rf_model.predict(input_data)[0]]
    return prediction

#  expose symptom list for dropdowns
def get_all_symptoms():
    return list(symptom_index.keys())
