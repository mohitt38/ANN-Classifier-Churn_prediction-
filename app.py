import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction App")

# User inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, value=5000.0)
credit_score = st.number_input("Credit Score")
tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=2)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

#  Use actual gender input (not string 'gender')
gender_encoded = label_encoder_gender.transform([gender])[0]

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode geography and merge properly
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Correct typo in concat and reset_index
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Output results
st.write(f"Churn Probability: {prediction_proba * 100:.2f}%")

if prediction_proba > 0.5:
    st.error(f"Customer is likely to churn with a probability of {prediction_proba * 100:.2f}%")
else:
    st.success(f"Customer is unlikely to churn with a probability of {(1 - prediction_proba) * 100:.2f}%")

st.caption("Thank you for using the Customer Churn Prediction App!")
