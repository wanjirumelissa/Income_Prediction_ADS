#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Suppress the warning related to st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the model
model_path = "best_model3.pkl"
loaded_model = joblib.load(model_path)

# Streamlit App
st.title("Income Prediction Model Explorer")

# Project Explanation
st.markdown("## Income Prediction Project\nPredicts income based on key features such as employment duration, car ownership, and more.")

# Upload CSV data through Streamlit
file_path = 'C:/Users/admin/Documents/Income_Prediction_ADS/dataset1.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
st.subheader("Dataset Preview")
st.write(df.head())

# Feature selection and scaling
features = ['employ', 'car', 'carcat', 'age', 'inccat']
df_selected = df[features]
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=features)

# Allow users to input their data
st.sidebar.subheader("Enter Your Data")
user_data = {}
for feature in features:
    user_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)

# Convert user input to DataFrame
user_df = pd.DataFrame([user_data])

# Scale user input
user_scaled = pd.DataFrame(scaler.transform(user_df), columns=features)

# Prediction using the linear regression model
st.subheader("Linear Regression Model Prediction")

# Make predictions for user input
user_predictions = loaded_model.predict(user_scaled)

# Add Predicted Income column to the DataFrame for the scatter plot
df_scatter = df.copy()
df_scatter['Predicted Income'] = loaded_model.predict(df_scaled)

# Scatter plot of predicted vs. actual income
st.subheader("Scatter Plot: Predicted vs. Actual Income")
scatter_chart = alt.Chart(df_scatter, height=400, width=600).mark_circle().encode(
    x='income',
    y=alt.Y('Predicted Income', title='Predicted Income'),
    color='income'
).interactive()

# Streamlit chart
st.altair_chart(scatter_chart, use_container_width=True)

# Model coefficients visualization
st.subheader("Model Coefficients")
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': loaded_model.coef_})
st.bar_chart(coef_df.set_index('Feature'))

# Evaluation metrics
st.subheader("Model Evaluation Metrics")
y_test = df["income"]
y_pred = loaded_model.predict(df_scaled)
st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
st.write("R-squared:", r2_score(y_test, y_pred))


# In[ ]:




