#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Prediction using the linear regression model
st.subheader("Linear Regression Model Prediction")

# Make predictions
predictions = loaded_model.predict(df_scaled)
df["Predicted Income"] = predictions
st.write(df[["income", "Predicted Income"]])

# Scatter plot of predicted vs. actual income
plt.figure(figsize=(8, 6))
sns.scatterplot(x='income', y='Predicted Income', data=df)
plt.title('Predicted vs. Actual Income')
plt.xlabel('Actual Income')
plt.ylabel('Predicted Income')
st.pyplot()

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




