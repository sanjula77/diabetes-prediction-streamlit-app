import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Load dataset & model
df = pd.read_csv('data/diabetes.csv')
model = joblib.load('best_model_random_forest.pkl')

# Sidebar menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualizations", "Prediction", "Model Performance"])

# Home page
if menu == "Home":
    st.title("Diabetes Prediction App")
    st.write("""
    This app helps explore diabetes dataset, visualize data, 
    and predict diabetes outcome using a trained Random Forest model.
    """)

# Data Exploration
elif menu == "Data Exploration":
    st.header("Dataset Overview")
    st.write(f"Dataset shape: {df.shape}")
    st.write(df.head())

    # Filter example (filter by Age)
    age_filter = st.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (21, 80))
    filtered_df = df[(df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1])]
    st.write(filtered_df)

# Visualizations
elif menu == "Visualizations":
    st.header("Data Visualizations")

    st.subheader("Outcome Distribution")
    st.bar_chart(df['Outcome'].value_counts())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Histograms")
    selected_feature = st.selectbox("Select feature for histogram", df.columns[:-1])
    fig2, ax2 = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax2)
    st.pyplot(fig2)

# Prediction
elif menu == "Prediction":
    st.header("Make a Prediction")

    # Input widgets for features
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.slider("Glucose", 40, 200, 120)
    blood_pressure = st.slider("BloodPressure", 30, 130, 70)
    skin_thickness = st.slider("SkinThickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 10.0, 70.0, 30.0)
    dpf = st.slider("DiabetesPedigreeFunction", 0.0, 3.0, 0.5, step=0.01)
    age = st.slider("Age", 20, 90, 33)

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.error(f"The model predicts: Diabetes (Positive) with probability {prediction_proba[0][1]:.2f}")
        else:
            st.success(f"The model predicts: No Diabetes (Negative) with probability {prediction_proba[0][0]:.2f}")

# Model Performance
elif menu == "Model Performance":
    st.header("Model Performance Metrics")

    # You can pre-calculate these in your notebook or here
    # For demo, just show hardcoded or load from a file
    
    accuracy = 0.76
    precision = 0.66
    recall = 0.67
    f1 = 0.66
    roc_auc = 0.84

    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")
    st.write(f"ROC AUC: {roc_auc}")

    # Confusion matrix plot example
    # (Replace y_test, y_pred with your test labels and predictions)
    y_test = df['Outcome']  # Replace with your test set labels in real app
    y_pred = model.predict(df.drop('Outcome', axis=1))  # Example, replace with test data prediction

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
