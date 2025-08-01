import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score

# --- Load trained model ---
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("notebooks/random_forest_model.pkl")

model = load_model()

# --- Load dataset ---
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('data/diabetes.csv')  # Adjust path as needed
    return df

df = load_data()

# --- App Title & Description ---
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("Diabetes Prediction Application")
st.markdown("""
This interactive app allows users to explore the Pima Indians Diabetes Dataset, visualize key insights,
and predict diabetes risk based on input features using a tuned Random Forest model.
""")

# --- Sidebar Navigation ---
menu = st.sidebar.radio("Navigate", ["Data Exploration", "Visualizations", "Prediction", "Model Performance"])

# --- Helper function: Show dataset overview ---
def show_data_overview(data):
    st.subheader("Dataset Overview")
    st.write(f"Shape: {data.shape}")
    st.write("Columns and Data Types:")
    st.write(pd.DataFrame(data.dtypes.astype(str), columns=["Data Type"]))
    st.write("Sample Data:")
    st.dataframe(data.head())

# --- Helper function: Data filtering ---
def filter_data(data):
    st.subheader("Filter Data")
    # Example filters - customize based on dataset features
    min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
    age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    
    glucose_threshold = st.number_input("Minimum Glucose Level", min_value=int(data['Glucose'].min()), 
                                        max_value=int(data['Glucose'].max()), value=int(data['Glucose'].min()))
    
    filtered = data[(data['Age'] >= age_range[0]) & (data['Age'] <= age_range[1]) & (data['Glucose'] >= glucose_threshold)]
    st.write(f"Filtered Data Shape: {filtered.shape}")
    st.dataframe(filtered)

# --- Helper function: Visualizations ---
def visualizations(data):
    st.subheader("Visualizations")
    chart_type = st.selectbox("Select Chart Type", ["Histogram", "Correlation Heatmap", "Scatter Plot"])
    
    if chart_type == "Histogram":
        feature = st.selectbox("Select Feature for Histogram", data.columns[:-1])
        bins = st.slider("Number of bins", 5, 50, 20)
        fig, ax = plt.subplots()
        sns.histplot(data[feature], bins=bins, kde=True, ax=ax)
        st.pyplot(fig)
    
    elif chart_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    elif chart_type == "Scatter Plot":
        x_feature = st.selectbox("X-axis", data.columns[:-1], index=0)
        y_feature = st.selectbox("Y-axis", data.columns[:-1], index=1)
        color_feature = st.selectbox("Color By (optional)", [None] + list(data.columns))
        fig = px.scatter(data, x=x_feature, y=y_feature, color=color_feature, title=f"{y_feature} vs {x_feature}")
        st.plotly_chart(fig, use_container_width=True)

# --- Helper function: User input for prediction ---
def user_input_features():
    st.sidebar.header("Input Features for Prediction")
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0.0, 900.0, 80.0)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
    age = st.sidebar.slider("Age", 10, 100, 30)
    
    # Create dataframe
    features = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    return features

# --- Helper function: Display prediction ---
def prediction_ui(features):
    st.subheader("Model Prediction")
    if st.button("Predict Diabetes Risk"):
        with st.spinner('Predicting...'):
            # Load training columns
            train_cols = pd.read_csv('data/X_train.csv').columns
            # Align user input to training columns
            user_X = features.reindex(columns=train_cols, fill_value=0)
            # Predict
            pred = model.predict(user_X)[0]
            pred_proba = model.predict_proba(user_X)[0][1]
        
        if pred == 1:
            st.error(f"**Positive for Diabetes** with confidence: {pred_proba:.2%}")
        else:
            st.success(f"**Negative for Diabetes** with confidence: {(1-pred_proba):.2%}")


def model_performance(data):
    st.subheader("Model Performance Metrics")
    
    # Load the saved model
    model = joblib.load("notebooks/random_forest_model.pkl")
    
    # 🔄 Use test data instead of training data!
    X = pd.read_csv('data/X_test.csv')
    y = pd.read_csv('data/y_test.csv').squeeze()

    # Make predictions
    y_pred = model.predict(X)

    # Compute accuracy and F1
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Display metrics
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
    
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

# ------------- Main app logic -------------

if menu == "Data Exploration":
    show_data_overview(df)
    filter_data(df)

elif menu == "Visualizations":
    visualizations(df)

elif menu == "Prediction":
    input_features = user_input_features()
    prediction_ui(input_features)

elif menu == "Model Performance":
    model_performance(df)