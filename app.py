import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .success-result {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-result {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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

# --- App Header ---
st.markdown('<h1 class="main-header">🏥 Diabetes Prediction Application</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    This interactive app allows users to explore the Pima Indians Diabetes Dataset, visualize key insights,
    and predict diabetes risk based on input features using a tuned Random Forest model.
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
menu = st.sidebar.radio("Navigate", ["Data Exploration", "Visualizations", "Prediction", "Model Performance"])

# # --- Helper function: Show dataset overview ---
def show_data_overview(data):
    st.markdown("""
        <style>
        .metric-card {
            background-color: #2c3e50;
            border-radius: 12px;
            padding: 1.3rem;
            text-align: left;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            height: 100%;
        }
        .metric-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #f8f9fa;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color:rgb(213, 214, 215);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)

    # Create 4 metric columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📏 Dataset Size</div>
            <div class="metric-value">{data.shape[0]} × {data.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        target_dist = data['Outcome'].value_counts().to_dict()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Distribution</div>
            <div class="metric-value">{{0: {target_dist.get(0, 0)}, 1: {target_dist.get(1, 0)}}}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📋 Features</div>
            <div class="metric-value">{len(data.columns)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        missing = data.isnull().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🔍 Missing Values</div>
            <div class="metric-value">{missing}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Data types and sample data
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">📋 Data Types</h3>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame(data.dtypes.astype(str), columns=["Data Type"])
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">📄 Sample Data</h3>', unsafe_allow_html=True)
        st.dataframe(data.head(9), use_container_width=True)

# --- Helper function: Data filtering with enhanced UI ---
def filter_data(data):
    st.markdown('<h2 class="section-header">🔍 Data Filtering</h2>', unsafe_allow_html=True)
    
    # Filter controls in columns
    col1, col2 = st.columns(2)
    
    with col1:
        min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
        age_range = st.slider(
            "👥 Age Range", 
            min_value=min_age, 
            max_value=max_age, 
            value=(min_age, max_age),
            help="Select age range to filter the dataset"
        )
    
    with col2:
        glucose_threshold = st.number_input(
            "🩸 Minimum Glucose Level", 
            min_value=int(data['Glucose'].min()), 
            max_value=int(data['Glucose'].max()), 
            value=int(data['Glucose'].min()),
            help="Filter records with glucose level above this threshold"
        )
    
    # Apply filters
    filtered = data[
        (data['Age'] >= age_range[0]) & 
        (data['Age'] <= age_range[1]) & 
        (data['Glucose'] >= glucose_threshold)
    ]
    
    # Display filtered results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Filtered Records</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: #1f77b4;">{len(filtered)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Original Records</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: #1f77b4;">{len(data)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<h3 class="section-header">📋 Filtered Data</h3>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True)

# # --- Helper function: Visualizations ---
def visualizations(data):
    st.markdown('<h2 class="section-header">📈 Data Visualizations</h2>', unsafe_allow_html=True)
    
    chart_type = st.selectbox(
        "📊 Select Chart Type", 
        ["📊 Histogram", "🔥 Correlation Heatmap", "📈 Scatter Plot", "📊 Box Plot"],
        help="Choose the type of visualization to display"
    )
    
    if chart_type == "📊 Histogram":
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature", data.columns[:-1])
        
        with col2:
            bins = st.slider("Number of bins", 5, 50, 35)
        
        fig = px.histogram(
            data, 
            x=feature, 
            nbins=bins,
            title=f"Distribution of {feature}",
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "🔥 Correlation Heatmap":
        fig = px.imshow(
            data.corr(),
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu"
        )
        fig.update_layout(
            title_x=0.5,
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "📈 Scatter Plot":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_feature = st.selectbox("X-axis", data.columns[:-1], index=0)
        
        with col2:
            y_feature = st.selectbox("Y-axis", data.columns[:-1], index=1)
        
        with col3:
            color_feature = st.selectbox("Color By", [None] + list(data.columns))
        
        fig = px.scatter(
            data, 
            x=x_feature, 
            y=y_feature, 
            color=color_feature,
            title=f"{y_feature} vs {x_feature}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            title_x=0.5,
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "📊 Box Plot":
        feature = st.selectbox("Select Feature for Box Plot", data.columns[:-1])
        
        fig = px.box(
            data, 
            y=feature,
            title=f"Box Plot of {feature}",
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# # --- Helper function: User input for prediction ---
def user_input_features():
    st.sidebar.markdown('<h3 class="sidebar-header">🔮 Prediction Inputs</h3>', unsafe_allow_html=True)
    
    # Organize inputs in logical groups
    st.sidebar.markdown("**👶 Pregnancy & Age**")
    pregnancies = st.sidebar.number_input(
        "Pregnancies", 
        0, 20, 1,
        help="Number of times pregnant"
    )
    age = st.sidebar.slider(
        "Age", 
        10, 100, 30,
        help="Age in years"
    )
    
    st.sidebar.markdown("**🩸 Blood Tests**")
    glucose = st.sidebar.slider(
        "Glucose Level", 
        0, 200, 120,
        help="Plasma glucose concentration (mg/dL)"
    )
    blood_pressure = st.sidebar.slider(
        "Blood Pressure", 
        0, 140, 70,
        help="Diastolic blood pressure (mm Hg)"
    )
    insulin = st.sidebar.slider(
        "Insulin", 
        0.0, 900.0, 80.0,
        help="2-Hour serum insulin (mu U/ml)"
    )
    
    st.sidebar.markdown("**📏 Physical Measurements**")
    skin_thickness = st.sidebar.slider(
        "Skin Thickness", 
        0, 100, 20,
        help="Triceps skin fold thickness (mm)"
    )
    bmi = st.sidebar.slider(
        "BMI", 
        0.0, 70.0, 25.0,
        help="Body mass index (weight in kg/(height in m)²)"
    )
    diabetes_pedigree = st.sidebar.slider(
        "Diabetes Pedigree Function", 
        0.0, 3.0, 0.5, 
        step=0.01,
        help="Diabetes pedigree function"
    )
    
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

# --- Helper function: prediction display ---
def prediction_ui(features):
    st.markdown('<h2 class="section-header">🔮 Diabetes Risk Prediction</h2>', unsafe_allow_html=True)
    
    # Display input summary
    st.markdown('<h3 class="section-header">📋 Input Summary</h3>', unsafe_allow_html=True)
    
    # Create input summary cards
    cols = st.columns(4)
    feature_names = {
        'Pregnancies': '👶 Pregnancies',
        'Glucose': '🩸 Glucose',
        'BloodPressure': '💓 Blood Pressure',
        'SkinThickness': '📏 Skin Thickness',
        'Insulin': '💉 Insulin',
        'BMI': '⚖️ BMI',
        'DiabetesPedigreeFunction': '🧬 Pedigree',
        'Age': '👤 Age'
    }
    
    for i, (col, value) in enumerate(features.iloc[0].items()):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{feature_names[col]}</h4>
                <p style="font-size: 1.2rem; font-weight: bold; color: #1f77b4;">{value}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction button and results
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
            with st.spinner('🔍 Analyzing data...'):
                # Load training columns
                train_cols = pd.read_csv('data/X_train.csv').columns
                # Align user input to training columns
                user_X = features.reindex(columns=train_cols, fill_value=0)
                # Predict
                pred = model.predict(user_X)[0]
                pred_proba = model.predict_proba(user_X)[0][1]
            
            st.markdown("---")
            
            # Display prediction result
            if pred == 1:
                st.markdown(f"""
                <div class="prediction-result error-result">
                    ⚠️ <strong>Positive for Diabetes</strong><br>
                    Confidence: {pred_proba:.2%}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-result success-result">
                    ✅ <strong>Negative for Diabetes</strong><br>
                    Confidence: {(1-pred_proba):.2%}
                </div>
                """, unsafe_allow_html=True)
            
            # Add confidence bar
            confidence = pred_proba if pred == 1 else (1 - pred_proba)
            st.progress(confidence)
            st.caption(f"Model Confidence: {confidence:.2%}")

# --- Helper function: model performance display ---
def model_performance(data):
    st.markdown('<h2 class="section-header">📋 Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    # Load the saved model
    model = joblib.load("notebooks/random_forest_model.pkl")
    
    # Use test data
    X = pd.read_csv('data/X_test.csv')
    y = pd.read_csv('data/y_test.csv').squeeze()

    # Make predictions
    y_pred = model.predict(X)

    # Compute metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style = "color:  #2c3e50;">🎯 Accuracy</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: #28a745;">{accuracy:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style = "color:  #2c3e50;">📊 F1 Score</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: #17a2b8;">{f1:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style = "color:  #2c3e50;">📈 Test Samples</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: #ffc107;">{len(y)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style = "color:  #2c3e50;">🔍 Features</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: #dc3545;">{X.shape[1]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confusion Matrix
    st.markdown('<h3 class="section-header">🔍 Confusion Matrix</h3>', unsafe_allow_html=True)
    
    cm = confusion_matrix(y, y_pred)
    
    # Create Plotly confusion matrix
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Diabetes', 'Predicted Diabetes'],
        y=['Actual No Diabetes', 'Actual Diabetes'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        title_x=0.5,
        title_font_size=16,
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Report
    with st.expander("📊 Detailed Classification Report"):
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

# ------------- Main app logic -------------

if menu == "Data Exploration":
    show_data_overview(df)
    st.markdown("---")
    filter_data(df)

elif menu == "Visualizations":
    visualizations(df)

elif menu == "Prediction":
    input_features = user_input_features()
    prediction_ui(input_features)

elif menu == "Model Performance":
    model_performance(model)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏥 Diabetes Prediction Application | Built with Streamlit & Machine Learning</p>
    <p>📊 Pima Indians Diabetes Dataset | Random Forest Model</p>
</div>
""", unsafe_allow_html=True)