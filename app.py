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

st.markdown('<h1 class="main-header"> Diabetes Prediction Application</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="main-description">
    This interactive app allows users to explore the Pima Indians Diabetes Dataset, visualize key insights,
    and predict diabetes risk based on input features using a tuned Random Forest model.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
            
    .main-description {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 2rem;
        padding: 0 2rem;
    }
            
    /* Input Summary Cards */
    .input-summary-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .input-summary-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
            
    .input-card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    
    .input-card-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f77b4;
    }

    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1f77b4, #2ca02c, #ff7f0e);
    }
    
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6c757d;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.25rem;
    }
    
    .metric-subtitle {
        font-size: 0.8rem;
        color: #95a5a6;
        font-weight: 400;
    }

    /* Divider Styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #1f77b4, transparent);
        margin: 2rem 0;
    }
            
    /* Prediction Result Cards */
    .prediction-result {
        padding: 2rem;
        border-radius: 16px;
        margin: 0.7rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .prediction-result:hover {
        transform: translateY(-2px);
    }
            
    .success-result {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        color: #155724;
    }
    
    .error-result {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        color: #721c24;
    }
            
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4, #2ca02c);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(31, 119, 180, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(31, 119, 180, 0.4);
    }


    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 12px;
        margin-top: 2rem;
        border-top: 3px solid #1f77b4;
    }     

    /* Sidebar section label */
    .sidebar-title {
        font-size: 2.5rem;
        font-weight: 800;
        padding: 0.75rem 1rem 0.25rem 1rem;
        color: #4B4B4B;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 0.75rem;
    
    }


    /* Radio buttons themselves */
    section[data-testid="stSidebar"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    /* Radio label styling */
    section[data-testid="stSidebar"] label {
        font-size: 1rem;
        font-weight: 500;
        color: #333333;
        padding: 0.4rem 0.6rem;
        border-radius: 0.375rem;
        transition: all 0.2s ease-in-out;
    }

    /* Highlight active radio option */
    section[data-testid="stSidebar"] label[data-selected="true"] {
        background-color: #2563EB20; /* Light blue */
        color: #2563eb;
        font-weight: 600;
    }

    /* Hover effect */
    section[data-testid="stSidebar"] label:hover {
        background-color: #2563EB20;
        width: 100%;
        color: #2563eb;
        cursor: pointer;
    }       

    .sidebar-title {
    font-size: 1.6rem;  
    font-weight: 600;
    padding: 1rem rem 2.25rem 1.4rem;
    color: #ffffff;
    border-bottom: 1px solid #3a3a3a;
    margin-bottom: 0.75rem;
    }
   
   
    .section-header-center {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        text-align: center;
    }

</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.markdown('<div class="sidebar-title">📚 Navigation</div>', unsafe_allow_html=True)

menu = st.sidebar.radio(
    label="Main Navigation",
    options=[
        "📊 Data Exploration",
        "📈 Visualizations",
        "🔮 Prediction",
        "📋 Model Performance"
    ],
    index=0,
    label_visibility="collapsed"
)
# # --- Helper function: Show dataset overview ---
def show_data_overview(data):
    
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
        diabetes_pct = (target_dist.get(1, 0) / len(data) * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Diabetes Cases</div>
            <div class="metric-value">{target_dist.get(1, 0)} ({diabetes_pct:.1f}%)</div>
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
            <div class="metric-title">🔍 Data Quality</div>
            <div class="metric-value">{"✅ Clean" if missing == 0 else f"⚠️ {missing}"}</div>
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
    st.markdown("---")
    
    result_cols = st.columns(4)
    
    with result_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Filtered Records</div>
            <div class="metric-value">{len(filtered)}</div>
            <div class="metric-subtitle">patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with result_cols[1]:
        reduction_pct = ((len(data) - len(filtered)) / len(data) * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📉 Reduction</div>
            <div class="metric-value">{reduction_pct:.1f}%</div>
            <div class="metric-subtitle">filtered out</div>
        </div>
        """, unsafe_allow_html=True)
    
    with result_cols[2]:
        diabetes_in_filtered = filtered['Outcome'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Diabetes Cases</div>
            <div class="metric-value">{diabetes_in_filtered}</div>
            <div class="metric-subtitle">in filtered data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with result_cols[3]:
        if len(filtered) > 0:
            diabetes_rate = (diabetes_in_filtered / len(filtered) * 100)
        else:
            diabetes_rate = 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📈 Diabetes Rate</div>
            <div class="metric-value">{diabetes_rate:.1f}%</div>
            <div class="metric-subtitle">in selection</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h3 class="section-header">📋 Filtered Data</h3>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, hide_index=True) 

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

        # Add statistical insights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="input-summary-card">
                <div class="input-card-title">📊 Mean</div>
                <div class="input-card-value">{data[feature].mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="input-summary-card">
                <div class="input-card-title">📈 Median</div>
                <div class="input-card-value">{data[feature].median():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="input-summary-card">
                <div class="input-card-title">📏 Std Dev</div>
                <div class="input-card-value">{data[feature].std():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="input-summary-card">
                <div class="input-card-title">🎯 Range</div>
                <div class="input-card-value">{data[feature].max() - data[feature].min():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    
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
    st.markdown('<div class="section-header-center">🔮 Diabetes Risk Prediction</div>', unsafe_allow_html=True)
    
    # Enhanced input summary with better organization
    st.markdown('<h3 class="section-header">📋 Patient Profile Summary</h3>', unsafe_allow_html=True)
    
    # Create organized input summary
    demo_col, lab_col, phys_col, genetic_col = st.columns(4)
    
    with demo_col:
        st.markdown("""
        <div class="input-summary-card">
            <div class="input-card-title">👶 Demographics</div>
            <div style="font-size: 0.9rem; color: #6c757d;">
                Age: <strong>{}</strong><br>
                Pregnancies: <strong>{}</strong>
            </div>
        </div>
        """.format(features.iloc[0]['Age'], features.iloc[0]['Pregnancies']), unsafe_allow_html=True)
    
    with lab_col:
        st.markdown("""
        <div class="input-summary-card">
            <div class="input-card-title">🩸 Lab Results</div>
            <div style="font-size: 0.9rem; color: #6c757d;">
                Glucose: <strong>{}</strong><br>
                Insulin: <strong>{}</strong>
            </div>
        </div>
        """.format(features.iloc[0]['Glucose'], features.iloc[0]['Insulin']), unsafe_allow_html=True)
    
    with phys_col:
        st.markdown("""
        <div class="input-summary-card">
            <div class="input-card-title">📏 Physical</div>
            <div style="font-size: 0.9rem; color: #6c757d;">
                BMI: <strong>{}</strong><br>
                BP: <strong>{}</strong>
            </div>
        </div>
        """.format(features.iloc[0]['BMI'], features.iloc[0]['BloodPressure']), unsafe_allow_html=True)
    
    with genetic_col:
        st.markdown("""
        <div class="input-summary-card">
            <div class="input-card-title">🧬 Genetic</div>
            <div style="font-size: 0.9rem; color: #6c757d;">
                Pedigree: <strong>{:.3f}</strong><br>
                Skin: <strong>{}</strong>
            </div>
        </div>
        """.format(features.iloc[0]['DiabetesPedigreeFunction'], features.iloc[0]['SkinThickness']), unsafe_allow_html=True)
    
    st.markdown("---")

    # Enhanced prediction button and results
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Analyze Diabetes Risk", type="primary", use_container_width=True):
            with st.spinner('🧠 AI Model Processing...'):
                # Load training columns
                train_cols = pd.read_csv('data/X_train.csv').columns
                # Align user input to training columns
                user_X = features.reindex(columns=train_cols, fill_value=0)
                # Predict
                pred = model.predict(user_X)[0]
                pred_proba = model.predict_proba(user_X)[0][1]
            
            st.markdown("---")
            
            # Enhanced prediction result display
            if pred == 1:
                st.markdown(f"""
                <div class="prediction-result error-result">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
                    <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">POSITIVE FOR DIABETES RISK</div>
                    <div style="font-size: 1.1rem;">Model Confidence: <strong>{pred_proba:.1%}</strong></div>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">
                        Recommendation: Consult healthcare provider immediately
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-result success-result">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">✅</div>
                    <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">LOW DIABETES RISK</div>
                    <div style="font-size: 1.1rem;">Model Confidence: <strong>{(1-pred_proba):.1%}</strong></div>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">
                        Recommendation: Maintain healthy lifestyle habits
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced confidence visualization
            confidence = pred_proba if pred == 1 else (1 - pred_proba)
            
            # Create confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Model Confidence"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 50], 'color': "#d4edda"},
                        {'range': [50, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced Risk Factor Analysis with modern card layout
            st.markdown('<div style="text-align: center;"><h4 class="section-header">🔍 Risk Factor Analysis</h4></div>', unsafe_allow_html=True)
            
            # Create modern card layout for risk factors
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                # High Risk Factors Card
                st.markdown(
                    """
                    <div style="background-color:#fff3f3; border-left: 6px solid #dc3545; padding: 16px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
                        <h5 style="color:#dc3545; margin-top:0;">🚨 High Risk Factors</h5>
                    """, unsafe_allow_html=True)

                high_risk = []
                if features.iloc[0]['Glucose'] > 140:
                    high_risk.append("• Glucose level above normal (>140 mg/dL)")
                if features.iloc[0]['BMI'] > 30:
                    high_risk.append("• BMI indicates obesity (>30)")
                if features.iloc[0]['Age'] > 45:
                    high_risk.append("• Age above 45 years")
                if features.iloc[0]['BloodPressure'] > 90:
                    high_risk.append("• High blood pressure (>90 mmHg)")

                if high_risk:
                    for risk in high_risk:
                        st.markdown(f"<div style='color: #dc3545;'>{risk}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #28a745;'>• No major high-risk factors detected</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
            
            with risk_col2:
                # Protective Factors Card
                st.markdown(
                    """
                    <div style="background-color:#f3fff3; border-left: 6px solid #28a745; padding: 16px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
                        <h5 style="color:#28a745; margin-top:0;">
                        ✅ Protective Factors</h5>
                    """, unsafe_allow_html=True)

                protective = []
                if features.iloc[0]['BMI'] < 25:
                    protective.append("• Healthy BMI (<25)")
                if features.iloc[0]['Age'] < 30:
                    protective.append("• Young age (<30)")
                if features.iloc[0]['Glucose'] < 100:
                    protective.append("• Normal glucose level (<100 mg/dL)")
                if features.iloc[0]['BloodPressure'] < 80:
                    protective.append("• Optimal blood pressure (<80 mmHg)")

                if protective:
                    for factor in protective:
                        st.markdown(f"<div style='color: #28a745;'>{factor}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #6c757d;'>• Consider lifestyle improvements</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

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

if menu == "📊 Data Exploration":
    show_data_overview(df)
    st.markdown("---")
    filter_data(df)

elif menu == "📈 Visualizations":
    visualizations(df)

elif menu == "🔮 Prediction":
    input_features = user_input_features()
    prediction_ui(input_features)

elif menu == "📋 Model Performance":
    model_performance(model)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
        🏥 Diabetes Prediction Application
    </div>
    <div style="margin-bottom: 1rem;">
        <span style="color: #1f77b4;">🤖 Powered by Machine Learning</span> | 
        <span style="color: #2ca02c;">📊 Built with Streamlit</span> | 
        <span style="color: #ff7f0e;">🔬 Pima Indians Dataset</span>
    </div>
    <div style="font-size: 0.9rem; color: #6c757d;">
        <strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Always consult healthcare professionals for medical decisions.
    </div>
</div>
""", unsafe_allow_html=True)
