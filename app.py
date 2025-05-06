import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import io
import time
from PIL import Image

# Import custom modules
from ml_pipeline import preprocess_data, feature_engineering, split_data, train_model, evaluate_model
from visualizations import plot_missing_values, plot_split, plot_feature_importance, plot_regression_results, plot_classification_results, plot_clusters
from utils import load_animation_url, validate_stock_ticker, show_notification

# Set page configuration with professional layout and theme
st.set_page_config(
    page_title="Financial ML Pipeline Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom ChromaWave theme configuration
st.markdown("""
<style>
    /* ChromaWave theme colors */
    :root {
        --primary-color: #6366F1;
        --secondary-color: #4F46E5;
        --background-color: #F9FAFB;
        --text-color: #111827;
        --accent-color: #8B5CF6;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --info-color: #3B82F6;
        --gradient-start: #6366F1;
        --gradient-end: #8B5CF6;
        --card-bg: rgba(255, 255, 255, 0.9);
        --dark-bg: #1F2937;
    }
    
    /* Main interface styling with subtle gradient */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: var(--background-color);
        background-image: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
    }
    
    /* Sidebar styling with subtle gradient */
    section[data-testid="stSidebar"] > div {
        background-image: linear-gradient(180deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.1);
        padding-top: 1rem;
    }
    
    /* Glassmorphism effect for cards */
    .card {
        background: var(--card-bg);
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    /* Modernized button styling */
    .stButton > button {
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
        background-image: linear-gradient(to right, var(--gradient-start), var(--gradient-end));
        color: white;
        border: none !important;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4);
        background-image: linear-gradient(to right, var(--gradient-end), var(--gradient-start));
    }
    
    /* Card styling */
    div.stAlert {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: none;
    }
    
    /* Step progress styling */
    .step-container {
        margin: 20px 0;
        padding: 15px;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    h4, h5, h6 {
        color: var(--accent-color);
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-weight: bold;
        background: linear-gradient(to right, var(--gradient-start), var(--gradient-end));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Container styling */
    div.stTabs [data-baseweb="tab-panel"] {
        padding: 1rem;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Plotly chart styling */
    div[data-testid="stPlotlyChart"] {
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1.5rem;
        background-color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    /* Animation styling */
    .animation-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    
    .animation-container img {
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
    }
    
    .animation-container img:hover {
        transform: scale(1.02);
    }
    
    /* Custom alert messages */
    .stAlert-success {
        background-color: var(--success-color);
        color: white;
        border-radius: 12px;
    }
    
    .stAlert-warning {
        background-color: var(--warning-color);
        color: white;
        border-radius: 12px;
    }
    
    .stAlert-error {
        background-color: var(--error-color);
        color: white;
        border-radius: 12px;
    }
    
    .stAlert-info {
        background-color: var(--info-color);
        color: white;
        border-radius: 12px;
    }
    
    /* Form elements styling */
    .stSelectbox > div,
    .stNumberInput > div,
    .stTextInput > div,
    .stFileUploader > div {
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div:hover,
    .stNumberInput > div:hover,
    .stTextInput > div:hover,
    .stFileUploader > div:hover {
        border: 1px solid var(--accent-color);
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.1);
    }
    
    /* Data frame styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stDataFrame"] table {
        border-radius: 12px;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] th {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--text-color);
        font-weight: 600;
    }
    
    /* Progress bar styling */
    progress {
        background-color: #E5E7EB;
        border-radius: 10px;
        height: 8px;
    }
    
    progress::-webkit-progress-bar {
        background-color: #E5E7EB;
        border-radius: 10px;
        height: 8px;
    }
    
    progress::-webkit-progress-value {
        background-image: linear-gradient(to right, var(--gradient-start), var(--gradient-end));
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Add ChromaWave-specific styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(to right, var(--gradient-start), var(--gradient-end));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.2);
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--secondary-color);
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.01em;
    }
    
    .card {
        background: var(--card-bg);
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.1);
        padding: 25px;
        margin-bottom: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.15);
    }
    
    .step-title {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: var(--primary-color);
        font-weight: 600;
        border-left: 4px solid var(--primary-color);
    }
    
    .highlight {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        padding: 5px 8px;
        border-radius: 6px;
        color: var(--primary-color);
        font-weight: 500;
    }
    
    .sidebar .decoration {
        margin: 30px 0;
        text-align: center;
    }
    
    .step-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        color: var(--primary-color);
        border-radius: 50%;
        width: 36px;
        height: 36px;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
        transition: all 0.3s ease;
    }
    
    .step-indicator:hover {
        transform: scale(1.1);
    }
    
    .complete {
        background: var(--success-color);
        color: white;
    }
    
    .current {
        background: linear-gradient(to right, var(--gradient-start), var(--gradient-end));
        color: white;
    }
    
    /* New feature badges */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(to right, var(--gradient-start), var(--gradient-end));
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
        box-shadow: 0 2px 5px rgba(99, 102, 241, 0.3);
    }
    
    /* Tooltip-style info boxes */
    .info-tooltip {
        position: relative;
        display: inline-block;
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-color);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .info-tooltip::before {
        content: 'ℹ️';
        margin-right: 8px;
        font-weight: bold;
    }
    
    /* Enhanced stat cards */
    .stat-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        padding: 15px;
        border-top: 4px solid var(--primary-color);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(to right, var(--gradient-start), var(--gradient-end));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: var(--text-color);
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None

# Variables for dataset comparison functionality
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}  # Store multiple datasets
if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = None  # Currently active dataset
if 'models' not in st.session_state:
    st.session_state.models = {}  # Store trained models for each dataset
if 'evaluation_results_by_dataset' not in st.session_state:
    st.session_state.evaluation_results_by_dataset = {}  # Store evaluation results for each dataset
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False  # Toggle for comparison mode
if 'model_results_by_dataset' not in st.session_state:
    st.session_state.model_results_by_dataset = {}  # Track model performances for comparison

# Main app header
st.markdown("<div class='main-header'>Financial Machine Learning Pipeline Pro</div>", unsafe_allow_html=True)

# Show welcome animation in the beginning
if st.session_state.step == 0:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Welcome to the Financial ML Pipeline App!</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='font-size: 1.1rem; line-height: 1.6;'>
        This professional application guides you through a complete machine learning pipeline for financial data analysis:
        
        <ol style='margin-left: 1.5rem;'>
            <li><span class='highlight'>Load financial data</span> from Kragle datasets or Yahoo Finance</li>
            <li><span class='highlight'>Preprocess and clean</span> your data</li>
            <li><span class='highlight'>Engineer relevant features</span> for better model performance</li>
            <li><span class='highlight'>Split data</span> into training and testing sets</li>
            <li><span class='highlight'>Train</span> a machine learning model</li>
            <li><span class='highlight'>Evaluate model performance</span> with metrics and visualizations</li>
            <li><span class='highlight'>Visualize results</span> and predictions</li>
        </ol>
        
        <p style='margin-top: 1rem;'>This intuitive pipeline helps you identify patterns and make predictions using financial data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Display finance GIF
        st.markdown(f"<img src='{load_animation_url('finance')}' width='100%' style='border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add quick start guide
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Quick Start Guide</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='display: flex; gap: 20px; flex-wrap: wrap;'>
        <div style='flex: 1; min-width: 200px; background-color: #E3F2FD; padding: 15px; border-radius: 5px;'>
            <h4 style='color: #1E88E5;'>📈 Yahoo Finance Data</h4>
            <p>Enter a stock ticker (e.g., AAPL, MSFT) to analyze stock market data and predict future prices.</p>
        </div>
        <div style='flex: 1; min-width: 200px; background-color: #E8F5E9; padding: 15px; border-radius: 5px;'>
            <h4 style='color: #43A047;'>📊 Custom Datasets</h4>
            <p>Upload your own CSV files to perform custom financial analysis using machine learning.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar for navigation and parameters
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #1E88E5;'>ML Pipeline Navigator</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Progress tracker
    if st.session_state.step > 0:
        progress_percent = min(100, int((st.session_state.step / 7) * 100))
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <p style="margin-bottom: 5px; font-size: 0.9rem; color: #666;">Pipeline Progress</p>
            <div style="background-color: #E8EAF6; border-radius: 10px; height: 10px; width: 100%;">
                <div style="background-color: #1E88E5; border-radius: 10px; height: 10px; width: {progress_percent}%;"></div>
            </div>
            <p style="text-align: right; font-size: 0.8rem; color: #666; margin-top: 5px;">{progress_percent}% Complete</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add professional step indicator in sidebar
        # Enhanced ChromaWave Steps Card
        st.markdown("<div class='card' style='padding: 20px; margin-top: 20px;'>", unsafe_allow_html=True)
        st.markdown("<div class='step-title'>Pipeline Steps</div>", unsafe_allow_html=True)
        
        # List the steps with detailed descriptions
        steps = [
            {"name": "Load Data", "description": "Import financial data from Yahoo Finance or CSV files"},
            {"name": "Preprocessing", "description": "Clean data and handle missing values"},
            {"name": "Feature Engineering", "description": "Create and select important features"},
            {"name": "Train/Test Split", "description": "Divide data for training and validation"},
            {"name": "Model Training", "description": "Train regression or classification models"},
            {"name": "Evaluation", "description": "Analyze model performance metrics"},
            {"name": "Results", "description": "Visualize predictions and insights"}
        ]
        
        # Create ChromaWave-styled step indicator
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 15px; margin-top: 15px;">
        """, unsafe_allow_html=True)
        
        for i, step in enumerate(steps, 1):
            # Determine the status of each step
            if i < st.session_state.step:
                status_class = "complete"
                icon = "✓"
                line_color = "var(--success-color)"  # Green for completed
                opacity = "1"
            elif i == st.session_state.step:
                status_class = "current"
                icon = str(i)
                line_color = "var(--gradient-start)"  # Primary for current
                opacity = "1"
            else:
                status_class = ""
                icon = str(i)
                line_color = "#E0E0E0"  # Gray for future
                opacity = "0.6"
            
            # Add gradient connector line
            connector_style = "display: none;" if i == len(steps) else ""
            
            st.markdown(f"""
            <div style="display: flex; position: relative; margin-bottom: 10px; opacity: {opacity}; transition: all 0.3s ease;">
                <div class="step-indicator {status_class}" style="z-index: 2; flex-shrink: 0;">{icon}</div>
                <div style="margin-left: 15px;">
                    <div style="font-weight: 600; font-size: 1rem;">{step['name']}</div>
                    <div style="font-size: 0.85rem; color: var(--text-color); opacity: 0.8; margin-top: 2px;">{step['description']}</div>
                </div>
                <div style="position: absolute; top: 36px; left: 18px; height: 30px; width: 2px; background-color: {line_color}; {connector_style}"></div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add ML Pipeline options
        st.markdown("<div class='step-title' style='margin-top: 30px;'>Options & Settings</div>", unsafe_allow_html=True)
        
        # Model selection
        model_options = {
            "Linear Regression": "For continuous target prediction",
            "Logistic Regression": "For binary classification tasks",
            "K-Means Clustering": "For identifying patterns"
        }
        
        selected_model = st.selectbox(
            "Model Type",
            options=list(model_options.keys()),
            index=0,
            format_func=lambda x: f"{x} - {model_options[x]}"
        )
        
        # Feature engineering options
        if st.session_state.step >= 2:
            st.markdown("<div class='info-tooltip'>Choose technical indicators to include in your model</div>", unsafe_allow_html=True)
            tech_indicators = st.multiselect(
                "Technical Indicators",
                options=["Moving Averages", "RSI", "MACD", "Bollinger Bands", "Volume Features"],
                default=["Moving Averages", "RSI"],
                help="Select technical indicators to generate for financial data"
            )
        
        # Data preprocessing options
        if st.session_state.step >= 1:
            preprocessing_options = st.expander("Advanced Preprocessing", expanded=False)
            with preprocessing_options:
                missing_val_method = st.radio(
                    "Handle Missing Values",
                    options=["Drop rows with missing values", "Fill numeric with mean", "Fill with median", "Fill with mode"],
                    index=0
                )
                handle_outliers = st.checkbox("Remove Outliers", value=False, 
                                            help="Use IQR method to detect and remove outliers")
                normalize_data = st.checkbox("Normalize Features", value=True,
                                           help="Scale numeric features to 0-1 range")
        
        # Model hyperparameters
        if st.session_state.step >= 4:
            model_params = st.expander("Model Hyperparameters", expanded=False)
            with model_params:
                if selected_model == "Linear Regression":
                    fit_intercept = st.checkbox("Fit Intercept", value=True)
                    normalize = st.checkbox("Normalize", value=False)
                elif selected_model == "Logistic Regression":
                    C_value = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01, 
                                      help="Lower values specify stronger regularization")
                    max_iter = st.number_input("Max Iterations", 100, 1000, 200, 50)
                elif selected_model == "K-Means Clustering":
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3, 1)
                    n_init = st.number_input("Number of Initializations", 5, 20, 10, 1)
        
        # Add preset templates
        st.markdown("<div class='step-title' style='margin-top: 20px;'>Quick Presets</div>", unsafe_allow_html=True)
        preset_cols = st.columns(3)
        with preset_cols[0]:
            st.button("Stock Price Prediction", use_container_width=True)
        with preset_cols[1]:
            st.button("Technical Analysis", use_container_width=True)
        with preset_cols[2]:
            st.button("Volatility Model", use_container_width=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card' style='padding: 15px;'>", unsafe_allow_html=True)
    # Data source selection
    st.markdown("<div class='step-title'>1. Choose Data Source</div>", unsafe_allow_html=True)
    data_source = st.radio("Select data source:", ["Upload Kragle Dataset", "Fetch Yahoo Finance Data"])
    
    if data_source == "Upload Kragle Dataset":
        # Dataset naming options for comparison functionality
        dataset_name = st.text_input("Dataset Name:", "Dataset 1", 
                                     help="Provide a name to identify this dataset for comparison")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            # Option to add to comparison collection
            add_to_comparison = st.checkbox("Add to dataset comparison collection", 
                                           value=False,
                                           help="Enable to add this dataset to your collection for later comparison")
            
            btn_load = st.button("📂 Load Dataset", use_container_width=True)
            if btn_load:
                try:
                    with st.spinner("Loading and processing dataset..."):
                        data = pd.read_csv(uploaded_file)
                        
                        # If adding to comparison collection
                        if add_to_comparison:
                            st.session_state.datasets[dataset_name] = {
                                'data': data,
                                'source': 'kragle',
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'rows': data.shape[0],
                                'columns': data.shape[1]
                            }
                            st.session_state.active_dataset = dataset_name
                            show_notification("success", f"Dataset '{dataset_name}' added to comparison collection!")
                        
                        # Set current working dataset
                        st.session_state.data = data
                        st.session_state.data_source = "kragle"
                        st.session_state.step = 1
                        show_notification("success", "Data successfully loaded! You can now proceed to preprocessing.")
                        st.rerun()
                except Exception as e:
                    show_notification("error", f"Error loading file: {str(e)}")
    
    elif data_source == "Fetch Yahoo Finance Data":
        # Dataset naming for comparison functionality
        dataset_name = st.text_input("Dataset Name:", "Yahoo Finance Data", 
                                     help="Provide a name to identify this dataset for comparison")
        
        ticker = st.text_input("Enter Stock Ticker Symbol:", placeholder="e.g., AAPL, MSFT, GOOGL")
        
        # Create two columns for date inputs for better spacing
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date:", datetime.now() - timedelta(days=365*2))
        with col2:
            end_date = st.date_input("End Date:", datetime.now())
        
        if ticker:
            # Option to add to comparison collection
            add_to_comparison = st.checkbox("Add to dataset comparison collection", 
                                           value=False,
                                           help="Enable to add this dataset to your collection for later comparison")
            
            btn_fetch = st.button("📈 Fetch Stock Data", use_container_width=True)
            if btn_fetch:
                with st.spinner("Validating ticker and fetching data..."):
                    if validate_stock_ticker(ticker):
                        try:
                            # Download data
                            data = yf.download(ticker, start=start_date, end=end_date)
                            if not data.empty:
                                # Reset index to make Date a column
                                data = data.reset_index()
                                
                                # If adding to comparison collection
                                if add_to_comparison:
                                    final_name = f"{dataset_name} - {ticker}"
                                    st.session_state.datasets[final_name] = {
                                        'data': data,
                                        'source': 'yahoo',
                                        'ticker': ticker,
                                        'start_date': start_date.strftime("%Y-%m-%d"),
                                        'end_date': end_date.strftime("%Y-%m-%d"),
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rows': data.shape[0],
                                        'columns': data.shape[1]
                                    }
                                    st.session_state.active_dataset = final_name
                                    show_notification("success", f"Dataset '{final_name}' added to comparison collection!")
                                
                                # Set current working dataset
                                st.session_state.data = data
                                st.session_state.data_source = "yahoo"
                                st.session_state.step = 1
                                show_notification("success", f"Successfully fetched data for {ticker}!")
                                st.rerun()
                            else:
                                show_notification("error", "No data available for the selected ticker and date range.")
                        except Exception as e:
                            show_notification("error", f"Error fetching data: {str(e)}")
                    else:
                        show_notification("error", "Invalid ticker symbol. Please enter a valid stock symbol.")
        else:
            st.info("Enter a stock ticker symbol to proceed.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card' style='padding: 15px; margin-top: 20px;'>", unsafe_allow_html=True)
    # ML model selection
    st.markdown("<div class='step-title'>2. Choose ML Model</div>", unsafe_allow_html=True)
    
    # Create model selection cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        linear_selected = st.button("📊 Linear", 
                                    disabled=st.session_state.step < 1,
                                    use_container_width=True)
    with col2:
        logistic_selected = st.button("📉 Logistic", 
                                      disabled=st.session_state.step < 1,
                                      use_container_width=True)
    with col3:
        kmeans_selected = st.button("🔍 Clusters", 
                                    disabled=st.session_state.step < 1,
                                    use_container_width=True)
    
    # Determine which model is selected
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "Linear Regression"
    
    if linear_selected:
        st.session_state.model_choice = "Linear Regression"
    elif logistic_selected:
        st.session_state.model_choice = "Logistic Regression"
    elif kmeans_selected:
        st.session_state.model_choice = "K-Means Clustering"
    
    # Display currently selected model
    st.markdown(f"""
    <div style='background-color: #E3F2FD; padding: 10px; border-radius: 5px; margin-top: 10px;'>
        <p style='margin: 0; font-weight: 500;'>Selected Model: <span style='color: #1E88E5;'>{st.session_state.model_choice}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model description - changes based on selection
    model_descriptions = {
        "Linear Regression": "Predicts continuous values based on linear relationships between variables. Ideal for price forecasting.",
        "Logistic Regression": "Classifies data into categories. Great for predicting binary outcomes like stock up/down movements.",
        "K-Means Clustering": "Groups similar data points. Useful for identifying market segments or trading patterns."
    }
    
    st.markdown(f"""
    <div style='font-size: 0.9rem; margin-top: 10px;'>
        {model_descriptions[st.session_state.model_choice]}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Reset application - at the bottom
    st.markdown("<div style='margin-top: 30px;'>", unsafe_allow_html=True)
    if st.button("🔄 Reset Application", use_container_width=True):
        for key in st.session_state.keys():
            if key != 'step':
                st.session_state[key] = None
        st.session_state.step = 0
        st.session_state.model_choice = "Linear Regression"
        show_notification("info", "Application has been reset!")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Main content based on current step
if st.session_state.step >= 1:
    # Check if we have datasets for comparison
    if len(st.session_state.datasets) > 1:
        with st.expander("📊 Dataset Comparison Options", expanded=False):
            st.markdown("""
            <div style="background-color: #F3F9FE; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="color: #1E88E5; margin-top: 0;">Dataset Comparison Tool</h4>
                <p style="margin-bottom: 5px;">You have multiple datasets available for comparison. Select options below to compare datasets.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Toggle comparison mode
            comparison_mode = st.checkbox("Enable Dataset Comparison Mode", 
                                         value=st.session_state.comparison_mode,
                                         help="Toggle to compare multiple datasets side by side")
            
            if comparison_mode != st.session_state.comparison_mode:
                st.session_state.comparison_mode = comparison_mode
                st.rerun()
                
            if comparison_mode:
                # Select datasets to compare
                datasets_to_compare = st.multiselect("Select Datasets to Compare:", 
                                                    options=list(st.session_state.datasets.keys()),
                                                    default=[list(st.session_state.datasets.keys())[0]])
                
                if datasets_to_compare:
                    comparison_tabs = st.tabs(["Dataset Overview", "Statistical Comparison", "Feature Comparison"])
                    
                    with comparison_tabs[0]:
                        st.subheader("Dataset Overview")
                        
                        # Create comparison dataframe
                        comparison_data = []
                        for ds_name in datasets_to_compare:
                            ds_info = st.session_state.datasets[ds_name]
                            comparison_data.append({
                                "Dataset": ds_name,
                                "Source": ds_info['source'].capitalize(),
                                "Rows": ds_info['rows'],
                                "Columns": ds_info['columns'],
                                "Loaded On": ds_info['timestamp']
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Dataset preview
                        if st.checkbox("Show Dataset Previews"):
                            for ds_name in datasets_to_compare:
                                st.markdown(f"##### Preview of {ds_name}")
                                st.dataframe(st.session_state.datasets[ds_name]['data'].head(5), use_container_width=True)
                    
                    with comparison_tabs[1]:
                        st.subheader("Statistical Comparison")
                        
                        selected_stat = st.selectbox("Select Statistic to Compare:", 
                                                    ["Shape", "Data Types", "Missing Values", "Statistical Summary"])
                        
                        if selected_stat == "Shape":
                            # Compare shapes
                            shape_data = []
                            for ds_name in datasets_to_compare:
                                ds_data = st.session_state.datasets[ds_name]['data']
                                shape_data.append({
                                    "Dataset": ds_name,
                                    "Rows": ds_data.shape[0],
                                    "Columns": ds_data.shape[1],
                                    "Size (KB)": round(ds_data.memory_usage(deep=True).sum() / 1024, 2)
                                })
                            shape_df = pd.DataFrame(shape_data)
                            st.dataframe(shape_df, use_container_width=True)
                            
                            # Shape visualization
                            fig = px.bar(shape_df, x="Dataset", y=["Rows", "Columns"], barmode="group",
                                        title="Dataset Dimensions Comparison",
                                        labels={"value": "Count", "variable": "Dimension"})
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif selected_stat == "Data Types":
                            # Show data types from each dataset - fix Arrow conversion issue
                            for ds_name in datasets_to_compare:
                                st.markdown(f"##### Data Types in {ds_name}")
                                ds_data = st.session_state.datasets[ds_name]['data']
                                # Convert dtypes to strings to avoid Arrow conversion issues
                                dtypes_dict = {col: str(dtype) for col, dtype in zip(ds_data.columns, ds_data.dtypes)}
                                dtypes_df = pd.DataFrame(list(dtypes_dict.items()), columns=["Column", "Data Type"])
                                st.dataframe(dtypes_df, use_container_width=True)
                        
                        elif selected_stat == "Missing Values":
                            # Compare missing values - Fix pd.DataFrame issue
                            for ds_name in datasets_to_compare:
                                st.markdown(f"##### Missing Values in {ds_name}")
                                ds_data = st.session_state.datasets[ds_name]['data']
                                missing_values = ds_data.isnull().sum()
                                missing_percent = (ds_data.isnull().sum() / len(ds_data)) * 100
                                
                                # Create with explicit index
                                missing_data = []
                                for col, miss, perc in zip(missing_values.index, missing_values.values, missing_percent.values):
                                    missing_data.append({
                                        "Column": col,
                                        "Missing Values": int(miss),
                                        "Percentage": round(float(perc), 2)
                                    })
                                
                                missing_df = pd.DataFrame(missing_data)
                                st.dataframe(missing_df, use_container_width=True)
                        
                        elif selected_stat == "Statistical Summary":
                            # Tabbed statistical summaries
                            summary_tabs = st.tabs(datasets_to_compare)
                            for i, ds_name in enumerate(datasets_to_compare):
                                with summary_tabs[i]:
                                    ds_data = st.session_state.datasets[ds_name]['data']
                                    st.dataframe(ds_data.describe(), use_container_width=True)
                    
                    with comparison_tabs[2]:
                        st.subheader("Feature Comparison")
                        
                        # Find common numeric columns for comparison
                        common_numeric_cols = set()
                        first = True
                        
                        for ds_name in datasets_to_compare:
                            ds_data = st.session_state.datasets[ds_name]['data']
                            numeric_cols = set(ds_data.select_dtypes(include=['number']).columns)
                            
                            if first:
                                common_numeric_cols = numeric_cols
                                first = False
                            else:
                                common_numeric_cols = common_numeric_cols.intersection(numeric_cols)
                        
                        common_numeric_cols = list(common_numeric_cols)
                        
                        if common_numeric_cols:
                            col_to_compare = st.selectbox("Select column to compare:", common_numeric_cols)
                            
                            if col_to_compare:
                                # Create comparison dataframe
                                comparison_data = []
                                for ds_name in datasets_to_compare:
                                    ds_data = st.session_state.datasets[ds_name]['data']
                                    series = ds_data[col_to_compare]
                                    comparison_data.append({
                                        "Dataset": ds_name,
                                        "Mean": series.mean(),
                                        "Median": series.median(),
                                        "Std Dev": series.std(),
                                        "Min": series.min(),
                                        "Max": series.max()
                                    })
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # Visualization
                                fig = px.box(title=f"Distribution of {col_to_compare} Across Datasets")
                                
                                for ds_name in datasets_to_compare:
                                    ds_data = st.session_state.datasets[ds_name]['data']
                                    fig.add_trace(go.Box(
                                        y=ds_data[col_to_compare],
                                        name=ds_name,
                                        boxmean=True
                                    ))
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No common numeric columns found across the selected datasets.")
    
                # Switch between datasets
                st.markdown("---")
                set_active_dataset = st.selectbox("Set Active Dataset for Analysis:", 
                                                 options=list(st.session_state.datasets.keys()),
                                                 index=list(st.session_state.datasets.keys()).index(st.session_state.active_dataset)
                                                 if st.session_state.active_dataset in st.session_state.datasets else 0)
                
                if set_active_dataset != st.session_state.active_dataset:
                    st.session_state.active_dataset = set_active_dataset
                    st.session_state.data = st.session_state.datasets[set_active_dataset]['data']
                    st.session_state.data_source = st.session_state.datasets[set_active_dataset]['source']
                    show_notification("info", f"Switched to dataset: {set_active_dataset}")
                    st.rerun()
    
    # Step 1: Data Preview
    if st.session_state.step == 1:
        st.header("Step 1: Data Preview")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Display data source info
            if st.session_state.data_source == "kragle":
                st.info("Data loaded from Kragle dataset.")
            else:
                st.info("Data loaded from Yahoo Finance.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Overview")
                st.write(f"Number of rows: {data.shape[0]}")
                st.write(f"Number of columns: {data.shape[1]}")
                
                # Display data types - fix similar Arrow conversion issue
                st.subheader("Data Types")
                dtypes_dict = {col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)}
                df_types = pd.DataFrame(list(dtypes_dict.items()), columns=["Column", "Data Type"])
                st.dataframe(df_types, use_container_width=True)
            
            with col2:
                st.subheader("Statistical Summary")
                st.dataframe(data.describe())
            
            # Preview the dataset
            st.subheader("Data Preview")
            st.dataframe(data.head(10))
            
            # Continue to preprocessing - styled card container
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-top: 30px;">
                <h4 style="margin-top: 0; margin-bottom: 15px; color: #424242;">Continue to Next Step</h4>
                <p style="color: #757575; margin-bottom: 15px;">The data has been loaded successfully. Click the button below to proceed to data preprocessing.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Continue to Preprocessing →", use_container_width=True):
                st.session_state.step = 2
                show_notification("success", "Moving to data preprocessing!")
                st.rerun()
    
    # Step 2: Data Preprocessing
    elif st.session_state.step == 2:
        st.header("Step 2: Data Preprocessing")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Missing Values Analysis")
                missing_values = data.isnull().sum()
                missing_percent = (missing_values / len(data)) * 100
                
                # Fix similar Arrow conversion issue
                missing_data = []
                for col, miss, perc in zip(missing_values.index, missing_values.values, missing_percent.values):
                    missing_data.append({
                        "Column": col,
                        "Missing Values": int(miss),
                        "Percentage": round(float(perc), 2)
                    })
                
                missing_df = pd.DataFrame(missing_data)
                st.dataframe(missing_df, use_container_width=True)
                
                # Display missing values chart
                fig = plot_missing_values(data)
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Preprocessing Options")
                
                handle_missing = st.selectbox(
                    "Handle Missing Values:",
                    ["Drop rows with missing values", "Fill numeric with mean", "Fill with median", "Fill with mode"]
                )
                
                handle_outliers = st.checkbox("Remove outliers (IQR method)")
                normalize_data = st.checkbox("Normalize numeric features")
                
            # Styled button for better UX
            st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
            if st.button("Apply Preprocessing", use_container_width=True, type="primary"):
                with st.spinner("Preprocessing data... This may take a moment."):
                    try:
                        # Add a progress bar for better UX
                        progress_bar = st.progress(0)
                        st.markdown("##### Processing stages:")
                        
                        # Stage 1: Preparing data
                        st.info("1️⃣ Preparing dataset...")
                        progress_bar.progress(25)
                        time.sleep(0.5)  # Simulate processing time
                        
                        # Stage 2: Handling missing values
                        st.info("2️⃣ Handling missing values...")
                        progress_bar.progress(50)
                        time.sleep(0.5)  # Simulate processing time
                        
                        # Stage 3: Processing outliers if needed
                        if handle_outliers:
                            st.info("3️⃣ Handling outliers...")
                            progress_bar.progress(75)
                            time.sleep(0.5)  # Simulate processing time
                        
                        # Final stage: Normalizing if needed
                        if normalize_data:
                            st.info("4️⃣ Normalizing features...")
                            
                        progress_bar.progress(100)
                        
                        # Actually do the preprocessing
                        preprocessed_data = preprocess_data(
                            data, 
                            handle_missing=handle_missing,
                            handle_outliers=handle_outliers,
                            normalize=normalize_data
                        )
                        
                        # Show comparison of before and after
                        st.subheader("Before vs After Preprocessing")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Before:**")
                            st.dataframe(data.head(5))
                            st.write(f"Shape: {data.shape}")
                        
                        with col2:
                            st.markdown("**After:**")
                            st.dataframe(preprocessed_data.head(5))
                            st.write(f"Shape: {preprocessed_data.shape}")
                        
                        # Update session state
                        st.session_state.data = preprocessed_data
                        st.session_state.step = 3
                        
                        show_notification("success", "Data preprocessing completed successfully!")
                        st.rerun()
                    except Exception as e:
                        show_notification("error", f"Error during preprocessing: {str(e)}")
    
    # Step 3: Feature Engineering
    elif st.session_state.step == 3:
        st.header("Step 3: Feature Engineering")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            st.subheader("Select Target Variable and Features")
            
            # For Yahoo Finance data, create some technical indicators automatically
            if st.session_state.data_source == "yahoo":
                st.info("For stock data, we'll calculate some common technical indicators.")
                
                create_indicators = st.checkbox("Create technical indicators", value=True)
                
                if create_indicators:
                    indicators = st.multiselect(
                        "Select technical indicators to create:",
                        ["Moving Averages", "RSI", "MACD", "Bollinger Bands", "Volume Features"],
                        default=["Moving Averages", "RSI"]
                    )
            
            # Select target variable
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
            
            # If it's Yahoo Finance data, suggest 'Close' as target by default
            # Set a safe default for target variable
            default_target = None
            if st.session_state.data is not None and len(numeric_columns) > 0:
                if st.session_state.data_source == "yahoo" and 'Close' in numeric_columns:
                    default_target = 'Close'
                else:
                    default_target = numeric_columns[0]
            
            target_variable = st.selectbox(
                "Select Target Variable:",
                numeric_columns,
                index=numeric_columns.index(default_target) if default_target in numeric_columns else 0
            )
            
            # Select features (excluding target)
            available_features = [col for col in data.columns if col != target_variable]
            selected_features = st.multiselect(
                "Select Features to Use:",
                available_features,
                default=available_features[:min(5, len(available_features))]
            )
            
            if st.button("Apply Feature Engineering"):
                with st.spinner("Engineering features..."):
                    try:
                        # Apply feature engineering
                        if st.session_state.data_source == "yahoo" and create_indicators:
                            features_df, target_series = feature_engineering(
                                data,
                                target_column=target_variable,
                                feature_columns=selected_features,
                                create_indicators=indicators if create_indicators else []
                            )
                        else:
                            features_df, target_series = feature_engineering(
                                data,
                                target_column=target_variable,
                                feature_columns=selected_features
                            )
                        
                        # Show results
                        st.subheader("Engineered Features")
                        st.dataframe(features_df.head())
                        
                        # Display correlation matrix if not too large
                        if features_df.shape[1] <= 20:
                            st.subheader("Feature Correlation Matrix")
                            fig = px.imshow(
                                features_df.corr(),
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale='RdBu_r'
                            )
                            st.plotly_chart(fig)
                        
                        # Update session state
                        st.session_state.features = features_df
                        st.session_state.target = target_series
                        st.session_state.step = 4
                        
                        show_notification("success", "Feature engineering completed successfully!")
                        st.rerun()
                    except Exception as e:
                        show_notification("error", f"Error during feature engineering: {str(e)}")
    
    # Step 4: Train/Test Split
    elif st.session_state.step == 4:
        st.header("Step 4: Train/Test Split")
        
        if st.session_state.features is not None and st.session_state.target is not None:
            st.subheader("Configure Data Split")
            
            test_size = st.slider("Test Size (% of data):", 10, 40, 20)
            random_state = st.number_input("Random State (for reproducibility):", 0, 100, 42)
            
            if st.button("Split Data"):
                with st.spinner("Splitting data into training and testing sets..."):
                    try:
                        # Split the data
                        X_train, X_test, y_train, y_test = split_data(
                            st.session_state.features,
                            st.session_state.target,
                            test_size=test_size/100,
                            random_state=random_state
                        )
                        
                        # Display split information
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Training Samples", f"{X_train.shape[0]}")
                            st.write(f"Number of features: {X_train.shape[1]}")
                        
                        with col2:
                            st.metric("Testing Samples", f"{X_test.shape[0]}")
                            st.write(f"Number of features: {X_test.shape[1]}")
                        
                        # Visualize the split
                        st.subheader("Train/Test Split Visualization")
                        fig = plot_split(X_train.shape[0], X_test.shape[0])
                        st.plotly_chart(fig)
                        
                        # Update session state
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.step = 5
                        
                        show_notification("success", "Data split completed successfully!")
                        st.rerun()
                    except Exception as e:
                        show_notification("error", f"Error during data splitting: {str(e)}")
    
    # Step 5: Model Training
    elif st.session_state.step == 5:
        st.header("Step 5: Model Training")
        
        if all(x is not None for x in [st.session_state.X_train, st.session_state.X_test, 
                                      st.session_state.y_train, st.session_state.y_test]):
            
            # Use model_choice instead of model_type
            st.subheader(f"Train {st.session_state.model_choice} Model")
            
            # Model parameters based on model type
            if st.session_state.model_choice == "Linear Regression":
                # Create a card layout for parameters
                st.markdown("""
                <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px;">
                    <h4 style="margin-top: 0; color: #1E88E5;">Model Parameters</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fit_intercept = st.checkbox("Fit Intercept", value=True, 
                                              help="Whether to calculate the intercept for this model")
                with col2:
                    # Note: 'normalize' parameter was removed in scikit-learn 1.0
                    # Instead, we'll inform users to use preprocessing techniques
                    st.markdown("""
                    <div style="background-color: #F3F4F6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p style="margin: 0; font-size: 0.9rem;">💡 <strong>Note:</strong> If you need to normalize data, use the preprocessing step instead.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                params = {
                    "fit_intercept": fit_intercept
                    # normalize parameter has been removed in scikit-learn 1.0+
                }
            
            elif st.session_state.model_choice == "Logistic Regression":
                # Create a card layout for parameters
                st.markdown("""
                <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px;">
                    <h4 style="margin-top: 0; color: #1E88E5;">Model Parameters</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"],
                                         help="Specify the norm of the penalty")
                    C = st.number_input("Regularization Strength (C)", 0.01, 10.0, 1.0,
                                      help="Inverse of regularization strength; smaller values specify stronger regularization")
                with col2:
                    max_iter = st.number_input("Maximum Iterations", 100, 1000, 100, 100,
                                             help="Maximum number of iterations for the solver to converge")
                    solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag"],
                                        help="Algorithm to use in the optimization problem")
                
                params = {
                    "penalty": penalty,
                    "C": C,
                    "max_iter": max_iter,
                    "solver": solver
                }
                
            elif st.session_state.model_choice == "K-Means Clustering":
                # Create a card layout for parameters
                st.markdown("""
                <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px;">
                    <h4 style="margin-top: 0; color: #1E88E5;">Model Parameters</h4>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3,
                                         help="The number of clusters to form and centroids to generate")
                with col2:
                    init = st.selectbox("Initialization Method", ["k-means++", "random"],
                                      help="Method for initialization of centroids")
                    n_init = st.number_input("Number of Initializations", 1, 20, 10,
                                           help="Number of times the algorithm will be run with different centroid seeds")
                
                params = {
                    "n_clusters": n_clusters,
                    "init": init,
                    "n_init": n_init,
                    "random_state": 42
                }
            
            # Professional styled train button
            st.markdown("<div style='margin-top: 30px;'>", unsafe_allow_html=True)
            train_btn = st.button("🚀 Train Model", 
                                  use_container_width=True, 
                                  type="primary")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if train_btn:
                with st.spinner(f"Training {st.session_state.model_choice} model..."):
                    try:
                        # Add a progress bar for better UX
                        progress_bar = st.progress(0)
                        st.markdown("##### Training progress:")
                        
                        # Stage 1: Preparing data
                        st.info("⚙️ Preparing data for training...")
                        progress_bar.progress(25)
                        time.sleep(0.7)  # Simulate processing time
                        
                        # Stage 2: Initializing model
                        st.info("🧠 Initializing model architecture...")
                        progress_bar.progress(50)
                        time.sleep(0.7)  # Simulate processing time
                        
                        # Stage 3: Fitting model
                        st.info("📊 Fitting model to training data...")
                        progress_bar.progress(75)
                        time.sleep(0.7)  # Simulate processing time
                        
                        # Stage 4: Finalizing model
                        st.info("✅ Finalizing and optimizing model...")
                        progress_bar.progress(100)
                        time.sleep(0.5)  # Simulate processing time
                        
                        # Train the model
                        model = train_model(
                            X_train=st.session_state.X_train,
                            y_train=st.session_state.y_train,
                            model_type=st.session_state.model_choice,
                            params=params
                        )
                        
                        # Display model information in a professional card layout
                        st.markdown("""
                        <div style="background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 25px; margin-top: 30px;">
                            <h3 style="color: #1E88E5; margin-top: 0; margin-bottom: 20px;">Model Information</h3>
                        """, unsafe_allow_html=True)
                        
                        if st.session_state.model_choice == "Linear Regression":
                            # Create columns for better layout
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("<h4 style='color: #424242;'>Model Coefficients</h4>", unsafe_allow_html=True)
                                coefficient_df = pd.DataFrame(
                                    {"Feature": st.session_state.X_train.columns, 
                                     "Coefficient": model.coef_}
                                ).sort_values(by="Coefficient", ascending=False)
                                st.dataframe(coefficient_df, use_container_width=True)
                            
                            with col2:
                                st.markdown("<h4 style='color: #424242;'>Model Parameters</h4>", unsafe_allow_html=True)
                                st.metric("Intercept", f"{model.intercept_:.4f}")
                                st.metric("Number of Features", f"{len(model.coef_)}")
                            
                            # Feature importance visualization
                            st.markdown("<h4 style='color: #424242; margin-top: 20px;'>Feature Importance</h4>", unsafe_allow_html=True)
                            
                            # Make sure we have feature names for visualization
                            feature_names = list(st.session_state.X_train.columns) if st.session_state.X_train is not None else []
                            if feature_names:
                                fig = plot_feature_importance(model, feature_names)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Feature importance visualization requires feature names. No feature data available.")
                            
                        elif st.session_state.model_choice == "Logistic Regression":
                            if hasattr(model, 'coef_'):
                                # Create columns for better layout
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("<h4 style='color: #424242;'>Model Parameters</h4>", unsafe_allow_html=True)
                                    st.metric("Intercept", f"{model.intercept_[0]:.4f}")
                                    st.metric("Number of Features", f"{model.coef_.shape[1]}")
                                
                                with col2:
                                    st.markdown("<h4 style='color: #424242;'>Decision Boundary</h4>", unsafe_allow_html=True)
                                    st.write("A positive coefficient means the feature increases the probability of the positive class.")
                                    st.write("A negative coefficient means the feature decreases the probability.")
                                
                                # Feature importance visualization
                                st.markdown("<h4 style='color: #424242; margin-top: 20px;'>Feature Importance</h4>", unsafe_allow_html=True)
                                # Make sure we have feature names for visualization
                                feature_names = list(st.session_state.X_train.columns) if st.session_state.X_train is not None else []
                                if feature_names:
                                    fig = plot_feature_importance(model, feature_names)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("Feature importance visualization requires feature names. No feature data available.")
                            
                        elif st.session_state.model_choice == "K-Means Clustering":
                            # Create columns for better layout
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("<h4 style='color: #424242;'>Clustering Statistics</h4>", unsafe_allow_html=True)
                                st.metric("Number of Clusters", f"{params.get('n_clusters', 'Not specified')}")
                                st.metric("Initialization Method", f"{params.get('init', 'k-means++')}")
                            
                            with col2:
                                st.markdown("<h4 style='color: #424242;'>Model Details</h4>", unsafe_allow_html=True)
                                st.metric("Number of Iterations", f"{model.n_iter_}")
                                st.metric("Inertia", f"{model.inertia_:.2f}")
                            
                            # Cluster centers information
                            st.markdown("<h4 style='color: #424242; margin-top: 20px;'>Cluster Centers</h4>", unsafe_allow_html=True)
                            centers_df = pd.DataFrame(
                                model.cluster_centers_, 
                                columns=st.session_state.X_train.columns
                            )
                            centers_df.index.name = "Cluster"
                            centers_df.index = [f"Cluster {i}" for i in range(len(centers_df))]
                            st.dataframe(centers_df, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Update session state
                        st.session_state.model = model
                        st.session_state.step = 6
                        
                        # Professional success message
                        st.markdown("""
                        <div style="background-color: #E8F5E9; border-left: 5px solid #4CAF50; padding: 15px; margin-top: 25px; border-radius: 4px;">
                            <h4 style="color: #2E7D32; margin-top: 0; margin-bottom: 10px;">✅ Model Trained Successfully!</h4>
                            <p style="color: #2E7D32; margin-bottom: 0;">Your {0} model has been trained and is ready for evaluation. Click below to proceed to the next step.</p>
                        </div>
                        """.format(st.session_state.model_choice), unsafe_allow_html=True)
                        
                        # Continue button
                        st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
                        if st.button("Continue to Model Evaluation →", use_container_width=True):
                            st.rerun()
                    except Exception as e:
                        show_notification("error", f"Error during model training: {str(e)}")
    
    # Step 6: Model Evaluation
    elif st.session_state.step == 6:
        st.header("Step 6: Model Evaluation")
        
        if st.session_state.model is not None:
            with st.spinner("Evaluating model..."):
                try:
                    # Add a progress bar for better UX
                    progress_bar = st.progress(0)
                    st.markdown("##### Evaluation progress:")
                    
                    # Stage 1: Preparing test data
                    st.info("🔍 Preparing test data...")
                    progress_bar.progress(25)
                    time.sleep(0.5)  # Simulate processing time
                    
                    # Stage 2: Making predictions
                    st.info("🧮 Making predictions...")
                    progress_bar.progress(50)
                    time.sleep(0.5)  # Simulate processing time
                    
                    # Stage 3: Calculating metrics
                    st.info("📏 Calculating performance metrics...")
                    progress_bar.progress(75)
                    time.sleep(0.5)  # Simulate processing time
                    
                    # Final stage: Finalizing evaluation
                    st.info("✅ Finalizing evaluation results...")
                    progress_bar.progress(100)
                    time.sleep(0.3)  # Simulate processing time
                    
                    # Evaluate model using the model_choice from session state
                    evaluation_results, predictions = evaluate_model(
                        model=st.session_state.model,
                        X_test=st.session_state.X_test,
                        y_test=st.session_state.y_test,
                        model_type=st.session_state.model_choice
                    )
                    
                    # Store evaluation results for dataset comparison if in comparison mode
                    if st.session_state.active_dataset and len(st.session_state.datasets) > 0:
                        st.session_state.evaluation_results_by_dataset[st.session_state.active_dataset] = {
                            'evaluation_results': evaluation_results,
                            'predictions': predictions,
                            'model_type': st.session_state.model_choice,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        # Also track performance metrics for comparison
                        st.session_state.model_results_by_dataset[st.session_state.active_dataset] = evaluation_results
                    
                    # Display evaluation metrics in a professional card
                    st.markdown("""
                    <div style="background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 25px; margin-top: 30px;">
                        <h3 style="color: #1E88E5; margin-top: 0; margin-bottom: 20px;">Model Performance Metrics</h3>
                    """, unsafe_allow_html=True)
                    
                    if st.session_state.model_choice == "Linear Regression":
                        # Create a metrics card with professional styling
                        st.markdown("""
                        <div style="background-color: #F5F7FF; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <h4 style="color: #424242; margin-top: 0;">Regression Metrics</h4>
                            <p style="color: #757575; font-size: 0.9rem; margin-bottom: 15px;">
                                These metrics evaluate how well the model's predictions match the actual values.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display metrics in an organized layout
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("R² Score", f"{evaluation_results['r2']:.4f}", 
                                     delta="Higher is better", delta_color="normal")
                            st.markdown("""
                            <div style="font-size: 0.8rem; color: #757575;">
                                Proportion of variance in the dependent variable that is predictable.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # For MSE, a lower value is better, so we use a negative delta
                            st.metric("Mean Squared Error", f"{evaluation_results['mse']:.4f}", 
                                     delta="Lower is better", delta_color="inverse")
                            st.markdown("""
                            <div style="font-size: 0.8rem; color: #757575;">
                                Average of the squares of the errors between predicted and actual values.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            # For MAE, a lower value is better, so we use a negative delta
                            st.metric("Mean Absolute Error", f"{evaluation_results['mae']:.4f}", 
                                     delta="Lower is better", delta_color="inverse")
                            st.markdown("""
                            <div style="font-size: 0.8rem; color: #757575;">
                                Average of the absolute differences between predicted and actual values.
                            </div>
                            """, unsafe_allow_html=True)
                    
                    elif st.session_state.model_choice == "Logistic Regression":
                        # Create a metrics card with professional styling
                        st.markdown("""
                        <div style="background-color: #F5F7FF; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <h4 style="color: #424242; margin-top: 0;">Classification Metrics</h4>
                            <p style="color: #757575; font-size: 0.9rem; margin-bottom: 15px;">
                                These metrics evaluate how well the model classifies data points into the correct categories.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display metrics in an organized layout
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy", f"{evaluation_results['accuracy']:.4f}", 
                                     delta="Higher is better", delta_color="normal")
                            st.markdown("""
                            <div style="font-size: 0.8rem; color: #757575;">
                                Proportion of correct predictions among the total number of cases.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Precision", f"{evaluation_results['precision']:.4f}", 
                                     delta="Higher is better", delta_color="normal")
                            st.markdown("""
                            <div style="font-size: 0.8rem; color: #757575;">
                                Proportion of positive identifications that were actually correct.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.metric("Recall", f"{evaluation_results['recall']:.4f}", 
                                     delta="Higher is better", delta_color="normal")
                            st.markdown("""
                            <div style="font-size: 0.8rem; color: #757575;">
                                Proportion of actual positives that were correctly identified.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confusion Matrix with better styling
                        st.markdown("<h4 style='color: #424242; margin-top: 25px;'>Confusion Matrix</h4>", unsafe_allow_html=True)
                        
                        # Create a more visually appealing confusion matrix
                        conf_matrix = evaluation_results['confusion_matrix'].values
                        fig = px.imshow(
                            conf_matrix,
                            labels=dict(x="Predicted Label", y="True Label"),
                            x=["Negative", "Positive"],
                            y=["Negative", "Positive"],
                            color_continuous_scale="Blues",
                            title="Confusion Matrix"
                        )
                        
                        # Add text annotations
                        for i in range(conf_matrix.shape[0]):
                            for j in range(conf_matrix.shape[1]):
                                fig.add_annotation(
                                    x=j, y=i,
                                    text=str(conf_matrix[i, j]),
                                    showarrow=False,
                                    font=dict(color="white" if conf_matrix[i, j] > conf_matrix.sum()/4 else "black")
                                )
                        
                        fig.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif st.session_state.model_choice == "K-Means Clustering":
                        # Create a metrics card with professional styling
                        st.markdown("""
                        <div style="background-color: #F5F7FF; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <h4 style="color: #424242; margin-top: 0;">Clustering Metrics</h4>
                            <p style="color: #757575; font-size: 0.9rem; margin-bottom: 15px;">
                                These metrics evaluate the quality of the clusters formed by the model.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display metrics in an organized layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Silhouette Score", f"{evaluation_results['silhouette']:.4f}", 
                                     delta="Higher is better", delta_color="normal")
                            st.markdown("""
                            <div style="font-size: 0.8rem; color: #757575;">
                                Measure of how similar an object is to its own cluster compared to other clusters.
                                Range from -1 to 1, with higher values indicating better clustering.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Inertia", f"{evaluation_results['inertia']:.2f}", 
                                     delta="Lower is better", delta_color="inverse")
                            st.markdown("""
                            <div style="font-size: 0.8rem; color: #757575;">
                                Sum of squared distances of samples to their closest cluster center.
                                Lower values indicate more compact clusters.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show cluster distribution
                        st.markdown("<h4 style='color: #424242; margin-top: 25px;'>Cluster Distribution</h4>", unsafe_allow_html=True)
                        
                        # Count samples in each cluster
                        cluster_counts = pd.Series(predictions).value_counts().sort_index()
                        
                        # Create a bar chart for cluster distribution
                        fig = px.bar(
                            x=[f"Cluster {i}" for i in cluster_counts.index],
                            y=cluster_counts.values,
                            labels={"x": "Cluster", "y": "Number of Samples"},
                            color=cluster_counts.values,
                            color_continuous_scale="Viridis",
                            title="Samples per Cluster"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Update session state
                    st.session_state.evaluation_results = evaluation_results
                    st.session_state.predictions = predictions
                    st.session_state.step = 7
                    
                    # Professional success message
                    st.markdown("""
                    <div style="background-color: #E8F5E9; border-left: 5px solid #4CAF50; padding: 15px; margin-top: 25px; border-radius: 4px;">
                        <h4 style="color: #2E7D32; margin-top: 0; margin-bottom: 10px;">✅ Model Evaluation Complete!</h4>
                        <p style="color: #2E7D32; margin-bottom: 0;">Your model has been successfully evaluated on the test data. Continue to the visualization step to see detailed results.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Continue button
                    st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
                    if st.button("Continue to Results Visualization →", use_container_width=True):
                        st.rerun()
                    
                except Exception as e:
                    show_notification("error", f"Error during model evaluation: {str(e)}")
    
    # Step 7: Results Visualization
    elif st.session_state.step == 7:
        st.header("Step 7: Results Visualization")
        
        # Model Comparison UI if multiple datasets with models are available
        if len(st.session_state.model_results_by_dataset) > 1:
            with st.expander("🔄 Model Comparison Across Datasets", expanded=False):
                st.markdown("""
                <div style="background-color: #F5F7FF; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h4 style="color: #1E88E5; margin-top: 0;">Cross-Dataset Model Performance</h4>
                    <p style="margin-bottom: 0;">Compare model performance metrics across different datasets</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get all datasets with evaluation results
                datasets_with_results = list(st.session_state.model_results_by_dataset.keys())
                
                # Create comparison data for different model types
                regression_datasets = []
                classification_datasets = []
                clustering_datasets = []
                
                for ds_name in datasets_with_results:
                    model_info = st.session_state.evaluation_results_by_dataset.get(ds_name, {})
                    model_type = model_info.get('model_type', '')
                    
                    if model_type == "Linear Regression":
                        regression_datasets.append(ds_name)
                    elif model_type == "Logistic Regression":
                        classification_datasets.append(ds_name)
                    elif model_type == "K-Means Clustering":
                        clustering_datasets.append(ds_name)
                
                # Create tabs for different model types
                if len(regression_datasets) + len(classification_datasets) + len(clustering_datasets) > 0:
                    model_tabs = []
                    if regression_datasets:
                        model_tabs.append("Regression Models")
                    if classification_datasets:
                        model_tabs.append("Classification Models")
                    if clustering_datasets:
                        model_tabs.append("Clustering Models")
                    
                    comparison_tabs = st.tabs(model_tabs)
                    
                    tab_index = 0
                    # Regression Models Comparison
                    if regression_datasets:
                        with comparison_tabs[tab_index]:
                            st.subheader("Regression Model Performance Comparison")
                            
                            # Create comparison dataframe for regression metrics
                            regression_data = []
                            for ds_name in regression_datasets:
                                metrics = st.session_state.model_results_by_dataset[ds_name]
                                regression_data.append({
                                    "Dataset": ds_name,
                                    "R² Score": round(metrics['r2'], 4),
                                    "MSE": round(metrics['mse'], 4),
                                    "MAE": round(metrics['mae'], 4)
                                })
                            
                            if regression_data:
                                regression_df = pd.DataFrame(regression_data)
                                st.dataframe(regression_df, use_container_width=True)
                                
                                # Visualization
                                fig = px.bar(
                                    regression_df, 
                                    x="Dataset", 
                                    y="R² Score",
                                    title="R² Score Comparison Across Datasets",
                                    color="R² Score",
                                    color_continuous_scale="Blues"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Error metrics comparison
                                error_df = pd.melt(
                                    regression_df, 
                                    id_vars=["Dataset"],
                                    value_vars=["MSE", "MAE"],
                                    var_name="Metric", 
                                    value_name="Value"
                                )
                                
                                fig = px.bar(
                                    error_df,
                                    x="Dataset",
                                    y="Value",
                                    color="Metric",
                                    barmode="group",
                                    title="Error Metrics Comparison (Lower is Better)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            tab_index += 1
                    
                    # Classification Models Comparison
                    if classification_datasets:
                        with comparison_tabs[tab_index if tab_index < len(comparison_tabs) else 0]:
                            st.subheader("Classification Model Performance Comparison")
                            
                            # Create comparison dataframe for classification metrics
                            classification_data = []
                            for ds_name in classification_datasets:
                                metrics = st.session_state.model_results_by_dataset[ds_name]
                                classification_data.append({
                                    "Dataset": ds_name,
                                    "Accuracy": round(metrics['accuracy'], 4),
                                    "Precision": round(metrics['precision'], 4),
                                    "Recall": round(metrics['recall'], 4),
                                    "F1 Score": round(metrics['f1'], 4)
                                })
                            
                            if classification_data:
                                classification_df = pd.DataFrame(classification_data)
                                st.dataframe(classification_df, use_container_width=True)
                                
                                # Visualization - radar chart for classification metrics
                                fig = go.Figure()
                                
                                for i, ds_name in enumerate(classification_datasets):
                                    metrics = st.session_state.model_results_by_dataset[ds_name]
                                    fig.add_trace(go.Scatterpolar(
                                        r=[metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']],
                                        theta=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                        fill='toself',
                                        name=ds_name
                                    ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 1]
                                        )
                                    ),
                                    title="Classification Metrics Comparison"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            tab_index += 1
                    
                    # Clustering Models Comparison
                    if clustering_datasets:
                        with comparison_tabs[tab_index if tab_index < len(comparison_tabs) else 0]:
                            st.subheader("Clustering Model Performance Comparison")
                            
                            # Create comparison dataframe for clustering metrics
                            clustering_data = []
                            for ds_name in clustering_datasets:
                                metrics = st.session_state.model_results_by_dataset[ds_name]
                                clustering_data.append({
                                    "Dataset": ds_name,
                                    "Silhouette Score": round(metrics['silhouette'], 4),
                                    "Inertia": round(metrics['inertia'], 2)
                                })
                            
                            if clustering_data:
                                clustering_df = pd.DataFrame(clustering_data)
                                st.dataframe(clustering_df, use_container_width=True)
                                
                                # Visualization
                                fig = px.bar(
                                    clustering_df, 
                                    x="Dataset", 
                                    y="Silhouette Score",
                                    title="Silhouette Score Comparison (Higher is Better)",
                                    color="Silhouette Score",
                                    color_continuous_scale="Viridis"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                # Options to export comparison results
                if st.button("Export Model Comparison", use_container_width=True):
                    # Create a comprehensive comparison dataframe
                    all_comparison_data = []
                    
                    for ds_name, metrics in st.session_state.model_results_by_dataset.items():
                        model_info = st.session_state.evaluation_results_by_dataset.get(ds_name, {})
                        model_type = model_info.get('model_type', '')
                        
                        comparison_row = {
                            "Dataset": ds_name,
                            "Model Type": model_type,
                            "Timestamp": model_info.get('timestamp', '')
                        }
                        
                        # Add appropriate metrics based on model type
                        if model_type == "Linear Regression":
                            comparison_row.update({
                                "R² Score": round(metrics.get('r2', 0), 4),
                                "MSE": round(metrics.get('mse', 0), 4),
                                "MAE": round(metrics.get('mae', 0), 4)
                            })
                        elif model_type == "Logistic Regression":
                            comparison_row.update({
                                "Accuracy": round(metrics.get('accuracy', 0), 4),
                                "Precision": round(metrics.get('precision', 0), 4),
                                "Recall": round(metrics.get('recall', 0), 4),
                                "F1 Score": round(metrics.get('f1', 0), 4)
                            })
                        elif model_type == "K-Means Clustering":
                            comparison_row.update({
                                "Silhouette Score": round(metrics.get('silhouette', 0), 4),
                                "Inertia": round(metrics.get('inertia', 0), 2)
                            })
                        
                        all_comparison_data.append(comparison_row)
                    
                    if all_comparison_data:
                        comparison_export_df = pd.DataFrame(all_comparison_data)
                        comparison_csv = comparison_export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Comparison Results",
                            data=comparison_csv,
                            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        if st.session_state.predictions is not None:
            st.subheader("Model Predictions Visualization")
            
            # Display results in a professional card layout
            st.markdown("""
            <div style="background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 25px; margin-top: 30px; margin-bottom: 30px;">
                <h3 style="color: #1E88E5; margin-top: 0; margin-bottom: 20px;">Interactive Results Visualization</h3>
            """, unsafe_allow_html=True)
            
            # Different visualization based on model type
            if st.session_state.model_choice == "Linear Regression":
                # Create tabs for different visualizations
                viz_tabs = st.tabs(["Predictions vs Actual", "Residuals Analysis", "Model Performance"])
                
                with viz_tabs[0]:
                    st.markdown("#### Predictions vs Actual Values")
                    st.markdown("""
                    <p style="color: #757575; font-size: 0.9rem; margin-bottom: 15px;">
                        This chart shows how well your model's predictions match the actual values. Points closer to the diagonal line indicate better predictions.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    # Actual vs Predicted Values - Enhanced with Plotly
                    fig = plot_regression_results(
                        y_true=st.session_state.y_test,
                        y_pred=st.session_state.predictions
                    )
                    
                    # Make the plot more interactive
                    fig.update_layout(
                        hovermode="closest",
                        hoverlabel=dict(
                            bgcolor="white",
                            font_size=12,
                            font_family="Arial"
                        ),
                        plot_bgcolor='rgba(240,240,240,0.2)',
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(220,220,220,0.4)',
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(220,220,220,0.4)',
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[1]:
                    st.markdown("#### Residuals Analysis")
                    st.markdown("""
                    <p style="color: #757575; font-size: 0.9rem; margin-bottom: 15px;">
                        Residuals are the differences between predicted and actual values. Ideally, they should be randomly scattered around zero with no clear pattern.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    # Create columns for different residual plots
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        # Residuals vs Predicted Values
                        residuals = st.session_state.y_test - st.session_state.predictions
                        
                        fig1 = px.scatter(
                            x=st.session_state.predictions, 
                            y=residuals,
                            labels={'x': 'Predicted Values', 'y': 'Residuals'},
                            title='Residuals vs Predicted Values',
                            color=abs(residuals),
                            color_continuous_scale='Viridis',
                        )
                        fig1.add_hline(y=0, line_dash="dash", line_color="red")
                        fig1.update_layout(
                            plot_bgcolor='rgba(240,240,240,0.2)',
                            coloraxis_colorbar=dict(title="Residual Magnitude"),
                            height=400
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with res_col2:
                        # Residuals Distribution
                        fig2 = px.histogram(
                            residuals, 
                            nbins=20,
                            labels={'value': 'Residual Value', 'count': 'Frequency'},
                            title='Residuals Distribution',
                            color_discrete_sequence=['#3366CC'],
                        )
                        fig2.add_vline(x=0, line_dash="dash", line_color="red")
                        fig2.update_layout(
                            plot_bgcolor='rgba(240,240,240,0.2)',
                            height=400
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                
                with viz_tabs[2]:
                    st.markdown("#### Model Performance Metrics")
                    
                    # Create a metrics summary card
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("R² Score", f"{st.session_state.evaluation_results['r2']:.4f}", 
                                 delta="Higher is better")
                    
                    with col2:
                        st.metric("MSE", f"{st.session_state.evaluation_results['mse']:.4f}", 
                                 delta="Lower is better", delta_color="inverse")
                    
                    with col3:
                        st.metric("MAE", f"{st.session_state.evaluation_results['mae']:.4f}", 
                                 delta="Lower is better", delta_color="inverse")
                    
                    # Add a feature importance plot
                    st.markdown("#### Feature Importance")
                    # Make sure we have feature names for visualization
                    feature_names = list(st.session_state.X_train.columns) if st.session_state.X_train is not None else []
                    if feature_names:
                        feature_fig = plot_feature_importance(st.session_state.model, feature_names)
                        feature_fig.update_layout(
                            plot_bgcolor='rgba(240,240,240,0.2)',
                            height=400
                        )
                        st.plotly_chart(feature_fig, use_container_width=True)
                    else:
                        st.error("Feature importance visualization requires feature names. No feature data available.")
                
            elif st.session_state.model_choice == "Logistic Regression":
                # Create tabs for different visualizations
                viz_tabs = st.tabs(["Classification Results", "Confusion Matrix", "ROC Curve"])
                
                with viz_tabs[0]:
                    st.markdown("#### Classification Results")
                    st.markdown("""
                    <p style="color: #757575; font-size: 0.9rem; margin-bottom: 15px;">
                        Overview of how well your model classifies the test data into different categories.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{st.session_state.evaluation_results['accuracy']:.4f}", 
                                 delta="Higher is better")
                    
                    with col2:
                        st.metric("Precision", f"{st.session_state.evaluation_results['precision']:.4f}", 
                                 delta="Higher is better")
                    
                    with col3:
                        st.metric("Recall", f"{st.session_state.evaluation_results['recall']:.4f}", 
                                 delta="Higher is better")
                    
                    # Add a prediction distribution plot
                    class_labels = np.unique(st.session_state.y_test)
                    class_counts = pd.Series(st.session_state.predictions).value_counts().sort_index()
                    
                    fig = px.pie(
                        values=class_counts.values,
                        names=[f"Class {int(i)}" for i in class_counts.index],
                        title="Prediction Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hole=0.4
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(240,240,240,0.2)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[1]:
                    st.markdown("#### Confusion Matrix")
                    st.markdown("""
                    <p style="color: #757575; font-size: 0.9rem; margin-bottom: 15px;">
                        The confusion matrix shows the count of correct and incorrect predictions for each class.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced confusion matrix with hover info
                    conf_matrix = st.session_state.evaluation_results['confusion_matrix'].values
                    labels = [f"Class {i}" for i in range(conf_matrix.shape[0])]
                    
                    # Calculate percentages for annotations
                    conf_matrix_percentages = conf_matrix / conf_matrix.sum() * 100
                    
                    # Create heatmap
                    fig = px.imshow(
                        conf_matrix,
                        labels=dict(x="Predicted Class", y="True Class"),
                        x=labels,
                        y=labels,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        aspect="auto"
                    )
                    
                    # Add hover information and styling
                    fig.update_traces(
                        hovertemplate="<b>True:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Count:</b> %{z}<br><b>Percentage:</b> %{text:.1f}%",
                        text=conf_matrix_percentages.flatten(),
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(240,240,240,0.2)',
                        height=500,
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[2]:
                    st.markdown("#### ROC Curve Analysis")
                    st.markdown("""
                    <p style="color: #757575; font-size: 0.9rem; margin-bottom: 15px;">
                        The ROC (Receiver Operating Characteristic) curve shows the performance of the classification model at different threshold settings.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    # ROC Curve with hover info
                    fig = px.line(
                        x=st.session_state.evaluation_results.get('fpr', [0, 1]), 
                        y=st.session_state.evaluation_results.get('tpr', [0, 1]),
                        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                        title=f"ROC Curve (AUC = {st.session_state.evaluation_results.get('auc', 0):.4f})"
                    )
                    
                    # Add the diagonal reference line
                    fig.add_shape(
                        type='line',
                        line=dict(dash='dash', color='gray'),
                        x0=0, x1=1, y0=0, y1=1
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(240,240,240,0.2)',
                        height=500,
                        hovermode="closest"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            elif st.session_state.model_choice == "K-Means Clustering":
                # Create tabs for different visualizations
                viz_tabs = st.tabs(["Cluster Visualization", "Cluster Statistics", "Feature Distribution"])
                
                with viz_tabs[0]:
                    st.markdown("#### Interactive Cluster Visualization")
                    st.markdown("""
                    <p style="color: #757575; font-size: 0.9rem; margin-bottom: 15px;">
                        Visualize how data points are grouped into different clusters. Select features to plot against each other.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    if st.session_state.X_test.shape[1] > 2:
                        # If more than 2 features, let user select which ones to visualize
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            features = st.session_state.X_test.columns.tolist()
                            feature_x = st.selectbox("X-axis feature:", features, index=0)
                        
                        with col2:
                            feature_y = st.selectbox("Y-axis feature:", features, index=min(1, len(features)-1))
                        
                        # Enhanced 3D clustering visualization if we have at least 3 features
                        if len(features) >= 3:
                            show_3d = st.checkbox("Show 3D visualization", value=False)
                            
                            if show_3d:
                                feature_z = st.selectbox("Z-axis feature:", features, 
                                                      index=min(2, len(features)-1))
                                
                                # Create 3D scatter plot
                                df_plot = pd.DataFrame({
                                    'x': st.session_state.X_test[feature_x],
                                    'y': st.session_state.X_test[feature_y],
                                    'z': st.session_state.X_test[feature_z],
                                    'cluster': [f"Cluster {c}" for c in st.session_state.predictions]
                                })
                                
                                fig = px.scatter_3d(
                                    df_plot, x='x', y='y', z='z',
                                    color='cluster',
                                    labels={'x': feature_x, 'y': feature_y, 'z': feature_z},
                                    title=f"3D Cluster Visualization",
                                    color_discrete_sequence=px.colors.qualitative.Bold
                                )
                                
                                fig.update_layout(
                                    scene=dict(
                                        xaxis_title=feature_x,
                                        yaxis_title=feature_y,
                                        zaxis_title=feature_z,
                                    ),
                                    height=700
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Regular 2D visualization with enhanced styling
                                fig = plot_clusters(
                                    X=st.session_state.X_test,
                                    clusters=st.session_state.predictions,
                                    x_feature=feature_x,
                                    y_feature=feature_y
                                )
                                
                                # Enhance the plot
                                fig.update_layout(
                                    plot_bgcolor='rgba(240,240,240,0.2)',
                                    height=500,
                                    legend_title="Cluster",
                                    hovermode="closest"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Regular 2D visualization
                            fig = plot_clusters(
                                X=st.session_state.X_test,
                                clusters=st.session_state.predictions,
                                x_feature=feature_x,
                                y_feature=feature_y
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        # If only 2 features, just plot them
                        fig = plot_clusters(
                            X=st.session_state.X_test,
                            clusters=st.session_state.predictions
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[1]:
                    st.markdown("#### Cluster Statistics")
                    
                    # Count samples in each cluster
                    cluster_counts = pd.Series(st.session_state.predictions).value_counts().sort_index()
                    cluster_percentage = (cluster_counts / len(st.session_state.predictions) * 100).round(1)
                    
                    # Display cluster distribution
                    cluster_df = pd.DataFrame({
                        'Cluster': [f"Cluster {i}" for i in cluster_counts.index],
                        'Count': cluster_counts.values,
                        'Percentage': cluster_percentage.values
                    })
                    
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.dataframe(cluster_df, use_container_width=True, height=400)
                    
                    with col2:
                        # Create a pie chart of cluster distribution
                        fig = px.pie(
                            cluster_df,
                            values='Count',
                            names='Cluster',
                            color_discrete_sequence=px.colors.qualitative.Bold,
                            hole=0.4,
                            title="Cluster Size Distribution"
                        )
                        
                        fig.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
                        )
                        
                        fig.update_layout(
                            height=400,
                            legend=dict(orientation="h", y=-0.1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[2]:
                    st.markdown("#### Feature Distribution by Cluster")
                    
                    # Select a feature to analyze
                    feature_to_analyze = st.selectbox(
                        "Select feature to analyze across clusters:",
                        st.session_state.X_test.columns.tolist()
                    )
                    
                    # Create a dataframe with feature values and cluster assignments
                    feature_cluster_df = pd.DataFrame({
                        'Feature Value': st.session_state.X_test[feature_to_analyze],
                        'Cluster': [f"Cluster {c}" for c in st.session_state.predictions]
                    })
                    
                    # Create visualization based on feature distribution
                    viz_type = st.radio(
                        "Select visualization type:",
                        ["Box Plot", "Violin Plot", "Histogram"],
                        horizontal=True
                    )
                    
                    if viz_type == "Box Plot":
                        fig = px.box(
                            feature_cluster_df,
                            x='Cluster',
                            y='Feature Value',
                            color='Cluster',
                            title=f"Distribution of {feature_to_analyze} by Cluster",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                    elif viz_type == "Violin Plot":
                        fig = px.violin(
                            feature_cluster_df,
                            x='Cluster',
                            y='Feature Value',
                            color='Cluster',
                            box=True,
                            title=f"Distribution of {feature_to_analyze} by Cluster",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                    else:  # Histogram
                        fig = px.histogram(
                            feature_cluster_df,
                            x='Feature Value',
                            color='Cluster',
                            marginal="rug",
                            barmode="overlay",
                            opacity=0.7,
                            title=f"Distribution of {feature_to_analyze} by Cluster",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(240,240,240,0.2)',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download options in a styled card
            st.markdown("""
            <div style="background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 25px; margin-top: 30px;">
                <h3 style="color: #1E88E5; margin-top: 0; margin-bottom: 20px;">
                    <i class="material-icons" style="vertical-align: middle; margin-right: 10px;">download</i>
                    Download Results
                </h3>
                <p style="color: #757575; margin-bottom: 20px;">
                    Download your model results and predictions for further analysis or reporting.
                </p>
            """, unsafe_allow_html=True)
            
            # Create tabs for different download options
            download_tabs = st.tabs(["Predictions", "Model Performance", "Visualization"])
            
            with download_tabs[0]:
                # Create DataFrame with predictions
                if st.session_state.model_choice in ["Linear Regression", "Logistic Regression"]:
                    results_df = pd.DataFrame({
                        'Actual': st.session_state.y_test,
                        'Predicted': st.session_state.predictions
                    })
                    
                    # Add error/difference column for regression
                    if st.session_state.model_choice == "Linear Regression":
                        results_df['Error'] = results_df['Actual'] - results_df['Predicted']
                else:  # K-Means
                    results_df = pd.DataFrame(st.session_state.X_test)
                    results_df['Cluster'] = st.session_state.predictions
                
                # Display a preview
                st.markdown("#### Preview of results")
                st.dataframe(results_df.head(5), use_container_width=True)
                
                # Create a download button for predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="🔽 Download Complete Predictions",
                    data=csv,
                    file_name="ml_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            
            with download_tabs[1]:
                # Create a performance metrics summary
                if st.session_state.model_choice == "Linear Regression":
                    metrics_df = pd.DataFrame({
                        'Metric': ['R² Score', 'Mean Squared Error', 'Mean Absolute Error'],
                        'Value': [
                            st.session_state.evaluation_results['r2'],
                            st.session_state.evaluation_results['mse'],
                            st.session_state.evaluation_results['mae']
                        ]
                    })
                elif st.session_state.model_choice == "Logistic Regression":
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
                        'Value': [
                            st.session_state.evaluation_results['accuracy'],
                            st.session_state.evaluation_results['precision'],
                            st.session_state.evaluation_results['recall'],
                            st.session_state.evaluation_results.get('f1', 0),
                            st.session_state.evaluation_results.get('auc', 0)
                        ]
                    })
                else:  # K-Means
                    metrics_df = pd.DataFrame({
                        'Metric': ['Silhouette Score', 'Inertia'],
                        'Value': [
                            st.session_state.evaluation_results['silhouette'],
                            st.session_state.evaluation_results['inertia']
                        ]
                    })
                
                # Display metrics
                st.markdown("#### Model Performance Summary")
                st.dataframe(metrics_df, use_container_width=True)
                
                # Create download button for metrics
                metrics_csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="🔽 Download Performance Metrics",
                    data=metrics_csv,
                    file_name="model_performance.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            
            with download_tabs[2]:
                st.markdown("#### Export Visualizations")
                
                st.markdown("""
                <div style="padding: 25px; background-color: rgba(99, 102, 241, 0.1); border-radius: 12px; margin-bottom: 25px; border-left: 4px solid var(--primary-color);">
                    <h4 style="margin-top: 0; color: var(--primary-color);">How to Export Visualizations</h4>
                    <p>All charts in this application can be easily exported as PNG images:</p>
                    
                    <div style="display: flex; align-items: center; margin: 15px 0; background-color: white; padding: 12px; border-radius: 8px;">
                        <div style="background-color: var(--primary-color); color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px;">1</div>
                        <div><strong>Hover over any chart</strong> to reveal the toolbar in the top-right corner</div>
                    </div>
                    
                    <div style="display: flex; align-items: center; margin: 15px 0; background-color: white; padding: 12px; border-radius: 8px;">
                        <div style="background-color: var(--primary-color); color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px;">2</div>
                        <div><strong>Click the camera icon</strong> in the toolbar to download the visualization as a PNG image</div>
                    </div>
                    
                    <div style="display: flex; align-items: center; margin: 15px 0; background-color: white; padding: 12px; border-radius: 8px;">
                        <div style="background-color: var(--primary-color); color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px;">3</div>
                        <div><strong>The image will be saved</strong> to your downloads folder</div>
                    </div>
                    
                    <div style="background-color: rgba(66, 66, 66, 0.1); padding: 15px; border-radius: 8px; margin-top: 20px;">
                        <h5 style="margin-top: 0; color: #424242;">Additional Export Options</h5>
                        <ul style="margin-bottom: 0;">
                            <li>You can also click the <strong>menu icon</strong> (three horizontal lines) in the toolbar to access more options</li>
                            <li>From this menu, you can export as SVG or WebP format</li>
                            <li>You can also zoom, pan, and reset the view of the visualization</li>
                        </ul>
                    </div>
                </div>
                
                <div style="background-color: #E8F5E9; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4CAF50;">
                    <h5 style="margin-top: 0; color: #2E7D32;">Pro Tip</h5>
                    <p style="margin-bottom: 0;">For reports and presentations, SVG format offers higher quality and scalability than PNG.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add local example of export functionality
                st.markdown("<p style='text-align: center; margin: 20px 0;'><strong>Example Visualization Export</strong></p>", unsafe_allow_html=True)
                
                # Create a sample chart that users can practice exporting
                sample_data = pd.DataFrame({
                    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                    'Return': [4.2, -2.1, 3.5, 5.9, -1.2, 4.8]
                })
                
                fig = px.bar(
                    sample_data, 
                    x='Month', 
                    y='Return',
                    title="Example Chart (Try Exporting This)",
                    color='Return',
                    color_continuous_scale='RdBu_r',
                    labels={'Return': 'Monthly Return (%)'},
                    text='Return'
                )
                
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(height=400)
                
                # Render the example chart
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show completion message and animation
            st.markdown("""
            <div style="background-color: #E8F5E9; border-radius: 10px; padding: 25px; margin-top: 30px; text-align: center;">
                <h2 style="color: #2E7D32; margin-bottom: 20px;">🎉 Congratulations!</h2>
                <p style="color: #2E7D32; font-size: 1.1rem; margin-bottom: 20px;">
                    You've successfully completed the entire Machine Learning pipeline from data loading to results visualization.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show celebration GIF
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin-top: 30px; margin-bottom: 30px;">
                <img src="{load_animation_url('celebrate')}" alt="Celebration" style="max-width: 100%; border-radius: 10px;">
            </div>
            """, unsafe_allow_html=True)
            
            # Show completion message and animation
            st.success("🎉 Congratulations! You've completed the full ML pipeline.")
            st.markdown(f"![Celebration]({load_animation_url('celebrate')})")
