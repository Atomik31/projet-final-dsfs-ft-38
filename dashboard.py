"""
Dashboard Wind Turbine - Turbine 1
Production Ready - Simple Model (8 raw features)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import plotly.graph_objects as go
import plotly.express as px
import glob
import os

st.set_page_config(page_title="🌬️ Turbine 1", layout="wide")
st.title("🌬️ Wind Turbine Maintenance - Turbine 1")

# ============================================================================
# AUTO-DETECT MODELS & DATASETS
# ============================================================================

# Find all .pkl files (models)
available_models = glob.glob("*.pkl")
if not available_models:
    available_models = ["model_simple.pkl"]  # Default fallback

# Find all .csv files (datasets)
available_datasets = glob.glob("*.csv")
if not available_datasets:
    available_datasets = ["wind_turbine_maintenance_data.csv"]  # Default fallback

# ============================================================================
# LOAD DATA & MODEL
# ============================================================================

@st.cache_data
def load_data(dataset_name):
    df = pd.read_csv(dataset_name)
    # Filter to Turbine 1 if it has Turbine_ID column
    if 'Turbine_ID' in df.columns:
        df = df[df['Turbine_ID'] == 1].reset_index(drop=True)
    return df

@st.cache_resource
def load_model(model_name):
    try:
        return joblib.load(model_name)
    except:
        return None

# Load based on selection (need to get sidebar values first)
# This will be done after sidebar definition

# Placeholder - will be updated after sidebar
df = None
model = None

# ============================================================================
# SIDEBAR - TIME WINDOW
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Model & Data Selection")
    
    # Model selection (auto-detected)
    model_choice = st.selectbox(
        "Select Model",
        available_models,
        help="Choose which model to use for predictions"
    )
    
    # Dataset selection (auto-detected)
    dataset_choice = st.selectbox(
        "Select Dataset",
        available_datasets,
        help="Choose dataset that matches the selected model"
    )
    
    st.info("⚠️ Make sure dataset matches the model training data!")
    
    st.divider()
    st.subheader("Time Window")
    time_choice = st.radio("Select time window", 
                          ["Predefined", "Custom Hours"],
                          horizontal=True)
    
    if time_choice == "Predefined":
        time_options = {
            "All Data": None,
            "Last 3 Hours": 3,
            "Last 6 Hours": 6,
            "Last 12 Hours": 12,
            "Last 24 Hours": 24,
            "Last 48 Hours": 48,
            "Last 72 Hours": 72,
            "Last 1 Week": 168,
            "Last 2 Weeks": 336
        }
        selected_time = st.selectbox("Choose preset", list(time_options.keys()), index=4)
        hours_limit = time_options[selected_time]
    else:
        hours_limit = st.slider("Select hours", min_value=1, max_value=336, value=24, step=1)
        selected_time = f"Last {hours_limit} Hours"

# ============================================================================
# LOAD SELECTED MODEL & DATASET
# ============================================================================

df = load_data(dataset_choice)
model = load_model(model_choice)

st.success(f"✅ Turbine 1: {df.shape[0]} rows | Model: {model_choice.split('/')[-1]} | {'✅ Ready' if model else '❌ Not found'}")

df_filtered = df.copy()
if hours_limit:
    df_filtered = df_filtered.tail(hours_limit).reset_index(drop=True)

st.write(f"📊 Time: {selected_time} | Rows: {len(df_filtered)}")

# Update slider in sidebar to show max hours available
st.sidebar.subheader("Select Specific Hour")
selected_hour_in_period = st.sidebar.slider("View status at hour:", 
                                            min_value=0, 
                                            max_value=len(df_filtered)-1,
                                            value=len(df_filtered)-1,
                                            step=1,
                                            help="Select which hour in the period to view")

# ============================================================================
# FEATURES FOR MODEL
# ============================================================================

# Auto-detect features based on dataset (exclude Turbine_ID and Maintenance_Label)
available_features = [col for col in df.columns 
                     if col.lower() not in ['turbine_id', 'maintenance_label']]

# For simple model, use only 8 raw features
if 'model_simple' in model_choice:
    feature_cols = ['Rotor_Speed_RPM', 'Wind_Speed_mps', 'Power_Output_kW', 
                    'Gearbox_Oil_Temp_C', 'Generator_Bearing_Temp_C', 
                    'Vibration_Level_mmps', 'Ambient_Temp_C', 'Humidity_pct']
else:
    # For complex model, use all available features except unnamed columns
    feature_cols = [col for col in available_features if not col.lower().startswith('unnamed')]
    
    # Adjust to exact model feature count
    if len(feature_cols) > model.n_features_in_ if model else False:
        feature_cols = feature_cols[:model.n_features_in_]

st.info(f"ℹ️ Using {len(feature_cols)} features | Model expects {model.n_features_in_ if model else '?'} features")

# ============================================================================
# SECTION 1: REAL-TIME STATUS
# ============================================================================

st.header("🚨 Real-Time Status")

if model is not None and len(df_filtered) > 0:
    try:
        # Get all predictions for filtered period
        X = df_filtered[feature_cols].fillna(0)
        predictions = model.predict(X)
        
        # Confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.ones(len(X))
        
        # Latest
        latest_pred = int(predictions[-1])
        latest_conf = float(confidence[-1])
        
        # Status at SELECTED HOUR
        selected_pred = int(predictions[selected_hour_in_period])
        selected_conf = float(confidence[selected_hour_in_period])
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Show status AT SELECTED HOUR
            if selected_pred == 0:
                st.success("🟢 **HEALTHY** - No action needed", icon="✅")
            elif selected_pred == 1:
                st.warning("🟡 **MAINTENANCE NEEDED** - Schedule soon", icon="⚠️")
            else:
                st.error("🔴 **CRITICAL** - Immediate action required", icon="🚨")
        
        with col2:
            st.metric("Confidence", f"{selected_conf:.1%}")
        
        with col3:
            st.metric("Hour #", f"{selected_hour_in_period}")
        
        # Latest measurements at SELECTED HOUR
        st.divider()
        st.subheader(f"Measurements at Hour #{selected_hour_in_period}")
        
        selected_row = df_filtered.iloc[selected_hour_in_period]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Wind Speed", f"{selected_row['Wind_Speed_mps']:.2f} m/s")
        col2.metric("Power Output", f"{selected_row['Power_Output_kW']:.2f} kW")
        col3.metric("Rotor Speed", f"{selected_row['Rotor_Speed_RPM']:.0f} RPM")
        col4.metric("Vibration", f"{selected_row['Vibration_Level_mmps']:.2f} mm/s")
        
        st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # ===== PREDICTION TIMELINE =====
        st.divider()
        st.subheader("Prediction Timeline (Period)")
        
        colors_map = {0: "#28a745", 1: "#ffc107", 2: "#dc3545"}
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=list(range(len(predictions))),
            y=predictions,
            mode='markers+lines',
            marker=dict(
                size=5,
                color=[colors_map.get(int(p), '#999') for p in predictions],
                line=dict(width=1, color='white')
            ),
            line=dict(color='rgba(0,0,0,0.1)'),
            name='Status'
        ))
        
        fig_timeline.update_yaxes(
            tickvals=[0, 1, 2],
            ticktext=['Healthy', 'Maintenance', 'Critical']
        )
        
        fig_timeline.update_layout(
            height=250,
            hovermode='x unified',
            xaxis_title="Sample",
            yaxis_title="Prediction",
            template='plotly_white'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # ===== STATISTICS =====
        st.subheader("Statistics")
        
        unique, counts = np.unique(predictions, return_counts=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Healthy", int(counts[unique==0][0] if 0 in unique else 0))
        col2.metric("🟡 Maintenance", int(counts[unique==1][0] if 1 in unique else 0))
        col3.metric("🔴 Critical", int(counts[unique==2][0] if 2 in unique else 0))
        
        # Accuracy
        if 'Maintenance_Label' in df_filtered.columns:
            actual = df_filtered['Maintenance_Label'].values
            accuracy = (predictions == actual).sum() / len(predictions)
            st.metric("Accuracy on Period", f"{accuracy:.1%}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.error("❌ Model or data not available")

# ============================================================================
# SECTION 2: FEATURE MONITORING
# ============================================================================

st.divider()
st.header("📊 Feature Monitoring")

tabs = st.tabs(feature_cols)

for tab, feature in zip(tabs, feature_cols):
    with tab:
        fig = px.line(df_filtered, y=feature, title=feature, markers=False)
        fig.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min", f"{df_filtered[feature].min():.2f}")
        col2.metric("Max", f"{df_filtered[feature].max():.2f}")
        col3.metric("Mean", f"{df_filtered[feature].mean():.2f}")
        col4.metric("Std", f"{df_filtered[feature].std():.2f}")

# ============================================================================
# SECTION 3: DATA TABLE
# ============================================================================

st.divider()
st.header("📋 Raw Data")

if st.checkbox("Show all rows"):
    st.dataframe(df_filtered, use_container_width=True, height=400)
else:
    st.dataframe(df_filtered.tail(20), use_container_width=True, height=400)

st.divider()
st.caption(f"Rows: {len(df_filtered)} | ⏰ {datetime.now().strftime('%H:%M:%S')}")