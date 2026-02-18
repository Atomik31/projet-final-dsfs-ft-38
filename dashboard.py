"""
Dashboard Wind Turbine - Turbine 1
Local file loading - Windows + Streamlit Cloud compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="🌬️ Turbine 1", layout="wide")
st.title("🌬️ Wind Turbine Maintenance - Turbine 1")

# ============================================================================
# 🔧 CONFIGURATION DES CHEMINS - AUTO-DETECTION
# ============================================================================

def find_data_dir():
    """Cherche le dossier 'data' en remontant depuis le script"""
    # Commencer par le répertoire du script
    current = Path(__file__).parent.absolute()
    
    # Remonter jusqu'à 3 niveaux pour trouver 'data/'
    for _ in range(3):
        if (current / "data").exists():
            return current / "data"
        current = current.parent
    
    # Fallback: répertoire courant
    return Path.cwd() / "data"

DATA_DIR = find_data_dir()
MODELS_DIR = DATA_DIR / "models"

# ============================================================================
# AUTO-DETECT MODELS & DATASETS LOCALLY
# ============================================================================

def get_local_files(directory, extension):
    """List files locally with given extension"""
    if not directory.exists():
        return []
    files = [f.name for f in directory.glob(f'*{extension}')]
    return sorted(files)

available_models = get_local_files(MODELS_DIR, '.pkl')
available_datasets = get_local_files(DATA_DIR, '.csv')

# Ne jamais mettre un faux nom si rien n'est trouvé
if not available_models:
    st.error("❌ No .pkl models found in data/models/")
    st.stop()

if not available_datasets:
    st.error("❌ No .csv datasets found in data/")
    st.stop()

# ============================================================================
# LOAD DATA & MODEL FROM LOCAL FILES
# ============================================================================

@st.cache_data(ttl=300)
def load_data(dataset_name):
    """Load CSV locally"""
    try:
        filepath = DATA_DIR / dataset_name
        df = pd.read_csv(filepath)
        if 'Turbine_ID' in df.columns:
            df = df[df['Turbine_ID'] == 1].reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_model(model_name):
    """Load model locally"""
    try:
        filepath = MODELS_DIR / model_name
        return joblib.load(filepath)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ============================================================================
# SIDEBAR - TIME WINDOW
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Model & Data Selection")
    
    model_choice = st.selectbox(
        "Select Model",
        available_models,
        help="Choose which model to use"
    )
    
    dataset_choice = st.selectbox(
        "Select Dataset",
        available_datasets,
        help="Choose dataset matching the model"
    )
    
    st.info("⚠️ Make sure dataset matches model training data!")
    
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
# LOAD DATA & MODEL
# ============================================================================

# Force cache clear when dataset changes
if 'last_dataset' not in st.session_state:
    st.session_state.last_dataset = dataset_choice

if st.session_state.last_dataset != dataset_choice:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.last_dataset = dataset_choice

df = load_data(dataset_choice)
model = load_model(model_choice)

df_filtered = df.copy()
if hours_limit:
    df_filtered = df_filtered.tail(hours_limit).reset_index(drop=True)

st.write(f"📊 Time: {selected_time} | Rows: {len(df_filtered)}")

st.sidebar.subheader("Select Specific Hour")
selected_hour_in_period = st.sidebar.slider("View status at hour:", 
                                            min_value=0, 
                                            max_value=len(df_filtered)-1,
                                            value=len(df_filtered)-1,
                                            step=1)

# Initialize engineering flag
use_engineering = False

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create all 45 features from raw data to match model training"""
    df = df.copy()
    
    # 1. Time features
    df['Hour_of_Day'] = df.index % 24
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour_of_Day'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour_of_Day'] / 24)
    df['Day_of_Week'] = (df.index // 24) % 7
    df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
    
    # 2. Ratio features
    df['Power_per_Wind'] = df['Power_Output_kW'] / (df['Wind_Speed_mps'] + 0.1)
    df['Power_per_Rotor'] = df['Power_Output_kW'] / (df['Rotor_Speed_RPM'] + 0.1)
    
    # 3. Temperature delta
    df['Temp_Delta'] = df['Gearbox_Oil_Temp_C'] - df['Ambient_Temp_C']
    df['Gen_vs_Gear_Temp'] = df['Generator_Bearing_Temp_C'] - df['Gearbox_Oil_Temp_C']
    
    # 4. Moving averages (3, 6, 12 hours)
    for window in [3, 6, 12]:
        df[f'wind_MA{window}'] = df['Wind_Speed_mps'].rolling(window, min_periods=1).mean()
        df[f'power_MA{window}'] = df['Power_Output_kW'].rolling(window, min_periods=1).mean()
        df[f'vibration_MA{window}'] = df['Vibration_Level_mmps'].rolling(window, min_periods=1).mean()
        df[f'gearbox_temp_MA{window}'] = df['Gearbox_Oil_Temp_C'].rolling(window, min_periods=1).mean()
    
    # 5. Standard deviation (6, 12 hours)
    for window in [6, 12]:
        df[f'vibration_Std{window}'] = df['Vibration_Level_mmps'].rolling(window, min_periods=1).std().fillna(0)
        df[f'power_Std{window}'] = df['Power_Output_kW'].rolling(window, min_periods=1).std().fillna(0)
        df[f'temp_Std{window}'] = df['Gearbox_Oil_Temp_C'].rolling(window, min_periods=1).std().fillna(0)
    
    # 6. Differences (lag 1)
    df['wind_speed_diff'] = df['Wind_Speed_mps'].diff().fillna(0)
    df['power_output_diff'] = df['Power_Output_kW'].diff().fillna(0)
    df['vibration_diff'] = df['Vibration_Level_mmps'].diff().fillna(0)
    df['gearbox_temp_diff'] = df['Gearbox_Oil_Temp_C'].diff().fillna(0)
    
    # 7. Max values (6, 12 hours)
    for window in [6, 12]:
        df[f'vibration_Max{window}'] = df['Vibration_Level_mmps'].rolling(window, min_periods=1).max()
        df[f'temp_Max{window}'] = df['Gearbox_Oil_Temp_C'].rolling(window, min_periods=1).max()
    
    # 8. Interaction feature
    df['Wind_x_Vib'] = df['Wind_Speed_mps'] * df['Vibration_Level_mmps']
    
    return df

# ============================================================================
# FEATURES FOR MODEL
# ============================================================================

# Check if model is a dict with feature_columns
if isinstance(model, dict) and 'feature_columns' in model:
    # Complex model: use dict structure
    feature_cols = model['feature_columns']
    model_obj = model['model']
    scaler = model.get('scaler', None)
    expected_features = len(feature_cols)
    use_engineering = True
    
elif isinstance(model, dict):
    st.error("❌ Dict model format not recognized")
    feature_cols = []
    model_obj = None
    scaler = None
    expected_features = 0
    use_engineering = False
    
else:
    # Simple model: use 8 raw features (no engineering)
    feature_cols = ['Rotor_Speed_RPM', 'Wind_Speed_mps', 'Power_Output_kW', 
                    'Gearbox_Oil_Temp_C', 'Generator_Bearing_Temp_C', 
                    'Vibration_Level_mmps', 'Ambient_Temp_C', 'Humidity_pct']
    model_obj = model
    scaler = None
    expected_features = len(feature_cols)
    use_engineering = False

# Apply feature engineering ONLY for complex models
if use_engineering and isinstance(model, dict) and 'feature_columns' in model:
    df_filtered = engineer_features(df_filtered)

# ============================================================================
# SECTION 1: REAL-TIME STATUS
# ============================================================================

st.header("🚨 Real-Time Status")

if model is not None and len(df_filtered) > 0 and model_obj is not None:
    try:
        # Prepare features
        X = df_filtered[feature_cols].fillna(0)
        
        # Apply scaler if available
        if scaler:
            X = scaler.transform(X)
        
        # Get predictions
        predictions = model_obj.predict(X)
        
        # Confidence
        if hasattr(model_obj, 'predict_proba'):
            proba = model_obj.predict_proba(X)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.ones(len(X))
        
        # Status at selected hour
        selected_pred = int(predictions[selected_hour_in_period])
        selected_conf = float(confidence[selected_hour_in_period])
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
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
        
        # Measurements
        st.divider()
        st.subheader(f"Measurements at Hour #{selected_hour_in_period}")
        
        selected_row = df_filtered.iloc[selected_hour_in_period]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Wind Speed", f"{selected_row['Wind_Speed_mps']:.2f} m/s")
        col2.metric("Power Output", f"{selected_row['Power_Output_kW']:.2f} kW")
        col3.metric("Rotor Speed", f"{selected_row['Rotor_Speed_RPM']:.0f} RPM")
        col4.metric("Vibration", f"{selected_row['Vibration_Level_mmps']:.2f} mm/s")
        
        st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Timeline
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
        
        fig_timeline.update_yaxes(tickvals=[0, 1, 2], ticktext=['Healthy', 'Maintenance', 'Critical'])
        fig_timeline.update_layout(height=250, hovermode='x unified', template='plotly_white')
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Statistics
        st.subheader("Statistics")
        unique, counts = np.unique(predictions, return_counts=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Healthy", int(counts[unique==0][0] if 0 in unique else 0))
        col2.metric("🟡 Maintenance", int(counts[unique==1][0] if 1 in unique else 0))
        col3.metric("🔴 Critical", int(counts[unique==2][0] if 2 in unique else 0))
        
        if 'Maintenance_Label' in df_filtered.columns:
            actual = df_filtered['Maintenance_Label'].values
            accuracy = (predictions == actual).sum() / len(predictions)
            st.metric("Accuracy on Period", f"{accuracy:.1%}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())

else:
    st.error("❌ Model or data not available")

# ============================================================================
# SECTION 2: FEATURE MONITORING
# ============================================================================

st.divider()
st.header("📊 Feature Monitoring")

# Only show tabs for numeric features that exist in df_filtered
numeric_features = [col for col in feature_cols if col in df_filtered.columns]

if numeric_features:
    tabs = st.tabs(numeric_features)
    
    for tab, feature in zip(tabs, numeric_features):
        with tab:
            if pd.api.types.is_numeric_dtype(df_filtered[feature]):
                fig = px.line(df_filtered, y=feature, title=feature, markers=False)
                fig.update_layout(height=300, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Min", f"{df_filtered[feature].min():.2f}")
                col2.metric("Max", f"{df_filtered[feature].max():.2f}")
                col3.metric("Mean", f"{df_filtered[feature].mean():.2f}")
                col4.metric("Std", f"{df_filtered[feature].std():.2f}")
            else:
                st.info(f"📊 {feature}: Non-numeric column")
else:
    st.warning("⚠️ No numeric features available for monitoring")

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