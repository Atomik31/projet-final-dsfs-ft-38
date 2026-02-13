"""
Dashboard Streamlit - Wind Turbine Maintenance Prediction
S3 + CSV → Graphiques features + Prédictions
"""

"""
Dashboard Streamlit - Wind Turbine Maintenance Prediction
S3 + CSV → Graphiques features + Prédictions
"""

import streamlit as st

# Check requirements at startup
import subprocess
import sys

try:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import joblib
    import os
    import boto3
    from dotenv import load_dotenv
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ModuleNotFoundError as e:
    st.error(f"❌ Missing package: {e}")
    st.info("Installing missing packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "plotly", "boto3", "python-dotenv"])
    st.rerun()

st.set_page_config(page_title="🌬️ Turbine Monitor", layout="wide")

st.title("🌬️ Wind Turbine Maintenance Prediction")
st.subheader("Real-time Monitoring & ML Predictions")

# ============================================================================
# LOAD S3 DATASET
# ============================================================================

@st.cache_data(ttl=3600)  # Cache pour 1 heure
def load_dataset_from_s3():
    """Load dataset from S3 (cached)"""
    try:
        load_dotenv()
        acces_key = os.getenv("AWS_ACCESS_KEY_ID")
        s_acces_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        session = boto3.Session(
            aws_access_key_id=acces_key,
            aws_secret_access_key=s_acces_key
        )
        
        s3 = session.resource("s3")
        bucket = s3.Bucket("projet-certif-dsfs-ft-38")
        
        obj = bucket.Object("dataset/Wind Turbine Predictive Maintenance_KAGGLE/wind_turbine_maintenance_data.csv")
        df = pd.read_csv(obj.get()['Body'])
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading from S3: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache pour 1 heure
def load_dataset_from_csv(csv_path):
    """Load dataset from local CSV (cached)"""
    return pd.read_csv(csv_path)

@st.cache_resource  # Cache permanent pour le modèle
def load_model_from_s3(model_s3_path="models/turbine_model.pkl"):
    """Load ML model from S3 (cached)"""
    try:
        load_dotenv()
        acces_key = os.getenv("AWS_ACCESS_KEY_ID")
        s_acces_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        session = boto3.Session(
            aws_access_key_id=acces_key,
            aws_secret_access_key=s_acces_key
        )
        
        s3 = session.client("s3")
        
        # Télécharger le modèle en mémoire
        import io
        model_bytes = io.BytesIO()
        s3.download_fileobj("projet-certif-dsfs-ft-38", model_s3_path, model_bytes)
        model_bytes.seek(0)
        
        # Charger avec joblib
        model = joblib.load(model_bytes)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model from S3: {str(e)}")
        return None

@st.cache_resource  # Cache permanent pour le modèle
def load_model_from_local(model_path):
    """Load ML model from local file (cached)"""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ============================================================================
# SIDEBAR - Configuration
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["S3 (AWS)", "Local CSV"],
        index=0
    )
    
    if data_source == "Local CSV":
        csv_file = st.text_input(
            "Dataset Path (CSV)",
            value="wind_turbine_maintenance_data.csv"
        )
    else:
        csv_file = None
    
    # Model source selection
    st.subheader("Model Source")
    model_source = st.radio(
        "Load model from",
        ["S3 (AWS)", "Local File"],
        index=0,
        help="Choose where to load the model"
    )
    
    if model_source == "Local File":
        model_file = st.text_input(
            "Model Path (pkl)",
            value="./turbine_model.pkl",
            help="Local path to model file (e.g., ./turbine_model.pkl or ./models/turbine_model.pkl)"
        )
        model_s3_path = None
    else:
        model_file = None
        model_s3_path = "dataset/Wind Turbine Predictive Maintenance_KAGGLE/turbine_model.pkl"  # Path correct S3
    
    # Turbine selection
    st.subheader("Turbine Selection")
    turbines_to_display = st.multiselect(
        "Select Turbines",
        options=[1, 2],
        default=[1, 2],
        help="Select which turbines to display"
    )
    
    # Time window selection
    st.subheader("Time Window")
    time_options = {
        "Last Hour": 1,
        "Last 6 Hours": 6,
        "Last 12 Hours": 12,
        "Last 24 Hours": 24,
        "Last 48 Hours": 48,
        "Last 72 Hours": 72,
        "Last 7 Days": 168,
        "Last 14 Days": 336
    }
    selected_time = st.selectbox(
        "Select time window",
        options=list(time_options.keys()),
        index=6,  # Default: Last 7 Days
        help="Select how many hours to display"
    )
    hours_to_display = time_options[selected_time]

# ============================================================================
# LOAD DATA
# ============================================================================

import time

if data_source == "S3 (AWS)":
    with st.spinner("⏳ Loading dataset from S3..."):
        df = load_dataset_from_s3()
        if df is None:
            st.stop()
else:
    try:
        df = load_dataset_from_csv(csv_file)
    except FileNotFoundError:
        st.error(f"❌ CSV not found: {csv_file}")
        st.info("Make sure the CSV file is in the working directory")
        st.stop()

# ============================================================================
# FILTER BY SELECTED TURBINES & LAST 7 DAYS (CACHED)
# ============================================================================

@st.cache_data(ttl=3600)
def filter_turbine_data(df, turbine_ids, hours_to_display=168):
    """Filter data for selected turbines and last N hours (cached)"""
    filtered_dfs = []
    for turbine_id in turbine_ids:
        turbine_data = df[df['Turbine_ID'] == turbine_id].reset_index(drop=True)
        
        if len(turbine_data) == 0:
            continue
        
        # Prendre les N dernières heures de cette turbine
        if len(turbine_data) > hours_to_display:
            turbine_data = turbine_data.tail(hours_to_display).reset_index(drop=True)
        
        filtered_dfs.append(turbine_data)
    
    if not filtered_dfs:
        return None
    
    return pd.concat(filtered_dfs, ignore_index=True)

# Appeler la fonction cachée avec la fenêtre temps sélectionnée
df_filtered = filter_turbine_data(df, tuple(turbines_to_display), hours_to_display)

if df_filtered is None:
    st.error("❌ No data found for selected turbines")
    st.stop()

st.toast(f"✅ Displaying Turbines {turbines_to_display} - Last {selected_time} ({len(df_filtered)} rows total)", icon="✅")
df = df_filtered

# Garder une copie des colonnes initiales pour l'affichage
df_display = df.copy()

# ============================================================================
# FEATURE ENGINEERING (EN ARRIÈRE-PLAN POUR LE MODÈLE)
# ============================================================================

def engineer_features(df):
    """Créer les features engineered pour le modèle"""
    df = df.copy()
    
    # Features de base (excluant Turbine_ID et Maintenance_Label)
    exclude_cols = ['Turbine_ID', 'Maintenance_Label']
    base_features = [col for col in df_display.columns if col not in exclude_cols]
    
    # Lag features (1h, 6h, 24h)
    for lag in [1, 6, 24]:
        for col in base_features:
            df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    # Rolling statistics (6h, 12h, 24h)
    for window in [6, 12, 24]:
        for col in base_features:
            df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window).mean()
            df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window).std()
            df[f'{col}_rolling_min_{window}h'] = df[col].rolling(window=window).min()
            df[f'{col}_rolling_max_{window}h'] = df[col].rolling(window=window).max()
    
    # Interaction features
    for i, col1 in enumerate(base_features):
        for col2 in base_features[i+1:]:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    # Trend features (pente sur 6h)
    for col in base_features:
        df[f'{col}_trend_6h'] = df[col].diff(6)
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

# Feature engineering pour les prédictions
df_model = engineer_features(df)

# ============================================================================
# LOAD MODEL
# ============================================================================

if model_source == "S3 (AWS)":
    with st.spinner("⏳ Loading model from S3..."):
        model = load_model_from_s3(model_s3_path)
        if model is not None:
            st.sidebar.success(f"✅ Model loaded from S3")
            model_loaded = True
        else:
            st.sidebar.warning(f"⚠️ Model not found in S3: {model_s3_path}")
            model_loaded = False
else:
    model = load_model_from_local(model_file)
    if model is not None:
        st.sidebar.success(f"✅ Model loaded from local")
        model_loaded = True
    else:
        st.sidebar.warning(f"⚠️ Model not found: {model_file}")
        model_loaded = False

# ============================================================================
# IDENTIFY COLUMNS
# ============================================================================

# Exclure Turbine_ID et Maintenance_Label (case-sensitive!)
exclude_cols = ['Turbine_ID', 'Maintenance_Label']
feature_cols = [col for col in df_display.columns if col not in exclude_cols]

# ============================================================================
# MAIN DASHBOARD - FEATURES
# ============================================================================

st.header("📊 All Features Over Time")

# Créer graphique pour chaque feature dans des onglets
if len(feature_cols) > 0:
    # Ajouter un onglet "All Features"
    all_tabs = ["All Features"] + feature_cols
    tabs = st.tabs(all_tabs)
    
    # Premier onglet: Toutes les features normalisées
    with tabs[0]:
        st.subheader("All Features - Individual Scales")
        
        # Créer des subplots (2 colonnes, autant de lignes que nécessaire)
        num_features = len(feature_cols)
        num_rows = (num_features + 1) // 2  # Arrondir à l'entier supérieur
        
        # Adapter l'espacement vertical selon le nombre de rows
        # Maximum autorisé = 1 / (rows - 1)
        max_v_spacing = 1 / (num_rows - 1) if num_rows > 1 else 0.5
        v_spacing = min(0.08, max_v_spacing * 0.9)  # Utiliser 90% du max
        
        fig_all = make_subplots(
            rows=num_rows,
            cols=2,
            subplot_titles=[col.replace('_', ' ') for col in feature_cols],
            vertical_spacing=v_spacing,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Plotly
        
        for idx, col in enumerate(feature_cols):
            row = (idx // 2) + 1
            col_num = (idx % 2) + 1
            
            fig_all.add_trace(
                go.Scatter(
                    y=df[col],
                    mode='lines',
                    name=col.replace('_', ' '),
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate='<b>' + col.replace('_', ' ') + '</b><br>Value: %{y:.2f}<extra></extra>',
                    showlegend=True
                ),
                row=row,
                col=col_num
            )
        
        fig_all.update_layout(
            title_text="All Features - Normalized Display",
            height=200 * num_rows,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        # Update y-axes labels
        for idx in range(1, num_features + 1):
            fig_all.update_yaxes(title_text="Value", row=(idx-1)//2 + 1, col=(idx-1)%2 + 1)
        
        st.plotly_chart(fig_all, use_container_width=True)
        
        # Stats pour chaque feature
        st.subheader("Features Statistics")
        cols_stats = st.columns(4)
        
        for idx, col in enumerate(feature_cols):
            with cols_stats[idx % 4]:
                st.metric(
                    col.replace('_', ' '), 
                    f"{df[col].mean():.2f}", 
                    delta=f"±{df[col].std():.2f}"
                )
    
    # Onglets individuels
    for tab, col in zip(tabs[1:], feature_cols):
        with tab:
            # Graphique ligne
            fig = px.line(
                df,
                y=col,
                title=f"{col.replace('_', ' ')}",
                markers=True,
                line_shape='linear'
            )
            fig.update_layout(
                height=350,
                hovermode='x unified',
                showlegend=False,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min", f"{df_display[col].min():.2f}")
            with col2:
                st.metric("Max", f"{df_display[col].max():.2f}")
            with col3:
                st.metric("Mean", f"{df_display[col].mean():.2f}")
            with col4:
                st.metric("Std", f"{df_display[col].std():.2f}")
else:
    st.error("No features found!")

# ============================================================================
# MODEL PREDICTIONS
# ============================================================================

st.divider()
st.header("🔮 Model Predictions")

if model_loaded and model is not None:
    try:
        # Utiliser les features engineered pour le modèle
        exclude_cols = ['Turbine_ID', 'Maintenance_Label']
        feature_cols_model = [col for col in df_model.columns if col not in exclude_cols]
        
        X = df_model[feature_cols_model].fillna(0)
        
        # Prédictions
        predictions = model.predict(X)
        
        # Confiance (si disponible)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.ones(len(X))
        
        # Ajouter les prédictions au dataframe d'affichage
        df_display['prediction'] = predictions
        df_display['confidence'] = confidence
        
        # Afficher dernière prédiction
        latest = df_display.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred = int(latest['prediction'])
            if pred == 0:
                st.success("🟢 HEALTHY")
                st.markdown("*No maintenance required*")
            elif pred == 1:
                st.warning("🟡 MAINTENANCE SUGGESTED")
                st.markdown("*Plan maintenance soon*")
            else:
                st.error("🔴 CRITICAL")
                st.markdown("*Immediate maintenance required*")
        
        with col2:
            st.metric("Model Confidence", f"{latest['confidence']:.1%}")
        
        with col3:
            if 'Turbine_ID' in df.columns:
                st.metric("Turbine ID", f"{int(latest['Turbine_ID'])}")
            st.metric("Total Rows", len(df_display))
        
        # Timeline prédictions
        st.subheader("Prediction Timeline")
        
        fig_pred = go.Figure()  # df_display
        
        colors = {0: "#28a745", 1: "#ffc107", 2: "#dc3545"}
        
        fig_pred.add_trace(go.Scatter(
            x=list(range(len(df_display))),
            y=df_display['prediction'].values,
            mode='markers+lines',
            marker=dict(
                size=6,
                color=[colors.get(int(p), '#999') for p in df['prediction']],
                line=dict(width=1, color='white')
            ),
            line=dict(color='rgba(0,0,0,0.1)'),
            name='Alert Level'
        ))
        
        fig_pred.update_yaxes(
            tickvals=[0, 1, 2],
            ticktext=['🟢 Healthy', '🟡 Maintenance', '🔴 Critical']
        )
        
        fig_pred.update_layout(
            height=300,
            hovermode='x unified',
            xaxis_title="Row Index",
            yaxis_title="Prediction",
            template='plotly_white'
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Confidence trend
        st.subheader("Model Confidence Trend")
        
        fig_conf = px.line(
            confidence=df['confidence'],
            title="Prediction Confidence",
            markers=True
        )
        fig_conf.add_trace(go.Scatter(
            y=df['confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#007bff', width=2),
            marker=dict(size=4)
        ))
        fig_conf.data = fig_conf.data[1:]  # Remove the empty trace
        
        fig_conf.update_yaxes(tickformat=".0%", range=[0, 1])
        fig_conf.update_layout(height=300, hovermode='x unified', template='plotly_white')
        
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Prediction Statistics")
        
        pred_counts = df['prediction'].value_counts().sort_index()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 Healthy", int(pred_counts.get(0, 0)))
        with col2:
            st.metric("🟡 Maintenance", int(pred_counts.get(1, 0)))
        with col3:
            st.metric("🔴 Critical", int(pred_counts.get(2, 0)))
        
        # Comparison: Actual vs Predicted (si Maintenance_Label existe)
        if 'Maintenance_Label' in df.columns:
            st.subheader("🎯 Actual vs Predicted")
            
            actual_counts = df['Maintenance_Label'].value_counts().sort_index()
            
            comparison_df = pd.DataFrame({
                'Alert Level': ['Healthy', 'Maintenance', 'Critical'],
                'Actual': [actual_counts.get(i, 0) for i in range(3)],
                'Predicted': [pred_counts.get(i, 0) for i in range(3)]
            })
            
            fig_comp = px.bar(
                comparison_df,
                x='Alert Level',
                y=['Actual', 'Predicted'],
                barmode='group',
                title="Actual vs Predicted Distribution",
                color_discrete_map={'Actual': '#6c757d', 'Predicted': '#007bff'}
            )
            
            fig_comp.update_layout(height=300, template='plotly_white')
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Accuracy metrics
            accuracy = (df['prediction'] == df['Maintenance_Label']).sum() / len(df_display)
            st.metric("Prediction Accuracy", f"{accuracy:.1%}")
    
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.info("Make sure model features match CSV columns")

else:
    st.warning("⚠️ Model not loaded")
    if model_source == "Local File":
        import os
        abs_path = os.path.abspath(model_file) if model_file else "Unknown"
        st.error(f"❌ Model file not found: {model_file}")
        st.info(f"Looking for: {abs_path}")
        st.info("Try placing the file in the same directory as dashboard.py or use the full path (e.g., ./models/turbine_model.pkl)")
    else:
        st.error(f"❌ Model not found in S3: {model_s3_path}")
        st.info("Check AWS credentials and S3 path")

# ============================================================================
# RAW DATA TABLE
# ============================================================================

st.divider()
st.header("📋 Raw Data")

if st.checkbox("Show full table", value=False):
    st.dataframe(df_display, use_container_width=True, height=500)
else:
    st.dataframe(df_display.head(20), use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    if data_source == "S3 (AWS)":
        st.caption("📁 File: S3 Dataset")
    else:
        st.caption(f"📁 File: {os.path.basename(csv_file) if csv_file else 'Unknown'}")
with col2:
    st.caption(f"📊 Total rows: {len(df_display)}")
with col3:
    st.caption(f"🎯 Features: {len(feature_cols)}")
with col4:
    st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
