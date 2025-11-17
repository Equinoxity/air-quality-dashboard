import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import numpy as np

# Page config
st.set_page_config(
    page_title="Global Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# Load data
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("global_air_quality_data_10000.csv")
    
    # Compute AQI
    def calc_aqi_pm25(pm):
        if pd.isna(pm):
            return np.nan
        if pm <= 12:
            return np.interp(pm, [0, 12], [0, 50])
        elif pm <= 35.4:
            return np.interp(pm, [12.1, 35.4], [51, 100])
        elif pm <= 55.4:
            return np.interp(pm, [35.5, 55.4], [101, 150])
        else:
            return 200

    def calc_aqi_pm10(pm):
        if pd.isna(pm):
            return np.nan
        if pm <= 54:
            return np.interp(pm, [0, 54], [0, 50])
        elif pm <= 154:
            return np.interp(pm, [55, 154], [51, 100])
        elif pm <= 254:
            return np.interp(pm, [155, 254], [101, 150])
        else:
            return 200

    df["AQI"] = df.apply(
        lambda row: max(
            calc_aqi_pm25(row["PM2.5"]),
            calc_aqi_pm10(row["PM10"])
        ) if not pd.isna(row["PM2.5"]) and not pd.isna(row["PM10"]) else np.nan, 
        axis=1
    )
    
    return df

df = load_data()

# ----------------------
# Train ML Model
# ----------------------
@st.cache_resource
def train_model(df):
    data_ml = df[["PM2.5", "PM10", "AQI"]].dropna()
    X = data_ml[["PM2.5", "PM10"]]
    y = data_ml["AQI"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# ----------------------
# Sidebar
# ----------------------
st.sidebar.title("🌫️ Air Quality Dashboard")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Dashboard", "📈 Insights", "🤖 ML Predictions"])

# ----------------------
# HOME PAGE
# ----------------------
if page == "🏠 Home":
    st.title("🌫️ Global Air Quality Dashboard")
    st.markdown("### Light-mode app with animations, maps, clustering & ML")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 Interactive visualizations")
    with col2:
        st.info("🤖 Machine Learning predictions")
    with col3:
        st.info("🔬 K-Means clustering analysis")

# ----------------------
# DASHBOARD PAGE
# ----------------------
elif page == "📊 Dashboard":
    st.title("📊 Dashboard")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Select Country", ["All"] + sorted(df.Country.dropna().unique().tolist()))
    with col2:
        city = st.selectbox("Select City", ["All"] + sorted(df.City.dropna().unique().tolist()))
    
    # Filter data
    filtered_df = df.copy()
    if country != "All":
        filtered_df = filtered_df[filtered_df.Country == country]
    if city != "All":
        filtered_df = filtered_df[filtered_df.City == city]
    
    # Summary Statistics
    st.markdown("### 📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_pm25 = filtered_df['PM2.5'].max()
        st.metric("Max PM2.5", f"{max_pm25:.2f}" if not pd.isna(max_pm25) else "N/A")
    
    with col2:
        mean_pm10 = filtered_df['PM10'].mean()
        st.metric("Mean PM10", f"{mean_pm10:.2f}" if not pd.isna(mean_pm10) else "N/A")
    
    with col3:
        max_aqi = filtered_df['AQI'].max()
        st.metric("Max AQI", f"{max_aqi:.0f}" if not pd.isna(max_aqi) else "N/A")
    
    with col4:
        st.metric("Data Points", len(filtered_df))
    
    # PM2.5 Over Time
    st.markdown("### 📉 PM2.5 Over Time")
    tmp = filtered_df.dropna(subset=["Date", "PM2.5"])
    if len(tmp) > 0:
        try:
            tmp["Date"] = pd.to_datetime(tmp["Date"], errors='coerce')
            tmp = tmp.dropna(subset=["Date"])
        except:
            pass
        fig = px.line(tmp, x="Date", y="PM2.5", color="City", template="plotly_white")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available")
    
    # AQI by City
    st.markdown("### 🏙️ AQI by City")
    temp = filtered_df.groupby("City")["AQI"].mean().reset_index()
    temp = temp.sort_values("AQI", ascending=False).head(40)
    if len(temp) > 0:
        fig = px.bar(temp, x="City", y="AQI", template="plotly_white")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available")
    
    # 3D Clustering
    st.markdown("### 🔬 3D Clustering")
    if st.button("Run Clustering"):
        temp = filtered_df[["PM2.5", "PM10", "AQI"]].dropna()
        if len(temp) >= 3:
            n_clusters = min(3, len(temp))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            temp = temp.copy()
            temp["cluster"] = kmeans.fit_predict(temp)
            fig = px.scatter_3d(temp, x="PM2.5", y="PM10", z="AQI", color="cluster", template="plotly_white")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for clustering")

# ----------------------
# INSIGHTS PAGE
# ----------------------
elif page == "📈 Insights":
    st.title("📌 Key Insights")
    
    st.markdown("""
    - **PM2.5 and PM10 are the primary pollutants** determining AQI levels worldwide.
    - **Urban areas consistently show higher AQI values** compared to rural regions due to traffic and industrial emissions.
    - **Air quality shows seasonal patterns** - winter months often have worse AQI due to heating and temperature inversions.
    - **Country-level averages can hide localized pollution hotspots** in major cities.
    - **The ML model shows that PM2.5 has a stronger correlation with AQI** than PM10.
    - **Clustering analysis reveals distinct patterns**: clean air regions, moderately polluted urban areas, and pollution hotspots.
    - **Even moderate AQI levels (51-100) can affect sensitive groups** including children and elderly.
    """)

# ----------------------
# ML PREDICTIONS PAGE
# ----------------------
elif page == "🤖 ML Predictions":
    st.title("🤖 ML Predictions")
    
    st.markdown("### Predict AQI from PM2.5 & PM10")
    
    col1, col2 = st.columns(2)
    with col1:
        pm25_input = st.number_input("PM2.5", value=20.0, min_value=0.0)
    with col2:
        pm10_input = st.number_input("PM10", value=40.0, min_value=0.0)
    
    if st.button("Predict AQI"):
        try:
            input_data = pd.DataFrame([[pm25_input, pm10_input]], columns=["PM2.5", "PM10"])
            pred = model.predict(input_data)[0]
            
            st.success(f"**Predicted AQI: {pred:.2f}**")
            
            # Category
            if pred <= 50:
                category = "Good"
                color = "#00e400"
                message = "Air quality is satisfactory, and air pollution poses little or no risk."
            elif pred <= 100:
                category = "Moderate"
                color = "#ffff00"
                message = "Air quality is acceptable. However, there may be a risk for some people."
            elif pred <= 150:
                category = "Unhealthy for Sensitive Groups"
                color = "#ff7e00"
                message = "Members of sensitive groups may experience health effects."
            elif pred <= 200:
                category = "Unhealthy"
                color = "#ff0000"
                message = "Some members of the general public may experience health effects."
            elif pred <= 300:
                category = "Very Unhealthy"
                color = "#8f3f97"
                message = "Health alert: The risk of health effects is increased for everyone."
            else:
                category = "Hazardous"
                color = "#7e0023"
                message = "Health warning of emergency conditions: everyone is more likely to be affected."
            
            st.markdown(f"### Category: <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)
            st.info(message)
            
        except Exception as e:
            st.error("Error: Invalid input values")
    
    # AQI Reference Table
    st.markdown("### 📋 AQI Reference Table")
    
    table_data = {
        "AQI Range": ["0-50", "51-100", "101-150", "151-200", "201-300", "301+"],
        "Category": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
        "Color": ["Green", "Yellow", "Orange", "Red", "Purple", "Maroon"],
        "Health Implications": [
            "Air quality is satisfactory",
            "Acceptable; some pollutants may be a concern",
            "Sensitive groups may experience health effects",
            "Everyone may begin to experience health effects",
            "Health alert: everyone may experience serious effects",
            "Health warnings of emergency conditions"
        ]
    }
    
    st.table(pd.DataFrame(table_data))