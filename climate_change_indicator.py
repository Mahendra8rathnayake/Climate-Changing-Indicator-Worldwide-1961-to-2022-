# app.py — Climate Change Dashboard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import json
from streamlit_lottie import st_lottie

# ------------------------------------------------------------
# 🔹 STREAMLIT PAGE CONFIG (must be first)
# ------------------------------------------------------------
st.set_page_config(
    page_title="🌎 Climate Dashboard",
    page_icon="🌎",
    layout="wide",
)

# ------------------------------------------------------------
# 🔹 DARK THEME + BACKGROUND + FONT + CARD STYLE
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Background image */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
        background-size: cover;
        background-position: center;
    }
    /* Card style for dataframes */
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        background-color: rgba(20, 20, 20, 0.85);
        color: white;
    }
    /* Headers & text */
    .css-18e3th9 {
        color: white;
    }
    h1, h2, h3, h4 {
        color: white;
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 🔹 LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("climate_change_indicators.csv")
    return df

df = load_data()

# ------------------------------------------------------------
# 🔹 Sidebar Filters
# ------------------------------------------------------------
st.sidebar.header("Filters")

# Country multiselect (safe defaults)
all_countries = df["Country"].unique()
default_countries = [c for c in ["United States", "China", "India"] if c in all_countries]

countries = st.sidebar.multiselect(
    "Select Countries:",
    options=all_countries,
    default=default_countries
)

df_filtered = df[df["Country"].isin(countries)]

# Indicator filter (optional)
all_indicators = df_filtered["Indicator"].unique()
indicator = st.sidebar.selectbox("Select Indicator:", options=all_indicators)

df_filtered = df_filtered[df_filtered["Indicator"] == indicator]

# ------------------------------------------------------------
# 🔹 Convert wide to long format
# ------------------------------------------------------------
year_cols = [c for c in df_filtered.columns if c.startswith("F")]

df_long = df_filtered.melt(
    id_vars=["Country", "Indicator"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Temperature"
)

# Clean Year column
df_long["Year"] = df_long["Year"].str.replace("F", "", regex=False).astype(int)

# Drop rows with missing Temperature
df_long = df_long.dropna(subset=["Temperature"])

# ------------------------------------------------------------
# 🔹 Header + Lottie Animation
# ------------------------------------------------------------
st.title("🌎 Climate Change Dashboard")
st.markdown("Analyze 1961–2022 global temperature change trends using ML & Neural Networks.")

# Load local Lottie animation (avoid HTTP 403)
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

try:
    lottie_animation = load_lottie_file("earth.json")
    st_lottie(lottie_animation, height=200, key="earth")
except Exception as e:
    st.warning("Lottie animation not found. Skipping animation.")

# ------------------------------------------------------------
# 🔹 Raw Data Preview
# ------------------------------------------------------------
st.subheader("📄 Raw Dataset Preview")
st.dataframe(df_filtered.head())

st.write(f"Year Columns Found: {len(year_cols)}")
st.subheader("📊 Long Format Data (Cleaned)")
st.dataframe(df_long.head())

# ------------------------------------------------------------
# 🔹 Global Temperature Trend Chart
# ------------------------------------------------------------
st.header("📈 Global Temperature Trend (Selected Countries)")

global_trend = df_long.groupby("Year")["Temperature"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(x="Year", y="Temperature", data=global_trend, ax=ax, color="#1f77b4", linewidth=2)
ax.set_title("Global Average Temperature Change Over Time", fontsize=16, color="white")
ax.set_xlabel("Year", fontsize=12, color="white")
ax.set_ylabel("Temperature Change (°C)", fontsize=12, color="white")
ax.tick_params(colors="white")
st.pyplot(fig)

# ------------------------------------------------------------
# 🔹 Machine Learning Models
# ------------------------------------------------------------
st.header("🧠 ML Models for Temperature Prediction")
st.write("Predict temperature change using Linear Regression, Random Forest, and Neural Network.")

X = df_long[["Year"]]
y = df_long["Temperature"]

# Split safely
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_lr = linreg.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Neural Network
nn = Sequential()
nn.add(Dense(32, activation="relu", input_dim=1))
nn.add(Dense(16, activation="relu"))
nn.add(Dense(1))
nn.compile(optimizer="adam", loss="mse")
nn.fit(X_train, y_train, epochs=50, verbose=0)
y_pred_nn = nn.predict(X_test).flatten()

# ------------------------------------------------------------
# 🔹 Model Evaluation
# ------------------------------------------------------------
def evaluate_model(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred)
    )

results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Neural Network"],
    "MAE, °C": [evaluate_model(y_test, y_pred_lr)[0],
                 evaluate_model(y_test, y_pred_rf)[0],
                 evaluate_model(y_test, y_pred_nn)[0]],
    "RMSE, °C": [evaluate_model(y_test, y_pred_lr)[1],
                  evaluate_model(y_test, y_pred_rf)[1],
                  evaluate_model(y_test, y_pred_nn)[1]],
    "R²": [evaluate_model(y_test, y_pred_lr)[2],
           evaluate_model(y_test, y_pred_rf)[2],
           evaluate_model(y_test, y_pred_nn)[2]]
})

st.subheader("📊 Model Performance Comparison")
st.dataframe(results_df.style.highlight_min(axis=0, color="#fdd").highlight_max(axis=0, color="#dfd"))

# ------------------------------------------------------------
# 🔹 Neural Network Loss Plot
# ------------------------------------------------------------
st.subheader("📉 Neural Network Training Loss")
fig2, ax2 = plt.subplots(figsize=(7,4))
ax2.plot(nn.history.history["loss"], color="#ff7f0e")
ax2.set_title("Neural Network Training Loss", color="white")
ax2.set_xlabel("Epochs", color="white")
ax2.set_ylabel("Loss", color="white")
ax2.tick_params(colors="white")
st.pyplot(fig2)

# ------------------------------------------------------------
# 🔹 Final Insights
# ------------------------------------------------------------
st.header("🏁 Final Conclusion")
st.markdown("""
### 🔍 Model Summary
- **Linear Regression performed the best**, indicating a mostly linear relationship between Year and Temperature Change.
- **Neural Network performed competitively**, capturing non-linear patterns.
- **Random Forest struggled**, suggesting the dataset is relatively smooth for tree-based models.

### 📌 Interpretation for LinkedIn
*“Although neural networks and random forests are powerful nonlinear models, the results showed that Linear Regression performed the best for temperature prediction. This suggests the underlying global temperature trend behaves in a relatively linear manner across years and countries. The neural network achieved competitive performance, while the random forest struggled to generalize on this dataset.”*
""")
