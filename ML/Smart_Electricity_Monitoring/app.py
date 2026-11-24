import streamlit as st
import pandas as pd

import joblib

from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="AI Smart Electricity Monitoring",
    page_icon="⚡",
    layout="wide"
)

# Load dataset and model
@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_appliance.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("results/ml_ready_dataset.csv")
    df = df[df["session_id"] != 7].copy()

    features = [
        "voltage_V",
        "current_A",
        "power_W",
        "frequency_Hz",
        "power_factor"
    ]

    X = df[features]
    y = df["appliance"]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_test, y_test, features

model = load_model()
X_test, y_test, features = load_data()

# Dashboard heading
st.title("⚡ AI Smart Electricity Monitoring System")
st.caption("Machine Learning Based Appliance Identification")
st.divider()

# Sidebar
st.sidebar.header("Control Panel")

st.sidebar.success("Random Forest Model Loaded")

st.sidebar.write("Model Test Accuracy")
st.sidebar.metric("Accuracy", "99.33%")

# Select reading
st.sidebar.subheader("Select Electrical Reading")

reading_number = st.sidebar.selectbox(
    "Choose a test reading",
    range(len(X_test)),
    format_func=lambda x: f"Reading {x + 1}"
)

reading = X_test.iloc[reading_number]
actual = y_test.iloc[reading_number]

# Main dashboard
st.subheader("Electrical Parameters")

col1, col2, col3 = st.columns(3)

col1.metric("Voltage", f"{reading['voltage_V']:.2f} V")
col2.metric("Current", f"{reading['current_A']:.3f} A")
col3.metric("Power", f"{reading['power_W']:.2f} W")

col4, col5 = st.columns(2)

col4.metric("Power Factor", f"{reading['power_factor']:.2f}")
col5.metric("Frequency", f"{reading['frequency_Hz']:.2f} Hz")

st.divider()

# Prediction button
st.subheader("AI Appliance Identification")

if st.button("Identify Appliance", type="primary"):

    input_data = pd.DataFrame(
        [reading],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100

    st.success(f"Predicted Appliance: {prediction}")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    if prediction == actual:
        st.success("Prediction matches the recorded appliance.")
    else:
        st.warning("Prediction does not match the recorded appliance.")

    st.write("Recorded Appliance:", actual)

    # Probability for each appliance
    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({
        "Appliance": model.classes_,
        "Probability (%)": probabilities * 100
    }).sort_values(
        "Probability (%)",
        ascending=False
    )

    st.bar_chart(
        probability_df.set_index("Appliance")
    )

st.divider()

st.subheader("Project Information")

st.write("""
This system uses a Random Forest machine learning algorithm
to identify electrical appliances based on:

- Voltage
- Current
- Power
- Power Factor
- Frequency

The current dashboard demonstrates predictions using previously
collected electrical readings. Live ESP32 integration will be
added in a subsequent stage.
""")

st.caption("Final Year Project | AI-Based Smart Electricity Monitoring")
