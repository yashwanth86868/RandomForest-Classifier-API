import os
import streamlit as st
import requests



API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🤖",
    layout="wide"
)

# Store prediction history during the session
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🤖 Customer Purchase Prediction")
st.caption("Machine Learning Dashboard | Random Forest Classifier")

st.divider()

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:
    salary = st.number_input(
        "Estimated Salary",
        min_value=0,
        value=50000
    )

if st.button("Predict Purchase", type="primary"):

    try:
        response = requests.post(
            API_URL,
            json={"age": age, "salary": salary},
            timeout=10
        )

        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        probability = result["probability"]

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success("Customer is predicted to purchase")
            else:
                st.warning("Customer is predicted not to purchase")

        with col2:
            st.metric(
                "Prediction Confidence",
                f"{probability:.2f}%"
            )

        # Save prediction to session history
        st.session_state.history.append({
            "Age": age,
            "Salary": salary,
            "Prediction": prediction,
            "Confidence": f"{probability:.2f}%"
        })

    except requests.exceptions.RequestException as error:
        st.error(f"API connection error: {error}")

st.divider()

# Display prediction history
st.subheader("📊 Prediction History")

if st.session_state.history:

    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No predictions yet. Make your first prediction!")

st.divider()

st.caption("Powered by Python | FastAPI | Streamlit | Scikit-learn")
