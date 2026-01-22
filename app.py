import streamlit as st
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Consumer Complaint Classifier",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("svc_text_classifier.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown("## Consumer Complaint Classification")
st.markdown(
    "Automatically classify consumer financial complaints using a machine learning NLP model."
)
st.markdown("---")

# -----------------------------
# Main Input Section
# -----------------------------
st.markdown("### Complaint Narrative")

user_input = st.text_area(
    label="Enter the consumer complaint text below",
    height=220,
    placeholder="Example: I found inaccurate information on my credit report and the bank is not resolving it..."
)

# -----------------------------
# Prediction Button
# -----------------------------
st.markdown("")

predict_button = st.button("Classify Complaint", use_container_width=True)

# -----------------------------
# Output Section
# -----------------------------
if predict_button:
    if user_input.strip() == "":
        st.warning("Please enter a complaint narrative before submitting.")
    else:
        prediction = model.predict([user_input])[0]

        st.markdown("### Prediction Result")
        st.success(f"Predicted Category: **{prediction}**")

# -----------------------------
# Sidebar (Clean & Minimal)
# -----------------------------
with st.sidebar:
    st.markdown("### About the Model")
    st.markdown(
        """
        This system uses:
        - TF-IDF text vectorization  
        - Linear Support Vector Machine  
        - Trained on real CFPB consumer complaints
        """
    )

    st.markdown("### Supported Categories")
    st.markdown(
        """
        - Credit Reporting  
        - Debt Collection  
        - Mortgages and Loans  
        - Credit Cards  
        - Retail Banking  
        """
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Machine Learning Project | NLP Text Classification | Streamlit Deployment"
)
