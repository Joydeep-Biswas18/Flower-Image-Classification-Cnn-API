import streamlit as st
import PIL
import requests
from PIL import Image
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="FlowerVision AI",
    page_icon="🌸",
    layout="wide"
)

# ---------------- CSS ----------------
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🌸 FlowerVision AI")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Classify Flower", "About Model"]
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient( 135deg,
        #fff0f6,
        #f3e8ff,
        #e0f2fe);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fde2e4, #f3d1f4);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.main > div {
    background: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
if menu == "Home":
    st.markdown("""
    <div class="hero">
        <h1>🌸 FlowerVision AI</h1>
        <p>Deep Learning powered flower classification with 102 categories</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Flower Classes", "102")
    c2.metric("Model Type", "CNN")
    c3.metric("Accuracy", "96%+")

    st.markdown("### 🌼 What can this app do?")
    st.write("""
    - Upload a flower image  
    - Identify the species instantly  
    - Get confidence scores  
    - Explore AI-powered plant recognition  
    """)

# ---------------- CLASSIFIER ----------------
elif menu == "Classify Flower":
    st.markdown("## 🌼 Flower Classification")

    uploaded_file = st.file_uploader(
        "Upload a flower image",
        type=["jpg", "png", "jpeg"]
    )
   

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)

        if st.button("🔍 Predict Flower"):
            files={"file": uploaded_file.getvalue()}

            with st.spinner("Analyzing petals & patterns..."):
                response = requests.post("http://127.0.0.1:8000/predict",
                                         files = files) 
                st.success("Prediction Complete 🌸")

                if response.status_code==200:
                    result = response.json()
                    predicted_class = result["Flower_class"]
                    confidence = result["Confidence"]
                    top5_predictions = result["Top Five Prediction"]
                    st.markdown(f"""
                                <div class="result-card">
                                <h2>{predicted_class}</h2>
                                <p>Confidence: {confidence*100:.2f}%</p>
                                </div>
                    """, unsafe_allow_html=True)
                    st.markdown("### 🔝 Top Predictions")
                    for item in top5_predictions[:5]:
                        name = item["class"]
                        p = item["confidence"]  # already a float
                        st.progress(p)
                        st.write(f"**{name}** — {p*100:.2f}%")
                else:
                    st.error(f"Prediction Failed...")


            

# ---------------- ABOUT ----------------
else:
    st.markdown("## 🤖 About the Model")
    st.write("""
    - Dataset: Oxford 102 Flower Dataset  
    - Architecture: CNN / Transfer Learning  
    - Input Size: 224×224  
    - Framework: TensorFlow / Keras  
    """)

    st.info("Model can be upgraded with EfficientNet, or MobileNet")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit & Deep Learning
</div>
""", unsafe_allow_html=True)


import base64
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


            