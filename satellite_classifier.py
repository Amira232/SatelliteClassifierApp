import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import os

# TensorFlow Import
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image

    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    st.error(f"TensorFlow Import Error: {e}")

# Page Configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide"
)

# Class Information
CLASS_INFO = {
    'Cloudy': {
        'description': 'Areas covered by clouds.',
        'color': '#87CEEB',
        'icon': '☁️'
    },
    'Desert': {
        'description': 'Arid land with little vegetation.',
        'color': '#F4A460',
        'icon': '🏜️'
    },
    'Green_Area': {
        'description': 'Vegetation, forests, grasslands.',
        'color': '#228B22',
        'icon': '🌳'
    },
    'Water': {
        'description': 'Lakes, rivers, oceans.',
        'color': '#4682B4',
        'icon': '💧'
    }
}

# Load Model
@st.cache_resource
def load_classification_model():

    if not TENSORFLOW_AVAILABLE:
        return None

    model_path = "Modelenv.v1.h5"

    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None

        st.sidebar.write("📂 Model found")

        model_size = os.path.getsize(model_path) / (1024 * 1024)
        st.sidebar.write(f"📦 Model Size: {model_size:.2f} MB")

        model = load_model(model_path, compile=False)

        st.sidebar.success("✅ TensorFlow Model Loaded Successfully")
        return model

    except Exception as e:
        st.error(f"❌ Model Loading Error:\n{e}")
        return None


# Image Preprocessing
def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((255, 255))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array


# Prediction Function
def predict_image(model, img_array):

    class_names = [
        'Cloudy',
        'Desert',
        'Green_Area',
        'Water'
    ]

    prediction = model.predict(img_array)

    st.write("Raw Prediction:", prediction[0])

    predicted_index = np.argmax(prediction[0])

    predicted_class = class_names[predicted_index]

    confidence = float(prediction[0][predicted_index])

    probabilities = {
        class_names[i]: float(prediction[0][i])
        for i in range(len(class_names))
    }

    return predicted_class, confidence, probabilities


# Confidence Chart
def create_confidence_chart(probabilities):

    classes = list(probabilities.keys())
    values = list(probabilities.values())

    fig = px.bar(
        x=classes,
        y=values,
        color=classes,
        color_discrete_map={
            cls: CLASS_INFO[cls]['color']
            for cls in classes
        },
        title="Prediction Confidence Scores"
    )

    fig.update_layout(
        showlegend=False,
        yaxis=dict(range=[0, 1]),
        height=400
    )

    return fig


# Main App
def main():

    st.title("🛰️ Satellite Image Classifier")

    st.markdown("""
    Upload a satellite image and classify it into:

    - ☁️ Cloudy
    - 🏜️ Desert
    - 🌳 Green Area
    - 💧 Water
    """)

    # Debug Information
    st.sidebar.header("System Information")

    st.sidebar.write("Current Directory:")
    st.sidebar.code(os.getcwd())

    st.sidebar.write("Files Available:")

    try:
        st.sidebar.code("\n".join(os.listdir()))
    except:
        pass

    # Load Model
    model = load_classification_model()

    if model is None:
        st.error("⚠️ Model could not be loaded.")
        st.stop()

    st.success("✅ Model Ready")

    uploaded_file = st.file_uploader(
        "Upload Satellite Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image_pil = Image.open(uploaded_file)

        st.image(
            image_pil,
            caption="Uploaded Image",
            use_container_width=True
        )

        with st.spinner("Classifying..."):

            img_array = preprocess_image(image_pil)

            predicted_class, confidence, probabilities = predict_image(
                model,
                img_array
            )

        st.markdown("---")

        icon = CLASS_INFO[predicted_class]["icon"]

        st.markdown(
            f"## {icon} Prediction: {predicted_class}"
        )

        st.markdown(
            f"### Confidence: {confidence:.2%}"
        )

        st.write(
            CLASS_INFO[predicted_class]["description"]
        )

        st.markdown("---")

        st.plotly_chart(
            create_confidence_chart(probabilities),
            use_container_width=True
        )

        df = pd.DataFrame({
            "Class": probabilities.keys(),
            "Confidence": [
                f"{v:.2%}" for v in probabilities.values()
            ]
        })

        st.dataframe(df)

        st.markdown("---")

        st.write(
            f"Image Size: {image_pil.width} x {image_pil.height}"
        )

        st.write(
            f"Image Mode: {image_pil.mode}"
        )

        st.write(
            f"Image Format: {image_pil.format}"
        )


if __name__ == "__main__":
    main()
