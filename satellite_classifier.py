import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px

# Try to import TensorFlow, fall back to demo mode if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow not found. Running in demo mode with simulated predictions.")

# Page configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class information
CLASS_INFO = {
    'Cloudy': {
        'description': 'Areas covered by clouds, typically appearing white or gray in satellite imagery.',
        'color': '#87CEEB',
        'icon': '☁️'
    },
    'Desert': {
        'description': 'Arid land areas with minimal vegetation, appearing as brown or tan regions.',
        'color': '#F4A460',
        'icon': '🏜️'
    },
    'Green_Area': {
        'description': 'Vegetated areas including forests, grasslands, and agricultural land.',
        'color': '#228B22',
        'icon': '🌳'
    },
    'Water': {
        'description': 'Bodies of water including oceans, lakes, rivers, and reservoirs.',
        'color': '#4682B4',
        'icon': '💧'
    }
}

@st.cache_resource
def load_classification_model():
    """Load the pre-trained model"""
    if not TENSORFLOW_AVAILABLE:
        return "demo_model"

    try:
        model = load_model('Modelenv.v1.h5')
        st.sidebar.success("✅ Real TensorFlow model loaded.")
        return model
    except Exception as e:
        st.sidebar.error(f"🚫 Error loading model: {str(e)}")
        st.sidebar.info("Please place 'Modelenv.v1.h5' in the same folder as this script.")
        return "demo_model"

def preprocess_image(img):
    """Preprocess image for prediction"""
    if not TENSORFLOW_AVAILABLE:
        return np.random.rand(1, 255, 255, 3)

    img = img.resize((255, 255))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

    if not TENSORFLOW_AVAILABLE or model == "demo_model":
        np.random.seed()
        prediction = np.random.dirichlet(np.ones(4))
        prediction = prediction.reshape(1, -1)
    else:
        prediction = model.predict(img_array)

    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]

    st.write("🔍 Raw model prediction output:", prediction[0])

    probabilities = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    return predicted_class, confidence, probabilities

def create_confidence_chart(probabilities):
    """Create an interactive confidence chart"""
    classes = list(probabilities.keys())
    values = list(probabilities.values())

    fig = px.bar(
        x=classes,
        y=values,
        color=classes,
        color_discrete_map={cls: CLASS_INFO[cls]['color'] for cls in classes},
        title="Prediction Confidence Scores",
        labels={'x': 'Land Cover Class', 'y': 'Confidence Score'}
    )

    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Land Cover Class",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1])
    )
    return fig

def main():
    st.title("🛰️ Satellite Image Classifier")

    st.markdown("""
        📸 Upload a satellite image and let the model classify it as:
        - ☁️ Cloudy
        - 🏜️ Desert
        - 🌳 Green Area
        - 💧 Water
    """)

    st.sidebar.header("🎯 Classification Categories")
    for name, data in CLASS_INFO.items():
        st.sidebar.markdown(f"- {data['icon']} **{name}**: {data['description']}")

    st.sidebar.header("🚀 How to Use")
    st.sidebar.markdown("1. Upload a satellite image\n2. View the classification\n3. Analyze confidence scores")

    # Load model
    model = load_classification_model()
    if model == "demo_model":
        st.warning("⚠️ Running in demo mode — model not loaded.")
    else:
        st.success("✅ Model is ready.")

    uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Classifying image..."):
            img_array = preprocess_image(image_pil)
            predicted_class, confidence, probabilities = predict_image(model, img_array)

        icon = CLASS_INFO[predicted_class]['icon']
        st.markdown(f"### 🎯 Prediction: {icon} **{predicted_class}**")
        st.markdown(f"**Confidence Score:** `{confidence:.2%}`")
        st.markdown(f"📄 {CLASS_INFO[predicted_class]['description']}")

        st.markdown("---")
        st.markdown("### 📊 Confidence Scores")
        st.plotly_chart(create_confidence_chart(probabilities), use_container_width=True)

        st.markdown("### 📈 Detailed Confidence Table")
        df = pd.DataFrame({
            "Class": list(probabilities.keys()),
            "Confidence": [f"{v:.2%}" for v in probabilities.values()]
        }).sort_values(by="Confidence", ascending=False)
        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.markdown(f"📏 **Image Details:** {image_pil.width} × {image_pil.height}, Mode: {image_pil.mode}, Format: {image_pil.format}")

if __name__ == "__main__":
    main()
