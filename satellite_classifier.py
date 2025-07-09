import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .instruction-text {
        font-size: 1.4rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        font-weight: 500;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    .prediction-box h3 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-box p {
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: bold;
        color: #FFD700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .upload-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

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
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure 'Modelenv.v1.h5' is in the same directory as this script.")
        return None

def preprocess_image(img):
    """Preprocess image for prediction"""
    if not TENSORFLOW_AVAILABLE:
        return np.random.rand(1, 255, 255, 3)  # Demo data
    
    # Resize image to model input size
    img = img.resize((255, 255))
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
    
    if not TENSORFLOW_AVAILABLE or model == "demo_model":
        # Demo mode - generate realistic-looking predictions
        np.random.seed(42)  # For consistent demo results
        prediction = np.random.dirichlet(np.ones(4) * 2)  # More realistic distribution
        prediction = prediction.reshape(1, -1)
    else:
        prediction = model.predict(img_array)
    
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    # Create prediction probabilities dictionary
    probabilities = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    
    return predicted_class, confidence, probabilities

def create_confidence_chart(probabilities):
    """Create an interactive confidence chart"""
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [CLASS_INFO[cls]['color'] for cls in classes]
    
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
    # Header
    st.markdown('<h1 class="main-header">🛰️ Satellite Image Classifier</h1>', unsafe_allow_html=True)
    
    # Clear instruction line
    st.markdown('''
    <div class="instruction-text">
        📸 Upload an image and let the model classify it as <strong>Cloudy</strong>, <strong>Water</strong>, <strong>Green Area</strong>, or <strong>Desert</strong>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Classification Categories")
    st.sidebar.markdown("""
    **The model can identify:**
    - ☁️ **Cloudy** - Cloud-covered areas
    - 🏜️ **Desert** - Arid, sandy regions  
    - 🌳 **Green Area** - Forests & vegetation
    - 💧 **Water** - Lakes, rivers, oceans
    """)
    
    st.sidebar.markdown("## 🚀 How to Use")
    st.sidebar.markdown("""
    1. **Upload** a satellite image
    2. **Wait** for processing
    3. **View** the classification result
    4. **Analyze** confidence scores
    """)
    
    st.sidebar.markdown("## 📊 Model Info")
    if TENSORFLOW_AVAILABLE:
        st.sidebar.success("✅ TensorFlow Model Loaded")
    else:
        st.sidebar.warning("⚠️ Demo Mode Active")
    
    # Load model
    model = load_classification_model()
    
    if model is None:
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('''
        <div class="upload-section">
            <h2 style="color: #2c3e50; margin-bottom: 1rem;">📤 Upload Your Satellite Image</h2>
            <p style="color: #34495e; font-size: 1.1rem;">Choose a satellite image file to classify</p>
        </div>
        ''', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a satellite image for land cover classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            with st.spinner("Classifying image..."):
                img_array = preprocess_image(image_pil)
                predicted_class, confidence, probabilities = predict_image(model, img_array)
            
            # Display prediction results
            st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Main prediction box
            icon = CLASS_INFO[predicted_class]['icon']
            st.markdown(f"""
            <div class="prediction-box">
                <h3>{icon} {predicted_class}</h3>
                <p>Confidence Score</p>
                <div class="confidence-score">{confidence:.1%}</div>
                <p style="margin-top: 1rem; font-size: 1.1rem;">{CLASS_INFO[predicted_class]['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<h2 class="sub-header">📊 Confidence Analysis</h2>', unsafe_allow_html=True)
            
            # Interactive confidence chart
            fig = create_confidence_chart(probabilities)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probabilities table
            st.markdown('<h3 class="sub-header">📈 Detailed Scores</h3>', unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                'Class': [f"{CLASS_INFO[cls]['icon']} {cls}" for cls in probabilities.keys()],
                'Confidence': [f"{prob:.2%}" for prob in probabilities.values()],
                'Score': list(probabilities.values())
            })
            
            # Sort by confidence
            prob_df = prob_df.sort_values('Score', ascending=False)
            
            st.dataframe(
                prob_df[['Class', 'Confidence']],
                use_container_width=True,
                hide_index=True
            )
    
    # Additional information section
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📈 Detailed Analysis</h2>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">📋 Image Information</h4>
                <p><strong>Dimensions:</strong> {image_pil.size[0]} × {image_pil.size[1]} pixels</p>
                <p><strong>Color Mode:</strong> {image_pil.mode}</p>
                <p><strong>File Format:</strong> {image_pil.format or 'Unknown'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Get top 2 predictions
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            top_class, top_conf = sorted_probs[0]
            second_class, second_conf = sorted_probs[1]
            
            st.markdown(f"""
            <div class="info-box">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">🏆 Top Predictions</h4>
                <p><strong>🥇 First:</strong> {top_class} ({top_conf:.1%})</p>
                <p><strong>🥈 Second:</strong> {second_class} ({second_conf:.1%})</p>
                <p><strong>📊 Confidence Gap:</strong> {(top_conf - second_conf):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 2rem;">
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
        <p>Environmental Monitoring and Land Cover Classification System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
