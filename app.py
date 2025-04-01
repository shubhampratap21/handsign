import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image

# Configuration
MODEL_PATH = "models/best_model.keras"
CLASS_NAMES = [chr(i) for i in range(65, 91)]  # A-Z

class ASLTranslator:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Load trained model
        self.model = tf.keras.models.load_model(MODEL_PATH)
        
    def predict(self, frame):
        """Process frame and return prediction"""
        # Convert and process image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(image)
        
        if results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Normalize and predict
            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = self.model.predict(landmarks, verbose=0)
            return CLASS_NAMES[np.argmax(prediction)], np.max(prediction)
        
        return "", 0.0

def main():
    st.set_page_config(page_title="Real-Time ASL Translator", layout="wide")
    st.title("Sign Language Translator 🤟")
    
    # Initialize translator
    if "translator" not in st.session_state:
        st.session_state.translator = ASLTranslator()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    app_mode = st.sidebar.radio("Select Input Mode:", 
                               ["Webcam", "Image Upload"])
    
    # Main content area
    if app_mode == "Webcam":
        webcam_interface()
    else:
        upload_interface()

def webcam_interface():
    """Real-time webcam processing"""
    st.header("Live Camera Translation")
    run = st.checkbox("Start Camera", key="webcam_run")
    FRAME_WINDOW = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        # Mirror display
        frame = cv2.flip(frame, 1)
        
        # Get prediction
        prediction, confidence = st.session_state.translator.predict(frame)
        
        # Add overlay
        cv2.putText(frame, f"{prediction} ({confidence*100:.1f}%)", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                   (0, 255, 0), 3, cv2.LINE_AA)
        
        # Display frame
        FRAME_WINDOW.image(frame, channels="BGR")
    
    cap.release()

def upload_interface():
    """Image upload processing"""
    st.header("Image Upload Translation")
    uploaded_file = st.file_uploader("Upload ASL Image", 
                                    type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Process image
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get prediction
        prediction, confidence = st.session_state.translator.predict(frame)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.success(f"Prediction: **{prediction}**")
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            if confidence < 0.7:
                st.warning("Low confidence prediction - ensure clear hand sign")

if __name__ == "__main__":
    main()
