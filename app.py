import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from mtcnn import MTCNN  # For face detection
import tempfile
import os

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

# Load the trained model
model_path = "violence_detection_model_.keras"
MoBiLSTM_model = tf.keras.models.load_model(model_path)

# Initialize MTCNN for face detection
detector = MTCNN()

# Function to extract frames from video
def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    
    video_reader.release()
    return frames_list

# Function to detect faces using MTCNN
def detect_faces(frame):
    faces = detector.detect_faces(frame)
    return faces

# Function to predict video
def predict_video(video_file_path):
    frames_list = frames_extraction(video_file_path)
    
    if len(frames_list) == 0:
        st.error("Error: No frames were extracted from the video.")
        return None, None, None
    
    frames_array = np.array(frames_list)
    frames_array = np.expand_dims(frames_array, axis=0)
    
    predicted_labels_probabilities = MoBiLSTM_model.predict(frames_array)[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    confidence = predicted_labels_probabilities[predicted_label]
    
    # Extract faces if violence is detected
    faces_list = []
    if predicted_class_name == "Violence":
        video_reader = cv2.VideoCapture(video_file_path)
        for frame_counter in range(SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * (int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT)) // SEQUENCE_LENGTH))
            success, frame = video_reader.read()
            if not success:
                break
            faces = detect_faces(frame)
            for face in faces:
                x, y, width, height = face['box']
                face_img = frame[y:y+height, x:x+width]
                faces_list.append(face_img)
        video_reader.release()
    
    return predicted_class_name, confidence, faces_list

# Streamlit App
st.set_page_config(page_title="Violence Detection", page_icon=":movie_camera:", layout="centered")

# Custom CSS for attractive UI
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1b1b1b;
        color: white;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader>div>div>button {
        background-color: #4CAF50;
        color: white;
    }
    .stFileUploader>div>div>button:hover {
        background-color: #45a049;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: white;
        text-align: center;
    }
    .stMarkdown p {
        color: white;
        text-align: center;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        .center {
            text-align: center;
        }
        .title {
            font-family: 'Arial', sans-serif;  /* Change to desired font */
            font-size: 50px;  /* Adjust size */
            font-weight: bold;
            color: #ff0000;  /* Adjust color */
        }
        .subtitle {
            font-family: 'Courier New', monospace; /* Example of a different font */
            font-size: 24px;
            color: #de4743;
        }
    </style>

    <div class="center">
        <h1 class="title">🚨 SafeWatch 🚨</h1>
        <h4 class="subtitle">Surveillance with a purpose</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# Title and description with centered alignment
image_path = "violence frame.jpeg"  # Replace with your local image path
try:
    image = Image.open(image_path)
    st.image(image, use_container_width=True)
except FileNotFoundError:
    st.error(f"Image not found at path: {image_path}")

st.write("Upload a video to detect whether it contains violence.")

# File uploader
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name
    
    # Display the video
    st.video(temp_file_path)
    
    # Predict the video
    predicted_class, confidence, faces_list = predict_video(temp_file_path)
    
    # Display the result
    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Display a message based on the prediction
    if predicted_class == "Violence":
        st.error("⚠️ Warning: Violence detected in the video!")
        
        # Display extracted faces
        if len(faces_list) > 0:
            st.subheader("Faces Detected in Violent Frames")
            cols = st.columns(4)  # Display 4 faces per row
            for i, face_img in enumerate(faces_list):
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                with cols[i % 4]:
                    st.image(face_img, caption=f"Face {i+1}", use_container_width=True)
        else:
            st.warning("No faces detected in violent frames.")
    else:
        st.success("✅ No violence detected in the video.")
    
    # Clean up temporary file
    os.unlink(temp_file_path)
