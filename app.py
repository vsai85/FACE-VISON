import streamlit as st
import numpy as np
import cv2
import cvlib as cv
from deepface import DeepFace
from PIL import Image, ImageDraw
import json
import os
from datetime import datetime
from streamlit_lottie import st_lottie

# Create folder to store images and videos
SAVE_DIR = "detected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Page Configuration
st.set_page_config(layout='wide', page_title="Face Vision", page_icon="👦")

# Load Lottie Animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

logo = load_lottiefile("animation.json")

# Sidebar UI
with st.sidebar:
    st.title('Experience the power of Computer Vision')
    st.success('Detect emotions in real-time using your webcam or an image.')
    mode = st.selectbox('Select Mode', ['<Select>', 'Capture', 'Webcam'])

# Main Title
st.markdown("""
    <h1 style='text-align: center;'>Face Vision</h1>
    <br>
    """, unsafe_allow_html=True)

# Display Lottie Animation
st_lottie(logo, speed=1, width=250, height=250)

if mode == '<Select>':
    st.warning('👈 Select a mode from the sidebar to start!')

# Image Capture Mode
if mode == 'Capture':
    image = st.camera_input("Capture a snapshot")
    if image is not None:
        image = Image.open(image)
        image_np = np.array(image)
        faces, confidences = cv.detect_face(image_np)
        draw = ImageDraw.Draw(image)
        
        for f in faces:
            (startX, startY, endX, endY) = f
            draw.rectangle(((startX, startY), (endX, endY)), outline="green", width=2)
            face_crop = image_np[startY:endY, startX:endX]
            
            # Emotion Detection
            obj = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            label = obj[0]['dominant_emotion'] if obj else "No emotion detected"
            draw.text((startX, startY-10), label, fill="green")
        
        # Save the captured image in detected_images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = os.path.join(SAVE_DIR, f"captured_{timestamp}.png")
        image.save(img_filename)

        st.image(image, caption=f"Processed Image (Saved as {img_filename})", use_column_width=True)
        st.write('**Emotion Detected:**', label)

# Live Webcam Mode
if mode == 'Webcam':
    st.write("### Real-Time Emotion Detection")
    
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    # Prepare video writer to save the webcam recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(SAVE_DIR, f"recorded_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

    stop_button = st.button("Stop Webcam")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam. Ensure it's properly connected.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, confidences = cv.detect_face(frame_rgb)

        for f in faces:
            (startX, startY, endX, endY) = f
            cv2.rectangle(frame_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)
            face_crop = frame_rgb[startY:endY, startX:endX]

            # Emotion Detection
            obj = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            label = obj[0]['dominant_emotion'] if obj else "No emotion detected"
            cv2.putText(frame_rgb, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_placeholder.image(frame_rgb, channels="RGB")

        # Save the frame to the video
        out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        # Stop webcam when button is pressed
        if stop_button:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    st.success(f"Video saved as {video_filename}")
