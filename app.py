import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import datetime
import os

# Load YOLO model
MODEL_PATH = "best.pt"  # Ensure best.pt is in the same directory
model = YOLO(MODEL_PATH)

# Function to detect fire/smoke
def detect_fire(frame):
    results = model(frame)
    annotated_frame = results[0].plot()  # Annotated image with boxes

    # Check if fire or smoke is detected
    if len(results[0].boxes) > 0:
        save_alert(frame)
        raise_alarm()

    return annotated_frame

# Function to raise an alarm
def raise_alarm():
    st.warning("🔥 Fire/Smoke Detected! Raising Alarm!")

    # Use Streamlit's built-in audio player instead of playsound()
    audio_file = "alarm.mp3"  # Ensure this file exists in the same directory
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")

# Function to log alerts
def save_alert(frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    log_entry = f"{timestamp} - Fire/Smoke Detected\n"

    # Append log to file
    with open("fire_logs.txt", "a") as log_file:
        log_file.write(log_entry)

    # Save detected frame
    alert_img_path = f"detected_{timestamp}.jpg"
    cv2.imwrite(alert_img_path, frame)

# Streamlit UI
st.title("🔥 Fire and Smoke Detection System")
st.sidebar.header("Upload Video, Image, or Enter CCTV Link")

option = st.sidebar.radio("Choose an Input", ("Image", "Video", "CCTV"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        detected_image = detect_fire(image)
        st.image(detected_image, channels="BGR", caption="Processed Image")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded_video:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detected_frame = detect_fire(frame)
            stframe.image(detected_frame, channels="BGR", caption="Processing Video")

        cap.release()
        os.remove(temp_video_path)

elif option == "CCTV":
    cctv_url = st.text_input("Enter CCTV Stream URL")
    if cctv_url:
        cap = cv2.VideoCapture(cctv_url)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detected_frame = detect_fire(frame)
            stframe.image(detected_frame, channels="BGR", caption="Live CCTV Feed")

        cap.release()
