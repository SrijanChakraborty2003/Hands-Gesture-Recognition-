import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import tempfile

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
model.load_state_dict(torch.load("gesture_mobilenetv2.pth", map_location=torch.device("cpu")))
model.eval()

# Class names
class_names = ['c', 'down', 'fist', 'fist_moved', 'index', 'l', 'ok', 'palm', 'palm_moved', 'thumb']

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🖐️ Hand Gesture Recognition App")

option = st.selectbox(
    "Choose input type:",
    ("Take a live photo", "Upload an image", "Upload a video")
)

def predict(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# 1. Live Photo Input
if option == "Take a live photo":
    img = st.camera_input("Capture hand gesture using webcam")
    if img:
        image = Image.open(img)
        prediction = predict(image)
        st.image(image, caption="Captured Image", use_column_width=True)
        st.success(f"Prediction: **{prediction}**")

# 2. Upload an Image
elif option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        prediction = predict(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"Prediction: **{prediction}**")

# 3. Upload a Video
elif option == "Upload a video":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = total_frames // 2  # Pick the middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            prediction = predict(image)
            st.image(image, caption="Frame from Uploaded Video", use_column_width=True)
            st.success(f"Prediction from frame {target_frame}: **{prediction}**")
        else:
            st.error("Could not extract frame from the video.")
