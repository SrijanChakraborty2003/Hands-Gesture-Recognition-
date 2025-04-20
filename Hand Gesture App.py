import streamlit as st
import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms
from PIL import Image
import numpy as np
import tempfile
class_names = ['c', 'down', 'fist', 'fist_moved', 'index', 'l', 'ok', 'palm', 'palm_moved', 'thumb']
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load("gesture_mobilenetv2.pth", map_location=torch.device('cpu')))
    model.eval()
    return model
model = load_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
st.title("Real-time Hand Gesture Recognition")
st.write("Show your hand gesture in front of your webcam and get predictions in real-time.")
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])
if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        input_tensor = transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_class = class_names[pred.item()]
        cv2.putText(frame, f"Prediction: {pred_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    st.write("Check the box to start the webcam.")