import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

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
    ("Take a live photo", "Upload an image")
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
