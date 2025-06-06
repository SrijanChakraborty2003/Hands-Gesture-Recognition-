import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
model.load_state_dict(torch.load("gesture_mobilenetv2.pth", map_location=torch.device("cpu")))
model.eval()
class_names = ['c', 'down', 'fist', 'fist_moved', 'index', 'l', 'ok', 'palm', 'palm_moved', 'thumb']
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
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
if option == "Take a live photo":
    img = st.camera_input("Capture hand gesture using webcam")
    if img:
        image = Image.open(img)
        prediction = predict(image)
        st.image(image, caption="Captured Image", use_column_width=True)
        st.success(f"Prediction: **{prediction}**")
elif option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        prediction = predict(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"Prediction: **{prediction}**")
