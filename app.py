import streamlit as st
import torch 
from torch import nn 
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

classes_name = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___Late_blight",
    4: "Potato___healthy",
    5: "Tomato_Bacterial_spot",
    6: "Tomato_Early_blight",
    7: "Tomato_Late_blight",
    8: "Tomato_Leaf_Mold",
    9: "Tomato_Septoria_leaf_spot",
    10: "Tomato_Spider_mites_Two_spotted_spider_mite",
    11: "Tomato__Target_Spot",
    12: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    13: "Tomato__Tomato_mosaic_virus",
    14: "Tomato_healthy"
}

num_classes = 15 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(weights=None)
target_layer = [model.layer4[-1]]
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("resnet50_finetuned(FULL).pth"))
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Image classifier")

uploaded_file = st.file_uploader("Upload an image (JPEG, PNG, WEBP)", type=["jpeg", "png", "webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image")
    
    # Ensure the image is in the correct format
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()
    
    st.subheader("Prediction")
    st.write(f"Confidence: {confidence:.2%}")
    st.write(f"Predicted class: {classes_name[predicted_class]}")
    
    st.subheader("CAM")
    with GradCAM(model=model, target_layers=target_layer) as cam:
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        
        # Convert image to tensor for normalization
        image_tensor = transform(image).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        rgb_img = image_tensor * std + mean
        rgb_img = rgb_img.permute(1, 2, 0)
        rgb_img = rgb_img.cpu().numpy()  # Move tensor to CPU and then convert to numpy
        rgb_img = rgb_img.clip(0, 1)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        st.subheader("Gradcam results: ")
        st.image(visualization, caption="Regions influencing the prediction")
else:
    st.info("Please upload an image")