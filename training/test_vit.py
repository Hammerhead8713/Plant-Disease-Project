import torch
from torchvision import transforms
from PIL import Image
from timm import create_model
import os

# Set the absolute path to one of the images in your dataset
img_path = r"C:\Users\User\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage\Tomato_Early_blight\00c5c908-fc25-4710-a109-db143da23112___RS_Erly.B 7778.jpg"

# Load the image
img = Image.open(img_path).convert("RGB")

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Load pretrained Vision Transformer
model = create_model("vit_base_patch16_224", pretrained=True)
model.eval()

# Run inference
with torch.no_grad():
    output = model(img_tensor)
    predicted_class = output.argmax().item()

print(f"Predicted class index: {predicted_class}")
