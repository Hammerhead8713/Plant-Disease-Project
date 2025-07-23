import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from timm import create_model

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "plant_disease_vit_full.pth"
IMAGE_SIZE = 224

# Class names
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
]

@st.cache_resource
def load_model():
    model = create_model('vit_base_patch16_224', pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

model = load_model()

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🌿 Plant Disease Classifier with Attention")
st.write("Upload a plant leaf image to identify its disease and optionally view the attention map.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
        predicted_class = CLASS_NAMES[pred.item()]
    
    st.success(f"🧪 Prediction: {predicted_class}")
    st.info(f"📊 Confidence: {confidence.item() * 100:.2f}%")

    if st.checkbox("🔍 Show Attention Map"):
        # Access model's internal attention weights
        def extract_attention(model, x):
            attn_weights = []
            for block in model.blocks:
                x_residual = x
                x = block.norm1(x)
                B, N, C = x.shape
                qkv = block.attn.qkv(x).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                attn_weights.append(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = block.attn.proj(x)
                x = x_residual + x
            return attn_weights

        with torch.no_grad():
            x = model.patch_embed(input_tensor)
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = model.pos_drop(x + model.pos_embed)
            attn_weights = extract_attention(model, x)

        # Use last layer, class token attention
        cls_attn = attn_weights[-1][0, :, 0, 1:].mean(0).reshape(14, 14).cpu().numpy()

        fig, ax = plt.subplots()
        ax.imshow(image.resize((224, 224)))
        ax.imshow(cls_attn, cmap='jet', alpha=0.5)
        ax.axis('off')
        ax.set_title("ViT Attention Map")
        st.pyplot(fig)

with st.expander("📈 Model Performance Summary"):
    st.markdown("""
    **Classification Metrics on Test Set (PlantVillage):**

    - **Overall Accuracy**: 94.0%
    - **Macro Average F1-Score**: 0.93
    - **Weighted Average Precision**: 0.94

    **Best Performing Classes:**
    - `Tomato_healthy`: F1-score 0.99
    - `Tomato__Tomato_YellowLeaf__Curl_Virus`: F1-score 0.98
    - `Potato___Early_blight`: F1-score 0.95

    **Most Confused Classes:**
    - `Tomato_Leaf_Mold` was sometimes misclassified as `Tomato_Bacterial_spot`
    - `Potato___healthy` had lower recall (0.80) due to visual similarity with early blight.

    *(Results from model evaluation on 4,127 test images using Google Colab)*
    """)
