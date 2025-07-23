# Plant-Disease-Project

### 🔗 Download the Trained Model
[Download plant_disease_vit_full.pth from Google Drive](https://drive.google.com/your-shareable-link)

# 🌿 Plant Disease Classification with Vision Transformer

This project uses a Vision Transformer (ViT) model to detect 15 plant disease classes using the PlantVillage dataset. It includes a Streamlit web app for real-time image classification and optional attention map visualization.

## 📚 Project Overview

The goal of this project is to develop a deep learning solution for plant disease detection using a Vision Transformer (ViT). The model achieves 94% accuracy across 15 classes and is deployed through a Streamlit web app. Users can upload images of plant leaves to detect disease type and visualize attention regions.

## 🧠 Model Summary

I implemented a ViT, specifically vit_base_patch16_224 using the timm library. The model was trained on a subset of the PlantVillage dataset (tomato, potato, bell pepper) which covered 15 disease and healthy classes. The training pipeline included data augmentation, proper normalization, and a carefully tuned learning rate schedule.

I considered a CNN baseline during the planning phase of the project timeline but I opted to prioritize ViT given its recent success in image classification tasks and the time constraints of this project. I trained for 5 epochs with mixed precision and optimization techniques. The model reached 94% accuracy on the validation set and had strong precision and recall across all classes. I then deployed a Streamlit app for real-time predictions that also implemented metrics and a heat map.

- Architecture: `vit_base_patch16_224` from `timm`
- Input size: 224x224
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Best Model Accuracy: **94.0%**

## 🌱 Dataset

- **Source:** [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Subset:** Tomato, Potato, and Bell Pepper (15 classes)
- **Train/Test Split:** 80/20
- **Preprocessing:** 
  - Resize: 224x224
  - Normalization: ImageNet mean & std
  - Augmentation: Horizontal flip, rotation (train set only)

## 🖼️ How to Run the Streamlit App

1. Install dependencies:
pip install -r requirements.txt

2. Start the app:
streamlit run streamlit_app.py

This opens http://localhost:8501/

3. Upload a leaf image to receive:
   - Predicted class
   - Confidence score
   - Optional attention map

## ✅ Key Results

| Metric     | Value |
|------------|-------|
| Accuracy   | 94%   |
| F1-Score   | ~0.93 |
| Classes    | 15    |

## 🙏 Credits & Acknowledgments

- [`timm`](https://github.com/rwightman/pytorch-image-models)
- [`transformers`](https://github.com/huggingface/transformers)
- [`PlantVillage`](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Special thanks to Troy University CS 6625
