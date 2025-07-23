import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from timm import create_model
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.datasets.folder import default_loader
import random

# --- CONFIG ---
ROOT_PATH = r"C:\Users\User\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"
SELECTED_CLASSES = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold']
BATCH_SIZE = 8
EPOCHS = 2
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- LOAD AND FILTER DATASET ---
all_data = torchvision.datasets.ImageFolder(ROOT_PATH, transform=transform)
original_class_to_idx = all_data.class_to_idx
selected_class_indices = [original_class_to_idx[c] for c in SELECTED_CLASSES]

# Filter and remap labels
filtered_samples = [
    (path, selected_class_indices.index(label))  # Remap 8,9,10 → 0,1,2
    for path, label in all_data.samples
    if label in selected_class_indices
]

# Shuffle and limit for dev speed
random.shuffle(filtered_samples)
filtered_samples = filtered_samples[:300]  # Use max 300 samples for speed

# --- CREATE REMAPPED DATASET CLASS ---
class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)

filtered_dataset = FilteredDataset(filtered_samples, transform)

# --- SPLIT TRAIN/VAL ---
train_size = int(0.8 * len(filtered_dataset))
val_size = len(filtered_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(filtered_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- MODEL SETUP ---
model = create_model("vit_base_patch16_224", pretrained=True)
model.head = nn.Linear(model.head.in_features, len(SELECTED_CLASSES))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# --- TRAINING LOOP ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# --- EVALUATION ---
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=SELECTED_CLASSES))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

MODEL_PATH = "plant_disease_vit_dev.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")