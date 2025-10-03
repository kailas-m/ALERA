# infer_fusion.py
"""
Interactive inference script for fusion model (image + metadata).
- Opens file dialog to pick an image
- Asks for metadata (age, sex, site) via CLI
- Provides site selection as a numbered menu
"""

import os
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import json
import tkinter as tk
from tkinter import filedialog

# ---------- CONFIG ----------
MERGED_CSV = r"D:\alera12\data\isic2019_merged.csv"
IMG_DIR = r"D:\alera12\data\images"
ENC_DIR = r"D:\alera12\models\encoders"
MODEL_PATH = r"D:\alera12\models\fusion_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# AMP availability
try:
    from torch.amp import autocast
    USE_AMP = DEVICE.type == "cuda"
except ImportError:
    autocast = None
    USE_AMP = False

# ---------- LOAD DATA & ENCODERS ----------
df = pd.read_csv(MERGED_CSV)
ohe = joblib.load(os.path.join(ENC_DIR, "ohe_sex_site.joblib"))
scaler = joblib.load(os.path.join(ENC_DIR, "scaler_age.joblib"))

# Label columns
site_cols = [c for c in df.columns if "anatom_site" in c]
metadata_cols = {"image", "age_approx", "sex", "anatom_site_general", "lesion_id"} | set(site_cols)
label_cols = [c for c in df.columns if c not in metadata_cols]
class_names = label_cols

# ---------- META VECTOR ----------
def build_meta_vector(age, sex, site):
    age_arr = np.array([[age if age is not None else scaler.mean_[0]]], dtype=float)
    age_scaled = scaler.transform(age_arr)
    cat = np.array([[sex if sex else "unknown", site if site else "unknown"]])
    cat_enc = ohe.transform(cat)
    return np.hstack([age_scaled.reshape(1, -1), cat_enc]).ravel().astype(np.float32)

# ---------- MODEL ----------
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.backbone = resnet
    def forward(self, x):
        return self.backbone(x)

class FusionModel(nn.Module):
    def __init__(self, feature_extractor, feature_dim, meta_dim, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + meta_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x, meta):
        feats = self.feature_extractor(x)
        if feats.dim() > 2:
            feats = feats.flatten(1)
        if meta.dtype != feats.dtype:
            meta = meta.to(feats.dtype)
        return self.fc(torch.cat([feats, meta], dim=1))

# Load model
weights = models.ResNet50_Weights.IMAGENET1K_V1 if hasattr(models, "ResNet50_Weights") else None
resnet = models.resnet50(weights=weights)
feature_dim = resnet.fc.in_features
resnet.fc = nn.Identity()

meta_dim = 1 + ohe.transform([["male", "torso"]]).shape[1]
model = FusionModel(ResNetFeatureExtractor(resnet), feature_dim, meta_dim, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

# ---------- TRANSFORMS ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- PREDICT ----------
def predict(image_path, age=None, sex=None, site=None, topk=3):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    meta_vec = build_meta_vector(age, sex, site)
    meta_t = torch.tensor(meta_vec).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if USE_AMP:
            with autocast(device_type="cuda"):
                logits = model(x, meta_t)
        else:
            logits = model(x, meta_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()

    idx = int(np.argmax(probs))
    primary = class_names[idx]

    return {
        "primary": primary,
        "primary_prob": float(probs[idx]),
        "probs": dict(zip(class_names, probs.round(4).tolist()))
    }

# ---------- INTERACTIVE CLI ----------
if __name__ == "__main__":
    print("=== Fusion Model Inference ===")

    # Open file dialog
    root = tk.Tk()
    root.withdraw()  # hide main window
    img_path = filedialog.askopenfilename(
        title="Select an image for inference",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not img_path:
        print("❌ No file selected.")
        exit()

    print("Selected image:", img_path)

    # Age
    try:
        age = int(input("Enter age (or leave blank): ") or -1)
        if age < 0: age = None
    except ValueError:
        age = None

    # Sex
    sex = input("Enter sex (male/female/unknown): ").strip().lower()
    if sex not in ["male", "female", "unknown"]:
        sex = "unknown"

    # Site menu
    valid_sites = sorted(df[site_cols[0]].dropna().unique()) if site_cols else []
    print("\nAvailable sites:")
    for i, s in enumerate(valid_sites, 1):
        print(f"{i}. {s}")
    print(f"{len(valid_sites)+1}. unknown")

    try:
        choice = int(input("Choose site number: "))
        if 1 <= choice <= len(valid_sites):
            site = valid_sites[choice-1]
        else:
            site = "unknown"
    except ValueError:
        site = "unknown"

    # Predict
    result = predict(img_path, age=age, sex=sex, site=site)
    print("\n=== Prediction Result ===")
    print(json.dumps(result, indent=2))
