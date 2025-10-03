# scripts/infer_with_metadata.py
import os, math
import torch, torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import numpy as np

# ---------- Config ----------
CSV_PATH = "data/isic2019_merged.csv"
IMG_DIR = "data/images"                # folder with .jpg images
RESNET_PATH = "models/resnet50_isic2019.pth"   # trained weights (optional)
DENSENET_PATH = "models/densenet121_isic2019.pth" # optional
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
# ----------------------------

# ---------- Helpers ----------
df = pd.read_csv(CSV_PATH)

# determine label columns automatically (everything except known metadata)
metadata_cols = {"image", "age_approx", "anatom_site_general", "lesion_id", "sex"}
label_cols = [c for c in df.columns if c not in metadata_cols]
class_names = label_cols

# Build metadata priors: P(class | site)
def build_priors(df):
    priors = {}
    # P(class) (global)
    global_prior = df[label_cols].mean().values + 1e-8
    priors["global"] = global_prior / global_prior.sum()
    # P(class | site)
    site_groups = df.groupby("anatom_site_general")
    priors["site"] = {}
    for site, g in site_groups:
        vec = g[label_cols].mean().values + 1e-8
        priors["site"][site] = vec / vec.sum()
    # P(class | sex)
    sex_groups = df.groupby("sex")
    priors["sex"] = {}
    for sex, g in sex_groups:
        vec = g[label_cols].mean().values + 1e-8
        priors["sex"][sex] = vec / vec.sum()
    return priors

PRIORS = build_priors(df)

# image transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# load models (if weights exist, else use pretrained ImageNet backbones as fallback)
def load_resnet(path=None):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model = model.to(DEVICE).eval()
    return model

def load_densenet(path=None):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model = model.to(DEVICE).eval()
    return model

# image -> probs
def image_to_probs(model, img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    return probs

# combine model probs (ensemble)
def ensemble_probs(probs_list, weights=None):
    arr = np.stack(probs_list, axis=0)
    if weights is None:
        weights = np.ones(arr.shape[0]) / arr.shape[0]
    weights = np.array(weights).reshape(-1,1)
    combined = (weights * arr).sum(axis=0)
    combined /= combined.sum()
    return combined

# apply metadata priors: multiply and renormalize
def apply_metadata_prior(probs, site=None, sex=None, alpha_site=1.0, alpha_sex=1.0):
    posterior = probs.copy()
    if site is not None and site in PRIORS["site"]:
        prior_site = PRIORS["site"][site]
        posterior = posterior * (prior_site ** alpha_site)
    if sex is not None and sex in PRIORS["sex"]:
        prior_sex = PRIORS["sex"][sex]
        posterior = posterior * (prior_sex ** alpha_sex)
    posterior = posterior + 1e-12
    posterior = posterior / posterior.sum()
    return posterior

# find nearest cases (by age and same site) for evidence
def nearest_cases(image_name, age, site, topk=5):
    subset = df.copy()
    if site in subset["anatom_site_general"].unique():
        subset = subset[subset["anatom_site_general"] == site]
    if pd.notnull(age):
        subset = subset.assign(age_diff=(subset["age_approx"].fillna(age) - age).abs())
        subset = subset.sort_values("age_diff")
    else:
        subset = subset.sample(frac=1).reset_index(drop=True)
    return subset.head(topk)[["image"] + label_cols + ["age_approx","sex","anatom_site_general"]]

# ---------- Main prediction function ----------
def predict_with_metadata(image_path, image_id=None, age=None, sex=None, site=None,
                          use_resnet=True, use_densenet=True):
    models_list = []
    probs_list = []
    if use_resnet:
        res = load_resnet(RESNET_PATH)
        models_list.append(res)
        probs_list.append(image_to_probs(res, image_path))
    if use_densenet:
        den = load_densenet(DENSENET_PATH)
        models_list.append(den)
        probs_list.append(image_to_probs(den, image_path))
    # ensemble
    probs = ensemble_probs(probs_list)
    # apply priors
    posterior = apply_metadata_prior(probs, site=site, sex=sex, alpha_site=1.0, alpha_sex=0.6)
    primary_idx = int(np.argmax(posterior))
    primary_name = class_names[primary_idx]
    primary_prob = float(posterior[primary_idx])
    # nearest evidence
    evidence = None
    try:
        evidence = nearest_cases(image_id, age, site).to_dict(orient="records")
    except Exception:
        evidence = []
    return {
        "primary_class": primary_name,
        "primary_prob": primary_prob,
        "posterior_probs": dict(zip(class_names, posterior.round(4).tolist())),
        "evidence_cases": evidence
    }

# Example usage:
if __name__ == "__main__":
    # pick an image file and optional metadata (age/sex/site)
    sample_img = os.path.join(IMG_DIR, "ISIC_0000000.jpg")   # change to actual filename
    # if you have corresponding metadata, give them; else pass None
    res = predict_with_metadata(sample_img, image_id="ISIC_0000000", age=55, sex="female", site="anterior torso")
    import json
    print(json.dumps(res, indent=2))
