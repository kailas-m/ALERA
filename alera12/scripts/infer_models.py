import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import pydicom
import numpy as np
import yaml
import os

# Load config.yaml
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# --- DICOM to RGB loader ---
def dicom_to_rgb(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)

    # Normalize to [0, 255]
    img -= np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img) * 255.0
    img = img.astype(np.uint8)

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:  # single channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 3:
        img = img  # already RGB
    else:
        raise ValueError(f"Unexpected DICOM format: {img.shape}")

    return img

# --- Model Loader ---
def load_model(model_name, model_path, num_classes=2):
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Invalid model name")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# --- Prediction ---
def predict(model, image_path, device="cpu"):
    img = dicom_to_rgb(image_path)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg["training"]["img_size"], cfg["training"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])   # ImageNet std
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return "Malignant" if predicted.item() == 1 else "Benign"

# --- Main ---
def main():
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    # Load both models
    resnet = load_model("resnet50", os.path.join(cfg["output"]["models_dir"], cfg["output"]["resnet_name"])).to(device)
    densenet = load_model("densenet121", os.path.join(cfg["output"]["models_dir"], cfg["output"]["densenet_name"])).to(device)

    image_path = input("Enter path to DICOM image: ").strip()

    if not os.path.exists(image_path):
        print("❌ Image path not found.")
        return

    resnet_pred = predict(resnet, image_path, device)
    densenet_pred = predict(densenet, image_path, device)

    print(f"\n✅ Predictions:")
    print(f"   ResNet50   ➡ {resnet_pred}")
    print(f"   DenseNet121 ➡ {densenet_pred}")

if __name__ == "__main__":
    main()
