from flask import Flask, request, jsonify, send_from_directory
import os
import requests
from dotenv import load_dotenv

# ML libraries
try:
    import tensorflow as tf
except Exception:
    tf = None
import numpy as np
import cv2
from PIL import Image
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    import joblib
    _torch_available = True
except Exception:
    _torch_available = False

# Load .env variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(DOTENV_PATH, override=True)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# ====== MODEL SETUP ======
TF_MODEL_PATH = "models/allergy_detector.h5"
FUSION_MODEL_PATH = os.path.join('models', 'fusion_model.pth')
BACKBONE_PATH = os.path.join('models', 'backbone.pth')
ENC_DIR = os.path.join('models', 'encoders')

_skin_model = None
_skin_ohe = None
_skin_scaler = None
_skin_device = None
_skin_transform = None
_skin_class_names = None

ISIC_2019_CLASSES = [
    'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'
]

def _try_load_skin_model():
    global _skin_model, _skin_ohe, _skin_scaler, _skin_device, _skin_transform, _skin_class_names
    if _skin_model is not None:
        return True
    if not _torch_available:
        return False
    if not (os.path.exists(FUSION_MODEL_PATH) and
            os.path.exists(os.path.join(ENC_DIR, 'ohe_sex_site.joblib')) and
            os.path.exists(os.path.join(ENC_DIR, 'scaler_age.joblib'))):
        return False
    try:
        _skin_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _skin_ohe = joblib.load(os.path.join(ENC_DIR, 'ohe_sex_site.joblib'))
        _skin_scaler = joblib.load(os.path.join(ENC_DIR, 'scaler_age.joblib'))

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

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if hasattr(models, 'ResNet50_Weights') else None
        resnet = models.resnet50(weights=weights)
        if os.path.exists(BACKBONE_PATH):
            try:
                bb_state = torch.load(BACKBONE_PATH, map_location='cpu')
                if isinstance(bb_state, dict) and 'state_dict' in bb_state:
                    bb_state = bb_state['state_dict']
                resnet.load_state_dict(bb_state, strict=False)
            except Exception:
                pass
        feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        env_classes = os.environ.get('SKIN_CLASS_NAMES', '').strip()
        if env_classes:
            _skin_class_names = [c.strip() for c in env_classes.split(',') if c.strip()]
        else:
            _skin_class_names = ISIC_2019_CLASSES

        meta_dim = 1 + _skin_ohe.transform([["unknown", "unknown"]]).shape[1]
        _skin_model = FusionModel(ResNetFeatureExtractor(resnet), feature_dim, meta_dim, len(_skin_class_names))
        state = torch.load(FUSION_MODEL_PATH, map_location=_skin_device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        _skin_model.load_state_dict(state)
        _skin_model = _skin_model.to(_skin_device).eval()

        _skin_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return True
    except Exception as e:
        print("[_try_load_skin_model] failed:", e)
        _skin_model = None
        return False

# Preload model
_fusion_ready = _try_load_skin_model()
print(f"[STARTUP] Fusion PyTorch model loaded: {_fusion_ready}")

# ====== ROUTES ======
@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

@app.route('/predict_skin', methods=['POST'])
def predict_skin():
    if not _try_load_skin_model():
        return jsonify({"error": "Skin model unavailable"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({"error": "Invalid image"}), 400
    x = _skin_transform(img).unsqueeze(0).to(_skin_device)
    age = request.form.get('age')
    sex = request.form.get('sex')
    site = request.form.get('site')
    meta_vec = _build_skin_meta(age, sex, site)
    if meta_vec is None:
        return jsonify({"error": "Failed to build metadata vector"}), 500
    meta_t = torch.tensor(meta_vec).unsqueeze(0).to(_skin_device)
    with torch.no_grad():
        logits = _skin_model(x, meta_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    return jsonify({"classes": _skin_class_names, "probs": probs.tolist()}), 200

# ====== MAIN ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
