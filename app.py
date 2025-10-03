from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
import requests
from dotenv import load_dotenv
import os
# Try to import TensorFlow optionally (we won't eagerly load a TF model)
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

# Explicit .env path in the same directory as this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')

_dotenv_loaded = False
_dotenv_encoding_used = None
_dotenv_error = None

# Try multiple encodings, including common Windows defaults
for enc in ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp1252']:
    try:
        if load_dotenv(DOTENV_PATH, override=True, encoding=enc):
            _dotenv_loaded = True
            _dotenv_encoding_used = enc
            break
    except Exception as e:
        _dotenv_error = str(e)

# Fallback: manual lightweight parser if python-dotenv couldn't load but file exists
if not _dotenv_loaded and os.path.exists(DOTENV_PATH):
    for enc in ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp1252']:
        try:
            with open(DOTENV_PATH, 'r', encoding=enc, errors='ignore') as f:
                for raw_line in f.readlines():
                    line = raw_line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value and key.upper() == key:
                        # Only set if not already present or when override wanted
                        os.environ[key] = value
                _dotenv_loaded = True
                _dotenv_encoding_used = f'manual:{enc}'
                _dotenv_error = None
                break
        except Exception as e:
            _dotenv_error = str(e)

# Prevent Flask CLI from reloading .env with strict UTF-8 (which caused UnicodeDecodeError)
os.environ.setdefault('FLASK_SKIP_DOTENV', '1')

app = Flask(__name__, static_folder='static', static_url_path='/static')

# --- Option A: disable eager TF .h5 loading (we use the PyTorch fusion model) ---
TF_MODEL_PATH = "models/allergy_detector.h5"  # kept for reference only
model = None  # legacy TF fallback disabled by default — app uses fusion_model.pth

# Default class names for the legacy TF model (kept for response structure)
CLASS_NAMES = ['No Allergy', 'Peanut Allergy', 'Lactose Intolerance']

# Optional: Load PyTorch fusion skin rash model if available
FUSION_MODEL_PATH = os.path.join('models', 'fusion_model.pth')
BACKBONE_PATH = os.path.join('models', 'backbone.pth')
ENC_DIR = os.path.join('models', 'encoders')
_skin_model = None
_skin_ohe = None
_skin_scaler = None
_skin_device = None
_skin_transform = None
_skin_class_names = None

# ISIC 2019 skin lesion class names (actual classes from dataset)
ISIC_2019_CLASSES = [
    'MEL',      # Melanoma
    'NV',       # Melanocytic nevus
    'BCC',      # Basal cell carcinoma
    'AK',       # Actinic keratosis
    'BKL',      # Benign keratosis
    'DF',       # Dermatofibroma
    'VASC',     # Vascular lesion
    'SCC',      # Squamous cell carcinoma
    'UNK'       # Unknown
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
        # Optionally load custom backbone weights
        if os.path.exists(BACKBONE_PATH):
            try:
                bb_state = torch.load(BACKBONE_PATH, map_location='cpu')
                # Support checkpoints saved with wrapper keys
                if isinstance(bb_state, dict) and 'state_dict' in bb_state:
                    bb_state = bb_state['state_dict']
                resnet.load_state_dict(bb_state, strict=False)
            except Exception:
                pass
        feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        # Use ISIC 2019 class names for skin lesion classification
        env_classes = os.environ.get('SKIN_CLASS_NAMES', '').strip()
        if env_classes:
            _skin_class_names = [c.strip() for c in env_classes.split(',') if c.strip()]
        else:
            # Use standard ISIC 2019 class names (9 classes)
            _skin_class_names = ISIC_2019_CLASSES

        # Estimate meta_dim: 1 for age + one-hot dims from ohe
        meta_dim = 1 + _skin_ohe.transform([["unknown", "unknown"]]).shape[1]
        _skin_model = FusionModel(ResNetFeatureExtractor(resnet), feature_dim, meta_dim, len(_skin_class_names))
        state = torch.load(FUSION_MODEL_PATH, map_location=_skin_device)
        # Support both raw state_dict and wrapped checkpoints
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
        # On any failure, ensure model is reset and return False
        _skin_model = None
        # helpful debug print
        try:
            print("[_try_load_skin_model] failed:", e)
        except Exception:
            pass
        return False

def _build_skin_meta(age, sex, site):
    if _skin_scaler is None or _skin_ohe is None:
        return None
    try:
        age_val = None
        if age is not None and str(age).strip() != '':
            try:
                age_val = float(age)
            except Exception:
                age_val = None
        if age_val is None:
            # Use scaler mean if available
            try:
                age_val = float(_skin_scaler.mean_[0])
            except Exception:
                age_val = 0.0
        age_arr = np.array([[age_val]], dtype=float)
        age_scaled = _skin_scaler.transform(age_arr)
        sex_val = (sex or 'unknown').strip().lower()
        site_val = (site or 'unknown').strip().lower()
        cat = np.array([[sex_val, site_val]])
        cat_enc = _skin_ohe.transform(cat)
        vec = np.hstack([age_scaled.reshape(1, -1), cat_enc]).ravel().astype(np.float32)
        return vec
    except Exception:
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128)) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape for the model
    return img

# --- Robust skin detection helpers ---
def _skin_stats(image_bgr: np.ndarray):
    if image_bgr is None or image_bgr.size == 0:
        return {"skin_ratio": 0.0, "largest_area_frac": 0.0, "mean_sat": 0.0, "mean_val": 0.0, "hue_std": 0.0}

    h, w = image_bgr.shape[:2]
    max_side = max(h, w)
    if max_side > 1000:
        scale = 1000.0 / float(max_side)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image_bgr.shape[:2]

    blur = cv2.GaussianBlur(image_bgr, (5, 5), 0)

    # HSV mask (two bands around red/orange)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,  30,  60], dtype=np.uint8)
    upper1 = np.array([20, 180, 255], dtype=np.uint8)
    lower2 = np.array([170, 30,  60], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)
    hsv_mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    # YCrCb mask (classic skin band on Cr/Cb)
    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    ycrcb_mask = cv2.inRange(Cr, 133, 173) & cv2.inRange(Cb, 77, 127)

    # Intersection for precision
    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    total = mask.size
    skin_pixels = int(cv2.countNonZero(mask))
    skin_ratio = (skin_pixels / float(total)) if total > 0 else 0.0

    # Largest contiguous area fraction
    largest_area = 0
    if skin_pixels > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        if num_labels > 1:
            largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    largest_area_frac = (largest_area / float(total)) if total > 0 else 0.0

    mean_sat = float(hsv[...,1].mean())
    mean_val = float(hsv[...,2].mean())
    hue_std = float(hsv[...,0][mask > 0].std()) if skin_pixels > 0 else 0.0

    return {
        "skin_ratio": float(skin_ratio),
        "largest_area_frac": float(largest_area_frac),
        "mean_sat": mean_sat,
        "mean_val": mean_val,
        "hue_std": hue_std
    }

def _is_skin_like(image_bgr: np.ndarray):
    s = _skin_stats(image_bgr)
    # Tunable heuristics
    min_ratio = 0.05
    min_blob = 0.01
    min_sat = 35.0
    min_val = 40.0
    max_val = 245.0
    min_hue_std = 3.0

    ok = (
        s["skin_ratio"] >= min_ratio and
        s["largest_area_frac"] >= min_blob and
        s["mean_sat"] >= min_sat and
        (min_val <= s["mean_val"] <= max_val) and
        s["hue_std"] >= min_hue_std
    )
    return ok, s
def _estimate_skin_ratio_bgr(image_bgr: np.ndarray) -> float:
    """Return proportion of skin-like pixels in [0.0, 1.0] using HSV thresholds.
    Uses two HSV ranges to capture skin tones, with basic noise reduction.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    # Resize large images to speed up and stabilize ratio (keep aspect)
    h, w = image_bgr.shape[:2]
    max_side = max(h, w)
    if max_side > 800:
        scale = 800.0 / float(max_side)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    image_blur = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)

    # HSV skin ranges (heuristic). Adjusted S/V lower bounds to exclude very dark/very desaturated regions.
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([20, 170, 255], dtype=np.uint8)
    lower2 = np.array([170, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 170, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    skin_pixels = int(cv2.countNonZero(mask))
    total_pixels = int(mask.size)
    if total_pixels == 0:
        return 0.0
    return float(skin_pixels) / float(total_pixels)

@app.route('/validate_skin', methods=['POST'])
def validate_skin():
    """Validate an uploaded image contains skin-like regions and age is valid.
    Expects multipart/form-data with keys:
      - file: image file
      - age: integer [1, 110]
      - debug: optional flag to include detector stats
    Returns JSON with acceptance and optional stats.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Validate age
    raw_age = (request.form.get('age') or '').strip()
    try:
        age_val = int(float(raw_age))
    except Exception:
        return jsonify({"error": "Invalid age. Provide a number between 1 and 110."}), 400
    if age_val < 1 or age_val > 110:
        return jsonify({"error": "Age out of range. Must be between 1 and 110."}), 400

    file = request.files['file']
    try:
        # Read file bytes and decode with OpenCV
        file_bytes = file.read()
        img_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"error": "Invalid image format."}), 400
    except Exception:
        return jsonify({"error": "Failed to read image."}), 400

    debug = (request.form.get('debug') or '').lower() in ('1','true','yes')
    ok, stats = _is_skin_like(image_bgr)
    if not ok:
        payload = {"error": "Image does not appear to contain skin-like regions.", "accepted": False}
        if debug:
            payload.update({**stats, "opencv_version": getattr(cv2, '__version__', 'unknown')})
        return jsonify(payload), 400

    payload = {"message": "Skin-like image accepted.", "accepted": True, "age": age_val}
    if debug:
        payload.update({**stats, "opencv_version": getattr(cv2, '__version__', 'unknown')})
    return jsonify(payload), 200

# --- Preload the fusion PyTorch model at startup (Option A) ---
_fusion_ready = _try_load_skin_model()
print(f"[STARTUP] Fusion PyTorch model loaded: {_fusion_ready}")
print(f"[STARTUP] Fusion path exists: {os.path.exists(FUSION_MODEL_PATH)}, "
      f"encoders exist: {os.path.exists(os.path.join(ENC_DIR,'ohe_sex_site.joblib')) and os.path.exists(os.path.join(ENC_DIR,'scaler_age.joblib'))}")

@app.route('/predict_skin', methods=['POST'])
def predict_skin():
    """Skin rash classification using the fusion PyTorch model.
    Accepts multipart/form-data with keys:
      - file: image file
      - age: optional number
      - sex: optional string (male/female/unknown)
      - site: optional string (anatomical site)
    Returns JSON with probabilities per class if model is available.
    """
    if not _try_load_skin_model():
        return jsonify({
            'error': 'Skin model unavailable. Ensure PyTorch is installed and fusion_model.pth with encoders exist.',
            'needs': {
                'torch': _torch_available,
                'fusion_model_path': FUSION_MODEL_PATH,
                'encoders_dir': ENC_DIR
            }
        }), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    x = _skin_transform(img).unsqueeze(0)
    x = x.to(_skin_device)

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

    result = {
        'classes': _skin_class_names,
        'probs': [float(p) for p in probs],
        'top_class': _skin_class_names[int(np.argmax(probs))],
        'top_prob': float(np.max(probs))
    }
    return jsonify(result), 200

@app.route('/')
def root():
    # Serve the main landing page
    return send_from_directory('.', 'index.html')

@app.route('/index.html')
def index_page():
    return send_from_directory('.', 'index.html')

@app.route('/about.html')
def about_page():
    return send_from_directory('.', 'about.html')

@app.route('/allergy.html')
def allergy_page():
    return send_from_directory('.', 'allergy.html')

@app.route('/contact.html')
def contact_page():
    return send_from_directory('.', 'contact.html')

@app.route('/login.html')
def login_page():
    return send_from_directory('.', 'login.html')

@app.route('/signup.html')
def signup_page():
    return send_from_directory('.', 'signup.html')

@app.route('/results.html')
def results_page():
    return send_from_directory('.', 'results.html')

@app.route('/uploadDNAimage.html')
def upload_page():
    return send_from_directory('.', 'uploadDNAimage.html')

@app.route('/images/<path:filename>')
def images(filename):
    # Serve assets in images/ via Flask
    return send_from_directory('images', filename)

@app.route('/styles.css')
def styles():
    # Serve the root stylesheet
    return send_from_directory('.', 'styles.css')

def _get_groq_api_key():
    """Return a sanitized API key from multiple possible env var names.
    Accepts GROQ_API_KEY, OPENAI_API_KEY, GROQ_KEY. Trims quotes/whitespace.
    """
    candidate_names = ['GROQ_API_KEY', 'OPENAI_API_KEY', 'GROQ_KEY']
    raw = None
    for name in candidate_names:
        raw = os.environ.get(name)
        if raw:
            break
    if not raw:
        return None
    # Sanitize common mistakes: surrounding quotes or whitespace
    sanitized = raw.strip().strip('"').strip("'")
    return sanitized or None

@app.route('/chat', methods=['POST'])
def chat():
    """Proxy chat requests to Groq LLM if GROQ_API_KEY is set.
    Expected JSON: {"message": "..."}
    Returns: {"reply": "..."}
    """
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get('message') or '').strip()
        if not user_message:
            return jsonify({"reply": "Please type a message."}), 200

        api_key = _get_groq_api_key()
        if not api_key:
            return jsonify({"reply": "Chat unavailable: missing GROQ_API_KEY. Ask your admin to set it in .env."}), 200

        # Groq Chat Completions API
        url = 'https://api.groq.com/openai/v1/chat/completions'
        headers = { 'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json' }
        payload = {
            # Use a widely available Groq model
            'model': 'llama-3.1-8b-instant',
            'messages': [
                { 'role': 'system', 'content': 'You are a concise, friendly assistant for an allergy and DNA image website. Keep responses brief and non-diagnostic. For medical issues, recommend consulting a professional dermatologist or allergist.' },
                { 'role': 'user', 'content': user_message }
            ],
            'temperature': 0.4,
            'max_tokens': 256
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code >= 400:
            # Map common errors to actionable messages
            status = resp.status_code
            try:
                err = resp.json()
            except Exception:
                err = { 'error': resp.text }
            if status == 401:
                return jsonify({"reply": "Invalid or missing GROQ_API_KEY. Update your .env and restart the server."}), 200
            if status == 429:
                return jsonify({"reply": "Chat rate limited. Please wait a moment and try again."}), 200
            if status == 400:
                return jsonify({"reply": "Chat request was rejected. Try rephrasing your question."}), 200
            # Log server-side for debugging
            try:
                print('[GROQ_ERROR]', status, err)
            except Exception:
                pass
            return jsonify({"reply": "Chat service error. Please try again later."}), 200
        j = resp.json()
        reply = (((j or {}).get('choices') or [{}])[0].get('message') or {}).get('content') or "I couldn't generate a reply."
        return jsonify({"reply": reply}), 200
    except Exception:
        return jsonify({"reply": "Chat temporarily unavailable."}), 200

@app.route('/chat/health', methods=['GET'])
def chat_health():
    """Report whether a chat API key is detected (does not expose the key)."""
    has_key = _get_groq_api_key() is not None
    return jsonify({
        'chat_ready': has_key,
        'accepted_env_vars': ['GROQ_API_KEY', 'OPENAI_API_KEY', 'GROQ_KEY'],
        'dotenv_path': DOTENV_PATH,
        'dotenv_exists': os.path.exists(DOTENV_PATH),
        'dotenv_loaded': bool(_dotenv_loaded),
        'dotenv_encoding_used': _dotenv_encoding_used,
        'dotenv_error': _dotenv_error
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Unified prediction endpoint.
    If fusion skin model is available, use it; otherwise fallback to legacy TF model.
    For fusion model, accepts optional form fields: age, sex, site.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Prefer the fusion model when available
    if _try_load_skin_model():
        file = request.files['file']
        try:
            img = Image.open(file.stream).convert('RGB')
        except Exception:
            return jsonify({"error": "Invalid image"}), 400

        # Early reject non-skin images before any model work
        try:
            file.stream.seek(0)
            cv_bytes = file.read()
            cv_img = cv2.imdecode(np.frombuffer(cv_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if cv_img is None:
                return jsonify({"error": "Invalid image"}), 400
            ok, stats = _is_skin_like(cv_img)
            if not ok:
                return jsonify({
                    "error": "Image does not appear to contain skin-like regions.",
                    "accepted": False,
                    "skin_ratio": float(stats.get("skin_ratio", 0.0)),
                    "largest_area_frac": float(stats.get("largest_area_frac", 0.0))
                }), 400
        finally:
            file.stream.seek(0)

        x = _skin_transform(img).unsqueeze(0)
        x = x.to(_skin_device)

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

        # Convert 9-class results to benign/malignant
        # Malignant: MEL, BCC, SCC, AK (melanoma, basal cell carcinoma, squamous cell carcinoma, actinic keratosis)
        # Benign: NV, BKL, DF, VASC, UNK (nevus, benign keratosis, dermatofibroma, vascular, unknown)
        malignant_indices = [0, 2, 7, 3]  # MEL, BCC, SCC, AK
        benign_indices = [1, 4, 5, 6, 8]  # NV, BKL, DF, VASC, UNK
        
        malignant_prob = sum(probs[i] for i in malignant_indices)
        benign_prob = sum(probs[i] for i in benign_indices)
        
        # Normalize probabilities
        total_prob = malignant_prob + benign_prob
        if total_prob > 0:
            malignant_prob = malignant_prob / total_prob
            benign_prob = benign_prob / total_prob
        
        skin_classes = ['Benign', 'Malignant']
        probabilities = [benign_prob, malignant_prob]
        
        # --- NEW: include the original per-class probabilities so UI can display MEL/NV/BCC/etc. ---
        try:
            original_diseases = { ISIC_2019_CLASSES[i]: float(probs[i]) for i in range(min(len(probs), len(ISIC_2019_CLASSES))) }
        except Exception:
            original_diseases = {}

        return jsonify({
            'classes': skin_classes,
            'probs': probabilities,
            'top_class': skin_classes[int(np.argmax(probabilities))],
            'top_prob': float(np.max(probabilities)),
            'original_diseases': original_diseases
        }), 200

    # Fallback to legacy TensorFlow allergy model
    if model is None:
        return jsonify({"error": "No available model. Provide fusion_model.pth or allergy_detector.h5."}), 500

    file = request.files['file']
    img = preprocess_image(file)
    prediction = model.predict(img)[0]
    results_array = [float(p) for p in prediction]
    
    # Convert to benign/malignant classification
    # Group the 3-class model results into benign vs malignant
    # MEL (Melanoma) and BCC (Basal cell carcinoma) are malignant
    # NV (Melanocytic nevus) is benign
    
    # Calculate malignant probability (MEL + BCC)
    malignant_prob = results_array[0] + results_array[2]  # MEL + BCC
    benign_prob = results_array[1]  # NV
    
    # Normalize probabilities
    total_prob = malignant_prob + benign_prob
    if total_prob > 0:
        malignant_prob = malignant_prob / total_prob
        benign_prob = benign_prob / total_prob
    
    skin_classes = ['Benign', 'Malignant']
    probabilities = [benign_prob, malignant_prob]
    
    return jsonify({
        "classes": skin_classes,
        "probs": probabilities,
        "top_class": skin_classes[int(np.argmax(probabilities))],
        "top_prob": float(np.max(probabilities)),
        "original_diseases": {
            "MEL": float(results_array[0]),
            "NV": float(results_array[1]), 
            "BCC": float(results_array[2])
        }
    }), 200

if __name__ == "__main__":
    # Run single server for pages + API
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
