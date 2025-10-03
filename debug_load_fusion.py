# debug_load_fusion.py
import os, traceback
print("CWD:", os.getcwd())
print("Listing models/ and models/encoders/")
for d in ["models", os.path.join("models","encoders")]:
    try:
        print(d, "->", os.path.exists(d))
        if os.path.exists(d):
            print("  contents:", os.listdir(d))
    except Exception as e:
        print("  cannot list:", e)

paths = [
    "models/fusion_model.pth",
    "models/backbone.pth",
    "models/encoders/ohe_sex_site.joblib",
    "models/encoders/scaler_age.joblib"
]
for p in paths:
    try:
        exists = os.path.exists(p)
        size = os.path.getsize(p) if exists else None
        print(p, "exists:", exists, "size:", size)
    except Exception as e:
        print(p, "check failed:", e)

print("\nTrying to import torch and joblib...")
try:
    import torch, joblib
    print("torch version:", torch.__version__, "cuda available:", torch.cuda.is_available())
except Exception as e:
    print("Import failed:", e)
    raise SystemExit(1)

pth = "models/fusion_model.pth"
if os.path.exists(pth):
    try:
        ckpt = torch.load(pth, map_location='cpu')
        print("\nLoaded fusion_model.pth — type:", type(ckpt))
        if isinstance(ckpt, dict):
            print("Keys (first 50):", list(ckpt.keys())[:50])
    except Exception as e:
        print("\nFailed loading fusion_model.pth:", e)
        traceback.print_exc()

# Try loading the joblib encoders if present
for f in ["models/encoders/ohe_sex_site.joblib","models/encoders/scaler_age.joblib"]:
    if os.path.exists(f):
        try:
            obj = joblib.load(f)
            print(f, "loaded type:", type(obj))
        except Exception as e:
            print(f, "joblib.load failed:", e)
            traceback.print_exc()
    else:
        print(f, "not found")
