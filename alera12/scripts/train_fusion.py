# train_fusion.py
"""
Fusion training script (image + metadata) using ResNet50 backbone.
Uses torch.amp safely when CUDA is available; falls back to standard FP32 otherwise.
"""

import os
import math
import random
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing

# Try import AMP components; will be used only if CUDA available
try:
    from torch.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None

# ---------- AUTO CONFIG ----------
def get_vram_gb():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
            shell=True
        )
        return int(output.decode().split("\n")[0]) / 1024  # GB
    except Exception:
        return None

CPU_CORES = multiprocessing.cpu_count()
GPU_VRAM_GB = get_vram_gb()

NUM_WORKERS_TRAIN = min(4, max(1, CPU_CORES // 2))
NUM_WORKERS_VAL = max(1, NUM_WORKERS_TRAIN // 2)

BATCH_SIZE = 8

print(f"[AutoConfig] CPU cores: {CPU_CORES} | GPU VRAM: {GPU_VRAM_GB or 'N/A'} GB")
print(f"[AutoConfig] Using batch size {BATCH_SIZE}, train workers {NUM_WORKERS_TRAIN}, val workers {NUM_WORKERS_VAL}")

# ---------- PATH CONFIG ----------
MERGED_CSV = r"D:\alera12\data\isic2019_merged.csv"
IMG_DIR    = r"D:\alera12\data\images"
ENC_DIR    = r"D:\alera12\models\encoders"
OUT_DIR    = r"D:\alera12\models"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 224

# Training schedule
WARMUP_EPOCHS = 3
FINETUNE = True
FINETUNE_EPOCHS = 20

LR_HEAD = 1e-4
LR_FINETUNE = 1e-5
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5

MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda") and (autocast is not None) and (GradScaler is not None)
PIN_MEMORY = True if DEVICE.type == "cuda" else False

torch.backends.cudnn.benchmark = True

# Null autocast context
class _NullAutocast:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False

# ---------- DATA HELPERS ----------
def build_meta_vector(row, site_cols, scaler, ohe):
    age = np.array([[row.get("age_approx", np.nan)]], dtype=float)
    age = np.nan_to_num(age, nan=float(scaler.mean_[0]))
    age_scaled = scaler.transform(age)
    site_col = site_cols[0] if site_cols else "anatom_site_general"
    cat = np.array([[row.get("sex", "unknown"), row.get(site_col, "unknown")]])
    cat_enc = ohe.transform(cat)
    return np.hstack([age_scaled.reshape(1, -1), cat_enc]).ravel().astype(np.float32)

class GaussianNoise(object):
    def __init__(self, std=0.02): self.std = std
    def __call__(self, tensor):
        if self.std <= 0: return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)

class FusionDataset(Dataset):
    def __init__(self, df, img_dir, label_cols, site_cols, scaler, ohe, train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_cols = label_cols
        self.site_cols = site_cols
        self.scaler = scaler
        self.ohe = ohe
        if train:
            self.transform = transforms.Compose([
                transforms.Resize(int(IMG_SIZE * 1.2)),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                GaussianNoise(std=0.01),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["image"]) + ".jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Cannot open image {img_path}: {e}")
        image = self.transform(image)
        meta = build_meta_vector(row, self.site_cols, self.scaler, self.ohe)
        meta_t = torch.tensor(meta, dtype=torch.float32)
        label_idx = int(np.argmax(row[self.label_cols].values.astype(float)))
        return image, meta_t, torch.tensor(label_idx, dtype=torch.long)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.backbone = resnet
    def forward(self, x): return self.backbone(x)

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
        if feats.dim() > 2: feats = torch.flatten(feats, 1)
        if meta.dtype != feats.dtype:
            meta = meta.to(feats.dtype)
        return self.fc(torch.cat([feats, meta], dim=1))

# ---------- TRAIN UTILITIES ----------
def compute_class_weights(df, label_cols, eps=1e-6):
    y_idx = df[label_cols].idxmax(axis=1)
    counts = y_idx.value_counts().reindex(label_cols).fillna(0).astype(float).values
    inv = 1.0 / (counts + eps)
    return torch.tensor(inv / inv.sum() * len(label_cols), dtype=torch.float32)

def to_one_hot(indices, num_classes):
    y = torch.zeros(indices.size(0), num_classes, device=indices.device)
    y.scatter_(1, indices.unsqueeze(1), 1.0)
    return y

def soft_cross_entropy(logits, soft_targets, class_weights=None):
    log_probs = torch.log_softmax(logits, dim=1)
    if class_weights is not None:
        log_probs = log_probs * class_weights.unsqueeze(0)
        log_probs = log_probs / class_weights.mean().clamp_min(1e-8)
    return -(soft_targets * log_probs).sum(dim=1).mean()

def mixup_batch(x, meta, y_idx, alpha, p=1.0):
    if alpha <= 0.0 or random.random() > p:
        return x, meta, y_idx, None
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * meta + (1 - lam) * meta[perm], y_idx, (y_idx[perm], lam)

def run_epoch(model, loader, optimizer, class_weights, num_classes, scaler_obj, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    # FIX: autocast now passes device_type
    if USE_AMP:
        _autocast_ctx = lambda: autocast(device_type=DEVICE.type)
    else:
        _autocast_ctx = _NullAutocast()

    for imgs, metas, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        metas = metas.to(DEVICE, non_blocking=True).to(torch.float32)
        labels = labels.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad()
            imgs_m, metas_m, y_idx, mix = mixup_batch(imgs, metas, labels, MIXUP_ALPHA, MIXUP_PROB)
            with _autocast_ctx():
                logits = model(imgs_m, metas_m)
                if mix is None:
                    loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
                    loss = loss_fn(logits, y_idx)
                else:
                    y2, lam = mix
                    y1_soft = to_one_hot(y_idx, num_classes)
                    y2_soft = to_one_hot(y2, num_classes)
                    soft = lam * y1_soft + (1 - lam) * y2_soft
                    loss = soft_cross_entropy(logits, soft, class_weights)

            if USE_AMP and scaler_obj is not None:
                scaler_obj.scale(loss).backward()
                scaler_obj.step(optimizer)
                scaler_obj.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad(), _autocast_ctx():
                logits = model(imgs, metas)
                loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)

        loss_sum += float(loss.item())
        _, preds = torch.max(logits, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return loss_sum / max(1, len(loader)), correct / max(1, total)

# ---------- MAIN ----------
def main():
    print("Device:", DEVICE)
    print("Use AMP:", USE_AMP)
    df = pd.read_csv(MERGED_CSV)
    df["image"] = df["image"].astype(str).str.strip()

    site_cols = [c for c in df.columns if "anatom_site" in c]
    metadata_cols = {"image", "age_approx", "sex", "anatom_site_general", "lesion_id"} | set(site_cols)
    label_cols = [c for c in df.columns if c not in metadata_cols]
    num_classes = len(label_cols)
    print("Detected classes:", label_cols)

    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df[label_cols].idxmax(axis=1), random_state=42
    )

    ohe = joblib.load(os.path.join(ENC_DIR, "ohe_sex_site.joblib"))
    scaler = joblib.load(os.path.join(ENC_DIR, "scaler_age.joblib"))

    train_ds = FusionDataset(train_df, IMG_DIR, label_cols, site_cols, scaler, ohe, train=True)
    val_ds = FusionDataset(val_df, IMG_DIR, label_cols, site_cols, scaler, ohe, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY
    )

    class_weights = compute_class_weights(train_df, label_cols).to(DEVICE)

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if hasattr(models, "ResNet50_Weights") else None
    backbone = models.resnet50(weights=weights)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    dummy_meta = build_meta_vector(train_df.iloc[0], site_cols, scaler, ohe)
    meta_dim = int(dummy_meta.shape[0])
    model = FusionModel(ResNetFeatureExtractor(backbone), feature_dim, meta_dim, num_classes).to(DEVICE)

    for p in model.feature_extractor.backbone.parameters(): p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    scaler_obj = GradScaler() if USE_AMP else None

    best_acc, epochs_no_improve = 0.0, 0

    # Warmup
    for epoch in range(WARMUP_EPOCHS):
        print(f"\n[Warmup] Epoch {epoch+1}/{WARMUP_EPOCHS}")
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, class_weights, num_classes, scaler_obj, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, None, class_weights, num_classes, scaler_obj, train=False)
        print(f"Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, epochs_no_improve = val_acc, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "fusion_model.pth"))
            torch.save(backbone.state_dict(), os.path.join(OUT_DIR, "backbone.pth"))
            print(f"✓ Saved best model (acc {best_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print("Early stopping (warmup).")
                return

    # Fine-tune
    if FINETUNE:
        print("\nUnfreezing backbone for fine-tune…")
        for p in model.feature_extractor.backbone.parameters(): p.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
        epochs_no_improve = 0
        for epoch in range(FINETUNE_EPOCHS):
            print(f"\n[Finetune] Epoch {epoch+1}/{FINETUNE_EPOCHS}")
            train_loss, train_acc = run_epoch(model, train_loader, optimizer, class_weights, num_classes, scaler_obj, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, None, class_weights, num_classes, scaler_obj, train=False)
            print(f"Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc, epochs_no_improve = val_acc, 0
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "fusion_model.pth"))
                torch.save(backbone.state_dict(), os.path.join(OUT_DIR, "backbone.pth"))
                print(f"✓ Saved best model (acc {best_acc:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print("Early stopping (finetune).")
                    break

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print("Training finished. Saved models in", OUT_DIR)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
