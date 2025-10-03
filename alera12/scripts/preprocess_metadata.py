# preprocess_metadata.py
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import inspect

# --- CONFIG - edit if needed ---
META_CSV = r"D:\alera12\data\ISIC_2019_Training_Metadata.csv"
GT_CSV   = r"D:\alera12\data\ISIC_2019_Training_GroundTruth.csv"
OUT_MERGED = r"D:\alera12\data\isic2019_merged.csv"
ENC_DIR = r"D:\alera12\models\encoders"
os.makedirs(ENC_DIR, exist_ok=True)

# --- Load & merge ---
meta = pd.read_csv(META_CSV)
gt = pd.read_csv(GT_CSV)
merged = pd.merge(meta, gt, on="image", how="inner")

# Normalize image id (remove extensions if any)
merged["image"] = merged["image"].astype(str).str.replace(".dcm", "", regex=False).str.strip()

# --- Metadata features to use ---
# Numerical: age_approx
# Categorical: sex, anatom_site_general (rename for convenience)
merged["sex"] = merged.get("sex", "").fillna("unknown").astype(str)
site_col_candidates = [c for c in merged.columns if "anatom_site" in c]
site_col = site_col_candidates[0] if site_col_candidates else "anatom_site_general"
merged[site_col] = merged.get(site_col, "").fillna("unknown").astype(str)

# --- Fit encoders/scalers ---
# OneHot encode sex + site together (compatible with sklearn old/new)
if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
else:
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

cat_array = merged[["sex", site_col]].values
ohe.fit(cat_array)
joblib.dump(ohe, os.path.join(ENC_DIR, "ohe_sex_site.joblib"))

# Age scaler
scaler = StandardScaler()
age_arr = merged[["age_approx"]].fillna(merged["age_approx"].median()).astype(float).values
scaler.fit(age_arr)
joblib.dump(scaler, os.path.join(ENC_DIR, "scaler_age.joblib"))

# --- Save merged CSV ---
merged.to_csv(OUT_MERGED, index=False)
print("✅ Saved merged CSV to:", OUT_MERGED)
print("✅ Saved encoders to:", ENC_DIR)
