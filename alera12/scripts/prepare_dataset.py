#!/usr/bin/env python3
"""
Prepare balanced dataset CSV for training from a master metadata CSV
Expected: config.data.metadata_csv exists with image_name,target columns
"""
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

cfg = yaml.safe_load(open("config.yaml"))
meta_path = cfg["data"]["metadata_csv"]
images_dir = cfg["data"]["images_dir"]
out_meta = "data/balanced_metadata.csv"
samples_per_class = cfg["training"]["samples_per_class"]

df = pd.read_csv(meta_path)
if "target" not in df.columns:
    raise SystemExit("metadata CSV must contain 'target' column (0/1).")

# optionally filter out images missing in folder
df = df[df["image_name"].apply(lambda x: os.path.exists(os.path.join(images_dir, x + ".jpg")))].reset_index(drop=True)
print(f"Found {len(df)} items with images on disk.")

mal = df[df["target"]==1]
ben = df[df["target"]==0]
n = min(samples_per_class, len(mal), len(ben))
mal_s = mal.sample(n, random_state=42)
ben_s = ben.sample(n, random_state=42)
balanced = pd.concat([mal_s, ben_s]).sample(frac=1, random_state=42).reset_index(drop=True)

# write balanced CSV
balanced.to_csv(out_meta, index=False)
print(f"Wrote balanced metadata to {out_meta} ({len(balanced)} rows)")
