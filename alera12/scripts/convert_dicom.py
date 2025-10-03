import os
import glob
import pydicom
from PIL import Image
import pandas as pd
import yaml
from tqdm import tqdm


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def convert_dicom_folder(root_dir, out_img_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    dcm_files = glob.glob(os.path.join(root_dir, "**", "*.dcm"), recursive=True)
    print(f"Found {len(dcm_files)} DICOM files. Converting...")

    converted_files = []
    for dcm_path in tqdm(dcm_files):
        try:
            ds = pydicom.dcmread(dcm_path)
            if hasattr(ds, 'pixel_array'):
                arr = ds.pixel_array
                img = Image.fromarray(arr).convert("RGB")
                base = os.path.splitext(os.path.basename(dcm_path))[0]
                out_name = base + ".jpg"
                out_path = os.path.join(out_img_dir, out_name)
                img.save(out_path)
                converted_files.append(base)
        except Exception as e:
            print(f"Warning: failed to process {dcm_path}: {e}")

    return converted_files


def main():
    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]
    images_dir = cfg["data"]["images_dir"]
    metadata_csv = cfg["data"]["metadata_csv"]
    output_metadata_csv = cfg["data"]["output_metadata_csv"]

    converted_files = convert_dicom_folder(raw_dir, images_dir)

    # Merge with metadata CSV
    if os.path.exists(metadata_csv):
        df_meta = pd.read_csv(metadata_csv)
        df_meta = df_meta[df_meta["image_name"].isin(converted_files)]
        df_meta.to_csv(output_metadata_csv, index=False)
        print(f"Merged metadata saved to {output_metadata_csv} with {len(df_meta)} entries.")
    else:
        print(f"Metadata CSV not found at {metadata_csv}. No labels merged.")


if __name__ == "__main__":
    main()
