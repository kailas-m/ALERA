import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
import yaml

# ---- Load config ----
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ---- Prepare ISIC 2019 metadata ----
def prepare_metadata(meta_csv, gt_csv, output_csv):
    meta = pd.read_csv(meta_csv)
    gt = pd.read_csv(gt_csv)

    # Merge on "image" column
    data = pd.merge(meta, gt, on="image")

    # Remove .dcm extension if present in CSV
    data["image"] = data["image"].str.replace(".dcm", "", regex=False)

    data.to_csv(output_csv, index=False)
    print(f"✅ Merged metadata saved at {output_csv}")

# ---- Dataset ----
class SkinLesionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        if 'image' not in self.data.columns:
            raise ValueError(f"No 'image' column found in {csv_file}")

        # Ensure labels are numeric
        self.image_ids = self.data['image']
        self.labels = self.data.drop(columns=['image']).apply(pd.to_numeric, errors='coerce').fillna(0).values
        self.num_classes = self.labels.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.image_ids.iloc[idx]
        img_path = os.path.join(self.img_dir, image_id + ".jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        # Convert one-hot to class index
        label_idx = torch.argmax(label).long()
        return image, label_idx

# ---- Model Training ----
def train_model(model, dataloader, criterion, optimizer, device, epochs=5):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Loss: {running_loss/len(dataloader):.4f} | "
              f"Acc: {100.*correct/total:.2f}%")

    return model

# ---- Main ----
def main():
    # Step 1: Merge ISIC 2019 metadata
    merged_csv = "data/isic2019_merged.csv"
    prepare_metadata(
        cfg["data"]["metadata_csv"], 
        cfg["data"]["labels_csv"], 
        merged_csv
    )

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((cfg["training"]["img_size"], cfg["training"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = SkinLesionDataset(
        csv_file=merged_csv,
        img_dir=cfg["data"]["images_dir"],
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    num_classes = dataset.num_classes

    # --- Train ResNet ---
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=cfg["training"]["lr"])
    print("\nTraining ResNet50...")
    resnet = train_model(resnet, dataloader, criterion, optimizer, device, epochs=cfg["training"]["epochs"])
    os.makedirs(cfg["output"]["models_dir"], exist_ok=True)
    torch.save(resnet.state_dict(), os.path.join(cfg["output"]["models_dir"], cfg["output"]["resnet_name"]))
    print(f"✅ ResNet50 saved at {os.path.join(cfg['output']['models_dir'], cfg['output']['resnet_name'])}")

    # --- Train DenseNet ---
    densenet = models.densenet121(weights=None)
    densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
    optimizer = optim.Adam(densenet.parameters(), lr=cfg["training"]["lr"])
    print("\nTraining DenseNet121...")
    densenet = train_model(densenet, dataloader, criterion, optimizer, device, epochs=cfg["training"]["epochs"])
    torch.save(densenet.state_dict(), os.path.join(cfg["output"]["models_dir"], cfg["output"]["densenet_name"]))
    print(f"✅ DenseNet121 saved at {os.path.join(cfg['output']['models_dir'], cfg['output']['densenet_name'])}")

if __name__ == "__main__":
    main()
