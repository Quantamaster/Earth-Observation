import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from torchmetrics.classification import MulticlassF1Score

# ---- 1. DATASET SETUP ----

class EO_Dataset(Dataset):
    def __init__(self, df, img_folder, label_map):
        self.df = df.reset_index(drop=True)
        self.folder = img_folder
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ])
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        fname = self.df.loc[i, 'filename']
        img = plt.imread(f"{self.folder}/{fname}")[:, :, :3]  # H,W,3
        img = self.tfm(img)
        label = self.df.loc[i, 'label']
        label_idx = self.label_map[label]
        return img, label_idx

# ---- 2. LABEL MAPPING (must be consistent across splits) ----

# Load your split datasets
train = pd.read_csv("data/train_split.csv")
test = pd.read_csv("data/test_split.csv")
# If needed, use your full clean dataset:
# imgs_clean = pd.read_csv("data/labelled_images_clean.csv")

unique_labels = sorted(train['label'].dropna().unique())
label_map = {v: i for i, v in enumerate(unique_labels)}
inv_label_map = {i: v for v, i in label_map.items()}

ESA_CODE_TO_LABEL = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse",
    70: "Snow/ice",
    80: "Water",
    90: "Wetland",
    95: "Mangrove",
    100: "Moss/lichen"
}

# ---- 3. DATALOADERS ----

train_ds = EO_Dataset(train, 'rgb', label_map)
test_ds = EO_Dataset(test, 'rgb', label_map)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

# ---- 4. MODEL ----

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, len(label_map))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), 1e-4)
epochs = 10

# ---- 5. TRAINING LOOP ----

import torch.nn.functional as F

for ep in range(epochs):
    model.train()
    running_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {ep+1}/{epochs}, Loss: {running_loss/len(train_dl):.4f}")

# ---- 6. EVALUATION ----

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(yb.numpy())

macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Custom Macro F1: {macro_f1:.3f}")

tm_f1 = MulticlassF1Score(num_classes=len(label_map), average='macro')
print("Torchmetrics Macro F1:", tm_f1(torch.tensor(all_preds), torch.tensor(all_labels)).item())

# Confusion matrix
fig, ax = plt.subplots(figsize=(8,8))
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=[ESA_CODE_TO_LABEL[inv_label_map[i]] for i in range(len(label_map))])
disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
plt.show()

# ---- 7. VISUALIZE EXAMPLES ----

def unnormalize(img):
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    img = img * std + mean
    return img.permute(1, 2, 0).cpu().numpy()

correct_idx = [i for i, (a, b) in enumerate(zip(all_labels, all_preds)) if a == b][:5]
wrong_idx = [i for i, (a, b) in enumerate(zip(all_labels, all_preds)) if a != b][:5]

for idx in correct_idx + wrong_idx:
    img, true_label = test_ds[idx]
    pred_label = all_preds[idx]
    plt.figure()
    plt.imshow(np.clip(unnormalize(img), 0, 1))
    plt.title(f"GT: {ESA_CODE_TO_LABEL[inv_label_map[true_label]]} | Pred: {ESA_CODE_TO_LABEL[inv_label_map[pred_label]]}")
    plt.axis('off')
    plt.show()
