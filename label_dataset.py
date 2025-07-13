import os
import json
import numpy as np
import pandas as pd
from rasterio.windows import Window
from rasterio.transform import from_origin
import rasterio
from collections import Counter

ESA_CLASS_MAP = {
    10: "Tree Cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/Sparse",
    70: "Snow/Ice",
    80: "Water",
    90: "Wetlands",
    95: "Mangroves",
    100: "Moss/Lichen"
}

def get_label_from_patch(patch):
    patch = patch.flatten()
    patch = patch[patch != 0]
    if len(patch) == 0:
        return None
    most_common = Counter(patch).most_common(1)[0][0]
    return ESA_CLASS_MAP.get(most_common, "Unknown")

def extract_labels(landcover_path, coords_path, output_csv="Dataset/labels.csv"):
    # ✅ FIX: initialize labels here
    labels = []

    with open(coords_path) as f:
        coords = json.load(f)

    dataset = rasterio.open(landcover_path)

    for fname, (lat, lon) in coords.items():
        try:
            row, col = dataset.index(lon, lat)
            window = Window(col - 64, row - 64, 128, 128)
            patch = dataset.read(1, window=window)
            label = get_label_from_patch(patch)
            if label:
                labels.append({"filename": fname, "label": label})
        except Exception as e:
            print(f"❌ Skipping {fname}: {e}")

    # ✅ Create output directory if missing
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.DataFrame(labels)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} entries to {output_csv}")
    return df
if __name__ == "__main__":
    df = extract_labels(
        landcover_path=r"C:\Users\abc\PycharmProjects\earth observation\Scenario-1\data\land_cover.tif",
        coords_path=r"C:\Users\abc\PycharmProjects\earth observation\Scenario-1\data\image_coords.json",
        output_csv=r"C:\Users\abc\PycharmProjects\earth observation\Scenario-1\Dataset\labels.csv"
    )

        # continue with split + plot...

    # Optional split & plot
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    os.makedirs("Dataset", exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df["label"])
    train_df.to_csv("Dataset/train.csv", index=False)
    test_df.to_csv("Dataset/test.csv", index=False)

    plt.figure(figsize=(10, 5))
    df["label"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Dataset/class_distribution.png")
    plt.show()
