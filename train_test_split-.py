import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load the Clean, Labeled Dataset ---

# Update path as needed
df = pd.read_csv("data/labelled_images_clean.csv")

# Check that label and class_str columns exist
print("Columns:", df.columns)

# Display class counts before filtering
print("Counts per class:\n", df['class_str'].value_counts())

# --- Step 2: Handle Extremely Rare Classes ---

label_counts = df['label'].value_counts()
rare_labels = label_counts[label_counts == 1].index.tolist()

if rare_labels:
    print(f"Warning: These classes have only one sample and will be dropped: {rare_labels}")
    df_split = df[~df['label'].isin(rare_labels)].reset_index(drop=True)
else:
    df_split = df

# --- Step 3: Stratified Train-Test Split (60% train, 40% test) ---

train, test = train_test_split(
    df_split,
    test_size=0.4,
    random_state=42,
    stratify=df_split['label']
)

print(f"\nTrain set: {len(train)} samples | Test set: {len(test)} samples")

# Optional: Save outputs for modeling scripts
train.to_csv("data/train_split.csv", index=False)
test.to_csv("data/test_split.csv", index=False)
print("Saved train and test splits to CSV.")

# --- Step 4: Visualize Training Set Class Distribution ---

plt.figure(figsize=(8, 4))
sns.countplot(x=train['class_str'], order=sorted(train['class_str'].unique()))
plt.xticks(rotation=45, ha='right')
plt.title('Training Set Land Cover Class Distribution')
plt.xlabel("Land Cover Class")
plt.ylabel("Sample Count")
plt.tight_layout()
plt.show()
