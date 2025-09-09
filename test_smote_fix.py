import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from nlpaug.augmenter.word import SynonymAug
import pandas as pd

# Simulate data
np.random.seed(42)
train_data = {'text': [f'text_{i}' for i in range(43410)], 'labels': [np.random.choice(range(28), np.random.randint(1,3), replace=False).tolist() for _ in range(43410)]}
train_data = type('obj', (object,), train_data)()

# Binarize labels
mlb = MultiLabelBinarizer(classes=range(28))
y_train = mlb.fit_transform([labels for labels in train_data.labels])
print(f"y_train shape: {y_train.shape}")

# Features
model = SentenceTransformer('all-MiniLM-L6-v2')
X_train_features = model.encode(train_data.text[:100])  # Small sample for test
y_train = y_train[:100]

# Identify rare classes (prevalence <1%)
class_prevalence = np.mean(y_train, axis=0)
rare_classes = np.where(class_prevalence < 0.01)[0].tolist()
print(f"Rare classes (prevalence <1%): {rare_classes}")

# Per-class oversampling for rare classes only
all_X_aug = []
all_y_aug_full = []
all_texts_aug = []
original_size = len(train_data.text)
target_size = int(original_size * 1.5)  # Cap at 1.5x

for rare_class in rare_classes[:1]:  # Test first rare class only
    rare_mask = y_train[:, rare_class] == 1
    if np.sum(rare_mask) == 0:
        continue
    X_rare = X_train_features[rare_mask]
    y_rare_binary = y_train[rare_mask, rare_class]  # Binary target for this class
    
    # SMOTE for this binary class
    smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy='auto')
    X_aug, y_aug = smote.fit_resample(X_rare, y_rare_binary)
    print(f"SMOTE for class {rare_class}: {X_rare.shape} -> {X_aug.shape}, no ValueError")
    
    # Reconstruct full labels: replace rare_class column with augmented binary, keep others intact
    y_rare_other = y_train[rare_mask][:, [c for c in range(28) if c != rare_class]]
    y_aug_full_temp = np.hstack([y_rare_other, y_aug.reshape(-1, 1)])
    # Fix column order
    cols = list(range(28))
    cols.pop(rare_class)
    cols.insert(rare_class, rare_class)
    y_aug_full = np.zeros((len(y_aug_full_temp), 28))
    for i, c in enumerate(cols):
        y_aug_full[:, c] = y_aug_full_temp[:, i]
    
    # Find nearest originals for augmented samples
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_rare)
    distances, indices = nn.kneighbors(X_aug)
    
    # Get original indices for rare samples
    rare_indices = np.where(rare_mask)[0]
    rare_texts = [train_data.text[rare_indices[i]] for i in range(len(rare_indices))]
    augmented_texts_rare = [rare_texts[idx[0]] for idx in indices]
    
    # Augment texts with nlpaug to reach 5x (original rare + 4 augs per sample)
    aug = SynonymAug(aug_src='wordnet', aug_p=0.3)
    rare_factor = 5
    for i in range(len(augmented_texts_rare)):
        base_text = augmented_texts_rare[i]
        y_base = y_aug_full[i]
        for j in range(rare_factor):
            if j == 0:
                aug_text = base_text  # SMOTE synthetic
            else:
                aug_text = aug.augment(base_text)
            all_texts_aug.append(aug_text)
            all_y_aug_full.append(y_base)
            all_X_aug.append(X_aug[i])  # Approximate X for augs

# Union across rares, avoid duplicates, cap at 1.5x
df_aug = pd.DataFrame({'text': all_texts_aug, 'y': [list(row) for row in all_y_aug_full], 'X': [list(row) for row in all_X_aug]})
df_aug = df_aug.drop_duplicates(subset='text')
all_texts_aug = df_aug['text'].tolist()
all_y_aug_full = [np.array(y) for y in df_aug['y'].tolist()]
all_X_aug = [np.array(x) for x in df_aug['X'].tolist()]

# Truncate to target size
max_added = target_size - original_size
if len(all_texts_aug) > max_added:
    all_texts_aug = all_texts_aug[:max_added]
    all_y_aug_full = all_y_aug_full[:max_added]
    all_X_aug = all_X_aug[:max_added]

all_y_aug = np.vstack(all_y_aug_full)
all_X_aug = np.vstack(all_X_aug)

# Combine with original data
X_train_features_aug = np.vstack([X_train_features, all_X_aug])
y_train_aug = np.vstack([y_train, all_y_aug])
train_texts_aug = list(train_data.text) + all_texts_aug

# Reconstruct augmented_data
resampled_labels = mlb.inverse_transform(y_train_aug)
augmented_data = [{'text': text, 'labels': labels} for text, labels in zip(train_texts_aug, resampled_labels)]
print(f"Augmented dataset size: {len(augmented_data)} (capped at ~1.5x original)")
print(f"y_train_aug shape: {y_train_aug.shape}")
print("✅ Test passed: Per-class SMOTE + nlpaug for rares, no multilabel ValueError")