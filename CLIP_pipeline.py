import clip
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm

# Path to your main folder containing "real" and "fake"
BASE_DIR = r"C:\Users\hedif\OneDrive\Documents\GitHub\Real-vs-Fake-audio\spectrograms_output"

# Check if folder exists
if not os.path.exists(BASE_DIR):
    raise Exception(f"Folder does not exist: {BASE_DIR}")
print(f"Found base folder: {BASE_DIR}")

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Collect image paths
img_paths = []
labels = {}  # keep track of real/fake for each image
for subdir in ["real", "fake"]:
    folder = os.path.join(BASE_DIR, subdir)
    if os.path.exists(folder):
        sub_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        img_paths.extend(sub_paths)
        for p in sub_paths:
            labels[p] = subdir
        print(f"Found {len(sub_paths)} images in {folder}")
    else:
        print(f" Skipping missing folder: {folder}")

print(f"Total images found: {len(img_paths)}")


# Encode images with CLIP
def get_img_features(img_path):
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(image)
    return feat / feat.norm(dim=-1, keepdim=True)  # normalize

img_features = {}
for path in tqdm(img_paths, desc="Encoding images"):
    try:
        img_features[path] = get_img_features(path).cpu().numpy()
    except Exception as e:
        print(f"Error encoding {path}: {e}")

# Compute pairwise similarity
keys = list(img_features.keys())
feats = np.vstack([img_features[k] for k in keys])
sim_matrix = cosine_similarity(feats, feats)

# Sort the filenames for the similarity DataFrame
sorted_keys = sorted(keys, key=lambda x: os.path.basename(x))

df_sim = pd.DataFrame(sim_matrix,
                      index=[os.path.basename(k) for k in sorted_keys],
                      columns=[os.path.basename(k) for k in sorted_keys])

# Save similarity matrix
df_sim.to_csv("img2img_similarity.csv")
print("Saved img2img_similarity.csv")

#  Compute stats 
real_idx = [i for i, k in enumerate(keys) if labels[k] == "real"]
fake_idx = [i for i, k in enumerate(keys) if labels[k] == "fake"]

real_vs_real = sim_matrix[np.ix_(real_idx, real_idx)]
fake_vs_fake = sim_matrix[np.ix_(fake_idx, fake_idx)]
real_vs_fake = sim_matrix[np.ix_(real_idx, fake_idx)]

def print_stats(name, mat):
    # exclude diagonal for self-similarity
    if mat.shape[0] == mat.shape[1]:
        mat = mat[np.triu_indices_from(mat, k=1)]
    print(f"\n{name} statistics:")
    print(f"Mean: {np.mean(mat):.4f}")
    print(f"Std:  {np.std(mat):.4f}")
    print(f"Min:  {np.min(mat):.4f}")
    print(f"Max:  {np.max(mat):.4f}")

print_stats("Real vs Real", real_vs_real)
print_stats("Fake vs Fake", fake_vs_fake)
print_stats("Real vs Fake", real_vs_fake)