import argparse
import pandas as pd
import torch
import json
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from scipy.spatial.distance import cdist
import torch.nn.functional as F

from model.constants import MODEL_PATH, DATASET_FILE
from model.siamese_network import SiameseNetwork
from model.transforms import base_transform

parser = argparse.ArgumentParser(description="Generate an inclusive HSK confusion map.")
parser.add_argument("--hsk", type=int, required=True,
                    help="Target HSK level (1-6). Includes all levels up to this one.")
args = parser.parse_args()

TARGET_HSK = args.hsk
OUTPUT_JSON = f"confusion_map/confusion_map_hsk_{TARGET_HSK}.json"

IMAGE_FOLDER = "data/" + DATASET_FILE
MODEL_FOLDER = "model"

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_FOLDER + "/" + MODEL_PATH, map_location=device))
model.eval()

# Load Excel Data
print("Loading Excel mapping...")
df = pd.read_excel("data/character_files/General_Standard_Hanzi.xlsx")

char_hsk_map = df.set_index('Character')['Hsk Level'].to_dict()

files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')])
vectors = []
char_labels = []

# Process Images
with torch.no_grad():
    for i, filename in enumerate(files):
        # Decode filename to get character
        code_half = filename.split('.')[0]
        hex_code = code_half.replace("U+", "")
        char = chr(int(hex_code, 16))

        # Lookup character HSK level. Default to 999 if not found in Excel
        char_hsk = char_hsk_map.get(char, 999)

        # Skip characters that are NaN or above our target HSK level
        if pd.isna(char_hsk) or char_hsk > TARGET_HSK:
            continue

        char_labels.append(char)

        # Load and transform
        path = os.path.join(IMAGE_FOLDER, filename)
        img = Image.open(path)
        t = base_transform(img).unsqueeze(0).to(device)

        # Get feature vector (convert from gpu)
        v = model.forward_once(t).cpu().numpy()[0]
        vectors.append(v)

        if len(char_labels) % 100 == 0:
            print(f"Processed {len(char_labels)} valid HSK<={TARGET_HSK} images...")

print(f"Total valid characters found for HSK {TARGET_HSK}: {len(char_labels)}")

# Calculate Distance Matrix
print("Calculating distance matrix...")
feature_matrix = np.array(vectors)
distance_matrix = cdist(feature_matrix, feature_matrix, metric='euclidean')

# Find Top Matches
print("Finding top lookalikes...")
confusion_map = {}

for i, target in enumerate(char_labels):
    distances = distance_matrix[i]

    top_indices = np.argsort(distances)[1:6]

    best_5 = [(char_labels[idx], round((1.0 - distances[idx] / 2.0) * 100, 2)) for idx in top_indices]
    confusion_map[target] = best_5

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(confusion_map, f, ensure_ascii=False, indent=4)

print(f"Done! Saved to {OUTPUT_JSON}")