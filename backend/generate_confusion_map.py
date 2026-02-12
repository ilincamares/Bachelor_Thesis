import torch
import json
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from scipy.spatial.distance import cdist

from model.constants import MODEL_PATH, DATASET_FILE
from model.siamese_network import SiameseNetwork

IMAGE_FOLDER = "../data/" + DATASET_FILE
OUTPUT_JSON = "big_confusion_map.json"
MODEL_FOLDER = "../model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_FOLDER + "/" + MODEL_PATH))
model.eval()

transform = T.Compose([
    T.Grayscale(),
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')])
vectors = []
char_labels = []

with torch.no_grad():
    for i, filename in enumerate(files):
        # Load and transform
        path = os.path.join(IMAGE_FOLDER, filename)
        img = Image.open(path)
        t = transform(img).unsqueeze(0).to(device)

        # Get feature vector (convert from gpu)
        v = model.forward_once(t).cpu().numpy()[0]
        vectors.append(v)

        # Decode filename to get character
        code_half = filename.split('.')[0]
        hex_code = code_half.replace("U+", "")
        char = chr(int(hex_code, 16))
        char_labels.append(char)

        if i % 100 == 0:
            print(f"Processed {i}/{len(files)} images...")

# Calculate distance between every single pair of characters' feature vectors
print("Calculating distance matrix...")
feature_matrix = np.array(vectors)
distance_matrix = cdist(feature_matrix, feature_matrix, metric='euclidean')

# Find top matches
print("Finding top lookalikes...")
confusion_map = {}

for i, target in enumerate(char_labels):
    distances = distance_matrix[i]

    top_indices = np.argsort(distances)[1:6]
    best_3 = [char_labels[idx] for idx in top_indices]

    confusion_map[target] = best_3

    if i % 10 == 0:
        print(f"Processed {i}/{len(char_labels)} characters...")


with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(confusion_map, f, ensure_ascii=False, indent=4)

print("Done!")