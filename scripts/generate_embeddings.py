import torch
import json
import os
from PIL import Image

from model.constants import MODEL_PATH, DATASET_FILE
from model.siamese_network import SiameseNetwork
from model.transforms import base_transform

OUTPUT_JSON = f"embeddings/embeddings.json"

IMAGE_FOLDER = "../data/" + DATASET_FILE
MODEL_FOLDER = "../model"

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_FOLDER + "/" + MODEL_PATH, map_location=device))
model.eval()

files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')])
vectors = {}

# Process Images
with torch.no_grad():
    for i, filename in enumerate(files):
        # Decode filename to get character
        code_half = filename.split('.')[0]
        hex_code = code_half.replace("U+", "")
        char = chr(int(hex_code, 16))

        # Load and transform
        path = os.path.join(IMAGE_FOLDER, filename)
        img = Image.open(path)
        t = base_transform(img).unsqueeze(0).to(device)

        # Get feature vector (convert from gpu)
        v = model.forward_once(t)
        vectors[char] = v.squeeze(0).detach().cpu().numpy().tolist()

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} images...")

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(vectors, f, ensure_ascii=False, indent=4)

print(f"Done! Saved to {OUTPUT_JSON}")