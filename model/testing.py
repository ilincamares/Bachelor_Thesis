import torch
import os
import random
from PIL import Image
import torch.nn.functional as F

from model.constants import MODEL_PATH
from model.siamese_network import SiameseNetwork
from model.transforms import base_transform, positive_transform

TEST_DATA_FOLDER = "../data/dataset_skeletons"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


# Test Loop
files = [f for f in os.listdir(TEST_DATA_FOLDER) if f.endswith('.png')]
print(f"Testing on {len(files)} characters...")

correct_count = 0
total_tests = 0

for i in range(3000):
    # Pick anchor
    anchor_file = random.choice(files)
    img_anchor = Image.open(os.path.join(TEST_DATA_FOLDER, anchor_file))

    # Create a Positive
    img_positive = img_anchor.copy()

    tensor_anchor = base_transform(img_anchor).unsqueeze(0).to(device)
    tensor_positive = positive_transform(img_positive).unsqueeze(0).to(device)

    # Pick a Negative
    neg_file = random.choice(files)
    while neg_file == anchor_file:
        neg_file = random.choice(files)

    img_negative = Image.open(os.path.join(TEST_DATA_FOLDER, neg_file))
    tensor_negative = base_transform(img_negative).unsqueeze(0).to(device)

    with torch.no_grad():
        vec_a = model.forward_once(tensor_anchor)
        vec_p = model.forward_once(tensor_positive)
        vec_n = model.forward_once(tensor_negative)

    # Calculate distances
    dist_positive = F.pairwise_distance(vec_a, vec_p).item()
    dist_negative = F.pairwise_distance(vec_a, vec_n).item()

    # Compare
    if dist_negative > dist_positive:
        correct_count += 1

    total_tests += 1

    if i % 20 == 0 and dist_negative > dist_positive:
        print(f"Test {i}: Dist(Pos)={dist_positive:.2f} vs Dist(Neg)={dist_negative:.2f} | correct")
    if dist_negative < dist_positive:
        code_half = anchor_file.split('.')[0]
        hex_code = code_half.replace("U+", "")
        anchor = chr(int(hex_code, 16))

        code_half = neg_file.split('.')[0]
        hex_code = code_half.replace("U+", "")
        negative = chr(int(hex_code, 16))

        print(f"Test {i}: Dist(Pos)={dist_positive:.2f} vs Dist(Neg)={dist_negative:.2f} | INCORRECT ====== anchor: {anchor} | negative: {negative}")

accuracy = (correct_count / total_tests) * 100
print(f"\nFINAL RESULT: {accuracy:.2f}% Accuracy")