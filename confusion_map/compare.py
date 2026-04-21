import json

from skimage.metrics import structural_similarity as ssim
import cv2

from model.constants import MODEL_PATH
from model.verification import calculate_distance

img1_path = "../data/dataset_skeletons/U+8BF7.png"
img2_path = "../data/dataset_skeletons/U+6E05.png"
model_path = "../model/" + MODEL_PATH

# 三兰 哀衰 more similar in ai
# 使吏 less similar in ai
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Calculate SSIM (Returns a score from -1 to 1, where 1 is a perfect match)
ssim_score, diff = ssim(img1, img2, full=True)

with open('confusion_map.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

ai_score = calculate_distance(img1_path, img2_path, model_path)

ssim_pct = ((ssim_score + 1.0) / 2.0) * 100
ai_pct = (1.0 - (ai_score / 2.0)) * 100

print(f"Mathematical SSIM Score: {ssim_pct:.1f}%")

print(f"AI Score: {ai_pct:.1f}%")