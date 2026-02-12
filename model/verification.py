import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

from model.constants import MODEL_PATH
from model.siamese_network import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

transform = T.Compose([
    T.Grayscale(),
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])


def calculate_distance(img_path1, img_path2):
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    t1 = transform(img1).unsqueeze(0).to(device)
    t2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        v1 = model.forward_once(t1)
        v2 = model.forward_once(t2)

    dist = F.pairwise_distance(v1, v2)
    return dist.item()

char_A = '咸'
char_B = '或'

char_A_path = "../data/dataset_images/" + f"U+{ord(char_A):X}" + ".png"
char_B_path = "../data/dataset_images/" + f"U+{ord(char_B):X}" + ".png"

# Test 1: Identity (Should be very close to 0)
dist_same = calculate_distance(char_A_path, char_A_path)
print(f"Similarity score between character A and itself: {100 - dist_same:.4f}%")
dist_same = calculate_distance(char_B_path, char_B_path)
print(f"Similarity score between character B and itself: {100 - dist_same:.4f}%")

# Test 2: Different (Should be large, > 1.0)
dist_diff = calculate_distance(char_A_path, char_B_path)
print(f"\nSimilarity score between the two: {100 - dist_diff:.4f}%")
