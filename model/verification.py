import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

from model.constants import MODEL_PATH
from model.siamese_network import SiameseNetwork
from model.transforms import base_transform


def calculate_distance(img_path1, img_path2, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    t1 = base_transform(img1).unsqueeze(0).to(device)
    t2 = base_transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        v1 = model.forward_once(t1)
        v2 = model.forward_once(t2)

    dist = F.pairwise_distance(v1, v2)
    return dist.item()

if __name__ == "__main__":
    char_A = '大'
    char_B = '犬'

    char_A_path = "../data/dataset_skeletons/" + f"U+{ord(char_A):X}" + ".png"
    char_B_path = "../data/dataset_skeletons/" + f"U+{ord(char_B):X}" + ".png"


    # Test 1: Identity (Should be very close to 0)
    dist_same = calculate_distance(char_A_path, char_A_path, MODEL_PATH)
    print(f"Distance between character A and itself: {dist_same:.4f}")
    dist_same = calculate_distance(char_B_path, char_B_path, MODEL_PATH)
    print(f"Distance between character B and itself: {dist_same:.4f}")

    # Test 2: Different (Should be large, > 1.0)
    dist_diff = calculate_distance(char_A_path, char_B_path, MODEL_PATH)
    print(f"\nDistance between the two: {dist_diff:.4f}")
