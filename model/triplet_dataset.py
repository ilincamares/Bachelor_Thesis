import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model.constants import DATASET_FILE
from model.transforms import positive_transform, base_transform


class TripletDataset(Dataset):
    def __init__(self, image_folder, file_list=None):
        self.image_folder = image_folder

        if file_list is not None:
            self.image_files = file_list
        else:
            self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

        # Base Transform For Anchor and Negative
        self.base_transform = base_transform

        # Distortions For Positive
        self.positive_transform = positive_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        anchor_path = os.path.join(self.image_folder, self.image_files[idx])
        anchor_image = Image.open(anchor_path).convert('L')

        # Anchor and Positive
        anchor_tensor = self.base_transform(anchor_image)
        positive_tensor = self.positive_transform(anchor_image)

        # Negative
        negative_filename = random.choice(self.image_files)
        while negative_filename == self.image_files[idx]:
            negative_filename = random.choice(self.image_files)

        negative_path = os.path.join(self.image_folder, negative_filename)
        negative_image = Image.open(negative_path).convert('L')
        negative_tensor = self.base_transform(negative_image)


        return anchor_tensor, positive_tensor, negative_tensor




# TEST
if __name__ == "__main__":
    dataset = TripletDataset(image_folder="../data/" + DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    anchors, positives, negatives = next(iter(dataloader))
    print(f"Anchors shape: {anchors.shape}")  # Expected: [32, 1, 96, 96]
    print(f"Positives shape: {positives.shape}")  # Expected: [32, 1, 96, 96]
    print(f"Negatives shape: {negatives.shape}")  # Expected: [32, 1, 96, 96]
    print("DataLoader is ready for the Siamese Network!")