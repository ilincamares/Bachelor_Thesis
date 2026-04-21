import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Convolutional Layers
        # Input: 96x96x1 images
        self.conv_layers = nn.Sequential(
            # Layer 1: 96x96 -> 48x48
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Layer 2: 48x48 -> 24x24
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Layer 3: 24x24 -> 12x12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fully Connected Layers
        flattened_size = 12 * 12 * 128

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

    def forward_once(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, anchor, positive, negative):
        anchor_vec = self.forward_once(anchor)
        positive_vec = self.forward_once(positive)
        negative_vec = self.forward_once(negative)
        return anchor_vec, positive_vec, negative_vec



#TEST
if __name__ == "__main__":
    model = SiameseNetwork()
    dummy_input = torch.randn(16, 1, 96, 96)
    output_vec = model.forward_once(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output vector shape: {output_vec.shape}")  # Should be [16, 128]