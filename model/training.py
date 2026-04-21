import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.constants import MODEL_PATH, DATASET_FILE
from model.triplet_dataset import TripletDataset
from model.siamese_network import SiameseNetwork

def train_model(model, device, dataloader, max_epochs=100, patience=5):
    # Loss Function with Euclidean Distance
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print("Training...")

    # Training Loop
    # Epochs
    for epoch in range(max_epochs):
        running_loss = 0.0

        # Batch Loop
        for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
            # Move to GPU
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            # Zero Gradients
            optimizer.zero_grad()

            # Forward Pass
            anchor_vec, positive_vec, negative_vec = model(anchors, positives, negatives)

            # Calculate Loss
            loss = criterion(anchor_vec, positive_vec, negative_vec)

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}")

        # --- THE SMART CHECK ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("New Best!")
            # Save immediately just in case
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            print(f" No improvement ({patience_counter}/{patience})")

        # --- EARLY STOPPING TRIGGER ---
        if patience_counter >= patience:
            print(f"\n Stopping Early! Loss hasn't improved for {patience} epochs.")
            print(f"Best Loss was: {best_loss:.4f}")
            break

    print("Training Complete!")

    torch.save(best_model_wts, MODEL_PATH)
    print("Model saved!")



def train():
    train_dataset = TripletDataset(image_folder="../data/" + DATASET_FILE)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Model loaded!")
    else:
        print("Model created!")

    model.train()

    train_model(model, device, train_dataloader, max_epochs=30, patience=5)

if __name__ == "__main__":
    train()

