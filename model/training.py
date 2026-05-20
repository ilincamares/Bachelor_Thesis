import os

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from model.constants import MODEL_PATH, DATASET_FILE
from model.siamese_network import SiameseNetwork
from model.triplet_dataset import TripletDataset


def train_model_pro(model, train_loader, val_loader, max_epochs=50, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Training (Max Epochs: {max_epochs}, Patience: {patience})")

    for epoch in range(max_epochs):
        # 1. TRAINING PHASE
        model.train()
        running_train_loss = 0.0

        for anchors, positives, negatives in train_loader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            optimizer.zero_grad()
            vec_a, vec_p, vec_n = model(anchors, positives, negatives)
            loss = criterion(vec_a, vec_p, vec_n)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # 2. VALIDATION PHASE
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for anchors, positives, negatives in val_loader:
                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

                vec_a, vec_p, vec_n = model(anchors, positives, negatives)
                loss = criterion(vec_a, vec_p, vec_n)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{max_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}",
              end="")

        # 3. EARLY STOPPING CHECK
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("New Best!")
            torch.save(model.state_dict(), "siamese_hanzi_best.pth")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nStopping Early! Validation loss hasn't improved for {patience} epochs.")
            print(f"The model has officially learned as much as it can without memorizing.")
            break

    model.load_state_dict(best_model_wts)
    print(f"\nTraining Complete. Best Validation Loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    IMAGE_FOLDER = "../data/" + DATASET_FILE

    all_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')]

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    print(f"Training on {len(train_files)} unique characters...")
    print(f"Validating on {len(val_files)} unique unseen characters...")

    train_dataset = TripletDataset(image_folder=IMAGE_FOLDER, file_list=train_files)
    val_dataset = TripletDataset(image_folder=IMAGE_FOLDER, file_list=val_files)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SiameseNetwork()
    trained_model = train_model_pro(model, train_loader, val_loader)

    torch.save(trained_model.state_dict(), MODEL_PATH)
    print("Model saved!")