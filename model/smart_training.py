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
    # Setup device and optimizer (includes the L2 weight decay we talked about!)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Tracking variables for Early Stopping
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Training (Max Epochs: {max_epochs}, Patience: {patience})")
    with open("logs.txt", "a") as f:
        f.write(f"Training (Max Epochs: {max_epochs}, Patience: {patience})")

    for epoch in range(max_epochs):
        # =========================
        # 1. TRAINING PHASE
        # =========================
        model.train()  # Tell the model it is learning time
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

        # =========================
        # 2. VALIDATION PHASE
        # =========================
        model.eval()  # Tell the model it is TEST time (turns off Dropout and BatchNorm updates)
        running_val_loss = 0.0

        # torch.no_grad() tells PyTorch not to calculate gradients.
        # It saves memory and makes the test run 2x faster!
        with torch.no_grad():
            for anchors, positives, negatives in val_loader:
                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

                vec_a, vec_p, vec_n = model(anchors, positives, negatives)
                loss = criterion(vec_a, vec_p, vec_n)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        # Print the progress for this epoch
        print(f"Epoch [{epoch + 1}/{max_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}",
              end="")
        with open("logs.txt", "a") as f:
            f.write(f"Epoch [{epoch + 1}/{max_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}" + "\n")

        # =========================
        # 3. EARLY STOPPING CHECK
        # =========================
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0  # Reset patience because we got a new high score!
            print(" ✅ New Best!")
            torch.save(model.state_dict(), "siamese_hanzi_best.pth")
        else:
            patience_counter += 1
            print(f" ⏳ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n🛑 Stopping Early! Validation loss hasn't improved for {patience} epochs.")
            print(f"The model has officially learned as much as it can without memorizing.")
            break

    # Load the best weights before giving the model back to you
    model.load_state_dict(best_model_wts)
    print(f"\n🎉 Training Complete. Best Validation Loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    IMAGE_FOLDER = "../data/" + DATASET_FILE

    # 1. Get ALL 8,100 characters
    all_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')]

    # 2. Split them: 80% Train (6480 chars), 20% Test/Validation (1620 chars)
    # random_state=42 ensures you get the exact same split every time you run the code
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    print(f"📚 Training on {len(train_files)} unique characters...")
    print(f"🧪 Validating on {len(val_files)} unique unseen characters...")

    # 3. Create the distinct Datasets
    train_dataset = TripletDataset(image_folder=IMAGE_FOLDER, file_list=train_files)
    val_dataset = TripletDataset(image_folder=IMAGE_FOLDER, file_list=val_files)

    # 4. Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SiameseNetwork()
    #
    # if os.path.exists(MODEL_PATH):
    #     model.load_state_dict(torch.load(MODEL_PATH))
    #     print("Model loaded!")
    # else:
    #     print("Model created!")

    trained_model = train_model_pro(model, train_loader, val_loader)

    torch.save(trained_model.state_dict(), MODEL_PATH)
    print("Model saved!")