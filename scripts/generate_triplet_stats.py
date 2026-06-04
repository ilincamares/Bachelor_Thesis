import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn import TripletMarginLoss

from model.constants import DATASET_FILE, MODEL_PATH
from model.siamese_network import SiameseNetwork
from model.transforms import base_transform

try:
    from model.transforms import positive_transform
except ImportError:
    positive_transform = None


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def build_validation_split(
    image_folder: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    files = sorted([
        file for file in image_folder.iterdir()
        if file.suffix.lower() == ".png"
    ])

    if len(files) < 2:
        raise ValueError("At least two image files are required for triplet evaluation.")

    rng = random.Random(seed)
    rng.shuffle(files)

    split_index = int(len(files) * train_ratio)

    train_files = files[:split_index]
    val_files = files[split_index:]

    if len(val_files) < 2:
        raise ValueError("Validation set is too small. Lower train_ratio or add more images.")

    return train_files, val_files


def get_embedding(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.forward_once(image_tensor)

    return embedding


def evaluate_triplets(
    model,
    val_files,
    device,
    margin: float = 0.5,
    repeats_per_image: int = 3,
    seed: int = 42,
):
    rng = random.Random(seed)

    criterion = TripletMarginLoss(margin=margin, p=2)

    positive_distances = []
    negative_distances = []
    gaps = []
    losses = []
    margin_satisfied = 0
    ordering_satisfied = 0
    total = 0

    model.eval()

    for anchor_path in val_files:
        for _ in range(repeats_per_image):
            negative_path = anchor_path

            while negative_path == anchor_path:
                negative_path = rng.choice(val_files)

            anchor_img = load_image(anchor_path)
            positive_img = load_image(anchor_path)
            negative_img = load_image(negative_path)

            anchor_tensor = base_transform(anchor_img)

            if positive_transform is not None:
                positive_tensor = positive_transform(positive_img)
            else:
                positive_tensor = base_transform(positive_img)

            negative_tensor = base_transform(negative_img)

            anchor_embedding = get_embedding(model, anchor_tensor, device)
            positive_embedding = get_embedding(model, positive_tensor, device)
            negative_embedding = get_embedding(model, negative_tensor, device)

            positive_distance = F.pairwise_distance(
                anchor_embedding,
                positive_embedding,
                p=2,
            ).item()

            negative_distance = F.pairwise_distance(
                anchor_embedding,
                negative_embedding,
                p=2,
            ).item()

            loss = criterion(
                anchor_embedding,
                positive_embedding,
                negative_embedding,
            ).item()

            gap = negative_distance - positive_distance

            positive_distances.append(positive_distance)
            negative_distances.append(negative_distance)
            gaps.append(gap)
            losses.append(loss)

            if positive_distance < negative_distance:
                ordering_satisfied += 1

            if positive_distance + margin < negative_distance:
                margin_satisfied += 1

            total += 1

    return {
        "validation_triplets_evaluated": total,
        "margin": margin,
        "mean_triplet_loss": mean(losses),
        "mean_anchor_positive_distance": mean(positive_distances),
        "mean_anchor_negative_distance": mean(negative_distances),
        "mean_distance_gap_neg_minus_pos": mean(gaps),
        "std_anchor_positive_distance": std(positive_distances),
        "std_anchor_negative_distance": std(negative_distances),
        "min_anchor_positive_distance": min(positive_distances),
        "max_anchor_positive_distance": max(positive_distances),
        "min_anchor_negative_distance": min(negative_distances),
        "max_anchor_negative_distance": max(negative_distances),
        "ordering_satisfaction_rate_percent": round(
            ordering_satisfied / total * 100,
            2,
        ),
        "margin_satisfaction_rate_percent": round(
            margin_satisfied / total * 100,
            2,
        ),
    }


def mean(values):
    return round(sum(values) / len(values), 6)


def std(values):
    if len(values) < 2:
        return 0.0

    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)

    return round(variance ** 0.5, 6)


def print_results(results: dict):
    print("\nTriplet Evaluation Results")
    print("-" * 40)

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nLaTeX table rows:")
    print("-" * 40)

    latex_labels = {
        "validation_triplets_evaluated": "Validation triplets evaluated",
        "mean_triplet_loss": "Mean triplet loss",
        "mean_anchor_positive_distance": "Mean anchor-positive distance",
        "mean_anchor_negative_distance": "Mean anchor-negative distance",
        "mean_distance_gap_neg_minus_pos": "Mean distance gap",
        "ordering_satisfaction_rate_percent": "Ordering satisfaction rate",
        "margin_satisfaction_rate_percent": "Margin satisfaction rate",
    }

    for key, label in latex_labels.items():
        value = results[key]

        if "percent" in key:
            print(f"{label} & {value}\\% \\\\")
        else:
            print(f"{label} & {value} \\\\")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Siamese Network using validation triplets."
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio used to reconstruct the validation split.",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Triplet loss margin.",
    )

    parser.add_argument(
        "--repeats-per-image",
        type=int,
        default=3,
        help="Number of random validation triplets generated per validation image.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for validation split and negative sampling.",
    )
    args = parser.parse_args()

    image_folder = Path("../data/" + DATASET_FILE)
    model_path = Path("../model/" + MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_files, val_files = build_validation_split(
        image_folder=image_folder,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    model = SiameseNetwork().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    results = evaluate_triplets(
        model=model,
        val_files=val_files,
        device=device,
        margin=args.margin,
        repeats_per_image=args.repeats_per_image,
        seed=args.seed,
    )

    results["validation_images"] = len(val_files)
    results["train_ratio"] = args.train_ratio
    results["repeats_per_image"] = args.repeats_per_image
    results["seed"] = args.seed

    with open("triplet_stats.json", "w+", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    print_results(results)

    print(f"\nSaved results")


if __name__ == "__main__":
    main()