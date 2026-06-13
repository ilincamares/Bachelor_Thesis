import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from model.constants import DATASET_FILE


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str, data):
    with open(path, "w+", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def parse_character_from_filename(path: Path) -> str | None:
    stem = path.stem

    if stem.startswith("U+"):
        hex_code = stem.replace("U+", "")
        try:
            return chr(int(hex_code, 16))
        except ValueError:
            return None

    if len(stem) == 1:
        return stem

    return None


def load_images(image_folder: Path):
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    images = {}

    files = sorted([
        file for file in image_folder.iterdir()
        if file.suffix.lower() == ".png"
    ])

    for file_path in files:
        character = parse_character_from_filename(file_path)

        if character is None:
            continue

        image = Image.open(file_path).convert("L")
        array = np.asarray(image, dtype=np.float32)

        # Normalize to [0, 1].
        if array.max() > 1:
            array = array / 255.0

        images[character] = array

    return images


def parse_snn_neighbor(neighbor):

    if isinstance(neighbor, dict):
        character = neighbor.get("character")
        score = neighbor.get("similarity")
        return character, score

    if isinstance(neighbor, (list, tuple)) and len(neighbor) >= 2:
        return neighbor[0], neighbor[1]

    return None, None


def get_snn_top_k(confusion_map: dict, target: str, k: int):
    neighbors = confusion_map.get(target, [])

    parsed = []

    for neighbor in neighbors:
        character, score = parse_snn_neighbor(neighbor)

        if character is None:
            continue

        parsed.append(character)

        if len(parsed) == k:
            break

    return parsed


def calculate_ssim_score(image_a, image_b):
    return float(
        ssim(
            image_a,
            image_b,
            data_range=1.0,
        )
    )


def compute_ssim_neighbors_for_target(
    target: str,
    images: dict[str, np.ndarray],
    candidate_characters: list[str],
    top_k: int,
):
    target_image = images[target]

    scores = []

    for candidate in candidate_characters:
        if candidate == target:
            continue

        candidate_image = images[candidate]

        score = calculate_ssim_score(target_image, candidate_image)

        if math.isfinite(score):
            scores.append(
                {
                    "character": candidate,
                    "ssim": round(score, 6),
                }
            )

    scores.sort(key=lambda item: item["ssim"], reverse=True)

    return scores[:top_k]


def load_hsk_filter(metadata_excel: Path | None, hsk_level: str | None):
    if metadata_excel is None or hsk_level is None or hsk_level.lower() == "all":
        return None

    if not metadata_excel.exists():
        raise FileNotFoundError(f"Metadata Excel file not found: {metadata_excel}")

    df = pd.read_excel(metadata_excel)

    if "Character" not in df.columns:
        raise ValueError("Metadata file must contain a 'Character' column.")

    if "HSK" not in df.columns:
        raise ValueError("Metadata file must contain an 'HSK' column.")

    allowed = set()

    for _, row in df.iterrows():
        character = row.get("Character")
        hsk = row.get("HSK")

        if pd.isna(character) or pd.isna(hsk):
            continue

        character = str(character).strip()
        hsk = str(hsk).strip()

        if hsk.endswith(".0"):
            hsk = hsk[:-2]

        if hsk == str(hsk_level):
            allowed.add(character)

    return allowed


def compare_neighbors(
    confusion_map: dict,
    images: dict[str, np.ndarray],
    top_k_values: list[int],
    max_targets: int | None = None,
    seed: int = 42,
):
    max_k = max(top_k_values)

    common_targets = sorted(
        set(confusion_map.keys()) & set(images.keys())
    )

    if max_targets is not None and len(common_targets) > max_targets:
        rng = random.Random(seed)
        common_targets = rng.sample(common_targets, max_targets)
        common_targets = sorted(common_targets)

    candidate_characters = sorted(images.keys())

    results_per_target = []

    for index, target in enumerate(common_targets, start=1):
        print(f"[{index}/{len(common_targets)}] Processing {target}")

        snn_top_max = get_snn_top_k(
            confusion_map=confusion_map,
            target=target,
            k=max_k,
        )

        if len(snn_top_max) == 0:
            continue

        ssim_neighbors = compute_ssim_neighbors_for_target(
            target=target,
            images=images,
            candidate_characters=candidate_characters,
            top_k=max_k,
        )

        ssim_top_max = [
            item["character"]
            for item in ssim_neighbors
        ]

        if len(ssim_top_max) < max_k:
            continue

        target_result = {
            "target": target,
            "snn_top": snn_top_max,
            "ssim_top": ssim_top_max,
            "ssim_scores": ssim_neighbors,
        }

        for k in top_k_values:
            if len(snn_top_max) < k or len(ssim_top_max) < k:
                target_result[f"overlap_at_{k}"] = None
                target_result[f"overlap_ratio_at_{k}"] = None
                continue

            snn_set = set(snn_top_max[:k])
            ssim_set = set(ssim_top_max[:k])

            overlap_count = len(snn_set & ssim_set)
            overlap_ratio = overlap_count / k

            target_result[f"overlap_at_{k}"] = overlap_count
            target_result[f"overlap_ratio_at_{k}"] = overlap_ratio

        snn_top_1 = snn_top_max[0]
        ssim_top_1 = ssim_top_max[0]

        target_result["top_1_match"] = snn_top_1 == ssim_top_1

        if snn_top_1 in ssim_top_max:
            target_result["snn_top_1_rank_in_ssim_top_k"] = ssim_top_max.index(snn_top_1) + 1
        else:
            target_result["snn_top_1_rank_in_ssim_top_k"] = None

        results_per_target.append(target_result)

    return results_per_target


def summarize_results(results_per_target, top_k_values):
    summary = {
        "targets_evaluated": len(results_per_target),
    }

    if not results_per_target:
        return summary

    top_1_matches = [
        result["top_1_match"]
        for result in results_per_target
    ]

    summary["top_1_exact_match_count"] = int(sum(top_1_matches))
    summary["top_1_exact_match_percent"] = round(
        sum(top_1_matches) / len(top_1_matches) * 100,
        2,
    )

    for k in top_k_values:
        ratios = [
            result[f"overlap_ratio_at_{k}"]
            for result in results_per_target
            if result.get(f"overlap_ratio_at_{k}") is not None
        ]

        counts = [
            result[f"overlap_at_{k}"]
            for result in results_per_target
            if result.get(f"overlap_at_{k}") is not None
        ]

        if ratios:
            summary[f"average_overlap_at_{k}"] = round(float(np.mean(counts)), 4)
            summary[f"average_overlap_percent_at_{k}"] = round(float(np.mean(ratios)) * 100, 2)
        else:
            summary[f"average_overlap_at_{k}"] = None
            summary[f"average_overlap_percent_at_{k}"] = None

    ranks = [
        result["snn_top_1_rank_in_ssim_top_k"]
        for result in results_per_target
        if result["snn_top_1_rank_in_ssim_top_k"] is not None
    ]

    if ranks:
        summary["average_rank_of_snn_top_1_in_ssim_top_k"] = round(float(np.mean(ranks)), 4)
    else:
        summary["average_rank_of_snn_top_1_in_ssim_top_k"] = None

    return summary


def make_examples(results_per_target, top_k=5, count=10):
    sorted_results = sorted(
        results_per_target,
        key=lambda result: result.get(f"overlap_ratio_at_{top_k}", 0),
        reverse=True,
    )

    examples = []

    # High-overlap examples.
    examples.extend(sorted_results[: count // 2])

    # Low-overlap examples.
    examples.extend(sorted_results[-(count // 2):])

    formatted = []

    for item in examples:
        formatted.append(
            {
                "target": item["target"],
                "snn_top_5": item["snn_top"][:5],
                "ssim_top_5": item["ssim_top"][:5],
                "overlap_at_5": item.get("overlap_at_5"),
                "overlap_percent_at_5": round(
                    item.get("overlap_ratio_at_5", 0) * 100,
                    2,
                ),
            }
        )

    return formatted


def print_summary(summary, top_k_values):
    print("\nSNN vs SSIM Comparison Summary")
    print("=" * 60)

    print(f"Targets evaluated: {summary['targets_evaluated']}")
    print(
        "Top-1 exact match:",
        f"{summary.get('top_1_exact_match_count')} "
        f"({summary.get('top_1_exact_match_percent')}%)",
    )

    for k in top_k_values:
        print(
            f"Average overlap@{k}: "
            f"{summary.get(f'average_overlap_at_{k}')} / {k} "
            f"({summary.get(f'average_overlap_percent_at_{k}')}%)"
        )

    print(
        "Average rank of SNN top-1 inside SSIM top-k:",
        summary.get("average_rank_of_snn_top_1_in_ssim_top_k"),
    )


def print_latex_table(summary, top_k_values):
    print("\nLaTeX table:")
    print("-" * 60)

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Comparison between SNN-based neighbours and SSIM-based neighbours}")
    print(r"\begin{tabular}{lr}")
    print(r"\hline")
    print(r"Metric & Value \\")
    print(r"\hline")

    print(f"Targets evaluated & {summary['targets_evaluated']} \\\\")
    print(
        f"Top-1 exact match & "
        f"{summary.get('top_1_exact_match_percent')}\\% \\\\"
    )

    for k in top_k_values:
        print(
            f"Average overlap@{k} & "
            f"{summary.get(f'average_overlap_percent_at_{k}')}\\% \\\\"
        )

    avg_rank = summary.get("average_rank_of_snn_top_1_in_ssim_top_k")

    if avg_rank is not None:
        print(f"Average rank of SNN top-1 in SSIM top-k & {avg_rank} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SNN confusion map neighbours with SSIM-based image neighbours."
    )

    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Neighbourhood sizes to compare.",
    )

    parser.add_argument(
        "--max-targets",
        type=int,
        default=300
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used if max-targets is set.",
    )

    args = parser.parse_args()

    confusion_map_path = Path("../scripts/confusion_maps/confusion_map.json")
    image_folder = Path("../data/" + DATASET_FILE)

    print("Loading confusion map...")
    confusion_map = load_json(confusion_map_path)

    print("Loading images...")
    images = load_images(image_folder)

    print(f"Loaded {len(confusion_map)} confusion map entries.")
    print(f"Loaded {len(images)} images.")

    results_per_target = compare_neighbors(
        confusion_map=confusion_map,
        images=images,
        top_k_values=args.top_k,
        max_targets=args.max_targets,
        seed=args.seed,
    )

    summary = summarize_results(
        results_per_target=results_per_target,
        top_k_values=args.top_k,
    )

    examples = make_examples(
        results_per_target=results_per_target,
        top_k=5,
        count=10,
    )

    output = {
        "summary": summary,
        "examples": examples,
        "per_target_results": results_per_target,
    }

    save_json("snn_vs_ssim_comparison.json", output)

    print_summary(summary, args.top_k)
    print_latex_table(summary, args.top_k)

    print("\nExample cases:")
    print("-" * 60)

    for example in examples:
        print(
            f"Target: {example['target']} | "
            f"SNN: {example['snn_top_5']} | "
            f"SSIM: {example['ssim_top_5']} | "
            f"overlap@5: {example['overlap_percent_at_5']}%"
        )

    print(f"\nSaved results")


if __name__ == "__main__":
    main()