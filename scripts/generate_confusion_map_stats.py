import argparse
import json
import math
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_metadata(excel_path: Path):
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)

    character_col = "Character"
    pinyin_col = "Pinyin"
    hsk_col = "Hsk Level"

    required = [character_col, pinyin_col, hsk_col]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    char_to_pinyin = {}
    char_to_hsk = {}

    for _, row in df.iterrows():
        character = row.get(character_col)
        pinyin = row.get(pinyin_col)
        hsk = row.get(hsk_col)

        if pd.isna(character):
            continue

        character = str(character).strip()

        if not character:
            continue

        if not pd.isna(pinyin):
            pinyin = str(pinyin).strip()
            if pinyin and pinyin.lower() != "nan":
                char_to_pinyin[character] = pinyin

        if not pd.isna(hsk):
            hsk = str(hsk).strip()
            if hsk and hsk.lower() != "nan":
                # Handles values like 1.0 from Excel.
                if hsk.endswith(".0"):
                    hsk = hsk[:-2]
                char_to_hsk[character] = hsk

    return char_to_pinyin, char_to_hsk


def parse_neighbor(neighbor):
    """
    Supports both formats:
    ["你", 0.92]
    {"character": "你", "similarity": 0.92}
    """

    if isinstance(neighbor, dict):
        return neighbor.get("character"), neighbor.get("similarity")

    if isinstance(neighbor, (list, tuple)) and len(neighbor) >= 2:
        return neighbor[0], neighbor[1]

    return None, None


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def summarize(values):
    values = [v for v in values if v is not None]

    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
        }

    series = pd.Series(values)

    return {
        "count": int(series.count()),
        "mean": round(float(series.mean()), 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "std": round(float(series.std()), 4),
    }


def analyze_confusion_map(confusion_map, char_to_pinyin, char_to_hsk, top_k_values):
    top_k_scores = {k: [] for k in top_k_values}

    valid_distractor_counts = []
    same_pinyin_neighbor_counts = []
    characters_with_at_least_3_valid = 0

    per_hsk = {}

    total_characters = len(confusion_map)

    for target, neighbors in confusion_map.items():
        target_pinyin = char_to_pinyin.get(target)
        target_hsk = char_to_hsk.get(target, "unknown")

        if target_hsk not in per_hsk:
            per_hsk[target_hsk] = {
                "characters": 0,
                "valid_distractor_counts": [],
                "characters_with_at_least_3_valid": 0,
                "same_pinyin_neighbor_counts": [],
            }

        per_hsk[target_hsk]["characters"] += 1

        parsed_neighbors = []

        for neighbor in neighbors:
            neighbor_char, similarity = parse_neighbor(neighbor)
            similarity = safe_float(similarity)

            if neighbor_char is None or similarity is None:
                continue

            parsed_neighbors.append((neighbor_char, similarity))

        # Similarity statistics for top-k neighbours.
        for k in top_k_values:
            top_k = parsed_neighbors[:k]
            scores = [similarity for _, similarity in top_k]
            top_k_scores[k].extend(scores)

        used_pinyins = set()

        if target_pinyin:
            used_pinyins.add(target_pinyin)

        valid_count = 0
        same_pinyin_count = 0

        for neighbor_char, similarity in parsed_neighbors:
            neighbor_pinyin = char_to_pinyin.get(neighbor_char)

            if not neighbor_pinyin:
                continue

            if target_pinyin and neighbor_pinyin == target_pinyin:
                same_pinyin_count += 1
                continue

            if neighbor_pinyin in used_pinyins:
                same_pinyin_count += 1
                continue

            used_pinyins.add(neighbor_pinyin)
            valid_count += 1

        valid_distractor_counts.append(valid_count)
        same_pinyin_neighbor_counts.append(same_pinyin_count)

        per_hsk[target_hsk]["valid_distractor_counts"].append(valid_count)
        per_hsk[target_hsk]["same_pinyin_neighbor_counts"].append(same_pinyin_count)

        if valid_count >= 3:
            characters_with_at_least_3_valid += 1
            per_hsk[target_hsk]["characters_with_at_least_3_valid"] += 1

    similarity_summary = {
        f"top_{k}_scores": summarize(top_k_scores[k])
        for k in top_k_values
    }

    valid_distractor_summary = summarize(valid_distractor_counts)
    same_pinyin_summary = summarize(same_pinyin_neighbor_counts)

    per_hsk_summary = {}

    for hsk_level, data in per_hsk.items():
        character_count = data["characters"]

        if character_count == 0:
            percentage_with_3 = 0.0
        else:
            percentage_with_3 = round(
                data["characters_with_at_least_3_valid"] / character_count * 100,
                2,
            )

        per_hsk_summary[hsk_level] = {
            "characters": character_count,
            "average_valid_distractors": summarize(
                data["valid_distractor_counts"]
            )["mean"],
            "characters_with_at_least_3_valid_distractors": data[
                "characters_with_at_least_3_valid"
            ],
            "percentage_with_at_least_3_valid_distractors": percentage_with_3,
            "average_same_pinyin_skipped": summarize(
                data["same_pinyin_neighbor_counts"]
            )["mean"],
        }

    if total_characters == 0:
        percentage_with_3_valid = 0.0
    else:
        percentage_with_3_valid = round(
            characters_with_at_least_3_valid / total_characters * 100,
            2,
        )

    return {
        "total_characters_in_confusion_map": total_characters,
        "similarity_summary": similarity_summary,
        "valid_distractor_summary": valid_distractor_summary,
        "same_pinyin_neighbor_summary": same_pinyin_summary,
        "characters_with_at_least_3_valid_distractors": characters_with_at_least_3_valid,
        "percentage_with_at_least_3_valid_distractors": percentage_with_3_valid,
        "per_hsk_summary": dict(
            sorted(
                per_hsk_summary.items(),
                key=lambda item: hsk_sort_key(item[0]),
            )
        ),
    }


def hsk_sort_key(value):
    if value == "unknown":
        return 999

    try:
        return int(value)
    except ValueError:
        return 998


def print_main_summary(results):
    print("\nConfusion Map Statistics")
    print("=" * 50)

    print(
        "Total characters in confusion map:",
        results["total_characters_in_confusion_map"],
    )

    print("\nSimilarity score statistics:")
    for name, stats in results["similarity_summary"].items():
        print(f"\n{name}")
        print(f"  count: {stats['count']}")
        print(f"  mean:  {stats['mean']}")
        print(f"  min:   {stats['min']}")
        print(f"  max:   {stats['max']}")
        print(f"  std:   {stats['std']}")

    print("\nValid distractor statistics:")
    valid_stats = results["valid_distractor_summary"]
    print(f"  average valid distractors: {valid_stats['mean']}")
    print(f"  min valid distractors:     {valid_stats['min']}")
    print(f"  max valid distractors:     {valid_stats['max']}")
    print(
        "  characters with at least 3 valid distractors:",
        results["characters_with_at_least_3_valid_distractors"],
    )
    print(
        "  percentage with at least 3 valid distractors:",
        f"{results['percentage_with_at_least_3_valid_distractors']}%",
    )

    print("\nSame-pinyin skipped neighbour statistics:")
    same_stats = results["same_pinyin_neighbor_summary"]
    print(f"  average same-pinyin skipped: {same_stats['mean']}")
    print(f"  max same-pinyin skipped:     {same_stats['max']}")

    print("\nPer-HSK summary:")
    print(
        "HSK | Characters | Avg valid distractors | "
        "Characters with >=3 | % with >=3 | Avg same-pinyin skipped"
    )
    print("-" * 90)

    for hsk, data in results["per_hsk_summary"].items():
        print(
            f"{hsk:>3} | "
            f"{data['characters']:>10} | "
            f"{data['average_valid_distractors']:>21} | "
            f"{data['characters_with_at_least_3_valid_distractors']:>19} | "
            f"{data['percentage_with_at_least_3_valid_distractors']:>9}% | "
            f"{data['average_same_pinyin_skipped']:>24}"
        )


def print_latex_tables(results):
    print("\nLaTeX table: similarity statistics")
    print("-" * 50)

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Similarity score statistics in the generated confusion map}")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\hline")
    print(r"Neighbour set & Mean & Minimum & Maximum & Std. dev. \\")
    print(r"\hline")

    for name, stats in results["similarity_summary"].items():
        label = name.replace("_scores", "").replace("_", "-")
        print(
            f"{label} & {stats['mean']} & {stats['min']} & "
            f"{stats['max']} & {stats['std']} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\nLaTeX table: valid distractors by HSK")
    print("-" * 50)

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Valid distractor availability by HSK level}")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\hline")
    print(
        r"HSK level & Characters & Avg. valid distractors & "
        r"Characters with $\geq 3$ & Percentage \\"
    )
    print(r"\hline")

    for hsk, data in results["per_hsk_summary"].items():
        print(
            f"{hsk} & "
            f"{data['characters']} & "
            f"{data['average_valid_distractors']} & "
            f"{data['characters_with_at_least_3_valid_distractors']} & "
            f"{data['percentage_with_at_least_3_valid_distractors']}\\% \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze confusion map statistics for HanziTwins."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-k neighbour groups to summarize.",
    )

    args = parser.parse_args()

    confusion_map_path = Path("../scripts/confusion_maps/confusion_map.json")
    metadata_excel_path = Path("../data/character_files/General_Standard_Hanzi.xlsx")

    confusion_map = load_json(confusion_map_path)
    char_to_pinyin, char_to_hsk = load_metadata(metadata_excel_path)

    results = analyze_confusion_map(
        confusion_map=confusion_map,
        char_to_pinyin=char_to_pinyin,
        char_to_hsk=char_to_hsk,
        top_k_values=args.top_k,
    )

    with open("stats/confusion_map_stats.json", "w+", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    print_main_summary(results)
    print_latex_tables(results)

    print(f"\nSaved results")


if __name__ == "__main__":
    main()