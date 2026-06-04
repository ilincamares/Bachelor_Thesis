import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_embeddings(data):
    """
    Supports formats such as:

    [
        {"character": "你", "embedding": [0.1, 0.2, ...]},
        ...
    ]

    or:

    [
        ["你", [0.1, 0.2, ...]],
        ...
    ]

    or:

    {
        "你": [0.1, 0.2, ...],
        ...
    }
    """

    result = {}

    if isinstance(data, dict):
        for character, vector in data.items():
            result[str(character)] = np.array(vector, dtype=float)
        return result

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                character = (
                    item.get("character")
                    or item.get("char")
                    or item.get("hanzi")
                    or item.get("target")
                )

                vector = (
                    item.get("embedding")
                    or item.get("vector")
                    or item.get("features")
                )

                if character is not None and vector is not None:
                    result[str(character)] = np.array(vector, dtype=float)

            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                character = item[0]
                vector = item[1]
                result[str(character)] = np.array(vector, dtype=float)

    return result


def parse_projection(data):
    """
    Supports projection files such as:

    [
        {"character": "你", "x": 0.1, "y": 0.2},
        {"character": "好", "umap_x": 0.3, "umap_y": 0.4},
        {"character": "学", "tsne_x": 0.5, "tsne_y": 0.6}
    ]
    """

    result = {}

    for item in data:
        character = (
            item.get("character")
            or item.get("char")
            or item.get("hanzi")
            or item.get("target")
        )

        if character is None:
            continue

        x = (
            item.get("x")
            if item.get("x") is not None
            else item.get("umap_x")
            if item.get("umap_x") is not None
            else item.get("tsne_x")
            if item.get("tsne_x") is not None
            else item.get("projection_x")
        )

        y = (
            item.get("y")
            if item.get("y") is not None
            else item.get("umap_y")
            if item.get("umap_y") is not None
            else item.get("tsne_y")
            if item.get("tsne_y") is not None
            else item.get("projection_y")
        )

        if x is None or y is None:
            continue

        result[str(character)] = np.array([float(x), float(y)], dtype=float)

    return result


def align_embeddings_and_projection(embeddings, projection):
    common_characters = sorted(set(embeddings.keys()) & set(projection.keys()))

    if len(common_characters) < 20:
        raise ValueError(
            "Too few common characters between embeddings and projection files. "
            f"Found only {len(common_characters)}."
        )

    x_high = np.array([embeddings[char] for char in common_characters])
    x_low = np.array([projection[char] for char in common_characters])

    return common_characters, x_high, x_low


def neighbour_preservation(x_high, x_low, k):
    """
    Computes average overlap between top-k neighbours in original embedding space
    and top-k neighbours in the 2D projection.
    """

    high_nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    low_nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")

    high_nn.fit(x_high)
    low_nn.fit(x_low)

    high_indices = high_nn.kneighbors(x_high, return_distance=False)
    low_indices = low_nn.kneighbors(x_low, return_distance=False)

    overlaps = []

    for i in range(len(x_high)):
        # Remove self-neighbour at index 0.
        high_set = set(high_indices[i][1:])
        low_set = set(low_indices[i][1:])

        overlap = len(high_set & low_set) / k
        overlaps.append(overlap)

    return float(np.mean(overlaps))


def evaluate_projection(name, x_high, x_low, k_values):
    rows = []

    for k in k_values:
        trust = trustworthiness(
            X=x_high,
            X_embedded=x_low,
            n_neighbors=k,
            metric="euclidean",
        )

        preservation = neighbour_preservation(
            x_high=x_high,
            x_low=x_low,
            k=k,
        )

        rows.append(
            {
                "projection": name,
                "k": k,
                "trustworthiness": round(float(trust), 6),
                "neighbour_preservation": round(float(preservation), 6),
                "neighbour_preservation_percent": round(float(preservation) * 100, 2),
            }
        )

    return rows


def print_latex_table(results):
    print("\nLaTeX table:")
    print("-" * 60)

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Comparison between UMAP and t-SNE projections}")
    print(r"\begin{tabular}{lrrr}")
    print(r"\hline")
    print(r"Projection & $k$ & Trustworthiness & Neighbour preservation \\")
    print(r"\hline")

    for row in results:
        projection = row["projection"]
        k = row["k"]
        trust = row["trustworthiness"]
        preservation = row["neighbour_preservation_percent"]

        print(f"{projection} & {k} & {trust} & {preservation}\\% \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UMAP and t-SNE projections using trustworthiness and neighbour preservation."
    )

    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[5, 10, 15],
        help="Neighbourhood sizes to evaluate.",
    )

    args = parser.parse_args()

    embeddings_data = load_json(Path("../scripts/embeddings/embeddings.json"))
    umap_data = load_json(Path("../scripts/embeddings/embedding_projection_umap.json"))
    tsne_data = load_json(Path("../scripts/embeddings/embedding_projection_tsne.json"))

    embeddings = parse_embeddings(embeddings_data)
    umap_projection = parse_projection(umap_data)
    tsne_projection = parse_projection(tsne_data)

    print(f"Loaded embeddings: {len(embeddings)}")
    print(f"Loaded UMAP points: {len(umap_projection)}")
    print(f"Loaded t-SNE points: {len(tsne_projection)}")

    common_umap, x_high_umap, x_low_umap = align_embeddings_and_projection(
        embeddings,
        umap_projection,
    )

    common_tsne, x_high_tsne, x_low_tsne = align_embeddings_and_projection(
        embeddings,
        tsne_projection,
    )

    print(f"Common characters for UMAP: {len(common_umap)}")
    print(f"Common characters for t-SNE: {len(common_tsne)}")

    results = []

    results.extend(
        evaluate_projection(
            name="UMAP",
            x_high=x_high_umap,
            x_low=x_low_umap,
            k_values=args.k,
        )
    )

    results.extend(
        evaluate_projection(
            name="t-SNE",
            x_high=x_high_tsne,
            x_low=x_low_tsne,
            k_values=args.k,
        )
    )

    df = pd.DataFrame(results)
    df.to_csv("projection_evaluation_results.csv", index=False)

    print("\nProjection evaluation results:")
    print(df.to_string(index=False))

    print_latex_table(results)

    print(f"\nSaved results")


if __name__ == "__main__":
    main()