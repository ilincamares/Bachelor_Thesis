import json
import os
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = "embeddings/embeddings.json"
METADATA_PATH = "../data/character_files/General_Standard_Hanzi.xlsx"
OUTPUT_PATH = "embeddings/embedding_projection.json"


def load_embeddings(path: str):
    """
    Loads embeddings from a JSON file.

    Expected format:
    {
        "你": [0.12, 0.34, ...],
        "好": [0.45, 0.67, ...]
    }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("embeddings.json must be a dictionary: character -> vector")

    characters = []
    vectors = []

    for character, vector in data.items():
        if not isinstance(vector, list):
            continue

        if len(vector) == 0:
            continue

        characters.append(character)
        vectors.append(vector)

    if not characters:
        raise ValueError("No valid embeddings were found.")

    return characters, vectors


def find_column(df: pd.DataFrame, column_name: str):
    """
    Finds the matching column for a given column name.
    """
    normalized_columns = {
        column.lower().strip().replace(" ", "_"): column
        for column in df.columns
    }

    normalized_column_name = column_name.lower().strip().replace(" ", "_")

    if column_name in df.columns:
        return normalized_columns[normalized_column_name]

    return None


def load_metadata(path: str):
    """
    Loads character metadata from the Excel file.
    """
    if not os.path.exists(path):
        print(f"Metadata file not found: {path}")
        return {}

    df = pd.read_excel(path)

    character_column = find_column(df, "Character")
    pinyin_column = find_column(df, "Pinyin")
    hsk_column = find_column(df, "Hsk Level")

    if character_column is None:
        raise ValueError("Could not find a character column in the metadata file.")

    metadata = {}

    for _, row in df.iterrows():
        character = row.get(character_column)

        if pd.isna(character):
            continue

        character = str(character).strip()

        pinyin = None
        hsk_level = None

        if pinyin_column is not None:
            value = row.get(pinyin_column)
            if not pd.isna(value):
                pinyin = str(value).strip()

        if hsk_column is not None:
            value = row.get(hsk_column)
            if not pd.isna(value):
                hsk_level = str(int(value)).strip()

        metadata[character] = {
            "pinyin": pinyin,
            "hsk_level": hsk_level,
        }

    return metadata


def generate_projection(characters, vectors):
    """
    Reduces high-dimensional embeddings to 2D using t-SNE.
    """
    number_of_samples = len(vectors)

    if number_of_samples < 3:
        raise ValueError("At least 3 embeddings are required for t-SNE.")

    scaled_vectors = StandardScaler().fit_transform(vectors)

    perplexity = min(30, max(2, number_of_samples // 3))

    projection = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    ).fit_transform(scaled_vectors)

    return projection


def generate_clusters(vectors):
    """
    Assigns each character to a visual cluster.
    """
    number_of_samples = len(vectors)
    number_of_clusters = min(20, max(2, number_of_samples // 50))

    clusters = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init="auto",
    ).fit_predict(vectors)

    return clusters


def main():
    print("Loading embeddings...")
    characters, vectors = load_embeddings(EMBEDDINGS_PATH)

    print(f"Loaded {len(characters)} character embeddings.")

    print("Loading metadata...")
    metadata = load_metadata(METADATA_PATH)

    print("Generating 2D projection...")
    projection = generate_projection(characters, vectors)

    print("Generating clusters...")
    clusters = generate_clusters(vectors)

    output = []

    for index, character in enumerate(characters):
        character_metadata = metadata.get(character, {})

        output.append(
            {
                "character": character,
                "pinyin": character_metadata.get("pinyin"),
                "hsk_level": character_metadata.get("hsk_level"),
                "x": float(projection[index][0]),
                "y": float(projection[index][1]),
                "cluster": int(clusters[index]),
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"Saved projection to: {OUTPUT_PATH}")
    print(f"Total points written: {len(output)}")

if __name__ == "__main__":
    main()