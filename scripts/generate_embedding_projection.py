import json

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap


EMBEDDINGS_PATH = "embeddings/embeddings.json"
METADATA_PATH = "../data/character_files/General_Standard_Hanzi.xlsx"

UMAP_OUTPUT_PATH = "embeddings/embedding_projection_umap.json"
TSNE_OUTPUT_PATH = "embeddings/embedding_projection_tsne.json"


def load_embeddings(path: str):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    characters = []
    vectors = []

    for character, vector in data.items():
        characters.append(character)
        vectors.append(vector)

    return characters, vectors


def load_metadata(path: str):
    df = pd.read_excel(path)

    metadata = {}

    for _, row in df.iterrows():
        character = row.get("Character")

        if pd.isna(character):
            continue

        character = str(character).strip()

        pinyin = row.get("Pinyin")
        hsk_level = row.get("Hsk Level")

        metadata[character] = {
            "pinyin": None if pd.isna(pinyin) else str(pinyin).strip(),
            "hsk_level": None if pd.isna(hsk_level) else str(int(hsk_level)),
        }

    return metadata


def generate_clusters(vectors):
    number_of_samples = len(vectors)
    number_of_clusters = min(20, max(2, number_of_samples // 50))

    return KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init="auto",
    ).fit_predict(vectors)


def build_output(characters, projection, clusters, metadata):
    output = []

    for index, character in enumerate(characters):
        character_metadata = metadata.get(character, {})

        output.append({
            "character": character,
            "pinyin": character_metadata.get("pinyin"),
            "hsk_level": character_metadata.get("hsk_level"),
            "x": float(projection[index][0]),
            "y": float(projection[index][1]),
            "cluster": int(clusters[index]),
        })

    return output


def save_projection(path: str, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} points to {path}")


def main():
    print("Loading embeddings...")
    characters, vectors = load_embeddings(EMBEDDINGS_PATH)

    print("Loading metadata...")
    metadata = load_metadata(METADATA_PATH)

    print("Scaling vectors...")
    scaled_vectors = StandardScaler().fit_transform(vectors)

    print("Generating clusters...")
    clusters = generate_clusters(scaled_vectors)

    print("Generating UMAP projection...")
    umap_projection = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric="euclidean",
        random_state=42,
    ).fit_transform(scaled_vectors)

    print("Generating t-SNE projection...")
    tsne_projection = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    ).fit_transform(scaled_vectors)

    umap_output = build_output(characters, umap_projection, clusters, metadata)
    tsne_output = build_output(characters, tsne_projection, clusters, metadata)

    save_projection(UMAP_OUTPUT_PATH, umap_output)
    save_projection(TSNE_OUTPUT_PATH, tsne_output)


if __name__ == "__main__":
    main()