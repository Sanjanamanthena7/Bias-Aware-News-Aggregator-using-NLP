
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[1]


def compare_articles(df, embeddings, article_index, top_n=3,
                     similarity_threshold=0.50):

    if len(df) != len(embeddings):
        raise ValueError(
            "The number of articles must match the number of embeddings."
        )

    if not 0 <= article_index < len(df):
        raise IndexError("Invalid article index.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    similarities = cosine_similarity(
        embeddings[article_index].reshape(1, -1),
        embeddings
    )[0]

    # Exclude the selected article itself
    similarities[article_index] = -1

    # Prefer articles meeting the threshold
    similar_indices = np.where(
        similarities >= similarity_threshold
    )[0]

    # If none meet the threshold, use the most similar articles
    if len(similar_indices) == 0:
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = similar_indices[
            similarities[similar_indices] >= 0
        ]

    # Sort by similarity, highest first
    similar_indices = similar_indices[
        np.argsort(similarities[similar_indices])[::-1]
    ]

    top_indices = similar_indices[:top_n]
    comparison_rows = []

    for index in top_indices:
        row = df.iloc[index]

        comparison_rows.append({
            "article_index": index,
            "title": row.get("title", "Unknown"),
            "source": row.get("source", "Unknown"),
            "semantic_similarity": round(
                float(similarities[index]), 3
            ),
            "sentiment": row.get("sentiment", "Unknown"),
            "sentiment_score": row.get("compound", np.nan),
            "emotional_word_count": row.get(
                "emotional_word_count", np.nan
            ),
            "emotional_ratio": row.get(
                "emotional_ratio", np.nan
            )
        })

    return pd.DataFrame(comparison_rows)


if __name__ == "__main__":

    data_file = (
        ROOT_DIR / "data" / "processed"
        / "news_with_emotional_language.csv"
    )

    embedding_file = ROOT_DIR / "models" / "news_embeddings.npy"

    df = pd.read_csv(data_file)
    embeddings = np.load(embedding_file)

    print("Articles:", len(df))

    article_index = 0

    print("\nSelected Article:")
    print(df.iloc[article_index]["title"])

    comparison = compare_articles(
        df,
        embeddings,
        article_index,
        top_n=5
    )

    print("\nRelated Articles:\n")

    if comparison.empty:
        print("No comparable articles were found.")
    else:
        print(comparison.to_string(index=False))

    output_file = (
        ROOT_DIR / "data" / "processed"
        / "article_comparison.csv"
    )

    comparison.to_csv(output_file, index=False)

    print(f"\nComparison saved to: {output_file}")