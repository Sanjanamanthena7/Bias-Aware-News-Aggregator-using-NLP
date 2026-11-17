
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT_DIR / "data" / "processed" / "news_with_emotional_language.csv"
EMBEDDING_FILE = ROOT_DIR / "models" / "news_embeddings.npy"


def retrieve_articles(query, top_k=3):

    # Validate input
    if not query or not query.strip():
        raise ValueError("Please enter a valid search query.")

    # Load article data and embeddings
    df = pd.read_csv(DATA_FILE)
    embeddings = np.load(EMBEDDING_FILE)

    # Check that article rows match embedding rows
    if len(df) != len(embeddings):
        raise ValueError(
            f"Article count ({len(df)}) does not match "
            f"embedding count ({len(embeddings)})."
        )

    # Load the semantic embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert user query into an embedding
    query_embedding = model.encode([query])

    # Calculate similarity against all articles
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Select the top matching articles
    top_k = max(1, min(int(top_k), len(df)))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:

        article = df.iloc[index]

        results.append({
            "title": article.get("title", ""),
            "source": article.get("source", "Unknown"),
            "url": article.get("url", ""),
            "similarity": round(
                float(similarities[index]), 3
            ),
            "sentiment": article.get("sentiment", "Unknown"),
            "sentiment_score": float(
                article.get("compound", 0.0)
            ),
            "emotional_ratio": float(
                article.get("emotional_ratio", 0.0)
            ),
            "description": article.get("description", "")
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    query = "AI technology and policy"

    print("Search query:")
    print(query)

    print("\nRetrieved Articles:\n")

    results = retrieve_articles(query, top_k=3)

    print(
        results[
            ["title", "source", "url", "similarity"]
        ].to_string(index=False)
    )