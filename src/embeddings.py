import os
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings(texts):
    """
    Convert articles into semantic embeddings.
    """

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


def find_similar_articles(
    df,
    embeddings,
    article_index,
    threshold=0.50
):
    """
    Find articles semantically similar to a selected article.
    """

    similarities = cosine_similarity(
        [embeddings[article_index]],
        embeddings
    )[0]

    similar_indices = [
        i
        for i, score in enumerate(similarities)
        if i != article_index and score >= threshold
    ]

    results = []

    for i in similar_indices:
        results.append({
            "article_index": i,
            "title": df.iloc[i]["title"],
            "source": df.iloc[i]["source"],
            "similarity_score": round(
                float(similarities[i]),
                3
            )
        })

    results.sort(
        key=lambda x: x["similarity_score"],
        reverse=True
    )

    return results


if __name__ == "__main__":

    input_file = "data/processed/news_articles_clean.csv"

    df = pd.read_csv(input_file)

    df["clean_text"] = df["clean_text"].fillna("")

    print("Loading embedding model...")

    embeddings = generate_embeddings(
        df["clean_text"].tolist()
    )

    print("\nEmbedding shape:")
    print(embeddings.shape)

    # Save embeddings for later use
    os.makedirs("models", exist_ok=True)

    embedding_path = "models/news_embeddings.npy"

    import numpy as np

    np.save(
        embedding_path,
        embeddings
    )

    print(f"\nEmbeddings saved to: {embedding_path}")

    # Analyze first article
    article_index = 0

    print("\nOriginal Article:")
    print(df.iloc[article_index]["title"])

    similar_articles = find_similar_articles(
        df,
        embeddings,
        article_index
    )

    print("\nSemantically Similar Articles:")

    if not similar_articles:
        print("No similar articles found.")

    for article in similar_articles:
        print(
            f"\nTitle: {article['title']}"
            f"\nSource: {article['source']}"
            f"\nSemantic Similarity: "
            f"{article['similarity_score']}"
        )