import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def cluster_articles(embeddings, n_clusters=4):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(embeddings)

    return labels


if __name__ == "__main__":

    data_file = "data/processed/news_articles_clean.csv"
    embedding_file = "models/news_embeddings.npy"

    df = pd.read_csv(data_file)
    embeddings = np.load(embedding_file)

    print("Articles:", len(df))
    print("Embedding shape:", embeddings.shape)

    labels = cluster_articles(
        embeddings,
        n_clusters=4
    )

    df["cluster"] = labels

    print("\nCluster assignments:\n")

    print(
        df[["title", "source", "cluster"]]
        .sort_values("cluster")
        .to_string(index=False)
    )

    output_file = "data/processed/news_with_clusters.csv"

    df.to_csv(output_file, index=False)

    print(f"\nClustered dataset saved to: {output_file}")

    print("\nCluster summary:")

    print(
        df["cluster"]
        .value_counts()
        .sort_index()
    )