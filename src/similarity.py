import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(input_path):

    # Load processed dataset
    df = pd.read_csv(input_path)

    # Replace missing values
    df["clean_text"] = df["clean_text"].fillna("")

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return df, similarity_matrix


def find_similar_articles(df, similarity_matrix, article_index, threshold=0.3):

    similarities = similarity_matrix[article_index]

    similar_indices = [
        i for i, score in enumerate(similarities)
        if i != article_index and score >= threshold
    ]

    results = []

    for i in similar_indices:
        results.append({
            "article_index": i,
            "title": df.iloc[i]["title"],
            "source": df.iloc[i]["source"],
            "similarity_score": round(
                similarities[i],
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

    df, similarity_matrix = calculate_similarity(
        input_file
    )

    print("\nNumber of articles:", len(df))

    print("\nSimilarity Matrix Shape:")
    print(similarity_matrix.shape)

    # Analyze the first article
    article_index = 0

    print("\nOriginal Article:")
    print(df.iloc[article_index]["title"])

    similar_articles = find_similar_articles(
        df,
        similarity_matrix,
        article_index
    )

    print("\nSimilar Articles:")

    if not similar_articles:
        print("No similar articles found.")

    for article in similar_articles:
        print(
            f"\nTitle: {article['title']}"
            f"\nSource: {article['source']}"
            f"\nSimilarity: {article['similarity_score']}"
        )