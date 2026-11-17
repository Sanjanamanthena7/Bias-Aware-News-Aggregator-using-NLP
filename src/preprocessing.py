import os
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


def clean_text(text):
    """
    Clean and normalize article text.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


def preprocess_dataset(input_path, output_path):

    df = pd.read_csv(input_path)

    # Combine title and description
    df["text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("")
    )

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Remove empty articles
    df = df[df["clean_text"].str.strip() != ""]

    # Remove duplicate articles
    df = df.drop_duplicates(subset=["url"])

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Processed articles: {len(df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":

    input_file = "data/raw/news_articles.csv"
    output_file = "data/processed/news_articles_clean.csv"

    preprocess_dataset(
        input_file,
        output_file
    )