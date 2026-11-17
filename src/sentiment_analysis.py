import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(str(text))

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return pd.Series({
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "compound": compound,
        "sentiment": sentiment
    })


if __name__ == "__main__":

    input_file = "data/processed/news_with_clusters.csv"
    output_file = "data/processed/news_with_sentiment.csv"

    df = pd.read_csv(input_file)

    # Combine title and description for analysis
    df["analysis_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("")
    )

    sentiment_results = df["analysis_text"].apply(analyze_sentiment)

    df = pd.concat(
        [df, sentiment_results],
        axis=1
    )

    df.to_csv(output_file, index=False)

    print("Sentiment analysis completed.")

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nSample results:")
    print(
        df[
            [
                "title",
                "source",
                "sentiment",
                "compound"
            ]
        ].head(10).to_string(index=False)
    )

    print(f"\nOutput saved to: {output_file}")