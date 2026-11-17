import pandas as pd
import re


EMOTIONAL_WORDS = {
    "anger": [
        "angry", "outrage", "furious", "rage", "attack",
        "blame", "condemn", "destroy", "threat"
    ],
    "fear": [
        "fear", "danger", "dangerous", "crisis", "threat",
        "risk", "warning", "alarming", "scary"
    ],
    "excitement": [
        "amazing", "breakthrough", "exciting", "revolutionary",
        "historic", "huge", "remarkable", "success"
    ],
    "urgency": [
        "urgent", "immediately", "rapidly", "emergency",
        "critical", "must", "now", "warning"
    ],
    "approval": [
        "benefit", "success", "progress", "improve",
        "positive", "advantage", "opportunity"
    ],
    "disapproval": [
        "failure", "problem", "controversy", "concern",
        "reject", "wrong", "bad", "negative"
    ]
}


def detect_emotional_language(text):
    text = str(text).lower()

    words = re.findall(r"\b[a-z]+\b", text)

    total_words = len(words)

    category_counts = {}

    for category, keywords in EMOTIONAL_WORDS.items():
        count = sum(
            1 for word in words
            if word in keywords
        )

        category_counts[category] = count

    total_emotional_words = sum(category_counts.values())

    if total_words > 0:
        emotional_ratio = total_emotional_words / total_words
    else:
        emotional_ratio = 0

    return pd.Series({
        **category_counts,
        "emotional_word_count": total_emotional_words,
        "emotional_ratio": emotional_ratio
    })


if __name__ == "__main__":

    input_file = "data/processed/news_with_sentiment.csv"
    output_file = "data/processed/news_with_emotional_language.csv"

    df = pd.read_csv(input_file)

    df["analysis_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("")
    )

    results = df["analysis_text"].apply(
        detect_emotional_language
    )

    df = pd.concat([df, results], axis=1)

    df.to_csv(output_file, index=False)

    print("Emotional language analysis completed.")

    print("\nSample results:")

    print(
        df[
            [
                "title",
                "source",
                "emotional_word_count",
                "emotional_ratio",
                "anger",
                "fear",
                "excitement",
                "urgency",
                "approval",
                "disapproval"
            ]
        ].head(10).to_string(index=False)
    )

    print(f"\nOutput saved to: {output_file}")