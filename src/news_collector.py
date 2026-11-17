
import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import date

# Load environment variables
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

API_KEY = os.getenv("NEWS_API_KEY")

BASE_URL = "https://newsapi.org/v2/everything"


def fetch_news(
    query,
    page_size=50,
    from_date="2026-10-03",
    to_date="2026-11-01"
):
    """Fetch English news articles within a date range."""

    if not API_KEY:
        raise ValueError(
            "NEWS_API_KEY is not set in the .env file."
        )

    # Validate dates
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date)

    if start > end:
        raise ValueError("from_date cannot be after to_date.")

    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100.")

    params = {
        "q": query,
        "from": from_date,
        "to": f"{to_date}T23:59:59",
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": API_KEY
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(
            data.get("message", "News API request failed.")
        )

    articles = []

    for article in data.get("articles", []):

        title = article.get("title")

        # Skip incomplete or removed articles
        if not title or title == "[Removed]":
            continue

        articles.append({
            "source": article.get("source", {}).get("name"),
            "author": article.get("author"),
            "title": title,
            "description": article.get("description"),
            "url": article.get("url"),
            "published_at": article.get("publishedAt"),
            "content": article.get("content")
        })

    df = pd.DataFrame(articles)

    if not df.empty:
        # Remove duplicate articles using their URLs
        df = df.drop_duplicates(
            subset=["url"],
            keep="first"
        )

        # Ensure newest articles appear first
        df["published_at"] = pd.to_datetime(
            df["published_at"],
            errors="coerce",
            utc=True
        )

        df = df.sort_values(
            "published_at",
            ascending=False
        ).reset_index(drop=True)

    return df


if __name__ == "__main__":

    search_query = "artificial intelligence"

    df = fetch_news(
        search_query,
        page_size=50,
        from_date="2026-10-03",
to_date="2026-11-01"
    )

    output_dir = ROOT_DIR / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "news_articles.csv"

    df.to_csv(output_path, index=False)

    print("\nNEWS COLLECTION COMPLETED")
    print("-" * 45)
    print(f"Search topic: {search_query}")
    print(f"Date range: October 3 to November 1, 2026")
    print(f"Articles collected: {len(df)}")

    if not df.empty:
        print("\nLatest articles:")
        print(
            df[["title", "source", "published_at"]]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("No articles found for this query and date range.")

    print(f"\nSaved to: {output_path}")
