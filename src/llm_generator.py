
from ollama import chat
from retrieval import retrieve_articles


def generate_analysis(query, top_k=3):

    # Retrieve relevant articles
    articles = retrieve_articles(query, top_k=top_k)

    context_parts = []

    for i, (_, row) in enumerate(articles.iterrows(), start=1):
        context_parts.append(
            f"""
SOURCE {i}
Title: {row['title']}
Source: {row['source']}
URL: {row['url']}
Similarity: {row['similarity']:.3f}
Sentiment: {row['sentiment']}
Sentiment Score: {row['sentiment_score']:.4f}
Emotional Ratio: {row['emotional_ratio']:.4f}
Description: {row['description']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a neutral news analysis assistant.

User query:
{query}

Retrieved articles:
{context}

Rules:
- Use only the supplied article information.
- Do not invent facts or details.
- Identify sources when describing their articles.
- Do not label any source objectively biased or unbiased.
- Sentiment and emotional ratios are indicators, not proof of bias.
- If details are missing, say so.

Write these sections:
1. OVERVIEW
2. SOURCE-WISE SUMMARY
3. COMPARISON
4. LIMITATIONS

Keep the response concise, neutral, and factual.
Do not calculate or invent numerical indicators; the application displays
the actual values separately.
"""

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.2
        }
    )

    answer = response["message"]["content"]

    return articles, answer


if __name__ == "__main__":

    query = "AI technology and policy"

    print("User Query:")
    print(query)

    print("\nGenerating source-grounded analysis...\n")

    articles, answer = generate_analysis(query, top_k=3)

    print("=" * 70)
    print("LLM ANALYSIS")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    print(
        articles[
            ["title", "source", "url", "similarity"]
        ].to_string(index=False)
    )