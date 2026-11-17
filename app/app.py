
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# Allow Python to find modules inside src/
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from llm_generator import generate_analysis
from article_comparison import compare_articles


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bias-Aware News Aggregator",
    page_icon="📰",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📰 Bias-Aware News Aggregator")

st.write(
    "Retrieve related news articles and generate "
    "source-grounded analysis using NLP, semantic retrieval, "
    "and a local LLM."
)

st.divider()


# --------------------------------------------------
# Search
# --------------------------------------------------

query = st.text_input(
    "🔎 Enter a news topic or question",
    placeholder="Example: AI technology and policy"
)

top_k = st.slider(
    "Number of articles to retrieve",
    min_value=1,
    max_value=5,
    value=3
)


# --------------------------------------------------
# Analyze button
# --------------------------------------------------

if st.button("🚀 Analyze News", type="primary"):

    if not query.strip():
        st.warning("Please enter a news topic.")
        st.stop()

    with st.spinner(
        "Retrieving articles and generating analysis..."
    ):
        try:
            articles, answer = generate_analysis(
                query,
                top_k=top_k
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    # --------------------------------------------------
    # LLM Analysis
    # --------------------------------------------------

    st.header("🧠 Source-Grounded Analysis")
    st.markdown(answer)
    st.divider()

    # --------------------------------------------------
    # Measurable Indicators
    # --------------------------------------------------

    st.header("📰 Retrieved Articles")
    st.subheader("📊 Measurable Language Indicators")

    indicator_columns = [
        "title",
        "source",
        "similarity",
        "sentiment",
        "sentiment_score",
        "emotional_ratio"
    ]

    missing_columns = [
        column for column in indicator_columns
        if column not in articles.columns
    ]

    if missing_columns:
        st.warning(
            "Some indicator columns are unavailable: "
            + ", ".join(missing_columns)
        )
    else:
        indicator_data = articles[indicator_columns].copy()

        indicator_data.columns = [
            "Article",
            "Source",
            "Semantic Similarity",
            "Sentiment",
            "Sentiment Score",
            "Emotional Ratio"
        ]

        st.dataframe(
            indicator_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Article": st.column_config.TextColumn(
                    "Article",
                    width="large"
                ),
                "Source": st.column_config.TextColumn(
                    "Source",
                    width="medium"
                ),
                "Semantic Similarity": st.column_config.NumberColumn(
                    "Semantic Similarity",
                    format="%.3f"
                ),
                "Sentiment": st.column_config.TextColumn(
                    "Sentiment",
                    width="small"
                ),
                "Sentiment Score": st.column_config.NumberColumn(
                    "Sentiment Score",
                    format="%.4f"
                ),
                "Emotional Ratio": st.column_config.NumberColumn(
                    "Emotional Ratio",
                    format="%.4f"
                )
            }
        )

    st.divider()

    # --------------------------------------------------
    # Related Article Comparison
    # --------------------------------------------------

    st.header("🔎 Related Article Comparison")

    st.write(
        "Compare the most relevant retrieved article with "
        "other articles in the dataset using semantic similarity, "
        "sentiment, and emotional-language indicators."
    )

    try:
        data_file = (
            ROOT_DIR / "data" / "processed"
            / "news_with_emotional_language.csv"
        )
        embedding_file = ROOT_DIR / "models" / "news_embeddings.npy"

        source_df = pd.read_csv(data_file)
        embeddings = np.load(embedding_file)

        if articles.empty:
            st.info("No articles were retrieved for comparison.")
        else:
            selected_title = str(articles.iloc[0]["title"]).strip()

            matching_indices = source_df.index[
                source_df["title"].astype(str).str.strip()
                == selected_title
            ].tolist()

            if not matching_indices:
                st.info(
                    "The first retrieved article could not be matched "
                    "to the comparison dataset. Try another topic."
                )
            else:
                article_index = matching_indices[0]

                st.caption(
                    f"Selected article: {selected_title}"
                )

                comparison = compare_articles(
                    source_df,
                    embeddings,
                    article_index,
                    top_n=5
                )

                if comparison.empty:
                    st.info(
                        "No related articles were found in the dataset."
                    )
                else:
                    display_comparison = comparison[
                        [
                            "title",
                            "source",
                            "semantic_similarity",
                            "sentiment",
                            "sentiment_score",
                            "emotional_word_count",
                            "emotional_ratio"
                        ]
                    ].copy()

                    display_comparison.columns = [
                        "Related Article",
                        "Source",
                        "Semantic Similarity",
                        "Sentiment",
                        "Sentiment Score",
                        "Emotional Word Count",
                        "Emotional Ratio"
                    ]

                    st.dataframe(
                        display_comparison,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Related Article":
                                st.column_config.TextColumn(
                                    width="large"
                                ),
                            "Source":
                                st.column_config.TextColumn(
                                    width="medium"
                                ),
                            "Semantic Similarity":
                                st.column_config.NumberColumn(
                                    format="%.3f"
                                ),
                            "Sentiment Score":
                                st.column_config.NumberColumn(
                                    format="%.4f"
                                ),
                            "Emotional Word Count":
                                st.column_config.NumberColumn(
                                    format="%.0f"
                                ),
                            "Emotional Ratio":
                                st.column_config.NumberColumn(
                                    format="%.4f"
                                )
                        }
                    )

        st.caption(
            "Note: Similarity, sentiment, and emotional-language "
            "measurements are analytical indicators, not proof of bias."
        )

    except Exception as e:
        st.warning(
            "The main analysis completed, but related article "
            f"comparison could not be displayed: {e}"
        )

    st.divider()

    # --------------------------------------------------
    # Individual Article Details
    # --------------------------------------------------

    st.header("📄 Article Details")

    for index, (_, row) in enumerate(articles.iterrows(), start=1):

        with st.expander(
            f"{index}. {row['title']}"
        ):
            st.write(f"**Source:** {row['source']}")

            url = row.get("url", "")

            if pd.notna(url) and str(url).strip():
                st.link_button(
                    "🔗 Read Original Article",
                    str(url)
                )

            if pd.notna(row.get("similarity")):
                st.write(
                    f"**Semantic Similarity:** "
                    f"{row['similarity']:.3f}"
                )

            st.write(f"**Sentiment:** {row['sentiment']}")

            if pd.notna(row.get("sentiment_score")):
                st.write(
                    f"**Sentiment Score:** "
                    f"{row['sentiment_score']:.4f}"
                )

            if pd.notna(row.get("emotional_ratio")):
                st.write(
                    f"**Emotional Ratio:** "
                    f"{row['emotional_ratio']:.4f}"
                )

            st.write("**Description:**")
            st.write(row.get("description", "No description available."))