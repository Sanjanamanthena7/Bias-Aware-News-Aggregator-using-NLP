# Bias-Aware News Aggregator Using NLP, Machine Learning and LLMs

## 1. Project Overview

The Bias-Aware News Aggregator is a web application that collects news articles from multiple sources and helps users compare coverage of related topics using Natural Language Processing (NLP), semantic retrieval, sentiment analysis, emotional-language indicators and a locally running Large Language Model (LLM).

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant articles and provide context to the LLM for generating summaries and comparisons.

**Note:** Similarity, sentiment and emotional-language measurements are analytical indicators only. They are not proof that an article or publisher is biased.

## 2. Project Objectives

- Collect news articles from multiple publishers.
- Preprocess and organize article data.
- Generate semantic embeddings.
- Retrieve and compare related articles.
- Analyze sentiment using VADER.
- Identify emotional-language indicators.
- Generate source-grounded summaries using RAG and a local LLM.
- Display results through an interactive Streamlit application.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NewsAPI | News collection |
| Pandas | Data processing |
| NLTK / VADER | Text processing and sentiment analysis |
| Sentence Transformers | Semantic embeddings |
| Scikit-learn | Similarity and clustering |
| NumPy | Embedding array operations |
| Ollama | Local LLM runtime |
| Llama 3.2 3B | Local text generation |
| RAG | Retrieve relevant articles and provide context |
| Streamlit | User interface |

## 4. Project Workflow

1. User enters a news topic or question.
2. NewsAPI retrieves available articles.
3. Article data is preprocessed.
4. Sentence Transformers generates 384-dimensional embeddings.
5. Similarity and clustering identify related articles.
6. VADER estimates sentiment.
7. Predefined emotional keywords are used to calculate emotional-language indicators.
8. RAG retrieves relevant articles for the query.
9. Ollama runs Llama 3.2 locally to generate analysis.
10. Streamlit displays summaries, comparisons, indicators and original article links.

## 5. Main Project Modules

- `src/news_collector.py` – Collects news articles.
- `src/preprocessing.py` – Prepares article data.
- `src/embeddings.py` – Generates semantic embeddings.
- `src/similarity.py` – Calculates article similarity.
- `src/clustering.py` – Groups related articles.
- `src/sentiment_analysis.py` – Performs sentiment analysis.
- `src/emotional_language.py` – Calculates emotional-language indicators.
- `src/article_comparison.py` – Compares related articles.
- `src/retrieval.py` – Retrieves relevant articles.
- `src/llm_generator.py` – Generates LLM-based analysis.
- `app/app.py` – Streamlit application.

## 6. Application Features

- News topic and question input.
- Configurable number of articles to retrieve.
- Source-grounded overview and summaries.
- Comparison of perspectives across news sources.
- Semantic similarity scores.
- Sentiment labels and scores.
- Emotional-language ratios.
- Related-article comparison.
- Article details and original article links.

## 7. Demonstrated Results

The application was tested using the sample query **AI technology and policy**.

The application retrieved three articles from different sources and generated:
- An overview of the topic.
- Source-wise article summaries.
- A comparison of different perspectives.
- Measurable language indicators.
- Related-article comparisons.
- Article details and original links.

The sample results demonstrate how the application helps users explore differences in news coverage.

## 8. Limitations

- Results depend on the articles available through the news API.
- VADER may miss context, sarcasm and nuance.
- Predefined emotional keywords may not identify every emotion.
- Similarity indicates relatedness, not agreement.
- LLM-generated analysis should be checked against original articles.
- These indicators do not independently prove bias.

## 9. Future Enhancements

- Add more news providers.
- Support multilingual news articles.
- Improve article deduplication.
- Evaluate retrieval and sentiment quality using labeled data.
- Improve explainability and source-grounding checks.
- Deploy the application online.

## 10. Conclusion

The Bias-Aware News Aggregator demonstrates the integration of NLP, machine learning, semantic retrieval, RAG and a locally running LLM. It helps users retrieve related news articles, examine language indicators and compare how different publishers present a topic.

## 11. GitHub Repository

[Bias-Aware News Aggregator – GitHub](https://github.com/Sanjanamanthena7/Bias-Aware-News-Aggregator-using-NLP)
