# Bias-Aware News Aggregator Using NLP, Machine Learning and LLMs

## 1. Project Overview

The **Bias-Aware News Aggregator** is an AI-based news analysis application developed using Python, Natural Language Processing (NLP), Machine Learning and Large Language Models (LLMs).

The main objective is to collect news articles from multiple sources and analyze how different publishers present the same or related topics.

The application uses semantic similarity, clustering, sentiment analysis and emotional language analysis to help users explore different perspectives. It also uses Retrieval-Augmented Generation (RAG) with a locally running LLM to generate article summaries and comparisons.

**Note:** This project provides indicators that can help users examine potentially biased or emotionally worded news. It does not prove that an article or publisher is biased.

## 2. Project Objectives

- Collect news articles from multiple sources using NewsAPI.
- Preprocess and organize the collected news data.
- Identify semantically similar news articles.
- Group related articles using Machine Learning.
- Analyze the sentiment of news articles.
- Identify predefined emotional language indicators.
- Retrieve relevant articles based on a user's search query.
- Generate summaries and comparisons using a local LLM and RAG.
- Display the analysis through an interactive Streamlit application.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NewsAPI | Collecting news articles |
| Pandas | Data manipulation and processing |
| NumPy | Numerical arrays and embedding storage |
| NLTK | Text processing and sentiment analysis |
| VADER | Positive, negative and neutral sentiment analysis |
| Scikit-learn | Clustering and similarity-related operations |
| Sentence Transformers | Generating semantic text embeddings |
| all-MiniLM-L6-v2 | Text embedding model |
| Ollama | Running the LLM locally |
| Llama 3.2 3B | Generating summaries and comparisons |
| RAG | Retrieving relevant articles and providing context to the LLM |
| Streamlit | Building the interactive user interface |

## 4. Project Workflow

### Step 1: News Collection
NewsAPI is used to collect news articles from multiple sources based on a search topic. Article details such as title, source, description, publication date and URL are stored.

### Step 2: Data Preprocessing
The collected news text is cleaned and prepared for further analysis. Pandas and NLTK are used for data handling and text preprocessing.

### Step 3: Text Embeddings
Sentence Transformers converts article text into numerical vectors called embeddings. The project uses the `all-MiniLM-L6-v2` model, which generates 384-dimensional embeddings.

### Step 4: Similarity Analysis
Cosine similarity is used to identify articles with similar meanings, even when they contain different words.

### Step 5: News Clustering
K-Means clustering groups related news articles based on their numerical representations.

### Step 6: Sentiment Analysis
NLTK's VADER sentiment analyzer calculates sentiment scores and classifies article language as positive, negative or neutral.

### Step 7: Emotional Language Analysis
A predefined keyword-based approach identifies words associated with categories such as anger, fear, excitement, urgency, approval and disapproval. Emotional word counts and ratios are calculated.

### Step 8: Article Comparison
Related articles are compared using semantic similarity, sentiment scores and emotional language indicators.

### Step 9: Retrieval-Augmented Generation
The user's query is converted into an embedding. Relevant articles are retrieved using cosine similarity. The retrieved article content is passed as context to Llama 3.2 through Ollama, which generates an overview, source-wise summaries and comparisons.

### Step 10: Streamlit Application
The final analysis is displayed through a Streamlit interface. Users can search for a topic, select the number of articles to retrieve, view generated analysis, examine indicators, compare related articles and access original news links.

## 5. Project Structure

```text
bias-aware-news-aggregator/
│
├── app/
│   └── app.py
│
├── src/
│   ├── news_collector.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── similarity.py
│   ├── clustering.py
│   ├── sentiment_analysis.py
│   ├── emotional_language.py
│   ├── article_comparison.py
│   ├── retrieval.py
│   └── llm_generator.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 6. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- Git
- A NewsAPI account and API key
- Ollama installed on your computer

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/bias-aware-news-aggregator.git
cd bias-aware-news-aggregator
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

```bash
python -m venv bias
```

Activate it on Windows PowerShell:

```powershell
.\bias\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the NewsAPI key

Create a `.env` file in the root folder and add:

```env
NEWS_API_KEY=your_actual_newsapi_key
```

Use your own API key. Do not upload this `.env` file to GitHub.

### Step 5: Install and run the local LLM

Install Ollama from its official website:

https://ollama.com/

Download the model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before using the LLM features.

## 7. Running the Project

Run the Python scripts from the root project folder in the following order:

```bash
python src/news_collector.py
python src/preprocessing.py
python src/embeddings.py
python src/clustering.py
python src/sentiment_analysis.py
python src/emotional_language.py
python src/article_comparison.py
```

After generating the required processed data and embeddings, start the Streamlit application:

```bash
streamlit run app/app.py
```

Open the local URL displayed in the terminal to use the application.

## 8. Application Features

- Topic-based news analysis.
- Retrieval of semantically relevant articles.
- AI-generated news overview.
- Source-wise article summaries.
- Comparison of related articles.
- Positive, negative and neutral sentiment indicators.
- Emotional language indicators.
- Original article links.
- Interactive Streamlit interface.

## 9. Limitations

- News collection depends on NewsAPI availability and subscription limits.
- Sentiment scores may not capture the full context of an article.
- Emotional language analysis uses predefined keywords and may not understand sarcasm or context.
- Semantic similarity does not guarantee that two articles report exactly the same event.
- LLM-generated summaries may contain errors and should be checked against original sources.
- The application provides bias-related indicators, not a definitive or validated bias classification.

## 10. Future Enhancements

- Support for additional news sources.
- Multilingual news analysis.
- Advanced bias-detection models trained on labelled datasets.
- Improved emotion and contextual analysis.
- More detailed visual comparisons between news sources.
- Better evaluation of retrieval and generated summaries.
- Online deployment of the application.

## 11. Conclusion

The Bias-Aware News Aggregator combines NLP, Machine Learning, semantic search and Large Language Models to analyze news articles from multiple sources. It helps users explore related news, understand language indicators and compare different perspectives through an interactive application.

**Developed using Python, NLP, Machine Learning, RAG, Ollama, Llama 3.2 and Streamlit.**