# 🎬 AI-Powered Movie Recommendation System

> A hybrid AI recommendation engine combining **collaborative filtering**, **semantic vector search (RAG)**, and **LLM-generated explanations** — powered by MovieLens 1M, TMDB 5000, and your choice of OpenAI GPT or Groq LLaMA.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)
![Colab](https://img.shields.io/badge/Google_Colab-compatible-F9AB00?logo=googlecolab&logoColor=white)

---

## ✨ Key Features

| | Feature | Description |
|---|---|---|
| 🎯 | **Smart Personalised Recommendations** | SVD matrix factorisation predicts ratings across the full movie catalogue and surfaces the highest-ranked unseen titles for each user |
| 🔍 | **Semantic Search** | FAISS vector index over sentence-transformer embeddings finds movies by mood, theme, director, or any free-text description |
| 🤖 | **Explainable AI** | An LLM generates concise, user-specific natural-language justifications for every recommendation, grounded in watch history |
| 💬 | **Conversational Interface** | RAG-powered Streamlit chat — ask anything from *"dark psychological thrillers"* to *"movies like Inception"* and get contextualised answers |

---

## 🏗️ Architecture

Three independent, composable layers work together to deliver recommendations:

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — Collaborative Filtering (SVD)                │
│  Predicts per-user ratings for all unrated movies       │
│  → returns ranked movie list                            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2 — Semantic Search / RAG (FAISS)                │
│  Encodes movie metadata as dense vectors;               │
│  retrieves similar movies and builds LLM context        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3 — LLM Explanation (OpenAI / Groq)              │
│  Generates natural-language explanations using          │
│  user history + retrieved context as RAG grounding      │
└─────────────────────────────────────────────────────────┘
```

### Layer Summary

| Layer | Responsibility | Technology |
|---|---|---|
| **1 — Collaborative Filtering** | Learns user & item latent factors; predicts ratings for unseen movies | `scikit-surprise` SVD |
| **2 — Semantic Search / RAG** | Dense vector retrieval over movie metadata; builds context strings for the LLM | `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS `IndexFlatIP` |
| **3 — LLM Explanation** | Produces per-movie explanations and answers conversational queries | OpenAI `gpt-3.5-turbo` or Groq `llama-3.3-70b-versatile` |

---

## 📦 Datasets

### MovieLens 1M
- **1,000,209 ratings** by 6,040 users on 3,706 movies (1–5 stars)
- Downloaded automatically by `data/download_data.py`
- Source: [grouplens.org/datasets/movielens/1m](https://grouplens.org/datasets/movielens/1m/)

### TMDB 5000
- **4,803 movies** with overviews, genres, keywords, cast, and crew
- Requires a free Kaggle account — see setup instructions below
- Source: [kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- An API key from [Groq (free)](https://console.groq.com) **or** [OpenAI](https://platform.openai.com)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/RahulKakani9999/AI-Powered-Netflix-Movie-Recommendation-System.git
cd AI-Powered-Netflix-Movie-Recommendation-System
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure your API key**
```bash
cp .env.example .env
# Open .env and set your key:
#   LLM_PROVIDER=groq
#   GROQ_API_KEY=your-key-here
```

**4. Download the datasets**
```bash
python data/download_data.py
# MovieLens 1M downloads automatically.
# Follow the printed instructions to add the TMDB 5000 CSV files.
```

**5. Preprocess the data**
```bash
python data/preprocess.py
# Outputs: data/processed/ratings.csv, movies.csv, movies_metadata.csv
```

**6. Train the SVD model**
```bash
python models/train_svd.py
# Runs 5-fold CV, trains on full dataset, saves svd_model.pkl + trainset.pkl
```

**7. Build the FAISS index**
```bash
python models/build_faiss_index.py
# Encodes movie metadata with sentence-transformers, saves faiss_index.bin
```

**8. Launch the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Select a User ID in the sidebar and click **Get Recommendations**, or type any movie question in the chat.

---

## 📁 Project Structure

```
AI-Powered-Netflix-Movie-Recommendation-System/
│
├── app.py                        # Streamlit chat interface
├── config.py                     # Centralised paths, model params, API config
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template (copy → .env)
│
├── data/
│   ├── download_data.py          # Downloads MovieLens 1M; guides TMDB setup
│   ├── preprocess.py             # Cleans, merges, and saves processed CSVs
│   ├── movielens/                # Raw MovieLens .dat files (git-ignored)
│   ├── tmdb/                     # Raw TMDB CSV files (git-ignored)
│   └── processed/                # Processed CSVs output (git-ignored)
│
├── models/
│   ├── train_svd.py              # SVD cross-validation + full training
│   ├── build_faiss_index.py      # Encodes metadata → FAISS index
│   └── saved/                    # Trained artefacts: *.pkl, *.bin (git-ignored)
│
├── utils/
│   ├── recommender.py            # SVDRecommender — Layer 1 inference
│   ├── vector_search.py          # VectorSearch — Layer 2 FAISS queries
│   ├── llm_explainer.py          # LLMExplainer — Layer 3 generation
│   └── pipeline.py               # RecommendationPipeline — orchestrator
│
└── evaluation/
    ├── evaluate_model.py         # RMSE + Precision@K automated metrics
    └── human_eval_template.md    # Rubric for human evaluation of LLM output
```

---

## 📊 Evaluation

### Automated Metrics

Run the evaluation script after training:

```bash
python evaluation/evaluate_model.py
```

| Metric | Expected Value | Description |
|---|---|---|
| **RMSE** | ~0.87 | Root Mean Squared Error on 20 % held-out ratings (lower = better) |
| **Precision@10** | ~0.73 | Fraction of top-10 recommendations with actual rating ≥ 4★ |
| **Human Eval** | ≥ 3.5 / 5 | Target score across Relevance, Specificity, Helpfulness, Naturalness |

### Human Evaluation

Open `evaluation/human_eval_template.md` for the full evaluator rubric. Recommended process: collect responses from **3–5 independent evaluators** and report the grand average across all four criteria.

---

## 🔮 Future Work

- **Feedback loop** — capture in-app thumbs-up/down signals and use them to fine-tune SVD user vectors in near-real-time
- **Neural Collaborative Filtering** — replace SVD with a two-tower neural model (e.g., NCF or LightGCN) for richer non-linear user–item interactions
- **Multi-language support** — swap `all-MiniLM-L6-v2` for a multilingual sentence-transformer model to serve non-English content and queries
- **A/B testing framework** — route a percentage of traffic to alternative recommendation strategies and measure click-through and watch-time uplift
- **Personalised RAG** — seed the FAISS query with the user's top-rated genres so vector search is taste-aware, not just text-aware
- **Content-based cold start** — for brand-new users with no history, fall back to content similarity instead of popularity ranking

---

## 📖 Key Terminology

| Term | Definition |
|---|---|
| **Hybrid AI System** | A recommendation approach that combines multiple AI techniques (here: collaborative filtering + semantic search + generative AI) to compensate for each method's individual weaknesses |
| **Matrix Factorisation (SVD)** | Decomposes the user–item rating matrix into low-dimensional latent factor vectors; the dot product of a user vector and an item vector approximates the user's rating for that item |
| **RAG (Retrieval-Augmented Generation)** | A pattern where relevant documents are retrieved from a vector store and prepended to an LLM prompt as grounding context, reducing hallucination and improving answer specificity |
| **Explainable AI (XAI)** | Techniques that make AI decisions interpretable to end users; here the LLM translates opaque SVD scores into human-readable, history-referenced justifications |
| **FAISS IndexFlatIP** | A Facebook AI Similarity Search index that computes exact inner products between query and stored vectors; after L2 normalisation, inner product equals cosine similarity |

---

## 📄 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).  
Free to use, modify, and distribute with attribution.

---

<div align="center">

Built with ❤️ using MovieLens · TMDB · sentence-transformers · FAISS · scikit-surprise · OpenAI / Groq · Streamlit

</div>
