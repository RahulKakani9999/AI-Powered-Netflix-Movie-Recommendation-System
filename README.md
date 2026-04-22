# CineMatch AI — Hybrid Movie Recommendation System

---

## Abstract

CineMatch AI is a three-layer hybrid recommendation system that combines collaborative filtering via Singular Value Decomposition (SVD), retrieval-augmented generation (RAG) using FAISS vector search with sentence-transformers, and large language model (LLM) explanations into a unified inference pipeline. The system generates personalised movie recommendations from the MovieLens 1M dataset, enriches them with structured metadata from TMDB 5000, and produces natural-language explanations grounded in retrieved context. The design directly addresses the EU AI Act's explainability requirements for high-impact AI systems by ensuring every recommendation is accompanied by a traceable, human-readable rationale.

---

## System Architecture

| Layer | Component | Technology | Role |
|-------|-----------|------------|------|
| Layer 1 | Recommendation Engine | SVD via Surprise library | Predicts user ratings using matrix factorisation on MovieLens 1M ratings |
| Layer 2 | Semantic Search (RAG) | FAISS + sentence-transformers | Encodes movie descriptions as dense vectors; retrieves semantically similar films at query time |
| Layer 3 | Explanation Engine | Groq / OpenAI LLM API | Generates natural-language explanations for recommendations and handles open-ended conversational queries |

---

## Datasets

### MovieLens 1M

- **Source:** GroupLens Research (grouplens.org)
- **Size:** 1,000,209 ratings from 6,040 users across 3,883 movies
- **Contents:** User ratings (1-5 scale), movie titles with release years, genre tags
- **Usage:** Training the SVD collaborative filtering model; providing movie-level genre metadata

### TMDB 5000

- **Source:** The Movie Database via Kaggle (`tmdb-movie-metadata`)
- **Size:** 4,803 movies with rich metadata
- **Contents:** Movie overviews, cast and crew, keywords, genres, production details
- **Usage:** Constructing `combined_text` fields for semantic embedding; enriching recommendation cards with director, cast, and overview data

### Merge Strategy

The two datasets are joined on title and year using a five-stage matching pipeline:

1. Exact match on normalised title and year
2. Normalised title with year offset +/- 1
3. Unique normalised title match (no year constraint)
4. Aggressive substitution match (ampersand, Roman numerals, articles) with year offset +/- 1
5. Fuzzy match via `difflib.SequenceMatcher` with similarity threshold >= 0.85 and year offset +/- 2

Final match rate: **31.2%** of MovieLens movies successfully matched to TMDB records.

---

## Tech Stack

| Library / Tool | Version | Role |
|----------------|---------|------|
| Python | >= 3.10 | Runtime |
| scikit-surprise | >= 1.1.3 | SVD matrix factorisation and cross-validation |
| sentence-transformers | >= 2.2.0 | Dense text embeddings (`all-MiniLM-L6-v2`) |
| faiss-cpu | >= 1.7.4 | Approximate nearest-neighbour vector search |
| Groq / OpenAI SDK | >= 0.4.0 / >= 1.0.0 | LLM API for explanation generation |
| Streamlit | >= 1.30.0 | Web interface |
| Pandas | >= 2.0.0 | Data loading, preprocessing, and merging |
| NumPy | >= 1.24.0 | Numerical operations and embedding arrays |

---

## Evaluation Metrics

| Metric | Value | Method |
|--------|-------|--------|
| RMSE | 0.8739 | 5-fold cross-validation on MovieLens 1M |
| Precision@10 | Computed on held-out test split | Threshold rating >= 4.0 considered relevant |
| Human Evaluation | 1-5 scale per criterion | Relevance, Specificity, Helpfulness, Naturalness — target >= 3.5 / 5 |

Human evaluation is conducted using the template at `evaluation/human_eval_template.md`. A minimum of three independent evaluators is recommended before treating scores as representative.

---

## Project Structure

```
AI-Powered-Netflix-Movie-Recommendation-System/
|
|-- app.py                        # Streamlit web interface (CineMatch AI)
|-- config.py                     # Centralised configuration and path management
|-- requirements.txt              # Python dependencies
|-- .env.example                  # Environment variable template
|-- colab_notebook.ipynb          # End-to-end Google Colab walkthrough
|
|-- data/
|   |-- download_data.py          # Downloads MovieLens 1M and TMDB 5000
|   |-- preprocess.py             # Cleans, merges, and builds combined_text fields
|   |-- raw/                      # Raw downloaded datasets (gitignored)
|   |-- processed/                # Cleaned CSVs: movies.csv, ratings.csv, movies_metadata.csv
|
|-- models/
|   |-- train_svd.py              # SVD training with 5-fold cross-validation
|   |-- build_faiss_index.py      # Encodes combined_text and writes FAISS index
|   |-- saved/                    # Persisted artefacts: svd_model.pkl, faiss_index.bin
|
|-- utils/
|   |-- recommender.py            # SVDRecommender class (Layer 1 inference)
|   |-- vector_search.py          # VectorSearch class (Layer 2 RAG retrieval)
|   |-- llm_explainer.py          # LLMExplainer class (Layer 3 generation)
|   |-- pipeline.py               # RecommendationPipeline orchestrating all three layers
|
|-- evaluation/
|   |-- evaluate_model.py         # RMSE and Precision@K computation
|   |-- human_eval_template.md    # Structured rubric for LLM explanation quality
```

---

## Setup and Usage

### Option A: Google Colab (Recommended)

1. Open `colab_notebook.ipynb` in Google Colab.
2. Navigate to **Secrets** (key icon in the left sidebar) and add the following:
   - `GROQ_API_KEY` — obtain from console.groq.com (free tier available)
   - `KAGGLE_USERNAME` and `KAGGLE_KEY` — obtain from kaggle.com/settings
3. Run all cells in order. The notebook handles dataset download, preprocessing, model training, index building, and launches the Streamlit app via `localtunnel`.

### Option B: Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/RahulKakani9999/AI-Powered-Netflix-Movie-Recommendation-System.git
cd AI-Powered-Netflix-Movie-Recommendation-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY or OPENAI_API_KEY

# 4. Download datasets
python data/download_data.py

# 5. Preprocess and merge datasets
python data/preprocess.py

# 6. Train the SVD model
python models/train_svd.py

# 7. Build the FAISS index
python models/build_faiss_index.py

# 8. Launch the interface
streamlit run app.py
```

---

## Interface

The CineMatch AI web interface is built with Streamlit and presents three views:

- **Discover** — Select a user profile and click "Curate My List" to generate six SVD-based recommendations with LLM explanations under the "Director's Notes" section. A conversational chat interface below supports free-form queries about movies, genres, and directors.
- **Watchlist** — Movies saved from the Discover view are stored in session state and displayed in a three-column card grid.
- **History** — Displays the selected user's highest-rated movies from the training set, sorted by rating descending, with up to 30 titles shown.

---

## Future Work

- Implement a feedback loop allowing user ratings of recommendations to trigger incremental SVD retraining
- Replace SVD with a neural collaborative filtering model (e.g., NeuMF or LightGCN) for improved accuracy on sparse interaction matrices
- Add multilingual support using a multilingual sentence-transformer model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`)
- Build an A/B testing framework to compare recommendation strategies by click-through and session engagement metrics
- Extend the RAG context window with structured knowledge graph data (Wikidata film entities)

---

## Key Terminology

**Hybrid AI System** — A recommendation architecture that combines two or more distinct modelling approaches (here: collaborative filtering and content-based retrieval) to compensate for the weaknesses of each individual method.

**Matrix Factorisation (SVD)** — A technique that decomposes a sparse user-item rating matrix into low-dimensional latent factor vectors, enabling prediction of unobserved ratings via dot-product reconstruction.

**Retrieval-Augmented Generation (RAG)** — A pattern in which a language model's response is conditioned on documents retrieved at inference time from an external knowledge base, reducing hallucination and grounding outputs in verifiable content.

**Explainable AI (XAI)** — A class of methods and practices that make AI system outputs interpretable to human stakeholders, enabling audit, accountability, and trust — a requirement under Article 13 of the EU AI Act for high-risk AI systems.

---

## License

MIT License. See `LICENSE` for full terms.
