# job-search-engine-IR
Hybrid Sparse–Dense Job Search Engine (BM25 + Embeddings)

---

## Team Members
- Nikhil Malige
- Pravinesh Gowrypalan
- Skarlin Salazar
- Oscar Gallegos

---

# Hybrid Job Search Engine (Information Retrieval Project)
This project implements a hybrid sparse–dense search engine for job retrieval, developed as part of the ECS736P Information Retrieval module.

The system retrieves and ranks job postings based on user queries by combining:
- **Lexical matching (BM25F)**
- **Semantic similarity (Transformer embeddings)**
- **Hybrid score fusion**

---

## Hybrid Search Engine Features
- Search using free-text queries (e.g., "python developer with REST API")
- Combines:
  - BM25 (exact keyword matching)
  - Dense embeddings (semantic understanding)
- Hybrid ranking using weighted score fusion
- Evaluation with:
  - nDCG@10 (primary metric)
  - Precision@k
  - Mean Reciprocal Rank (MRR)
- Queries include:
  - Keyword-based
  - Descriptive
  - Semantically varied

---

## System Architecture

  User Query
  
       ↓ 
       
BM25 + Dense Retrieval

       ↓
       
  Score Fusion
  
       ↓     

Ranked Job Results

---

## Project Structure

- data/ # Dataset (job postings)
- src/ # Core retrieval system
- notebooks/ # Experiments & testing
- evaluation/ # Metrics and evaluation
- demo/ # Demo scripts
- requirements.txt
- README.md

---

## Installation

1. Clone repository:

      - git clone https://github.com/oscar5198/job-search-engine-IR.git

      - cd job-search-engine-ir

2. Install Dependencies:
   
      - pip install -r requirements.txt

---

## Usage
Run the search engine:

 - python src/search.py

---

## Technologies Used
- Python
- rank-bm25
- sentence-transformers
- FAISS
- NumPy / Pandas
- scikit-learn
- NLTK
