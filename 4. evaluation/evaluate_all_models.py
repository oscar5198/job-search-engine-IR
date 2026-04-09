import os
import sys
import pandas as pd
import numpy as np

# ----------------------------
# Ensure local src folder is in Python path
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
SRC_PATH = os.path.join(PROJECT_ROOT, "2. src")
sys.path.insert(0, SRC_PATH)

# ----------------------------
# Local imports
# ----------------------------
from preprocessing import load_dataset, preprocess_corpus_for_bm25, preprocess_query
from bm25 import BM25Retriever
from dense import load_embedding_model, build_job_embeddings, search_jobs_dense
from relevance_judgments import load_qrels
from sklearn.metrics import ndcg_score

# ----------------------------
# Paths to data
# ----------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "1. data", "job_dataset.csv")
QRELS_PATH = os.path.join(PROJECT_ROOT, "4. evaluation", "qrels_pool.csv")

# ----------------------------
# Load dataset
# ----------------------------
df = load_dataset(DATA_PATH)

# Use Title as JobID if missing
if "JobID" not in df.columns:
    print("JobID column missing in dataset, using Title as JobID.")
    df["JobID"] = df["Title"]

# ----------------------------
# Load qrels
# ----------------------------
# Read qrels CSV
qrels_df = pd.read_csv(QRELS_PATH)

# If JobID missing, use Title
if "JobID" not in qrels_df.columns and "Title" in qrels_df.columns:
    print("JobID column missing in qrels, using Title as JobID.")
    qrels_df["JobID"] = qrels_df["Title"]
    # Save temporary CSV to pass to load_qrels
    temp_qrels_path = os.path.join(PROJECT_ROOT, "4. evaluation", "qrels_pool_temp.csv")
    qrels_df.to_csv(temp_qrels_path, index=False)
    QRELS_PATH_TO_USE = temp_qrels_path
else:
    QRELS_PATH_TO_USE = QRELS_PATH

# Load qrels
qrels = load_qrels(QRELS_PATH_TO_USE)

# ----------------------------
# Queries to evaluate
# ----------------------------
queries = {
    "Q1": "data scientist machine learning python",
    "Q2": "frontend developer react javascript ui",
    "Q3": "data engineer sql pipelines cloud"
}

# ----------------------------
# BM25 setup
# ----------------------------
corpus = preprocess_corpus_for_bm25(df)
bm25 = BM25Retriever(corpus)

# ----------------------------
# Dense embedding setup
# ----------------------------
embedding_model = load_embedding_model("all-MiniLM-L6-v2")
job_embeddings = build_job_embeddings(df, embedding_model)

# ----------------------------
# Metrics functions
# ----------------------------
def precision_at_5(retrieved_ids, relevant_ids):
    return sum(1 for r in retrieved_ids[:5] if r in relevant_ids) / 5

def ndcg_at_10(retrieved_ids, relevant_ids):
    rel = [1 if r in relevant_ids else 0 for r in retrieved_ids[:10]]
    return ndcg_score([rel], [rel])

def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, r in enumerate(retrieved_ids, start=1):
        if r in relevant_ids:
            return 1 / rank
    return 0

# ----------------------------
# Hybrid search function
# ----------------------------
def hybrid_search(query, top_k=20, alpha=0.5):
    # BM25 part
    query_tokens, _ = preprocess_query(query)
    bm25_results = bm25.search(query_tokens, top_k=top_k)
    bm25_scores = np.zeros(len(df))
    for idx, score in bm25_results:
        bm25_scores[idx] = score

    # Dense part
    query_embedding = embedding_model.encode([query])[0]
    dense_scores = np.dot(job_embeddings, query_embedding)

    # Combine scores
    hybrid_scores = alpha * dense_scores + (1 - alpha) * bm25_scores
    ranked_idx = np.argsort(hybrid_scores)[::-1]

    return df.iloc[ranked_idx[:top_k]]["JobID"].tolist()

# ----------------------------
# Evaluation loop
# ----------------------------
results = []

for qid, query in queries.items():
    relevant_ids = qrels[(qrels["query_id"] == qid) & (qrels["relevance"] == 1)]["JobID"].tolist()

    # BM25
    query_tokens, _ = preprocess_query(query)
    bm25_res = bm25.search(query_tokens, top_k=20)
    bm25_ids = [df.iloc[idx]["JobID"] for idx, _ in bm25_res]

    # Dense
    dense_res = search_jobs_dense(query, df, embedding_model, job_embeddings, top_k=20)
    dense_ids = dense_res["JobID"].tolist()

    # Hybrid
    hybrid_ids = hybrid_search(query, top_k=20)

    # Store metrics
    for model_name, retrieved_ids in zip(["BM25", "Dense", "Hybrid"], [bm25_ids, dense_ids, hybrid_ids]):
        results.append({
            "Query": qid,
            "Model": model_name,
            "P@5": precision_at_5(retrieved_ids, relevant_ids),
            "nDCG@10": ndcg_at_10(retrieved_ids, relevant_ids),
            "MRR": mean_reciprocal_rank(retrieved_ids, relevant_ids)
        })

# ----------------------------
# Results table
# ----------------------------
df_results = pd.DataFrame(results)
pivot_table = df_results.pivot(index="Query", columns="Model")
print("\n=== Evaluation Results ===\n")
print(pivot_table.round(3))