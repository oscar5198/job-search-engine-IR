from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from bm25 import BM25Retriever
from dense import load_embedding_model, build_job_embeddings, search_jobs as dense_search_jobs
from preprocessing import (
    load_dataset,
    preprocess_corpus_for_bm25,
    preprocess_query,
)

def _normalise_scores(score_dict: dict[int, float]) -> dict[int, float]:

    if not score_dict:
        return {}

    values = np.array(list(score_dict.values()), dtype=float)
    min_v, max_v = values.min(), values.max()

    if max_v == min_v:
        return {doc_id: 1.0 for doc_id in score_dict}

    return {
        doc_id: float((score - min_v) / (max_v - min_v))
        for doc_id, score in score_dict.items()
    }


def fuse_scores(
    bm25_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    alpha: float = 0.5,
    top_k: int = 10,
) -> list[tuple[int, float]]:

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # Step 1 – convert to dicts
    bm25_dict = dict(bm25_results)
    dense_dict = dict(dense_results)

    # Step 2 – normalise
    bm25_norm = _normalise_scores(bm25_dict)
    dense_norm = _normalise_scores(dense_dict)

    # Step 3 – union of all candidate doc_ids
    all_doc_ids = set(bm25_norm.keys()) | set(dense_norm.keys())

    # Step 4 – hybrid score
    fused = {}
    for doc_id in all_doc_ids:
        b_score = bm25_norm.get(doc_id, 0.0)
        d_score = dense_norm.get(doc_id, 0.0)
        fused[doc_id] = alpha * b_score + (1 - alpha) * d_score

    # Step 5 – sort and trim
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


class HybridSearchEngine:

    def __init__(
        self,
        data_path: str = "1. data/job_dataset.csv",
        alpha: float = 0.5,
        bm25_top_k: int = 50,
        dense_top_k: int = 50,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.data_path = data_path
        self.alpha = alpha
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.model_name = model_name

        # Set by build_index()
        self.df: pd.DataFrame | None = None
        self.bm25_retriever: BM25Retriever | None = None
        self.dense_model = None
        self.job_embeddings = None
        self._indexed = False


    def build_index(self) -> None:
        print("[1/4] Loading dataset …")
        self.df = load_dataset(self.data_path)

        # Build the combined "text" field required by build_job_embeddings in dense.py
        self.df["text"] = (
            self.df["Title"].fillna("").astype(str) + " "
            + self.df["Skills"].fillna("").astype(str) + " "
            + self.df["Responsibilities"].fillna("").astype(str) + " "
            + self.df["Keywords"].fillna("").astype(str)
        ).str.strip()

        print("[2/4] Preprocessing corpus for BM25 …")
        bm25_corpus = preprocess_corpus_for_bm25(self.df)
        self.bm25_retriever = BM25Retriever(documents=bm25_corpus)

        print("[3/4] Loading sentence-transformer model …")
        self.dense_model = load_embedding_model(self.model_name)

        print("[4/4] Encoding job postings into dense embeddings …")
        self.job_embeddings = build_job_embeddings(self.df, self.dense_model)

        self._indexed = True
        print(f"Index built. {len(self.df)} job postings indexed.\n")

    def search(
        self,
        query: str,
        alpha: float | None = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        effective_alpha = alpha if alpha is not None else self.alpha

        # --- Query preprocessing ---
        bm25_tokens, dense_text = preprocess_query(query)

        # --- Sparse retrieval ---
        bm25_raw = self.bm25_retriever.search(bm25_tokens, top_k=self.bm25_top_k)

        # --- Dense retrieval ---
        dense_df = dense_search_jobs(
            query=dense_text,
            df=self.df,
            model=self.dense_model,
            job_embeddings=self.job_embeddings,
            top_k=self.dense_top_k,
        )
        # convert dense results to (doc_id, score) tuples aligned with DataFrame index
        dense_raw = [
            (int(self.df[self.df["JobID"] == row["JobID"]].index[0]), float(row["SimilarityScore"]))
            for _, row in dense_df.iterrows()
        ]

        # --- Score fusion ---
        fused = fuse_scores(bm25_raw, dense_raw, alpha=effective_alpha, top_k=top_k)

        # --- Build result DataFrame ---
        bm25_score_map = dict(bm25_raw)
        dense_score_map = dict(dense_raw)

        rows = []
        for rank, (doc_id, hybrid_score) in enumerate(fused, start=1):
            job = self.df.iloc[doc_id]
            rows.append({
                "Rank": rank,
                "JobID": job["JobID"],
                "Title": job["Title"],
                "Skills": job["Skills"],
                "Keywords": job["Keywords"],
                "BM25Score": round(bm25_score_map.get(doc_id, 0.0), 4),
                "DenseScore": round(dense_score_map.get(doc_id, 0.0), 4),
                "HybridScore": round(hybrid_score, 4),
            })

        return pd.DataFrame(rows)

    def explain(self, query: str, doc_id: int) -> dict:
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() first.")
        bm25_tokens, _ = preprocess_query(query)
        return self.bm25_retriever.explain_score(bm25_tokens, doc_id)