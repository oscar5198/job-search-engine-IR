# Precision @k
def precision_at_k(results, relevant_docs, k):
    retrieved = results[:k]
    relevant_retrieved = [doc for doc in retrieved if doc in relevant_docs]
    return len(relevant_retrieved) / k if k > 0 else 0

# Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(results, relevant_docs):
    for i, doc in enumerate(results):
        if doc in relevant_docs:
            return 1 / (i + 1)
    return 0

# Normalized Discounted Cumulative Gain (nDCG@10)
import math

def dcg_at_k(results, relevance_scores, k):
    dcg = 0.0
    for i in range(min(k, len(results))):
        rel = relevance_scores.get(results[i], 0)
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg

def ndcg_at_k(results, relevance_scores, k):
    dcg = dcg_at_k(results, relevance_scores, k)

    # Ideal ranking
    ideal_rels = sorted(relevance_scores.values(), reverse=True)
    ideal_dcg = 0.0
    for i in range(min(k, len(ideal_rels))):
        ideal_dcg += (2**ideal_rels[i] - 1) / math.log2(i + 2)

    return dcg / ideal_dcg if ideal_dcg > 0 else 0

