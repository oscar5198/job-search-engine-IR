import math
from collections import defaultdict


class BM25Retriever:
    # BM25 for the job search engine


    def __init__(self, documents, field_weights=None, k1=1.5, b=0.75):
        # each document should  be tokenised in preprocessing.py
       

        self.documents = documents
        self.k1 = k1
        self.b = b

        # default field weights based on the report design
        self.field_weights = field_weights or {
            "title": 3.0,
            "keywords": 2.5,
            "skills": 2.0,
            "responsibilities": 1.0
        }

        self.N = len(documents)  # total number of documents

        # inverted index:
        self.inverted_index = defaultdict(dict)

        # document frequencies:
        self.doc_freqs = defaultdict(int)

        # weighted document lengths
        self.doc_lengths = {}

        # average weighted document length
        self.avg_doc_length = 0.0

        # build the index when the class is created
        self.build_index()

    def build_index(self):
        #  builds  inverted index and stores document statistics

        total_length = 0.0

        for doc_id, doc in enumerate(self.documents):
            # stores weighted term counts for one document
            weighted_term_counts = defaultdict(float)

            # stores the weighted length of one document
            weighted_doc_length = 0.0

            # goes through each field in the document
            for field, weight in self.field_weights.items():
                tokens = doc.get(field, [])

                # add weighted field length to total document length
                weighted_doc_length += len(tokens) * weight

                # count each token with the field weight
                for token in tokens:
                    weighted_term_counts[token] += weight

            # savs document length
            self.doc_lengths[doc_id] = weighted_doc_length
            total_length += weighted_doc_length

            # adds terms to inverted index
            for term, weighted_tf in weighted_term_counts.items():
                self.inverted_index[term][doc_id] = weighted_tf

            # update document frequency
            for term in weighted_term_counts:
                self.doc_freqs[term] += 1

        # calculate average document length
        if self.N > 0:
            self.avg_doc_length = total_length / self.N
        else:
            self.avg_doc_length = 0.0

    def compute_idf(self, term):
        # BM25 inverse document frequency


        df = self.doc_freqs.get(term, 0)

        if df == 0:
            return 0.0

        return math.log(((self.N - df + 0.5) / (df + 0.5)) + 1)

    def score(self, query_tokens, doc_id):
        # calculates BM25 score for one document

        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0.0)

        # avoid division by zero
        if self.avg_doc_length == 0:
            return 0.0

        for term in query_tokens:
            # skips term if it is not in the collection
            if term not in self.inverted_index:
                continue

            # gets weighted term frequency for this document
            tf = self.inverted_index[term].get(doc_id, 0.0)

            # if the term is not in this document, skips it
            if tf == 0:
                continue

            idf = self.compute_idf(term)

            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def search(self, query_tokens, top_k=10):
        # scores all documents and returns the top ranked

        results = []

        for doc_id in range(self.N):
            doc_score = self.score(query_tokens, doc_id)

            # only keep documents that scored above 0
            if doc_score > 0:
                results.append((doc_id, doc_score))

        # sort by score from highest to lowest
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def explain_score(self, query_tokens, doc_id):
        # shows how each term contributed

        explanation = []
        total_score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0.0)

        if self.avg_doc_length == 0:
            return {
                "doc_id": doc_id,
                "total_score": 0.0,
                "details": []
            }

        for term in query_tokens:
            if term not in self.inverted_index:
                explanation.append({
                    "term": term,
                    "tf": 0.0,
                    "idf": 0.0,
                    "term_score": 0.0
                })
                continue

            tf = self.inverted_index[term].get(doc_id, 0.0)
            idf = self.compute_idf(term)

            if tf == 0:
                explanation.append({
                    "term": term,
                    "tf": 0.0,
                    "idf": round(idf, 4),
                    "term_score": 0.0
                })
                continue

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )

            term_score = idf * (numerator / denominator)
            total_score += term_score

            explanation.append({
                "term": term,
                "tf": round(tf, 4),
                "idf": round(idf, 4),
                "term_score": round(term_score, 4)
            })

        return {
            "doc_id": doc_id,
            "total_score": round(total_score, 4),
            "details": explanation
        }
