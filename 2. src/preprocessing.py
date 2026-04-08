import re
import pandas as pd

import nltk

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _nltk_sw
    from nltk.tokenize import word_tokenize as _nltk_tokenize
    _NLTK_AVAILABLE = True
    _NLTK_STOPWORDS = set(_nltk_sw.words("english"))
except Exception:
    _NLTK_AVAILABLE = False
    _NLTK_STOPWORDS = set()
    _nltk_tokenize = None


_BUNDLED_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
    "won","wouldn"
}

STOPWORDS = _NLTK_STOPWORDS if _NLTK_STOPWORDS else _BUNDLED_STOPWORDS

# Fields used for BM25 indexing (must match field_weights keys in bm25.py)
INDEXED_FIELDS = ["title", "keywords", "skills", "responsibilities"]

# CSV column -> internal field name
COLUMN_MAP = {
    "Title": "title",
    "Keywords": "keywords",
    "Skills": "skills",
    "Responsibilities": "responsibilities",
}

def clean_text(text: str) -> str:
    # Lowercase and strip punctuation/extra whitespace.
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation -> space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str, remove_stopwords: bool = True) -> list:
    text = clean_text(text)
    if _NLTK_AVAILABLE and _nltk_tokenize:
        try:
            tokens = _nltk_tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()

    tokens = [t for t in tokens if t.isalpha() or t.isdigit()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

def preprocess_document_for_bm25(row) -> dict:
    doc = {}
    for csv_col, field_name in COLUMN_MAP.items():
        raw = row.get(csv_col, "")
        if isinstance(raw, str) and ";" in raw:
            parts = [p.strip() for p in raw.split(";") if p.strip()]
            tokens = []
            for part in parts:
                tokens.extend(tokenise(part))
        else:
            tokens = tokenise(str(raw) if raw else "")
        doc[field_name] = tokens
    return doc


def preprocess_corpus_for_bm25(df) -> list:
    # Apply preprocess_document_for_bm25 to the whole DataFrame.
    return [preprocess_document_for_bm25(row) for _, row in df.iterrows()]


def preprocess_document_for_dense(row) -> str:
    # Combine all relevant fields into one string for sentence-transformer encoding.
    parts = []
    for csv_col in ["Title", "Skills", "Keywords", "Responsibilities"]:
        val = row.get(csv_col, "")
        if isinstance(val, str) and val.strip():
            parts.append(val.replace(";", " ").strip())
    return " ".join(parts)


def preprocess_query(query: str):
    # Preprocess a user query for both retrievers.
    # Returns: (bm25_tokens list, dense_text string)

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    bm25_tokens = tokenise(query, remove_stopwords=True)
    dense_text = clean_text(query)
    return bm25_tokens, dense_text


def load_dataset(path: str):
    # Load and validate the job dataset CSV
    required_columns = ["JobID", "Title", "Skills", "Responsibilities", "Keywords"]
    df = pd.read_csv(path)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    for col in required_columns:
        df[col] = df[col].fillna("").astype(str)
    return df.reset_index(drop=True)
