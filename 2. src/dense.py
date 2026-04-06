import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "1. data/job_dataset.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the job dataset and create a single text field for dense retrieval.
    """
    df = pd.read_csv(path)

    required_columns = ["JobID", "Title", "Skills", "Responsibilities", "Keywords"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    df["text"] = (
        df["Title"].fillna("").astype(str) + " "
        + df["Skills"].fillna("").astype(str) + " "
        + df["Responsibilities"].fillna("").astype(str) + " "
        + df["Keywords"].fillna("").astype(str)
    ).str.strip()

    return df


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load the sentence-transformer model used for dense retrieval.
    """
    return SentenceTransformer(model_name)


def build_job_embeddings(
    df: pd.DataFrame, model: SentenceTransformer
):
    """
    Encode all jobs into dense embeddings.
    """
    return model.encode(df["text"].tolist(), show_progress_bar=True)


def search_jobs(
    query: str,
    df: pd.DataFrame,
    model: SentenceTransformer,
    job_embeddings,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Search for the most relevant jobs for a given query using cosine similarity.
    Returns a DataFrame with the top-k results.
    """
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, job_embeddings)[0]

    results = df.copy()
    results["SimilarityScore"] = scores
    results = results.sort_values(by="SimilarityScore", ascending=False).head(top_k)

    return results[
        ["JobID", "Title", "Skills", "Keywords", "Responsibilities", "SimilarityScore"]
    ]


def print_results(results: pd.DataFrame) -> None:
    """
    Print search results in a clear format for demo/testing.
    """
    print("\nTop matching jobs:\n")

    for _, row in results.iterrows():
        print("-----")
        print(f"JobID: {row['JobID']}")
        print(f"Title: {row['Title']}")
        print(f"Similarity Score: {row['SimilarityScore']:.4f}")
        print(f"Skills: {row['Skills']}")
        print(f"Keywords: {row['Keywords']}")

        responsibilities_preview = str(row["Responsibilities"])[:200]
        if len(str(row["Responsibilities"])) > 200:
            responsibilities_preview += "..."
        print(f"Responsibilities: {responsibilities_preview}")
    print()


def interactive_search() -> None:
    """
    Run the dense retrieval system in interactive mode.
    """
    print("Loading dataset...")
    df = load_dataset()

    print("Loading embedding model...")
    model = load_embedding_model()

    print("Encoding job descriptions...")
    job_embeddings = build_job_embeddings(df, model)

    print("\nDense retrieval system ready.")
    print("Type a query to search for jobs.")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Enter your job search query: ").strip()

        if query.lower() == "exit":
            print("Exiting search.")
            break

        if not query:
            print("Please enter a valid query.\n")
            continue

        try:
            results = search_jobs(query, df, model, job_embeddings, top_k=TOP_K)
            print_results(results)
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    interactive_search()