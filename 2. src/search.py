import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_this_dir, "src")
for _path in [_this_dir, _src_dir]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from hybrid import HybridSearchEngine


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "1. data", "job_dataset.csv")

DEFAULT_ALPHA = 0.5

DEFAULT_TOP_K = 10


def print_results(results, query: str) -> None:
    """Pretty-print ranked search results to stdout."""
    print(f"\n{'='*60}")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    if results.empty:
        print("No results found.")
        return

    for _, row in results.iterrows():
        print(f"\n  Rank #{int(row['Rank'])}")
        print(f"  JobID    : {row['JobID']}")
        print(f"  Title    : {row['Title']}")
        print(f"  Skills   : {str(row['Skills'])[:120]}")
        print(f"  Keywords : {str(row['Keywords'])[:100]}")
        print(f"  Scores   → BM25: {row['BM25Score']:.4f} | "
              f"Dense: {row['DenseScore']:.4f} | "
              f"Hybrid: {row['HybridScore']:.4f}")
        print(f"  {'-'*56}")

    print()

def run_interactive(engine: HybridSearchEngine) -> None:
    print("\nHybrid Job Search Engine — Interactive Mode")
    print(f"Current alpha (BM25 weight): {engine.alpha}")
    print("Commands: 'alpha <0-1>' to change weight | 'exit' to quit\n")

    last_query = ""

    while True:
        try:
            user_input = input("Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        # Change alpha on the fly
        if user_input.lower().startswith("alpha "):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    new_alpha = float(parts[1])
                    if 0.0 <= new_alpha <= 1.0:
                        engine.alpha = new_alpha
                        print(f"  Alpha updated to {new_alpha} "
                              f"(BM25 weight={new_alpha}, dense weight={1-new_alpha})")
                    else:
                        print("  alpha must be between 0.0 and 1.0")
                except ValueError:
                    print("  Invalid alpha value. Usage: alpha 0.6")
            continue

        # BM25 term explanation
        if user_input.lower().startswith("explain "):
            parts = user_input.split()
            if len(parts) == 2 and last_query:
                try:
                    doc_id = int(parts[1])
                    explanation = engine.explain(last_query, doc_id)
                    print(f"\n  BM25 explanation for doc_id={doc_id}, query='{last_query}'")
                    print(f"  Total BM25 score: {explanation['total_score']}")
                    for detail in explanation["details"]:
                        print(f"    term='{detail['term']}' | tf={detail['tf']} "
                              f"| idf={detail['idf']} | contribution={detail['term_score']}")
                    print()
                except ValueError:
                    print("  Usage: explain <doc_id>  (integer index)")
            else:
                print("  Run a search first, then: explain <doc_id>")
            continue

        # Standard search
        last_query = user_input
        try:
            results = engine.search(user_input, top_k=DEFAULT_TOP_K)
            print_results(results, user_input)
        except Exception as exc:
            print(f"  Error during search: {exc}")


def main() -> None:
    # Allow overriding data path via environment variable
    data_path = os.environ.get("JOB_DATA_PATH", DATA_PATH)
    alpha = float(os.environ.get("HYBRID_ALPHA", DEFAULT_ALPHA))

    print(f"Data path : {data_path}")
    print(f"Alpha     : {alpha}  (BM25 weight)")

    engine = HybridSearchEngine(
        data_path=data_path,
        alpha=alpha,
        bm25_top_k=50,
        dense_top_k=50,
        model_name="all-MiniLM-L6-v2",
    )

    engine.build_index()
    run_interactive(engine)


if __name__ == "__main__":
    main()
