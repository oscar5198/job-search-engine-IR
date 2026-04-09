import pandas as pd

def load_qrels(path):
    """
    Load relevance judgments from CSV file.
    Assumes columns: query_id, JobID, relevance
    """
    qrels_df = pd.read_csv(path)

    # Make sure column names are correct
    if "JobID" not in qrels_df.columns:
        raise ValueError("CSV must have a 'JobID' column")

    # Convert to list of relevant doc IDs per query
    qrels_dict = {}
    for _, row in qrels_df.iterrows():
        qid = row['query_id']
        doc_id = row['JobID']
        relevance = row['relevance']

        if qid not in qrels_dict:
            qrels_dict[qid] = []
        if relevance == 1:
            qrels_dict[qid].append(doc_id)

    return qrels_df  # or return qrels_dict if you prefer
    