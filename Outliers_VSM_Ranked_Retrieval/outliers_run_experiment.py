import sys
import tarfile
import time
import re
from pathlib import Path
import pandas as pd
import pyterrier as pt

if not pt.java.started():
    pt.java.init()

print(f"[+] PyTerrier initialized successfully (Version {pt.__version__}).")

from src.outliers_preprocess import parse_documents, preprocess_text, load_stemmed_stopwords, PorterStemmer
#from .outliers_porter import PorterStemmer

DATA_DIR = Path("./data/Cran")
TAR_PATH = Path("./data/cran.tar.gz")
STOPWORDS_PATH = Path("./data/stopwords.txt")
INDEX_PATH = str(Path("./outliers_index").resolve())

if TAR_PATH.exists() and not DATA_DIR.exists():
    print("[+] Extracting cran.tar.gz archive...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(DATA_DIR)

DOCS_PATH = DATA_DIR / "cran.all.1400"
QUERIES_PATH = DATA_DIR / "cran.qry"
QRELS_PATH = DATA_DIR / "cranqrel"

def prepare_documents():
    print("[+] Parsing and preprocessing documents...")
    stemmer = PorterStemmer()
    stemmed_stopwords = load_stemmed_stopwords(STOPWORDS_PATH, stemmer)
    documents = parse_documents(DOCS_PATH)

    processed_documents = []
    for doc_id, text in documents:
        tokens = preprocess_text(text, stemmer, stemmed_stopwords)
        processed_documents.append({
            "docno": str(doc_id),
            "text": " ".join(tokens)
        })
    return pd.DataFrame(processed_documents), stemmer, stemmed_stopwords

def parse_queries(file_path):
    queries = []
    with open(file_path, "r", encoding="ascii", errors="ignore") as f:
        lines = f.readlines()

    current_query = []
    in_query = False

    for line in lines:
        line = line.rstrip("\n")
        if re.match(r"\.I\s+\d+", line):
            if current_query:
                queries.append({
                    "qid": str(len(queries) + 1),
                    "query": " ".join(current_query).strip()
                })
            current_query = []
            in_query = False
        elif line.strip() == ".W":
            in_query = True
        elif in_query:
            current_query.append(line.strip())

    if current_query:
        queries.append({
            "qid": str(len(queries) + 1),
            "query": " ".join(current_query).strip()
        })
    return pd.DataFrame(queries)

def load_qrels(qrel_path):
    if not Path(qrel_path).exists():
        print(f"Qrels file not found: {qrel_path}")
        return None

    qrels = pd.read_csv(
        qrel_path,
        sep=r"\s+",
        header=None,
        names=["qid", "docno", "label"]
    )

    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    qrels["label"] = qrels["label"].astype(int)

    return qrels

def index_exists(index_path):
    index_path = Path(index_path)

    if not index_path.exists():
        return False

    return any(index_path.iterdir())

def main():
    docs_df, stemmer, stemmed_stopwords = prepare_documents()
    print(f"[+] Document loading complete: {len(docs_df)} records.")

    raw_queries = parse_queries(QUERIES_PATH)
    processed_queries = []
    for _, row in raw_queries.iterrows():
        tokens = preprocess_text(row["query"], stemmer, stemmed_stopwords)
        processed_queries.append({
            "qid": str(row["qid"]),
            "query": " ".join(tokens)
        })
    queries_df = pd.DataFrame(processed_queries)
    print(f"[+] Queries processing complete: {len(queries_df)} queries.")

    print("[+] Checking existing index...")

    if index_exists(INDEX_PATH):
        print("[+] Existing index found. Skipping indexing.")

        index = pt.IndexFactory.of(INDEX_PATH)

        index_time_file = Path(__file__).resolve().parent / "index_time.txt"

        if index_time_file.exists():
            with open(index_time_file, "r") as f:
                index_time = float(f.read().strip())

            print(f"[+] Previous indexing time: {index_time:.2f} seconds")
        else:
            print("[!] Existing index found, but indexing time was not saved.")
            index_time = None

    else:
        print("[+] Index not found. Building PyTerrier Index...")

        indexer = pt.index.IterDictIndexer(
            INDEX_PATH,
            meta=["docno"],
            text_attrs=["text"],
            overwrite=True
        )

        start_time = time.perf_counter()

        indexref = indexer.index(
            docs_df.to_dict("records")
        )

        index_time = time.perf_counter() - start_time

        print(f"[+] Indexing finished in {index_time:.2f} seconds.")

        index_time_file = Path(__file__).resolve().parent / "index_time.txt"

        with open(index_time_file, "w") as f:
            f.write(str(index_time))

        index = pt.IndexFactory.of(indexref)

    tfidf = pt.terrier.Retriever(index, wmodel="TF_IDF")
    rm3_optimal = pt.rewrite.RM3(index, fb_docs=30, fb_terms=15)
    prf_pipeline = tfidf >> rm3_optimal >> tfidf

    qrels_df = load_qrels(QRELS_PATH)
    if qrels_df is None:
        print(
            "[!] Cannot perform evaluation "
            "without qrels."
        )

        return

    print("\n" + "="*50)
    print(" RUNNING TF-IDF + RM3 RETRIEVAL")
    print("="*50)

    prf_results = prf_pipeline.transform(queries_df)
    
    evaluation = pt.Evaluate(
        prf_results,
        qrels_df,
        metrics=[
            "map",
            "recip_rank",
            "P.5",
            "P.10",
            "ndcg_cut.10"
        ]
    )

    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    print("=" * 60)

    print(
        f"MAP       : "
        f"{evaluation['map']:.6f}"
    )

    print(
        f"MRR       : "
        f"{evaluation['recip_rank']:.6f}"
    )

    print(
        f"P@5      : "
        f"{evaluation['P.5']:.6f}"
    )

    print(
        f"P@10     : "
        f"{evaluation['P.10']:.6f}"
    )

    print(
        f"nDCG@10  : "
        f"{evaluation['ndcg_cut.10']:.6f}"
    )

    print("\n" + "-" * 60)

    print("\n[+] Generating retrieval results...")
    pt.io.write_results(prf_results, "results.txt", format="trec")
    print("[+] Process finished! Output saved to 'results.txt'.")

if __name__ == "__main__":
    main()