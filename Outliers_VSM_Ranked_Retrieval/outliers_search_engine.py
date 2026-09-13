import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import pyterrier as pt

if not pt.java.started():
    pt.java.init()

from src.outliers_preprocess import parse_documents, preprocess_text, load_stemmed_stopwords, PorterStemmer
PROJECT_DIR = Path(__file__).resolve().parent

def load_index_time():
    index_time_file = PROJECT_DIR / "index_time.txt"

    if not index_time_file.exists():
        print("[-] Index time file not found.")
        print("[-] Please run 'python outliers_run_experiment.py' first.")
        sys.exit(1)

    with open(index_time_file, "r") as f:
        return float(f.read().strip())

def load_system():
    stemmer = PorterStemmer()
    stopwords_path = PROJECT_DIR/"data"/ "stopwords.txt"
    stemmed_stopwords = load_stemmed_stopwords(stopwords_path, stemmer)
    
    index_path = PROJECT_DIR / "outliers_index"
    if not Path(index_path).exists():
        print("[-] Index not found! Please run 'python run_experiment.py' first.")
        sys.exit(1)
        
    index = pt.IndexFactory.of(str(index_path))
    
    docs_path = PROJECT_DIR / "data" / "Cran" / "cran.all.1400"
    raw_docs_dict = dict(parse_documents(docs_path))
    
    return stemmer, stemmed_stopwords, index, raw_docs_dict

def query_system(query_text, top_k=5):
    stemmer, stemmed_stopwords, index, raw_docs = load_system()
    index_time = load_index_time()
    
    tokens = preprocess_text(query_text, stemmer, stemmed_stopwords)
    processed_query = " ".join(tokens)
    
    if not processed_query.strip():
        print("[-] Query contained no valid terms after preprocessing.")
        return

    q_df = pd.DataFrame([{"qid": "1", "query": processed_query}])

    tfidf = pt.terrier.Retriever(index, wmodel="TF_IDF")
    rm3 = pt.rewrite.RM3(index, fb_docs=30, fb_terms=15)
    pipeline = tfidf >> rm3 >> tfidf

    search_start = time.perf_counter()
    res = pipeline.transform(q_df)
    search_time = time.perf_counter() - search_start

    
    print("\n" + "="*70)
    print(f" Raw Query       : {query_text}")
    print(f" Processed Query : {processed_query}")
    print(f"Indexing Time   : {index_time:.6f} seconds")
    print(f"Search Time     : {search_time:.6f} seconds")
    print("="*70)

    if res.empty:
        print("[-] No matching documents found.")
        return

    results_to_show = res.head(top_k)
    for _, row in results_to_show.iterrows():
        doc_id = int(row['docno'])
        raw_text = raw_docs.get(doc_id, "Text not found.")
        snippet = " ".join(raw_text.split())[:200]
        
        print(f"Rank {int(row['rank']) + 1:2d} | DocID: {row['docno']:>4} | Score: {row['score']:.4f}")
        print(f"Snippet: {snippet}...\n" + "-"*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cranfield Ad-Hoc Search Engine")
    parser.add_argument("--query", "-q", type=str, required=True, help="Input search string")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of documents to display (default: 5)")
    args = parser.parse_args()

    query_system(args.query, args.top_k)