import argparse
import math
from lib.keyword_search import (
    BM25_B,
    BM25_K1,
    build_command,
    search_movies,
    bm25_idf_command,
    bm25_tf_command,
    InvertedIndex
)




def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index and store in the disk")
    term_freq = subparsers.add_parser("tf", help="Build the inverted index and store in the disk")
    inv_doc_freq = subparsers.add_parser("idf", help="get the freq of a word that are specific to a given dataset")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    tfidf = subparsers.add_parser("tfidf", help="Calculate teh TF-IDF score")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )

    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    inv_doc_freq.add_argument("term", type=str, help="Term")
    term_freq.add_argument("doc_id", type=int, help="Document ID")
    term_freq.add_argument("term", type=str, help="Term")
    tfidf.add_argument("doc_id", type=int, help="Document ID")
    tfidf.add_argument("term", type=str, help="Term")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    inverted = InvertedIndex()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_movies(args.query, inverted)
        case "build":
            build_command(inverted)
        case "tf":
            freq = inverted.get_tf(args.doc_id, args.term)
            print(freq)
        case "idf":
            if inverted.load() is None:
                print("Files don't exist. Run build first.")
                return
            tokens = inverted._tokenizer(args.term)
            total_doc_count = len(inverted.docmap)
            term_match_doc_count = 0
            for token in tokens:
                term_match_doc_count += len(inverted.get_documents(token)) 
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            if inverted.load() is None:
                print("Files don't exist. Run build first.")
                return
            tokens = inverted._tokenizer(args.term)
            total_doc_count = len(inverted.docmap)
            term_match_doc_count = 0
            for token in tokens:
                term_match_doc_count += len(inverted.get_documents(token)) 
            tf = inverted.get_tf(args.doc_id, args.term)
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term, inverted)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, inverted, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            if inverted.load() is None:
                print("Files don't exist. Run build first.")
                return
            scores = inverted.bm25_search(args.query, 5)
            for index, score in enumerate(scores, start=1):
                print(f"{index}: ({score[0]}) {inverted.docmap[score[0]]['title']} - Score: {score[1]:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()