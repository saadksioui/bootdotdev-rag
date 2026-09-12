import argparse
import pickle
import os
import math
from collections import Counter
from lib.keyword_search import (
    clean_punctuation_stopwords,
    load_file,
    BM25_B,
    BM25_K1,
    build_command,
    search_movies,
    bm25_idf_command,
    bm25_tf_command
)


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join("cache/", "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = clean_punctuation_stopwords(text)

        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(doc_id)

    def _tokenizer(self, term):
        tokens = clean_punctuation_stopwords(term)
        return tokens

    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)
 
    def get_documents(self, term):
        doc_ids = self.index.get(term, [])
        return sorted(doc_ids)

    def get_tf(self, doc_id, term):
        tokens = self._tokenizer(term)
        if len(tokens) != 1:
            raise ValueError("Term must contain exactly one searchable token")
        token = tokens[0]
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(token, 0)

    def build(self):
        movies = load_file("data/movies.json")
        for m in movies:
            doc_id = m["id"]
            self.docmap[doc_id] = m
            self.__add_document(doc_id, f"{m['title']} {m['description']}")

    def save(self):
        if not os.path.exists("cache"):
            os.mkdir("cache")
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)

            with open("cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)

            with open("cache/term_frequencies.pkl", "rb") as file:
                self.term_frequencies = pickle.load(file)

            with open("cache/doc_lengths.pkl", "rb") as file:
                self.doc_lengths = pickle.load(file)
            return True
        except FileNotFoundError:
            return None

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        basic_tf = self.get_tf(doc_id, term)
        bm25_tf = (basic_tf * (k1 + 1)) / (basic_tf + k1 * length_norm)
        return bm25_tf

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokens = self._tokenizer(query)
        scores = {}

        for doc_id in self.docmap:
            total_score = 0.0

            for token in tokens:
                total_score += self.bm25(doc_id, token)

            scores[doc_id] = total_score

        sorted_scores = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return sorted_scores[:limit]


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