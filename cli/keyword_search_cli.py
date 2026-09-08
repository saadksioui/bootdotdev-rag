import json
import argparse
import string
import pickle
import os
import math
from nltk.stem import PorterStemmer
from collections import Counter

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        tokens = clean_punctuation_stopwords(text)

        self.term_frequencies[doc_id] = Counter(tokens)
        for token in set(tokens):
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(doc_id)

    def _tokenizer(self, term):
        tokens = clean_punctuation_stopwords(term)
        return tokens

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

    def load(self):
        try:
            with open("cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)

            with open("cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)

            with open("cache/term_frequencies.pkl", "rb") as file:
                self.term_frequencies = pickle.load(file)
            return True
        except FileNotFoundError:
            return None

def build_command(inverted):
    inverted.build()
    inverted.save()

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def stop_words():
    with open("data/stopwords.txt", 'r') as file:
        stopwords = file.read().splitlines()
    return list(filter(lambda x: clean_punctuation(x), stopwords))


def check_common(list1, list2):
    if not list1 or not list2:
        return False

    query_words = set(list1)
    title_words = set(list2)
    return any(q_tok in t_tok for q_tok in query_words for t_tok in title_words)


def clean_punctuation(input_string):
    translator = str.maketrans('', '', string.punctuation)
    cleaned_string = input_string.translate(translator).lower()
    return cleaned_string


def clean_punctuation_stopwords(string):
    clean = clean_punctuation(string).split()
    stop = stop_words()
    tokens = list(filter(lambda x: x not in stop, clean))
    return stemming(tokens)


def load_file(file_path):
    with open(file_path, 'r') as file:
        movies = json.load(file)["movies"]
    return movies


def search_movies(query, inverted):
    if inverted.load() is None:
        print("Files don't exist. Run build first.")
        return
    tokens = clean_punctuation_stopwords(query)
    found_movies = []
    limit = 5
    for token in tokens:
        if token in inverted.index:
            for doc_id in inverted.index[token]:
                found_movies.append(doc_id)
                if len(found_movies) >= limit:
                    break
        if len(found_movies) >= limit:
            break
    for doc_id in found_movies:
        movie = inverted.docmap[doc_id]
        print(f"{movie['title']} ({doc_id})")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index and store in the disk")
    term_freq = subparsers.add_parser("tf", help="Build the inverted index and store in the disk")
    inv_doc_freq = subparsers.add_parser("idf", help="get the freq of a word that are specific to a given dataset")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    tfidf = subparsers.add_parser("tfidf", help="Calculate teh TF-IDF score")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()