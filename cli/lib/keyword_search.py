from nltk.stem import PorterStemmer
from collections import Counter
import json
import string
import pickle
import os
import math


BM25_K1 = 1.5
BM25_B = 0.75


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
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(term, 0)

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

def build_command(inverted):
    inverted.build()
    inverted.save()

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def stop_words():
    with open("data/stopwords.txt", 'r') as file:
        return {
            clean_punctuation(line)
            for line in file.read().splitlines()
            if clean_punctuation(line)
        }


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
    tokens = [token for token in clean if token not in stop]
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


def bm25_idf_command(term, inverted):
    if inverted.load() is None:
        print("Files don't exist. Run build first.")
        return
    tokens = inverted._tokenizer(term)
    return inverted.get_bm25_idf(tokens[0])


def bm25_tf_command(doc_id, term, inverted, k1=BM25_K1, b=BM25_B):
    if inverted.load() is None:
        print("Files don't exist. Run build first.")
        return

    tokens = inverted._tokenizer(term)
    return inverted.get_bm25_tf(doc_id, tokens[0], k1)
