import argparse
import json
import os
import pickle
import string
from nltk.stem import PorterStemmer


def tokenize_text(text: str) -> list[str]:
    table = str.maketrans("", "", string.punctuation)
    tokens = [
        token
        for token in text.lower().translate(table).split()
        if token
    ]

    stemmer = PorterStemmer()
    with open("data/stopwords.txt", "r") as file:
        stopwords = set(file.read().splitlines())

    return [stemmer.stem(token) for token in tokens if token not in stopwords]


def load_movies():
    with open("data/movies.json", "r") as file:
        movies = json.load(file)

    return [
        movie
        for movie_group in movies.values()
        for movie in movie_group
    ]


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

    def get_documents(self, term):
        return sorted(self.index.get(term, set()))

    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie

            self.__add_document(
                doc_id,
                f"{movie['title']} {movie['description']}"
            )

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)


def search_movies(query: str) -> None:
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Error: index has not been built yet.")
        return

    tokens = tokenize_text(query)
    results = []

    for token in tokens:
        documents = inverted_index.get_documents(token)

        for doc_id in documents:
            if doc_id not in results:
                results.append(doc_id)

            if len(results) == 5:
                break

        if len(results) == 5:
            break

    if results:
        print(f"Searching for: {query}")

        for doc_id in results:
            movie = inverted_index.docmap[doc_id]
            print(f"{movie['title']} ({doc_id})")


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using keywords"
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    subparsers.add_parser(
        "build",
        help="Build and save the inverted index"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_movies(args.query)
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()