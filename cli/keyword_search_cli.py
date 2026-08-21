import argparse
import json
import string

def remove_stopwords(tokens: list[str]) -> list[str]:
    with open("data/stopwords.txt", "r") as file:
        stopwords = set(file.read().splitlines())
    return [token for token in tokens if token not in stopwords]

def check_common(query: str, title: str) -> bool:
    table = str.maketrans('', '', string.punctuation)
    query_tokens = remove_stopwords([token for token in query.lower().translate(table).split() if token])
    title_tokens = remove_stopwords([token for token in title.lower().translate(table).split() if token])

    if not query_tokens or not title_tokens:
        return False

    query_words = set(query_tokens)
    title_words = set(title_tokens)
    return any(q_tok in t_tok for q_tok in query_words for t_tok in title_words)


def search_movies(query: str) -> None:
    with open("data/movies.json", "r") as file:
        movies = json.load(file)
    table = str.maketrans('', '', string.punctuation)
    results = [
        movie["title"] 
        for k in movies.values() 
        for movie in k 
        if check_common(query, movie["title"])
    ]
    if results:
        print(f"Searching for: {query}")
        for index, movie in enumerate(results, start=1):
            print(f"{index}. {movie}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_movies(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
