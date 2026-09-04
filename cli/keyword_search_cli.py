import json
import argparse
import string


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


def load_file(file_path):
    with open(file_path, 'r') as file:
        movies = json.load(file)["movies"]
    return movies


def search_movies(query):
    movies = load_file("data/movies.json")
    search_cleaned = clean_punctuation(query).split()
    found_movies = []
    for movie in movies:
        title_cleaned = clean_punctuation(movie['title']).split()
        if check_common(search_cleaned, title_cleaned):
            found_movies.append(movie)
    for i, movie in enumerate(found_movies, start=1):
        print(f"{i}: {movie['title']}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_movies(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()