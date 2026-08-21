import argparse
import json
import string

def search_movies(query: str) -> None:
    with open("data/movies.json", "r") as file:
        movies = json.load(file)
    table = str.maketrans('', '', string.punctuation)
    results = [
        movie["title"] 
        for k in movies.values() 
        for movie in k 
        if query.lower().translate(table) in movie["title"].lower().translate(table)
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
