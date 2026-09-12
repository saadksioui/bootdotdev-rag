import argparse
from lib.semantic_search import verify_model, verify_embeddings, embed_query_text, search_query



def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "verify", help="Verify the semantic search model"
    )
    subparsers.add_parser(
        "verify_embeddings", help="Movies Document Embeddings"
    )
    search_parser = subparsers.add_parser(
        "search", help="Search for movies"
    )
    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate the embeddings from a text"
    )
    chunk_parser = subparsers.add_parser(
        "chunk", help="Split a large text into a smaller chunks"
    )

    chunk_parser.add_argument(
        "text", type=str, help="The text that will be splitted"
    )
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="The number of each chunk"
    )
    search_parser.add_argument(
        "query", type=str, help="The query to use for search the movies"
    )
    search_parser.add_argument(
        "--limit", type=int, default=5, help="The number of movies to be retrieved"
    )
    embed_query_parser.add_argument(
        "text", type=str, help="Text to generate the embeddings"
    )
    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()
        case "embed_query":
            embed_query_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            search_query(args.query, args.limit)
        case "chunk":
            words = args.text.split()
            chunks = [" ".join(words[i:i + args.chunk_size]) for i in range(0, len(words), args.chunk_size)]
            print(f"Chunking {len(args.text)} characters")
            for index, chunk in enumerate(chunks, start=1):
                print(f"{index}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()