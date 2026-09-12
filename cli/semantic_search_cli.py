import argparse
import re
from lib.semantic_search import (
    verify_model,
    verify_embeddings,
    embed_query_text,
    search_query,
    semantic_chunk,
    embed_chunks,
    search_chunked
)


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
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split a large text into a smaller chunks using semantic chunking"
    )
    subparsers.add_parser(
        "embed_chunks", help="Chunks data/movies.json using semantic chunking and turn those chunks into embeddings"
    )
    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search in the embedding chunks"
    )

    search_chunked_parser.add_argument(
        "query", type=str, help="The query used to search"
    )
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="The number of each chunk"
    )
    semantic_chunk_parser.add_argument(
        "text", type=str, help="The text that will be splitted"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="The number of each chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="The number of overlap in each chunk"
    )
    chunk_parser.add_argument(
        "text", type=str, help="The text that will be splitted"
    )
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="The number of each chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="The number of overlap in each chunk"
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
            chunks = []
            for i in range(0, len(words), args.chunk_size):
                if i - args.overlap > 0:
                    chunks.append(" ".join(words[i-args.overlap:i + args.chunk_size]))
                else:
                    chunks.append(" ".join(words[i:i + args.chunk_size]))
            
            print(f"Chunking {len(args.text)} characters")
            for index, chunk in enumerate(chunks, start=1):
                print(f"{index}. {chunk}")
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for index, chunk in enumerate(chunks, start=1):
                print(f"{index}. {chunk}")
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()