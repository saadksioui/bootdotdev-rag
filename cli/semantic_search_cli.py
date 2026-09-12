import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings



def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser(
        "verify", help="Verify the semantic search model"
    )
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Movies Document Embeddings"
    )
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate the embeddings from a text"
    )

    embed_text_parser.add_argument(
        "text", type=str, help="Text to generate the embeddings"
    )
    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()