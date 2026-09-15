import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.multimodal_search import verify_image_embedding, image_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", 
        help="Generate an embedding for an image and print its shape"
    )
    verify_parser.add_argument(
        "image_path", 
        type=str, 
        help="Path to the image file"
    )

    image_search_parser = subparsers.add_parser(
        "image_search", 
        help="Search for movies using an image query"
    )
    image_search_parser.add_argument(
        "image_path", 
        type=str, 
        help="Path to the image file"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
            
        case "image_search":
            results = image_search_command(args.image_path)
            for i, res in enumerate(results, 1):
                title = res['title']
                sim = res['similarity']
                desc = res['description']
                
                if len(desc) > 100:
                    desc_str = desc[:100] + "..."
                else:
                    desc_str = desc
                    
                print(f"{i}. {title} (similarity: {sim:.3f})")
                print(f"   {desc_str}")
                if i < len(results):
                    print()
                    
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()