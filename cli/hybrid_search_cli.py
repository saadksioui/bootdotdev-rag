import argparse
from lib.hybrid_search import HybridSearch, normilaze
from lib.keyword_search import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    normalize_parser = subparsers.add_parser(
        'normalize', help="Normalize the scores"
    )
    weighted_parser = subparsers.add_parser(
        "weighted-search", 
        help="Run a hybrid search using an alpha weight."
    )
    rrf_parser = subparsers.add_parser(
        "rrf-search", 
        help="Perform a hybrid search using Reciprocal Rank Fusion (RRF)"
    )


    rrf_parser.add_argument(
        "query", 
        type=str, 
        help="The search query text"
    )
    rrf_parser.add_argument(
        "-k", 
        type=int, 
        default=60, 
        help="Number of top results to return (default: 60)"
    )
    rrf_parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="The limit of search result (Default: 0.5)"
    )
    weighted_parser.add_argument(
        "query", 
        type=str, 
        help="The search query string."
    )
    weighted_parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.5,
        help="Weighting factor: 1.0 is pure dense, 0.0 is pure sparse. (Default: 0.5)"
    )
    weighted_parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="The limit of search result (Default: 0.5)"
    )
    normalize_parser.add_argument(
        'nums', nargs='*', type=float
    )
    args = parser.parse_args()
    documents = load_file("data/movies.json")
    hybrid_search = HybridSearch(documents)
    match args.command:
        case 'normalize':
            results = normilaze(args.nums)
            for item in results:
                print(f"* {item:.4f}")
        case 'weighted-search':
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            for id, item in enumerate(results, start=1):
                title = hybrid_search.semantic_search.document_map[item[0]]['title']
                description = hybrid_search.semantic_search.document_map[item[0]]['description']
                print(f"{id}. {title}")
                print(f"Hybrid Score: {item[1].get('hybrid_score', 0.0):.3f}")
                print(f"BM25: {item[1].get('bm25_score', 0.0):.3f}, Semantic: {item[1].get('semantic_score', 0.0):.3f}")
                print(description)
        case 'rrf-search':
            results = hybrid_search.rrf_search(args.query, args.k, args.limit)
            for id, item in enumerate(results, start=1):
                title = hybrid_search.semantic_search.document_map[item[0]]['title']
                description = hybrid_search.semantic_search.document_map[item[0]]['description']
                print(f"{id}. {title}")
                print(f"RRF Score: {item[1].get('rrf_score', 0.0):.3f}")
                print(f"BM25: {item[1].get('bm25_rank', 0.0):.3f}, Semantic: {item[1].get('semantic_rank', 0.0):.3f}")
                print(description)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()