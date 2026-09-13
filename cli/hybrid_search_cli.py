from lib.hybrid_search import HybridSearch, normilaze
from lib.keyword_search import load_file
from lib.llm_queries import spell, rewrite, expand, individual
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI
import json
import argparse
import os
import time


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


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
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-Ranking method",
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
            if args.enhance == "spell":
                old_query = args.query
                messages = [
                    {
                        "role": "user",
                        "content": spell(args.query),
                    }
                ]
                response = client.chat.completions.create(model="openrouter/free", messages=messages)
                args.query = response.choices[0].message.content
                print(f"Enhanced query ({args.enhance}): '{old_query}' -> '{args.query}'\n")
            elif args.enhance == "rewrite":
                old_query = args.query
                messages = [
                    {
                        "role": "user",
                        "content": rewrite(args.query),
                    }
                ]
                response = client.chat.completions.create(model="openrouter/free", messages=messages)
                args.query = response.choices[0].message.content
                print(f"Enhanced query ({args.enhance}): '{old_query}' -> '{args.query}'\n")
            elif args.enhance == "expand":
                old_query = args.query
                messages = [
                    {
                        "role": "user",
                        "content": expand(args.query),
                    }
                ]
                response = client.chat.completions.create(model="openrouter/free", messages=messages)
                args.query = response.choices[0].message.content
                print(f"Enhanced query ({args.enhance}): '{old_query}' -> '{args.query}'\n")
            if args.rerank_method:
                search_limit = args.limit * 5
            else:
                search_limit = args.limit

            results = hybrid_search.rrf_search(args.query, args.k, search_limit)

            if args.rerank_method == "individual":
                for item in results:
                    doc = hybrid_search.semantic_search.document_map[item[0]]
                    messages = [
                        {
                            "role": "user",
                            "content": individual(args.query, doc),
                        }
                    ]
                    response = client.chat.completions.create(model="openrouter/free", messages=messages)
                    score = response.choices[0].message.content
                    item[1]['rank_score'] = int(score) if score.isdigit() else 0
                results = sorted(results, key=lambda item: item[1].get('rank_score', 0), reverse=True)
                for id, item in enumerate(results, start=1):
                    title = hybrid_search.semantic_search.document_map[item[0]]['title']
                    description = hybrid_search.semantic_search.document_map[item[0]]['description']
                    print(f"{id}. {title}")
                    print(f"Re-rank Score: {item[1].get('rank_score', 0.0):.3f}/10")
                    print(f"RRF Score: {item[1].get('rrf_score', 0.0):.3f}")
                    print(f"BM25: {item[1].get('bm25_rank', 0.0):.3f}, Semantic: {item[1].get('semantic_rank', 0.0):.3f}")
                    print(description)

            elif args.rerank_method == "batch":
                doc_lines = []
                for item in results:
                    doc = hybrid_search.semantic_search.document_map[item[0]]
                    doc_id = item[0]
                    title = doc.get('title', '')
                    desc = doc.get('description', '')
                    doc_lines.append(f"{doc_id}: {title}\n{desc}")
                doc_list_str = "\n\n".join(doc_lines)

                prompt = f"""Rank the movies listed below by relevance to the following search query.\n\nQuery: \"{args.query}\"\n\nMovies:\n{doc_list_str}\n\nReturn the movie IDs in order of relevance, best match first.\n\nYour response must be a raw JSON array of integers.\nDo not wrap the JSON in Markdown. Do not use a ```json code block.\nDo not include any explanatory text.\n\nFor example:\n[75, 12, 34, 2, 1]\n\nRanking:"""

                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(model="openrouter/free", messages=messages)
                content = response.choices[0].message.content.strip()
                try:
                    ranked_ids = json.loads(content)
                except Exception:
                    ranked_ids = []

                rank_map = {int(doc_id): idx + 1 for idx, doc_id in enumerate(ranked_ids) if isinstance(doc_id, int) or (isinstance(doc_id, str) and doc_id.isdigit())}
                default_rank = len(results) + 1
                for item in results:
                    doc_id = int(item[0])
                    item[1]['llm_rank'] = rank_map.get(doc_id, default_rank)

                results = sorted(results, key=lambda item: item[1].get('llm_rank', default_rank))

                print(f"Re-ranking top {args.limit} results using batch method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):\n")
                for id, item in enumerate(results[: args.limit], start=1):
                    doc = hybrid_search.semantic_search.document_map[item[0]]
                    title = doc.get('title', '')
                    description = doc.get('description', '')
                    print(f"{id}. {title}")
                    print(f"   Re-rank Rank: {item[1].get('llm_rank', 0)}")
                    print(f"   RRF Score: {item[1].get('rrf_score', 0.0):.3f}")
                    print(f"   BM25 Rank: {item[1].get('bm25_rank', 0.0):.3f}, Semantic Rank: {item[1].get('semantic_rank', 0.0):.3f}")
                    print(f"   {description}\n")

            elif args.rerank_method == "cross_encoder":
                pairs = []
                for item in results:
                    doc = hybrid_search.semantic_search.document_map[item[0]]
                    document_text = doc.get('document', doc.get('description', ''))
                    pairs.append([args.query, f"{doc.get('title', '')} - {document_text}"])

                try:
                    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                except Exception as exc:
                    if "gpu" in str(exc).lower() or "cuda" in str(exc).lower() or "hardware" in str(exc).lower():
                        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")
                    else:
                        raise

                scores = cross_encoder.predict(pairs)
                for item, score in zip(results, scores):
                    item[1]['cross_encoder_score'] = float(score)

                results = sorted(results, key=lambda item: item[1].get('cross_encoder_score', float('-inf')), reverse=True)

                print(f"Re-ranking top {args.limit} results using cross_encoder method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):\n")
                for id, item in enumerate(results[: args.limit], start=1):
                    doc = hybrid_search.semantic_search.document_map[item[0]]
                    title = doc.get('title', '')
                    description = doc.get('description', '')
                    print(f"{id}. {title}")
                    print(f"   Cross Encoder Score: {item[1].get('cross_encoder_score', 0.0):.3f}")
                    print(f"   RRF Score: {item[1].get('rrf_score', 0.0):.3f}")
                    print(f"   BM25 Rank: {item[1].get('bm25_rank', 0.0):.3f}, Semantic Rank: {item[1].get('semantic_rank', 0.0):.3f}")
                    print(f"   {description}\n")

            else:
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