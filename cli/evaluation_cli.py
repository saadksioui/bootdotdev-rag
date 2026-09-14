import argparse
import json
from lib.hybrid_search import HybridSearch
from lib.keyword_search import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
         "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", 'r') as file:
        content = json.load(file)
    documents = load_file("data/movies.json")
    search = HybridSearch(documents)
    results = []
    for item in content['test_cases']:
        result = search.rrf_search(item['query'], 60, limit)
        retrieved_names = [search.semantic_search.document_map[item[0]]['title'] for item in result]
        relevant_retrieved = [name for name in retrieved_names if name in item['relevant_docs']]
        precision = len(relevant_retrieved) / len(retrieved_names) if retrieved_names else 0.0
        total_relevant = len(item['relevant_docs'])
        recall = len(relevant_retrieved) / total_relevant if total_relevant > 0 else 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        results.append(
            {
                "Query": item['query'],
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Retrieved": ", ".join(retrieved_names),
                "Relevant": ", ".join(item['relevant_docs'])
            }
        )
    print(f"k={len(results)}")
    for item in results:
        print(f"- Query: {item['Query']}")
        print(f"    - Precision@{limit}: {item['Precision']:.4f}")
        print(f"    - Recall@{limit}: {item['Recall']:.4f}")
        print(f"    - F1 Score: {item['F1']:.4f}")
        print(f"    - Retrieved: {item['Retrieved']}")
        print(f"    - Relevant: {item['Relevant']}")

if __name__ == "__main__":
    main()