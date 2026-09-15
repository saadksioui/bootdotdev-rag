import argparse
from lib.keyword_search import load_file
from lib.hybrid_search import HybridSearch
from dotenv import load_dotenv
from openai import OpenAI
import os
import json


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform summarize (search + generate answer)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="The limit of search result (Default: 5)"
    )
    
    citations_parser = subparsers.add_parser(
        "citations", help="Perform citations (search + generate answer)"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="The limit of search result (Default: 5)"
    )
    
    question_parser = subparsers.add_parser(
        "question", help="Answer a question conversationally based on search results"
    )
    question_parser.add_argument("question", type=str, help="The question to answer")
    question_parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="The limit of search result (Default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            documents = load_file("data/movies.json")
            search = HybridSearch(documents)
            results = search.rrf_search(query, 5)
            docs_str = json.dumps(results, indent=2)
            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
                Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                Provide a comprehensive answer that addresses the user's query.

                Query: {query}

                Documents:
                {docs_str}

                Answer:"""
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model="openrouter/free", messages=messages)
            content = response.choices[0].message.content.strip()
            print("Search Results:")
            for res in results:
                print(f"- {search.semantic_search.document_map[res[0]].get('title', 'Unknown Title')}")
                
            print("\nRAG Response:")
            print(content)
            
        case "summarize":
            query = args.query
            documents = load_file("data/movies.json")
            search = HybridSearch(documents)
            results = search.rrf_search(query, 5)
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

                This should be tailored to Webflyx users. Webflyx is a movie streaming service.

                Query: {query}

                Search results:
                {results}

                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model="openrouter/free", messages=messages)
            content = response.choices[0].message.content.strip()
            print("Search Results:")
            for res in results:
                print(f"- {search.semantic_search.document_map[res[0]].get('title', 'Unknown Title')}")
                
            print("\nLLM Response:")
            print(content)
            
        case "citations":
            query = args.query
            documents = load_file("data/movies.json")
            search = HybridSearch(documents)
            results = search.rrf_search(query, 60, args.limit)
            formatted_docs = []
            for i, res in enumerate(results, start=1):
                doc_data = search.semantic_search.document_map[res[0]]
                formatted_docs.append(f"[{i}] {json.dumps(doc_data)}")
                
            docs_str = "\n".join(formatted_docs)
            prompt = f"""Answer the query below and give information based on the provided documents.

                The answer should be tailored to users of Webflyx, a movie streaming service.
                If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

                Query: {query}

                Documents:
                {docs_str}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources in the format [1], [2], etc. when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the provided documents, say "I don't have enough information"
                - Be direct and informative

                Answer:"""
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model="openrouter/free", messages=messages)
            content = response.choices[0].message.content.strip()
            print("Search Results:")
            for res in results:
                print(f"- {search.semantic_search.document_map[res[0]].get('title', 'Unknown Title')}")
                
            print("\nLLM Response:")
            print(content)
            
        case "question":
            question_query = args.question
            documents = load_file("data/movies.json")
            search = HybridSearch(documents)
            
            results = search.rrf_search(question_query, 60, args.limit)
            
            docs_list = [search.semantic_search.document_map[res[0]] for res in results]
            context = json.dumps(docs_list)
            
            prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question_query}

Documents:{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model="openrouter/free", messages=messages)
            content = response.choices[0].message.content.strip()
            
            print("Search Results:")
            for res in results:
                print(f"  - {search.semantic_search.document_map[res[0]].get('title', 'Unknown Title')}")
                
            print("\nAnswer:")
            print(content)
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()