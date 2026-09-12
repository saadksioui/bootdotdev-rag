from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from lib.keyword_search import load_file


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movies.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        file_path = Path("cache/movie_embeddings.npy")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        file_path = Path("cache/movie_embeddings.npy")
        if file_path.is_file():
            self.embeddings = np.load(file_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)
    
    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Error: The text shouldn't be empty or filled with whitespaces")
        encode = self.model.encode([text])
        return encode[0]

    def search(self, query, limit):
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores = []
        for index, doc_emb in enumerate(self.embeddings):
            scores.append((cosine_similarity(query_embedding, doc_emb), self.documents[index]))
        sorted_scores = sorted(scores, key=lambda item: item[0], reverse=True)
        return sorted_scores[:limit]


def verify_model():
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_query_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_file("data/movies.json")

    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_query(query, limit):
    semantic_search = SemanticSearch()
    documents = load_file("data/movies.json")
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for index, item in enumerate(results, start=1):
        print(f"{index}. {item[1]['title']} (score: {item[0]})")
        print(item[1]['description'])