from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from lib.keyword_search import load_file
import re
import json


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        all_chunks = []
        chunk_metadata = []
        for movie_idx, doc in enumerate(self.documents):
            text = doc['description']
            if not text.strip():
                continue
            doc_chunks = semantic_chunk(text, 4, 1)
            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(doc_chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        file_path = Path("cache/chunk_embeddings.npy")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as file:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks),
                },
                file,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        file_path = Path("cache/chunk_embeddings.npy")
        metadata_path = Path("cache/chunk_metadata.json")
        if file_path.is_file() and metadata_path.is_file():
            self.chunk_embeddings = np.load(file_path)
            with open("cache/chunk_metadata.json", "r") as file:
                self.chunk_metadata = json.load(file)["chunks"]
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embe = self.generate_embedding(query)
        chunk_scores = []
        for embedding_idx, chunk in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata[embedding_idx]
            similarity_score = cosine_similarity(chunk, query_embe)
            chunk_scores.append(
                {
                    "chunk_idx": metadata['chunk_idx'],
                    "movie_idx": metadata['movie_idx'],
                    "score": similarity_score 
                }
            )
        movies_score = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]

            if movie_idx not in movies_score or score > movies_score[movie_idx]:
                movies_score[movie_idx] = score
        ranked_movies = sorted(movies_score.items(), key=lambda item: item[1], reverse=True)[:limit]
        results = []
        for movie_idx, score in ranked_movies:
            movie = self.documents[movie_idx]
            results.append(
                format_search_result(
                    movie,
                    score,
                    {},
                )
            )
        return results


def format_search_result(movie, score, metadata):
    return {
        "id": movie["id"],
        "title": movie["title"],
        "document": movie["description"][:100],
        "score": score,
        "metadata": metadata or {},
    }



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


def semantic_chunk(text, chunk_size, overlap):
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    
    chunks = []
    if not sentences:
        return chunks
        
    step = chunk_size - overlap
    
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(sentences):
            break
            
    return chunks

def embed_chunks():
    embedding_chunks = ChunkedSemanticSearch()
    documents = load_file("data/movies.json")

    embeddings = embedding_chunks.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(query, limit):
    embedding_chunks = ChunkedSemanticSearch()
    documents = load_file("data/movies.json")

    embedding_chunks.load_or_create_chunk_embeddings(documents)
    results = embedding_chunks.search_chunks(query, limit)
    for i, item in enumerate(results, start=1):
        print(f"\n{i}. {item['title']} (score: {item['score']:.4f})")
        print(f"   {item['document']}...")
    