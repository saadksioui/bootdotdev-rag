import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        
        if self.texts:
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        else:
            self.text_embeddings = []

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)
        
        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            dot_product = np.dot(image_embedding, text_embedding)
            norm_a = np.linalg.norm(image_embedding)
            norm_b = np.linalg.norm(text_embedding)
            
            if norm_a == 0 or norm_b == 0:
                score = 0.0
            else:
                score = dot_product / (norm_a * norm_b)
                
            doc = self.documents[i]
            results.append({
                "id": doc.get("id"),
                "title": doc.get("title", "Unknown Title"),
                "description": doc.get("description", ""),
                "similarity": float(score)
            })
            
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]

def verify_image_embedding(image_path: str):
    search = MultimodalSearch([])
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str):
    with open("data/movies.json", "r", encoding="utf-8") as f:
        documents = json.load(f)['movies']
        
    search = MultimodalSearch(documents)
    return search.search_with_image(image_path)