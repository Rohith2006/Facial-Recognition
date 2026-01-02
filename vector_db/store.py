import faiss
import numpy as np
import os
import pickle
from utils.logger import get_logger

logger = get_logger()

class VectorStore:
    def __init__(self, dim: int, index_path: str = "vector_db/faiss.index", embeddings_path: str = "vector_db/embeddings.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.embeddings_path = embeddings_path

        # Load index if exists
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatIP(dim)

        # Load embeddings if exists
        if os.path.exists(embeddings_path):
            with open(embeddings_path, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            self.embeddings = []

    def add_embedding(self, embedding: np.ndarray):
        if embedding.shape[0] != self.dim:
            return "Embedding dimension does not match store dimension"
        embedding = embedding / np.linalg.norm(embedding)
        self.index.add(np.expand_dims(embedding, axis=0))
        self.embeddings.append(embedding)

        logger.info(f"Added new embedding. Total embeddings: {len(self.embeddings)}")
        # Save to disk immediately
        self.save()
        return len(self.embeddings) - 1  # Return index of embedding

    def search(self, query_embedding: np.ndarray, top_k: int = 1, threshold: float = 0.3):
        if query_embedding.shape[0] != self.dim:
            return "Query embedding dimension does not match store dimension"
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        D, I = self.index.search(np.expand_dims(query_embedding, axis=0), top_k)
        results = [(idx, score) for idx, score in zip(I[0], D[0]) if score >= threshold]
        logger.info(f"Search results: {results}")
        return results

    def save(self):
        """Persist FAISS index and embeddings."""
        faiss.write_index(self.index, self.index_path)
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(self.embeddings, f)
            logger.info("VectorStore saved to disk.")
