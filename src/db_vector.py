import chromadb
from typing import List, Dict, Any

class VectorManager:
    """Handles semantic text processing and nearest-neighbor vector space lookups."""
    def __init__(self, path: str = "data/vector_store"):
        # Local embedded vector database initialization
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name="product_knowledge")

    def index_product_notes(self, product_id: str, text_description: str):
        """Generates vectors and indexes text content automatically."""
        self.collection.add(
            documents=[text_description],
            ids=[product_id]
        )

    def search_similar_products(self, query: str, top_n: int = 1) -> List[str]:
        """Performs a mathematical cosine similarity calculation across embeddings."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n
        )
        return results["documents"][0] if results["documents"] else []
