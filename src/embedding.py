import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

class Embedding:
    """Handles document Embedding Generation using Sentence Transformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Intialize Embedding Manager Utility

        Args:
            model_name: HuggingFace Model Name for Sentence Embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Sentence Transformer Model"""
        try:
            print(f"Loading Embedding Model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model Loaded successfully. Embedding Dimension: {self.model.get_sentence_embedding_dimension}")
        except Exception as e:
            print(f"Error Loading Model {self.model_name}: {e}")
            raise 

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate Embeddings for a List of texts

        Args:
            texts: List of text strings to embed
        
        Returns:
            numpy array for embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not Loaded")

        print(f"Generating Embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated Embeddings with shape: {embeddings.shape}")

        return embeddings