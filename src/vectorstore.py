import chromadb
import os
import numpy as np
import uuid
from typing import List, Dict, Any, Tuple
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """Manages Document Embeddings in Chroma DB"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        """
        Intialize the Vector Store

        Args:
            collection_name: Name of the ChromaDB Collection
            persist_directory: Directory to persist the Vector Store
        """

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._intialize_store()

    def _intialize_store(self):
        """Intialize ChromaDB Client and Collection"""
        try:
            #Create Persistent ChromaDB Client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            #Get or Create Collection
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {'description': "PDF document embeddings for RAG"}
            )

            print(f"Vector Store intialized. Collection: {self.collection_name}")
            print(f"Existing documents in the Collection: {self.collection.count()}")
        
        except Exception as e:
            print(f"Error Intializing the Vector Store: {e}")
            raise 

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the Vector Store

        Args:
            documents: List of LangChain Documents
            embeddings: Corresponding Embeddings for the documents
        """

        if len(documents) != len(embeddings):
            raise ValueError("Number of Documents must match the number of embeddings")

        print(f"Adding {len(documents)} documents to the Vector Store...")
        
        #Prepare Data for Chroma DB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            #Generate Unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #Prepare Metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            #Documents Content
            documents_text.append(doc.page_content)

            #Embedding
            embeddings_list.append(embedding.tolist())

        #Add to Collection
        try:
            self.collection.add(
                ids = ids,
                embeddings = embeddings_list,
                metadatas = metadatas,
                documents = documents_text
            )

            print(f"Successfully Loaded {len(documents)} documents to the Vector Store")
            print(f"Total Documents in the Collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding Documents to the Vector Store: {e}")
            raise
