from src.loaders.PDFLoader import PDFLoader
from src.vectorstore import VectorStore
from src.embedding import Embedding

###Ingestion Pipeline 
#Load all PDF Documents
data_src_path = "/Workspace/Users/ashishchauhan3301@gmail.com/RAG/data/" 
loader = PDFLoader(data_src_path)
all_pdf_documents = loader.process_all_pdfs()
chunks = loader.split_documents(all_pdf_documents)

#Load Embedding Utility
embedding_manager = Embedding()

#Load Vector Store
vectorstore = VectorStore()

#Convert Text to Embeddings
texts = [doc.page_content for doc in chunks]

#Generate the Embeddings
embeddings = embedding_manager.generate_embeddings(texts)

#Store in Vector Store
vectorstore.add_documents(chunks, embeddings)
