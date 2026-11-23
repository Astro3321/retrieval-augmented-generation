from src.loaders.PDFLoader import PDFLoader
from src.vectorstore import VectorStore
from src.embedding import Embedding
from src.utils.config_loader import load_config 

#Load Parameter Config 
config = load_config('/Workspace/Users/ashishchauhan3301@gmail.com/RAG/params/parameters.yml')

###Ingestion Pipeline 
#Load all PDF Documents
loader = PDFLoader(config)
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
