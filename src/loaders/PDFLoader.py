import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader

class PDFLoader:
    def __init__(self, config): 
        self.pdf_dir = Path(config['paths']['root'], config['paths']['pdf_dir'])
        self.chunk_size = config['chunking']['chunk_size']
        self.chunk_overlap = config['chunking']['chunk_overlap']

    #Read all the PDFs from the Directory
    def process_all_pdfs(self):
        all_documents = []
        pdf_dir = Path(self.pdf_dir)

        #Find all PDF Files recursively
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF Files to process..")

        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()

                #Add Source information to Metadata
                for doc in documents:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_type'] = 'pdf'

                all_documents.extend(documents)
                print(f"Loaded {len(documents)} pages")

            except Exception as e:
                print(f"Error: {e}")

        print(f"\nTotal Documents Loaded: {len(all_documents)}")
        return all_documents
    
    #Text Splitting 
    def split_documents(self, documents):
        """Split Documents into smaller Chunks for better RAG performance"""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        split_docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        #Show example of Chunks 
        if split_docs:
            print(f"\nExample Chunk:")
            print(f"Content: {split_docs[0].page_content[:200]}...")
            print(f"Metadata: {split_docs[0].metadata}")

        return split_docs 