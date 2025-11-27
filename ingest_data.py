import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
# Multilingual model to handle Hindi/English variations
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

def create_vector_db():
    """
    Ingests PDF documents, chunks them, and creates a FAISS vector store.
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH}. Please add PDF files there.")
        return

    # 1. Load Documents
    print("Loading PDF documents from 'data/'...")
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("No documents found! Please add PDFs to the 'data/' folder.")
        return

    # 2. Split Text (Chunking)
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=70
    )
    texts = text_splitter.split_documents(documents)
    print(f"Generated {len(texts)} chunks.")

    # 3. Generate Embeddings
    print("Loading embedding model (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    # 4. Create and Save Vector Store
    print("Creating FAISS vector database...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Success! Vector database saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
