import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

SOURCE_DIR = "./my_files" 
CHROMA_DIR = "./db"

def build_engine():
    print("Reading files...")
    # Using PyPDFLoader specifically for PDFs to avoid the 'unstructured' errors
    loader = DirectoryLoader(SOURCE_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error loading files: {e}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building search index... (This may take a moment)")
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    print("Done! Engine is ready.")
    return db

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        print(f"Created {SOURCE_DIR}. Drop your files there and run again.")
    else:
        vector_db = build_engine()
        if vector_db:
            while True:
                query = input("\nSearch your files (or type 'exit'): ")
                if query.lower() == 'exit': break
                results = vector_db.similarity_search(query, k=3)
                print("\n--- TOP MATCHES ---")
                for res in results:
                    print(f"\n[Source: {res.metadata.get('source')}]")
                    print(res.page_content[:500] + "...")