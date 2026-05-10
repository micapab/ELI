import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Document Search", page_icon="🔍", layout="wide")
st.title("🔍 Intelligent Document Search Engine")
st.markdown("Search across all your PDFs and Word documents using AI.")

SOURCE_DIR = "./my_files"
CHROMA_DIR = "./db"

# Ensure the upload directory exists
if not os.path.exists(SOURCE_DIR):
    os.makedirs(SOURCE_DIR)

# --- SIDEBAR: UPLOAD & INDEXING ---
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDFs or Word Docs", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(SOURCE_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} files to folder!")

    if st.button("🚀 Re-build Search Index"):
        with st.spinner("Reading documents and building 'brain'..."):
            # Load PDFs
            pdf_loader = DirectoryLoader(SOURCE_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
            # Load Word Docs
            word_loader = DirectoryLoader(SOURCE_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader)
            
            docs = pdf_loader.load() + word_loader.load()
            
            if not docs:
                st.error("No documents found in the folder!")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                split_docs = text_splitter.split_documents(docs)
                
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_db = Chroma.from_documents(
                    split_docs, 
                    embeddings, 
                    persist_directory=CHROMA_DIR
                )
                st.success("Indexing complete! You can now search.")

# --- MAIN AREA: SEARCH ---
query = st.text_input("💬 Enter your question or keywords:")

if query:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the existing database
    if os.path.exists(CHROMA_DIR):
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        results = db.similarity_search(query, k=3)
        
        st.subheader("💡 Top Results Found:")
        
        for i, res in enumerate(results):
            source_path = res.metadata.get('source', 'Unknown')
            file_name = os.path.basename(source_path)
            
            # Create a nice box for each result
            with st.container():
                st.markdown(f"### Result {i+1} from `{file_name}`")
                st.info(res.page_content)
                
                # Add the Download Button for the source file
                if os.path.exists(source_path):
                    with open(source_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download {file_name}",
                            data=f,
                            file_name=file_name,
                            key=f"btn_{i}" # Unique key for Streamlit
                        )
                st.divider()
    else:
        st.warning("The search index hasn't been built yet. Please upload files and click 'Re-build Search Index' in the sidebar.")