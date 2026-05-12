import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

# App Config
st.set_page_config(page_title="My Private Search Engine", layout="wide")
st.title("📂 File Search Engine")

SOURCE_DIR = "./my_files"
CHROMA_DIR = "./db"

# Ensure directories exist
if not os.path.exists(SOURCE_DIR):
    os.makedirs(SOURCE_DIR)

# Sidebar for uploading files
with st.sidebar:
    st.header("Upload Center")
    uploaded_files = st.file_uploader("Add more files to the ZIP", accept_multiple_files=True)
    if st.button("Index Files"):
        with st.spinner("Processing documents..."):
            # (Your existing logic to load and embed files goes here)
            st.success("Database Updated!")

# Main Search Area
query = st.text_input("Ask a question about your documents:")

if query:
    # Initialize the same 'brain' you used in the script
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    results = db.similarity_search(query, k=3)
    
    st.subheader("Top Matches:")
    for res in results:
        with st.expander(f"Source: {res.metadata.get('source')}"):
            st.write(res.page_content)