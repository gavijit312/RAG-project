import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

documents = []

# PDF file path
pdf_path = "Syllabus for AI-ML.pdf"

# Check if file exists
if os.path.exists(pdf_path):
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())
else:
    print(f"File not found: {pdf_path}")

print(f"Loaded {len(documents)} documents")

# Stop if no documents
if len(documents) == 0:
    print("No documents found — please add files to the `data/` folder or update `pdf_path`.")
    sys.exit(1)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# Stop if no chunks
if len(chunks) == 0:
    print("No document chunks found. Please check the input file paths and formats.")
    sys.exit(1)

# Load embedding model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
print("Creating vector store...")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save vector store
vectorstore.save_local("vectorstore")

print("Vector store saved successfully!")