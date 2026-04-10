import os
import sys
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Ingest a PDF into a local FAISS vectorstore")
    parser.add_argument(
        "--pdf",
        default=os.path.join(base_dir, "Syllabus for AI-ML.pdf"),
        help="Path to a PDF file to ingest",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(base_dir, "vectorstore"),
        help="Output folder for the FAISS vectorstore",
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    out_dir = args.out

    documents = []

    if os.path.exists(pdf_path):
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    else:
        print(f"File not found: {pdf_path}")

    print(f"Loaded {len(documents)} documents")

    if len(documents) == 0:
        print("No documents found — please update --pdf to point to a real PDF.")
        return 1

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    if len(chunks) == 0:
        print("No document chunks found. Please check the input file path and format.")
        return 1

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(out_dir, exist_ok=True)
    vectorstore.save_local(out_dir)
    print(f"Vector store saved successfully: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())