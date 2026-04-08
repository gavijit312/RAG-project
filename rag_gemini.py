import os
import json
import requests
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_vectorstore(path: str = "vectorstore") -> FAISS:
    """Load a local FAISS vectorstore created by `ingest.py`."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.load_local(path, embeddings)
    return store


def build_prompt(docs: List[str], query: str) -> str:
    """Construct a simple RAG prompt combining retrieved docs and the user query."""
    context = "\n\n---\n\n".join(docs)
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question. "
        "If the answer is not in the context, say you don't know and avoid hallucination.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    )
    return prompt


def call_gemini_via_rest(prompt: str, model: str = None, api_key: str = None) -> str:
    """
    Call Google Generative Language REST endpoint (works with PaLM / Gen AI models).

    This function uses the public REST endpoint and an API key. Set the environment
    variable `GEMINI_API_KEY` to your API key and optionally `GEMINI_MODEL` to the
    model name (default: text-bison-001). If you have a different Gemini/Vertex
    setup, replace this function with your project's client code.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    model = model or os.environ.get("GEMINI_MODEL", "text-bison-001")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateText?key={api_key}"
    payload = {
        "prompt": {"text": prompt},
        "temperature": 0.2,
        "maxOutputTokens": 512,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Response usually has candidates list with output text
    candidate = data.get("candidates")
    if candidate and isinstance(candidate, list) and len(candidate) > 0:
        return candidate[0].get("output", "")
    # Fallback for different response shapes
    return data.get("output", "") or json.dumps(data)


def answer_query(query: str, top_k: int = 4) -> str:
    """Retrieve context from the vectorstore and ask Gemini to generate an answer."""
    store = load_vectorstore()
    docs = store.similarity_search(query, k=top_k)
    # Extract text from retrieved documents
    texts = [d.page_content for d in docs]
    prompt = build_prompt(texts, query)
    answer = call_gemini_via_rest(prompt)
    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the RAG index using Gemini/PaLM REST API")
    parser.add_argument("query", nargs="?", help="Question to ask the RAG system (optional)")
    parser.add_argument("--top_k", type=int, default=4, help="Number of retrieved docs to use")
    args = parser.parse_args()
    # Support running without a positional `query` argument:
    # - If input is piped, read from stdin
    # - Otherwise prompt interactively
    query = args.query
    if not query:
        import sys
        if not sys.stdin.isatty():
            query = sys.stdin.read().strip()
        else:
            try:
                query = input("Enter your query: ").strip()
            except EOFError:
                query = ""

    if not query:
        parser.print_help()
        raise SystemExit(2)

    try:
        out = answer_query(query, top_k=args.top_k)
        print("\n=== Answer ===\n")
        print(out)
    except Exception as e:
        print("Error:", e)
