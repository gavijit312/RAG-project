import os
import json
import requests
from typing import List

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_vectorstore(path: str = "vectorstore") -> FAISS:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = path if os.path.isabs(path) else os.path.join(base_dir, path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
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
    Call Google Generative Language REST endpoint (Gemini).

    This function uses the public REST endpoint and an API key.

    Required env var:
      - GEMINI_API_KEY

    Optional env vars:
      - GEMINI_MODEL (default: gemini-1.5-flash)
      - GEMINI_API_VERSION (default: v1beta)
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    model = model or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    api_version = os.environ.get("GEMINI_API_VERSION", "v1beta")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    # Modern Gemini API: :generateContent
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
        },
    }
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    # If the endpoint/model combo is wrong, try older :generateText once.
    if resp.status_code == 404 or resp.status_code == 400:
        legacy_model = os.environ.get("GEMINI_LEGACY_MODEL")
        if legacy_model:
            legacy_url = (
                f"https://generativelanguage.googleapis.com/v1beta2/models/{legacy_model}:generateText"
                f"?key={api_key}"
            )
            legacy_payload = {
                "prompt": {"text": prompt},
                "temperature": 0.2,
                "maxOutputTokens": 512,
            }
            legacy_resp = requests.post(legacy_url, headers=headers, json=legacy_payload, timeout=60)
            legacy_resp.raise_for_status()
            legacy_data = legacy_resp.json()
            candidates = legacy_data.get("candidates")
            if candidates and isinstance(candidates, list):
                return candidates[0].get("output", "") or ""
            return legacy_data.get("output", "") or json.dumps(legacy_data)

    resp.raise_for_status()
    data = resp.json()

    # Expected shape:
    # {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
    candidates = data.get("candidates")
    if candidates and isinstance(candidates, list):
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if parts and isinstance(parts, list):
            text = parts[0].get("text")
            if isinstance(text, str):
                return text

    # Fallback for different response shapes
    return data.get("text", "") or data.get("output", "") or json.dumps(data)


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
