"""
Persistent Vector Store Management using FAISS

This module provides a persistent vector store implementation that:
1. Uses FAISS for efficient similarity search
2. Caches the vector store to disk to avoid re-indexing
3. Only rebuilds the index when source documents change
4. Provides a simple interface for all RAG agents

Usage:
    from officebuddy.vector_store import get_vector_store, get_retriever

    # Get the vector store (loads from cache or builds if needed)
    vector_store = get_vector_store()

    # Get a retriever with custom search parameters
    retriever = get_retriever(k=5)
"""

import hashlib
from pathlib import Path
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

# Default URLs for document sources
DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

# Directory for storing FAISS index
VECTOR_STORE_DIR = Path(__file__).parent.parent.parent / ".vector_store"
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# File paths
INDEX_PATH = VECTOR_STORE_DIR / "faiss_index"
URLS_HASH_FILE = VECTOR_STORE_DIR / "urls_hash.txt"


# ============================================================================
# Helper Functions
# ============================================================================

def compute_urls_hash(urls: List[str]) -> str:
    """
    Compute a hash of the URLs to detect changes.

    Args:
        urls: List of document URLs

    Returns:
        str: SHA256 hash of the URLs
    """
    urls_string = "|".join(sorted(urls))
    return hashlib.sha256(urls_string.encode()).hexdigest()


def needs_rebuild(urls: List[str]) -> bool:
    """
    Check if the vector store needs to be rebuilt.

    The store needs rebuilding if:
    - The index doesn't exist
    - The URLs have changed

    Args:
        urls: List of document URLs

    Returns:
        bool: True if rebuild is needed
    """
    # Check if index exists
    if not INDEX_PATH.exists():
        return True

    # Check if URLs hash file exists
    if not URLS_HASH_FILE.exists():
        return True

    # Check if URLs have changed
    current_hash = compute_urls_hash(urls)
    with open(URLS_HASH_FILE, "r") as f:
        stored_hash = f.read().strip()

    return current_hash != stored_hash


def load_and_split_documents(urls: List[str]):
    """
    Load documents from URLs and split them into chunks.

    Args:
        urls: List of document URLs

    Returns:
        List of document chunks
    """
    print(f"📥 Loading documents from {len(urls)} URLs...")

    # Load documents
    docs = []
    for url in urls:
        print(f"  - Loading: {url}")
        docs.extend(WebBaseLoader(url).load())

    print(f"✅ Loaded {len(docs)} documents")

    # Split into chunks
    print("🔪 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs)

    print(f"✅ Created {len(doc_splits)} document chunks")

    return doc_splits


def build_vector_store(urls: List[str]) -> FAISS:
    """
    Build a new FAISS vector store from documents.

    Args:
        urls: List of document URLs

    Returns:
        FAISS: The built vector store
    """
    print("\n🏗️  Building new vector store...")

    # Load and split documents
    doc_splits = load_and_split_documents(urls)

    # Create embeddings and vector store
    print("🔢 Creating embeddings and FAISS index...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(
        documents=doc_splits,
        embedding=embeddings
    )

    # Save to disk
    print(f"💾 Saving vector store to {INDEX_PATH}")
    vector_store.save_local(str(INDEX_PATH))

    # Save URLs hash
    urls_hash = compute_urls_hash(urls)
    with open(URLS_HASH_FILE, "w") as f:
        f.write(urls_hash)

    print("✅ Vector store built and saved successfully!\n")

    return vector_store


def load_vector_store() -> FAISS:
    """
    Load the FAISS vector store from disk.

    Returns:
        FAISS: The loaded vector store
    """
    print(f"📂 Loading vector store from {INDEX_PATH}")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded from cache\n")
    return vector_store


# ============================================================================
# Public API
# ============================================================================

def get_vector_store(urls: List[str] = None, force_rebuild: bool = False) -> FAISS:
    """
    Get the vector store, loading from cache or building if needed.

    This is the main entry point for getting a vector store. It will:
    1. Check if a cached version exists and is up to date
    2. Load from cache if available
    3. Build a new index if cache is missing or URLs changed

    Args:
        urls: List of document URLs (defaults to DEFAULT_URLS)
        force_rebuild: If True, rebuild even if cache exists

    Returns:
        FAISS: The vector store instance
    """
    if urls is None:
        urls = DEFAULT_URLS

    if force_rebuild or needs_rebuild(urls):
        return build_vector_store(urls)
    else:
        return load_vector_store()


def get_retriever(urls: List[str] = None, k: int = 4, force_rebuild: bool = False):
    """
    Get a retriever instance configured with the vector store.

    Args:
        urls: List of document URLs (defaults to DEFAULT_URLS)
        k: Number of documents to retrieve
        force_rebuild: If True, rebuild vector store even if cache exists

    Returns:
        Retriever: Configured retriever instance
    """
    vector_store = get_vector_store(urls=urls, force_rebuild=force_rebuild)
    return vector_store.as_retriever(search_kwargs={"k": k})


def clear_cache():
    """
    Clear the cached vector store.

    Use this to force a rebuild on the next get_vector_store() call.
    """
    import shutil

    if VECTOR_STORE_DIR.exists():
        print(f"🗑️  Clearing vector store cache at {VECTOR_STORE_DIR}")
        shutil.rmtree(VECTOR_STORE_DIR)
        VECTOR_STORE_DIR.mkdir(exist_ok=True)
        print("✅ Cache cleared")
    else:
        print("ℹ️  No cache to clear")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Persistent Vector Store")
    print("=" * 70)

    # Test 1: Build vector store
    print("\n📝 Test 1: Initial build")
    vector_store = get_vector_store()

    # Test 2: Load from cache
    print("\n📝 Test 2: Load from cache")
    vector_store = get_vector_store()

    # Test 3: Use retriever
    print("\n📝 Test 3: Test retrieval")
    retriever = get_retriever(k=3)
    results = retriever.invoke("What is reward hacking?")
    print(f"✅ Retrieved {len(results)} documents")
    print(f"First document preview: {results[0].page_content[:200]}...")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
