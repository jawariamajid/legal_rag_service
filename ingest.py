"""
ingest.py
-----------

Script to ingest domain‑specific documents into a Pinecone vector index.  Each
document in the `data/` directory is read, embedded using a SentenceTransformer
model, and upserted into the specified Pinecone index along with metadata.

This script should be run once (or whenever new documents are added) before
serving queries with `main.py`.
"""

import os
import glob
import uuid
from typing import Iterable, Tuple

import pinecone  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore


def load_documents(directory: str) -> Iterable[Tuple[str, str]]:
    """Yield `(path, text)` for each `.txt` file in the given directory."""
    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            yield filepath, text
        except Exception as exc:
            print(f"[WARN] Failed to read {filepath}: {exc}")


def ingest_documents(index_name: str, directory: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    """
    Embed all text files under `directory` and upsert them into the Pinecone index
    named `index_name`.  Uses the specified SentenceTransformer model.

    If the index does not exist, it will be created with the appropriate dimension.
    """
    # Load embedding model once
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    # Initialise Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_env = os.environ.get("PINECONE_ENVIRONMENT")
    if not pinecone_api_key or not pinecone_env:
        raise RuntimeError("PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables must be set")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    # Create index if needed
    if index_name not in pinecone.list_indexes():
        print(f"[INFO] Creating index '{index_name}' with dimension {dim}")
        pinecone.create_index(name=index_name, dimension=dim, metric="cosine")
    else:
        print(f"[INFO] Using existing index '{index_name}'")

    index = pinecone.Index(index_name)

    # Read and embed documents
    upserts = []
    for path, text in load_documents(directory):
        # embed text
        embedding = model.encode(text, show_progress_bar=False).tolist()
        # assign a unique ID
        doc_id = str(uuid.uuid4())
        # store file path in metadata
        metadata = {"source": path}
        upserts.append((doc_id, embedding, metadata))

    if upserts:
        print(f"[INFO] Upserting {len(upserts)} documents into index '{index_name}'...")
        index.upsert(vectors=upserts)
        print("[INFO] Ingestion complete.")
    else:
        print("[WARN] No documents found to ingest.")


if __name__ == "__main__":
    index_name = os.environ.get("PINECONE_INDEX", "legal-knowledge-index")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    ingest_documents(index_name, data_dir)