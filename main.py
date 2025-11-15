"""
main.py
-------

FastAPI application for the legal research assistant.  This service exposes
two routes: a GET `/` that serves a simple HTML form and a POST `/query`
that processes a user query using retrieval‑augmented generation.  The query
is embedded and used to search a Pinecone index for relevant documents; the
contents of the top matches are concatenated to form context for a prompt
that is sent to the OpenAI API.

Before running this service, ensure that documents have been ingested into the
index with `ingest.py`, and that the environment variables for Pinecone and
OpenAI are configured.  See README.md for details.
"""

import os
from typing import List

import pinecone  # type: ignore
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer  # type: ignore
import openai  # type: ignore


# Create FastAPI instance
app = FastAPI(title="Legal Research & Knowledge Assistant")

# Load environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "legal-knowledge-index")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# Initialise Pinecone
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise RuntimeError("Missing Pinecone configuration. Ensure PINECONE_API_KEY and PINECONE_ENVIRONMENT are set.")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX)

# Load embedding model (open‑source model for cost & security)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure OpenAI
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OpenAI configuration. Ensure OPENAI_API_KEY is set.")
openai.api_key = OPENAI_API_KEY

# Set up Jinja templates (for the HTML form and results)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


def fetch_contexts(matches: List[dict]) -> str:
    """
    Given a list of Pinecone matches, read the source documents and
    concatenate their contents to form a single context string.  If a file
    cannot be read, it is silently skipped.
    """
    contexts: List[str] = []
    for match in matches:
        meta = match.get("metadata") or {}
        source_path = meta.get("source")
        if not source_path:
            continue
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                contexts.append(f.read())
        except Exception:
            continue
    return "\n\n".join(contexts)


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """Serve the main page with the query form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query", response_class=HTMLResponse)
async def handle_query(request: Request, query: str = Form(...)):
    """
    Handle form submission.  Embed the query, search Pinecone, build a prompt
    with retrieved context, call the OpenAI API and return the answer.
    """
    # Compute query embedding
    query_vector = embed_model.encode(query, show_progress_bar=False).tolist()
    # Retrieve top 5 matches with metadata
    search_response = index.query([query_vector], top_k=5, include_metadata=True)
    matches = search_response.get("matches", [])
    # Fetch the context from matched documents
    context = fetch_contexts(matches)
    # Construct prompt for the language model
    prompt = (
        "You are a helpful legal research assistant. Use the provided context from "
        "our document collection to answer the user's question. If the context does "
        "not contain relevant information, you may say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    # Call OpenAI Chat Completion API
    try:
        completion = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        answer = completion.choices[0].message["content"].strip()
    except Exception as exc:
        answer = f"Error fetching answer from OpenAI: {exc}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": query,
            "answer": answer,
        },
    )