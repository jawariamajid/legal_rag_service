# Legal Research & Knowledge Assistant (RAG)

This repository contains a simple retrieval‑augmented generation (RAG) service designed as a proof‑of‑concept for building a legal research assistant.  It demonstrates how to combine a vector database with a generative language model to answer questions about a private collection of documents.

## Overview

1. **Ingestion:**  Load all `.txt` files from the `data/` directory, convert them to dense vectors using an open‑source embedding model (`all‑MiniLM‑L6‑v2` from the [Sentence‑Transformers](https://github.com/UKPLab/sentence-transformers) project) and store them in a Pinecone index.
2. **Retrieval:**  For each user query, encode the query with the same embedding model and retrieve the top relevant documents from the Pinecone index.
3. **Augmentation:**  Concatenate the retrieved document text into a single context string and append the user’s query to build a prompt for the language model.
4. **Generation:**  Send the prompt to the OpenAI API to generate a concise, context‑aware answer.

The service exposes a simple web interface (powered by FastAPI and Jinja2 templates) where you can type a question, submit it and see the generated answer.

## Setup

1. **Clone this repository** and change into its directory.
2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set required environment variables:**

   The application relies on a Pinecone index and the OpenAI API.  You need to create accounts on both services and obtain API keys.

   ```bash
   export PINECONE_API_KEY="your-pinecone-api-key"
   export PINECONE_ENVIRONMENT="us-east-1-aws"  # or your chosen region
   export PINECONE_INDEX="legal-knowledge-index"  # default index name used in this repo
   export OPENAI_API_KEY="your-openai-api-key"
   export OPENAI_MODEL="gpt-3.5-turbo"  # or another chat model, e.g. gpt-4o
   ```

5. **Prepare your data:**

   Place domain‑specific `.txt` files in the `legal_rag_service/data/` directory.  Each file should contain the full text of a document (e.g. statutes, case law, contracts, policies).  PDF or other formats must be converted to plain text before ingestion.

## Usage

1. **Ingest documents into Pinecone:**

   ```bash
   python ingest.py
   ```

   This script will create the specified Pinecone index if it does not already exist, encode all `.txt` files under `data/` and upsert the resulting vectors along with metadata.

2. **Start the web service:**

   ```bash
   uvicorn main:app --reload
   ```

   The service will be available at `http://127.0.0.1:8000/`.  Navigate to that URL in your browser, enter a legal question and submit it to receive an answer.

## Notes

* The embedding model (`all‑MiniLM‑L6‑v2`) and the Pinecone client must be installed locally.  See `requirements.txt` for the complete list of dependencies.
* This project is intended as a starting point; you can extend it by adding caching, better prompt engineering, or using other vector stores (e.g. FAISS) if Pinecone is not available.
* When deploying to production, consider using environment managers like Docker or Heroku and secure your API keys appropriately.
