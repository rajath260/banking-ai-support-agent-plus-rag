# RAG Part 4: End-to-end indexing and retrieval

## What is now connected?

The pipeline module connects the earlier building blocks:

- `document_loader.py`: reads FAQ Markdown files and creates chunks.
- `embeddings.py`: converts each chunk and each user query into vectors.
- `vector_store.py`: persists chunk text, metadata and vectors in Chroma.
- `pipeline.py`: orchestrates indexing and retrieval.
- `demo.py`: provides a local interactive demonstration.

## Indexing

Run the indexing flow when the knowledge base is first loaded or updated:

1. Read the Markdown files.
2. Split them into overlapping chunks.
3. Generate an embedding for each chunk.
4. Upsert each chunk into Chroma using a stable source-and-chunk identifier.

The use of upsert makes repeated indexing of unchanged files idempotent for the same source filename and chunk index. If a document is shortened or removed, old records that are no longer represented are not automatically deleted; production indexing needs an explicit stale-record cleanup/versioning strategy.

## Retrieval

For each user question:

1. Generate its embedding using the same model used for document chunks.
2. Query Chroma for the nearest vectors.
3. Return the original text, source metadata and cosine distance.

The distance is a ranking signal, not a confidence score. The next milestone will pass the retrieved text to an LLM with instructions to answer only from supplied context and to acknowledge when the knowledge base does not contain the answer.

## Run locally

From the repository root with the virtual environment activated and optional RAG dependencies installed:

```powershell
python -m app.rag.demo
```

Try questions such as:

- Someone stole my ATM card. How do I stop it?
- My UPI payment is pending. Should I pay again?
- Can I share my OTP with a bank employee?
- What should I do about suspicious account access?

The first run downloads the small Sentence Transformers model if it is not cached, then runs inference locally. Chroma persists its index under `data/chroma`. This demo does not call a hosted chat model or require paid inference credits.

## Important limitation

This milestone retrieves relevant FAQ passages; it does not yet generate a natural-language answer with an LLM. It is a retrieval demonstration, not a complete answer-generation RAG chain.
