# RAG Part 2: Embeddings

## What is an embedding?

An embedding model converts text into a dense numerical vector. Texts with related meanings are generally represented by vectors that are closer under a suitable similarity measure.

For example, "block my lost debit card" and "I misplaced my ATM card; how can I deactivate it?" use different words but express a similar intent.

## How this project uses embeddings

1. Load FAQ documents.
2. Split documents into chunks.
3. Convert each chunk into a vector.
4. Convert a user's question into a vector using the same model.
5. Compare the query vector with stored chunk vectors.
6. Retrieve the most relevant chunks.

## Model choice

This project uses `sentence-transformers/all-MiniLM-L6-v2` as a small, commonly used sentence embedding model. It produces 384-dimensional embeddings.

The model is downloaded from the Hugging Face Hub the first time it is initialized, then inference runs locally. This is different from calling a hosted Hugging Face Inference API: no paid hosted inference call is made. The initial model download still requires internet access, and local inference uses CPU/RAM (or a compatible GPU).

## Install the optional dependency

The core agent dependencies remain unchanged. Install the embedding runtime separately:

```powershell
pip install -r requirements-rag.txt
```

## Why normalize vectors?

The embedding adapter requests normalized vectors. For unit-length vectors, cosine similarity is equivalent to the dot product, which makes similarity comparison convenient in the next milestone.

## Testing

Unit tests use a fake SentenceTransformer implementation. They verify the adapter without downloading the real model or making network requests.

## Production considerations

Model versioning, embedding dimensions, document versioning and re-embedding strategy matter. If the embedding model changes, existing stored vectors should generally be regenerated with the same model before comparing them with new query vectors.
