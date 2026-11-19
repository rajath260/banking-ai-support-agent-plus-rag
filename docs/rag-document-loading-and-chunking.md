# RAG Part 1: Document Loading and Chunking

## Goal

Prepare source documents so a later retrieval system can search small, meaningful pieces instead of sending entire files to a language model.

## Pipeline so far

Markdown FAQ files -> Document objects -> overlapping chunks

## Key concepts

- **Document loading:** reads source files into the application.
- **Chunking:** splits long documents into smaller pieces.
- **Chunk size:** the maximum size of one piece. This first exercise measures size in characters, not tokens.
- **Overlap:** repeated text shared by adjacent chunks. It helps preserve context when an answer-relevant sentence is near a chunk boundary.
- **Metadata:** information such as source filename and chunk index. We will use it later to identify which source supports a retrieved answer.

## Why not send whole documents to the model?

Large knowledge bases can exceed a model's context window, increase inference cost, and include irrelevant information. Retrieval systems first select relevant chunks and provide only those chunks as context.

## Important limitation

Character-based chunking is intentionally simple for learning. Production systems often use token-aware or structure-aware chunking, preserve headings and sections, and tune chunk size against retrieval quality.

The FAQ content in knowledge_base/ is fictional demonstration material, not real banking policy.
