# GraphRAG with local inference for long technical documents

Bachelor's thesis (Computer Engineering, Universidad de Valladolid, 2026) by **Daniel Gómez Arranz**.

A question-answering system for long public-tender documents (*pliegos técnicos*) that runs entirely on local models. It keeps each document's real structure as a graph so that answers carry the context of the section they come from, which plain vector RAG loses.

📄 Full thesis (Spanish): [`docs/TFG_DanielGomezArranz.pdf`](docs/TFG_DanielGomezArranz.pdf)

## Results

On a benchmark of 30 questions over 6 public documents (150 pages):

| Configuration | Precision@3 |
|---|---|
| **Full system (GraphRAG)** | **0.83** |
| Without query rewriting by the agent | 0.67 |
| Without cross-encoder reranking | 0.70 |
| Without relevance propagation through the graph | 0.60 |
| Plain vector RAG (fixed-size chunks, no graph) | 0.53 |

- The strategy agent chose the right answering mode in 48 of 50 test queries (96%).
- Answers in under 30 seconds with Llama 3.1 8B through Ollama on a consumer GPU.
- Also tested on public cybersecurity guides (CCN-CERT, INCIBE) to check that it generalises beyond tenders.

The evaluation set is small, so treat these numbers as indicative rather than definitive; the thesis discusses the limitations in detail.

## How it works

1. **Structure-aware ingestion.** Detects the table of contents (internal links first, text patterns as fallback), infers the section hierarchy and splits the document by real sections instead of fixed-size chunks.
2. **Document graph** (NetworkX). Sections are nodes; edges connect parent and child sections, consecutive sections, and semantically similar ones.
3. **Hybrid retrieval.** BM25 plus embedding similarity, then a cross-encoder reranks the best candidates.
4. **Relevance propagation.** A short subsection inherits relevance from its parent and siblings, so it isn't lost when it lacks keywords on its own.
5. **LLM agents (LangGraph).** One agent decides whether to answer from retrieved sections or read the whole document; another rewrites the user's question before searching.
6. **Answer modes.** One answer per document, or a single consolidated answer across several documents.

## Repository layout

```
src/                  pipeline and experiments
  main.py             main GraphRAG pipeline (local models via Ollama)
  chunk_extractor*.py structure-aware PDF ingestion
  document_strategy_agent*.py  strategy-selection agent
  eval_metrics.py     automatic evaluation (Token F1, ROUGE-L, BERTScore, RAGAS)
  V3_OpenAI_ph.py     same pipeline answering with gpt-4o-mini, for comparison
data/                 put the input PDFs here (see data/README.md)
docs/                 thesis PDF, defence slides and LaTeX sources
```

## Running it

Requires Python 3.10+, [Ollama](https://ollama.com) with `llama3.1:8b` pulled, and ideally an NVIDIA GPU with 8 GB of VRAM.

```bash
pip install -r src/requirements.txt
ollama pull llama3.1:8b
python src/main.py          # run from the repository root
```

The OpenAI comparison script reads the key from the `OPENAI_API_KEY` environment variable; no key is stored in this repository.

## Limitations

- Assumes PDFs with a parseable table of contents; scanned documents (no text layer) need OCR, which is not included.
- Small evaluation set (30 questions for the ablation) and a single local model family.
- The knowledge base lives in memory; persistence (e.g. ChromaDB or FAISS) is future work.

## Note

This is research code written for a thesis, kept close to how it was used for the experiments. Comments and some file names are in Spanish.
