# DocTalk 

I wanted to understand the concept RAG-It's a technique that gives an LLM access to your own documents when answering questions, instead of relying only on what it learned during training. 

So I built DocTalk from scratch. DocTalk lets you upload any PDF and ask questions about it in plain English. No keyword search. It understands what you're asking and finds the right answer from your document.

Same core architecture as NotebookLM. Built from scratch.

---

## How it works

```
PDF → extract → chunk → embed → ChromaDB
                                    ↓
         question → embed → semantic search → Groq → answer
```

---

## Stack

- **FastAPI** — API framework
- **PyMuPDF** — PDF parsing
- **LangChain** — text chunking
- **sentence-transformers** — embeddings (all-MiniLM-L6-v2)
- **ChromaDB** — vector database
- **Groq** — LLM (llama-3.3-70b-versatile)

---

## Run it locally

Hit `http://localhost:8000/docs` to test.

## Future updates

- Minimal web UI (for a non-technical person)
- Persistent vector DB (ChromaDB Cloud)
- Multi-PDF support

---

Built by [Ayush Mishra]
(https://github.com/mishra-codes)
