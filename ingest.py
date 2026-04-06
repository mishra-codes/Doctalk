import argparse
import uuid
import fitz
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

#Configuration

CHROMA_PATH = "./chroma_db"
COLLECTION = "doctalk"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"

#Function- Extracting text from pdf

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    print(f" Extracted {len(full_text)} characters from {pdf_path}")
    return full_text

#Function - chunking the text

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n","\n", ".", "", ""]
    )
    chunks = splitter.split_text(text)
    print(f" Created {len(chunks)} chunks")
    return chunks

#Function - embedding the text

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()
    print(f" Generated {len(embeddings)} embeddings (dim= {len(embeddings[0])})")
    return embeddings

#Function - Storing in ChromaDB

def store_in_chroma(chunks: list[str], embeddings: list[list[float]], pdf_name: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": pdf_name, "chunk_index": i} for i, _ in  enumerate(chunks)]

    collection.add (
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f" Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}'")
    print(f"  Collection '{COLLECTION}' now has {collection.count()} total docs")

#main function

if __name__ == "__main__":
    print("Starting......")
    parser = argparse.ArgumentParser(description="Doctalk ingestion pipeline")
    parser.add_argument("--pdf", required=True, help="Path to Pdf files")
    args = parser.parse_args()

    text = extract_text(args.pdf)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_in_chroma(chunks, embeddings, pdf_name=args.pdf)

    print("\n Ingestion complete. Ready to Query.")