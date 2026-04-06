import argparse
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION  = "doctalk"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 3

# Function - Load ChromaDB collection

def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION)
    print(f"Loaded collection '{COLLECTION}' with {collection.count()} docs")
    return collection

# Function - Embed the question

def embed_question(question: str) -> list[float]:
    model = SentenceTransformer(EMBED_MODEL)
    embedding = model.encode([question]).tolist()[0]
    print(f" Question embedded")
    return embedding

# Function - Querying  ChromaDB

def query_collection(collection, question_embedding: list[float]) -> list[str]:
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )
    chunks = results["documents"][0]
    print(f"Retrieved {len(chunks)} chunks")
    return chunks

# Main Function

if __name__ == "__main__":
    print("Finding Answers...")
    parser = argparse.ArgumentParser(description="DocTalk query pipeline")
    parser.add_argument("--question", required=True, help="Question to ask")
    args = parser.parse_args()

    collection = load_collection()
    question_embedding = embed_question(args.question)
    chunks = query_collection(collection, question_embedding)

    print("\n--- Retrieved Chunks ---")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}]:\n{chunk}")