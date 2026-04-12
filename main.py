import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import fitz
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import os
from groq import Groq
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cohere


load_dotenv()
groq_client= Groq(api_key=os.getenv("GROQ_API_KEY"))
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Configuration
COLLECTION  = "doctalk"
CHUNK_SIZE  = 512
CHUNK_OVERLAP = 50
TOP_K       = 3


# Global references (set during startup)

collection = None

# New Embedding model

def get_embedding(texts, input_type="search_document"):
    response = co.embed(
        texts= texts,
        model="embed-english-light-v3.0",
        input_type=input_type
    )
    return response.embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    try:
        print("Loading model and connecting to ChromaDB...")
        client = chromadb.CloudClient(
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
            api_key=os.getenv("CHROMA_API_KEY")
    )
        
        collection = client.get_or_create_collection(name=COLLECTION)
        print("Ready!")

    except Exception as e:
        print(f"Startup error: {e}")

    yield

#initializing Fastapi
app = FastAPI(title="DocTalk API",lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")

#adding Pydantic model for query request
class QueryRequest(BaseModel):
    question: str

#ingesting the pdf part

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(full_text)

    embeddings = get_embedding(chunks, input_type="search_document")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": file.filename, "chunk_index": i} for i, _ in enumerate(chunks)]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return {"message": f"Ingested {len(chunks)} chunks from {file.filename}"}


# Querying 

@app.post("/query")
async def query_pdf(request: QueryRequest):
    question_embedding = get_embedding([request.question], input_type="search_query")[0]
    
    
    if collection is None:
        return {"error": "Database not initialized"}

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )

    
    chunks = results["documents"][0]
    context = "\n\n".join(chunks)

    prompt =f"""You are a helpful assistant. Answer the Questions based only on the context provided below.

Context:
{context}

Question: {request.question}

Answer:"""
    
    try: 
        response = groq_client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages=[{"role":"user","content": prompt}]
        )
        answer = response.choices[0].message.content

    except Exception as e:
        print(f"Groq error: {e}")
        answer = f"Error: {str(e)}"

    return {
        "question": request.question,
        "answer": answer,
        "chunks": chunks
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"starting server on port {port}...")

    uvicorn.run(
        "main:app",
         host="0.0.0.0", 
         port=port, 
         reload =True
    )