import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import fitz
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import uuid
import os
from groq import Groq
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()
groq_client= Groq(api_key=os.getenv("GROQ_API_KEY"))

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION  = "doctalk"
CHUNK_SIZE  = 512
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 3

#initializing Fastapi
app = FastAPI(title="DocTalk API")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

#initializing model and ChromaDB once at startup
model = SentenceTransformer(EMBED_MODEL)
client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)
collection = client.get_or_create_collection(name=COLLECTION)

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

    embeddings = model.encode(chunks).tolist()

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
    question_embedding = model.encode([request.question]).tolist()[0]
    
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
    print("starting server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload = True)