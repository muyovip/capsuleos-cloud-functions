import os
import io
from google.cloud import storage
from flask import Flask, request
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.readers.file import PDFReader
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone

app = Flask(__name__)

print("Starting Flask app on PORT:", os.getenv("PORT", "8080"))
print("Binding to host: 0.0.0.0")

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")
index = pinecone.Index("stratheum-hoi-poi-index")

@app.route("/", methods=["POST"])
def ingest_pdf():
    print("Received trigger:", request.json)
    data = request.json
    bucket_name = data["bucket"]
    name = data["name"]
    
    if not name.startswith("raw/") or not name.endswith(".pdf"):
        print("Ignored:", name)
        return "Ignored", 200
    
    print("Processing:", name)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(name)
    pdf_bytes = blob.download_as_bytes()
    
    documents = PDFReader().load_data(io.BytesIO(pdf_bytes))
    embed_model = GeminiEmbedding(model_name="models/embedding-001")
    vector_store = PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    
    new_name = name.replace("raw/", "processed/")
    bucket.rename_blob(blob, new_name)
    
    print(f"Success: {len(documents)} chunks to Pinecone")
    return f"Success: {len(documents)} chunks", 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
    print("Flask app running on 0.0.0.0:", port)
