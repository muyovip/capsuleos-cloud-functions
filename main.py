import functions_framework
from google.cloud import storage
import io
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.readers.file import PDFReader
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone
import os

# Init Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")
index = pinecone.Index("stratheum-hoi-poi-index")

@functions_framework.cloud_event
def ingest_pdf(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    name = data["name"]
    
    if not name.startswith("raw/") or not name.endswith(".pdf"):
        print(f"Ignored: {name}")
        return
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(name)
    pdf_bytes = blob.download_as_bytes()
    
    documents = PDFReader().load_data(io.BytesIO(pdf_bytes))
    embed_model = GeminiEmbedding(model_name="models/embedding-001")
    vector_store = PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    
    # Move to processed
    new_name = name.replace("raw/", "processed/")
    bucket.rename_blob(blob, new_name)
    
    print(f"INGESTED {name} → {len(documents)} chunks → Pinecone")
    return "success"
