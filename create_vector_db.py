import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


# Folder containing medical PDFs
pdf_folder = "data/medical_documents"


# Find all PDF files
pdf_files = [
    file for file in os.listdir(pdf_folder)
    if file.lower().endswith(".pdf")
]

print("PDF files found:", len(pdf_files))


# Load all PDF documents
all_documents = []

for pdf_file in pdf_files:

    pdf_path = os.path.join(pdf_folder, pdf_file)

    print(f"\nLoading: {pdf_file}")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    all_documents.extend(documents)

    print("Pages loaded:", len(documents))


# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(all_documents)


print("\nTotal pages:", len(all_documents))
print("Total chunks:", len(chunks))


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")


# Delete old collection if it exists
try:
    client.delete_collection(name="medical_documents")
    print("\nOld vector database deleted.")
except Exception:
    print("\nNo old collection found.")


# Create fresh collection
collection = client.create_collection(
    name="medical_documents"
)


# Store chunks and embeddings
for i, chunk in enumerate(chunks):

    embedding = model.encode(
        chunk.page_content
    ).tolist()

    collection.add(
        ids=[str(i)],
        documents=[chunk.page_content],
        embeddings=[embedding],
        metadatas=[chunk.metadata]
    )


print("\nChunks stored:", len(chunks))
print("Vector database created successfully!")