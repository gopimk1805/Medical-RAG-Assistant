import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# Folder containing medical PDFs
pdf_folder = "data/medical_documents"

# Find all PDF files
pdf_files = [
    file for file in os.listdir(pdf_folder)
    if file.lower().endswith(".pdf")
]

print("PDF files found:", len(pdf_files))

all_documents = []

# Load every PDF
for pdf_file in pdf_files:

    pdf_path = os.path.join(pdf_folder, pdf_file)

    print(f"\nLoading: {pdf_file}")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    all_documents.extend(documents)

    print("Pages loaded:", len(documents))


# Split all documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(all_documents)


print("\n==============================")
print("Total pages:", len(all_documents))
print("Total chunks:", len(chunks))
print("==============================")


# Show first 3 chunks
print("\nFirst 3 chunks:\n")

for i, chunk in enumerate(chunks[:3]):

    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)


# Show metadata
print("\nChunk metadata:")
print(chunks[0].metadata)


# Test embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode(chunks[0].page_content)

print("\nEmbedding length:", len(embedding))