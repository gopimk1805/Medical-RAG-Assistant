from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="medical_documents"
)

# Get user's question
question = input("Ask a medical question: ")

# Convert question into embedding
query_embedding = model.encode(question).tolist()

# Retrieve more relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("\nQuestion:", question)

print("\nRetrieved information:\n")

for i, document in enumerate(results["documents"][0]):

    print(f"--- Result {i + 1} ---")

    print(document)

    print()