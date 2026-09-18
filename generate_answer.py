from dotenv import load_dotenv
import os
from google import genai
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# Gemini API setup
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_collection(
    name="medical_documents"
)

# Get user's question
question = input("Ask a medical question: ")

# Convert question into an embedding
query_embedding = model.encode(question).tolist()

# Retrieve relevant medical information
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

# Check whether retrieved information is relevant
distances = results["distances"][0]

if distances[0] > 1.0:
    print("\nMedical Assistant:")
    print("I don't have enough information in the provided medical documents to answer this question.")
    exit()

# Combine retrieved chunks into context
context = "\n\n".join(results["documents"][0])

# Grounding prompt
prompt = f"""
You are a medical information assistant.

Answer the user's question ONLY using the information provided in the CONTEXT.

CONTEXT:
{context}

USER QUESTION:
{question}

RULES:
1. Use only the information present in the CONTEXT.
2. Do not use outside medical knowledge.
3. Do not guess or make up information.
4. If the answer is not available in the CONTEXT, say:
"I don't have enough information in the provided medical documents to answer this question."
5. Give a clear and easy-to-understand answer.
6. This is for informational purposes only and does not replace advice from a qualified healthcare professional.
"""

# Send grounded prompt to Gemini
interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)

# Display answer
print("\nMedical Assistant:")
print(interaction.output_text)

# Display sources
print("\nSources:")

for i, document in enumerate(results["documents"][0]):
    print(f"\n--- Source {i + 1} ---")
    print(document)