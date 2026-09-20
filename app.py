from flask import Flask, render_template, request

from dotenv import load_dotenv
import os

from google import genai
import chromadb


app = Flask(__name__)

load_dotenv()


# --------------------------------------------------
# Gemini Configuration
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    gemini_client = genai.Client(
        api_key=api_key,
        http_options={
            "api_version": "v1",
            "timeout": 30000
        }
    )
else:
    gemini_client = None


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

# Load the model only when a question is asked.
# This helps Render start the web server faster.

model = None


def get_embedding_model():

    global model

    if model is None:

        print("Loading embedding model...")

        # Load only when needed
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded.")

    return model


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="medical_documents"
)

print("ChromaDB connected.")

print(
    "Documents in database:",
    collection.count()
)


# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    answer = None
    sources = []
    question = ""

    if request.method == "POST":

        print("\nPOST request received.")

        question = request.form.get(
            "question",
            ""
        ).strip()

        print("Question:", question)


        # --------------------------------------------------
        # Empty Question
        # --------------------------------------------------

        if not question:

            answer = (
                "Please enter a medical question."
            )


        else:

            # --------------------------------------------------
            # Load Embedding Model
            # --------------------------------------------------

            embedding_model = (
                get_embedding_model()
            )


            # --------------------------------------------------
            # Create Question Embedding
            # --------------------------------------------------

            print(
                "Creating question embedding..."
            )

            query_embedding = (
                embedding_model
                .encode(question)
                .tolist()
            )


            # --------------------------------------------------
            # Search ChromaDB
            # --------------------------------------------------

            print(
                "Searching ChromaDB..."
            )

            results = collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=3
            )


            documents = results[
                "documents"
            ][0]

            metadatas = results[
                "metadatas"
            ][0]

            distances = results[
                "distances"
            ][0]


            print(
                "Retrieved chunks:",
                len(documents)
            )


            if distances:

                print(
                    "Best similarity distance:",
                    distances[0]
                )


            # --------------------------------------------------
            # Out-of-Knowledge-Base Handling
            # --------------------------------------------------

            if (
                not documents
                or not distances
                or distances[0] > 1.0
            ):

                answer = (
                    "I don't have enough information "
                    "in the provided medical documents "
                    "to answer this question."
                )


            else:

                # --------------------------------------------------
                # Build Context
                # --------------------------------------------------

                context_parts = []

                for i, document in enumerate(
                    documents
                ):

                    context_parts.append(
                        f"Source {i + 1}:\n"
                        f"{document}"
                    )


                context = "\n\n".join(
                    context_parts
                )


                # --------------------------------------------------
                # Collect Sources
                # --------------------------------------------------

                for metadata in metadatas:

                    source = metadata.get(
                        "source",
                        "Unknown source"
                    )

                    page = metadata.get(
                        "page",
                        None
                    )


                    if page is not None:

                        page = int(page) + 1

                        source_text = (
                            f"{source} - Page {page}"
                        )

                    else:

                        source_text = source


                    if source_text not in sources:

                        sources.append(
                            source_text
                        )


                # --------------------------------------------------
                # Gemini
                # --------------------------------------------------

                if gemini_client is None:

                    answer = (
                        "Gemini API key was not found. "
                        "Please check your .env file."
                    )


                else:

                    prompt = f"""
You are a medical information assistant.

Your task is to answer the user's question using ONLY
the information provided in the CONTEXT below.

CONTEXT:
{context}

USER QUESTION:
{question}

IMPORTANT RULES:

1. Use only information contained in the CONTEXT.
2. Do not use outside medical knowledge.
3. Do not invent facts, symptoms, treatments, medicines,
   dosages, or schedules.
4. Do not diagnose the user.
5. Do not prescribe medicines.
6. Do not create a personalized treatment plan.
7. If the CONTEXT does not contain enough information
   to answer the question, say:

"I don't have enough information in the provided medical
documents to answer this question."

8. Give the answer in simple and clear language.
9. When relevant, organize the answer using short
   headings and bullet points.
10. If the CONTEXT contains warning signs or information
    about when medical attention is needed, include it.
11. Clearly mention that this is general medical
    information and does not replace advice from a qualified
    healthcare professional.

Now answer the user's question.
"""


                    try:

                        print(
                            "Calling Gemini..."
                        )


                        interaction = (
                            gemini_client
                            .interactions
                            .create(
                                model="gemini-3.8-flash",
                                input=prompt,
                                generation_config={
                                    "thinking_level": "low"
                                }
                            )
                        )


                        print(
                            "Gemini response received."
                        )


                        answer = (
                            interaction.output_text
                        )


                    except Exception as error:

                        error_text = str(error)

                        print(
                            "Gemini error:",
                            error_text
                        )


                        # --------------------------------------------------
                        # Quota Error
                        # --------------------------------------------------

                        if (
                            "429" in error_text
                            or
                            "RESOURCE_EXHAUSTED"
                            in error_text
                        ):

                            answer = (
                                "Gemini API quota is "
                                "currently exhausted. "
                                "The medical information "
                                "was retrieved successfully, "
                                "but the AI-generated answer "
                                "cannot be generated until "
                                "the API quota becomes "
                                "available again."
                            )


                        # --------------------------------------------------
                        # Timeout Error
                        # --------------------------------------------------

                        elif (
                            "timeout"
                            in error_text.lower()
                            or
                            "timed out"
                            in error_text.lower()
                        ):

                            answer = (
                                "The Gemini request timed out. "
                                "The medical information was "
                                "retrieved successfully, but "
                                "the AI answer could not be "
                                "generated right now. "
                                "Please try again."
                            )


                        # --------------------------------------------------
                        # Other Gemini Errors
                        # --------------------------------------------------

                        else:

                            answer = (
                                "The medical information was "
                                "retrieved successfully, but "
                                "Gemini could not generate "
                                "the answer right now. "
                                "Please try again later."
                            )


    # --------------------------------------------------
    # Render HTML
    # --------------------------------------------------

    return render_template(
        "index.html",
        answer=answer,
        sources=sources,
        question=question
    )


# --------------------------------------------------
# Local Development
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                5000
            )
        )
    )