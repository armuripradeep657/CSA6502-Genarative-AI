from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

documents = [
    "Artificial intelligence enables machines to perform tasks that normally require human intelligence.",
    "Machine learning allows computers to learn patterns from data and make predictions.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables computers to analyze images and videos.",
    "Cybersecurity protects computer systems and data from digital threats."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(documents)

ai_assistant = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("===== AI DOCUMENT ASSISTANT =====")
print("Type 'exit' to stop.")

while True:

    question = input("\nEnter your question: ")

    if question.lower() == "exit":
        print("Assistant: Goodbye!")
        break

    query_embedding = model.encode([question])

    similarity_scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    top_k = 3

    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    context = ""

    for index in top_indices:
        context += documents[index] + " "

    prompt = (
        "Answer the question using only the provided documents. "
        "Do not use outside information.\n\n"
        "Documents: " + context +
        "\nQuestion: " + question +
        "\nAnswer:"
    )

    response = ai_assistant(
        prompt,
        max_new_tokens=100
    )

    print("\nAssistant:", response[0]["generated_text"])
