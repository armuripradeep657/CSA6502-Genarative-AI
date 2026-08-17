from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

documents = [
    "Artificial intelligence is a technology that enables machines to perform tasks that normally require human intelligence.",
    "Machine learning is a branch of artificial intelligence that allows computers to learn from data.",
    "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand and process human language.",
    "Computer vision enables machines to understand and analyze images and videos."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(documents)

question = input("Enter your question: ")

question_embedding = model.encode([question])

similarities = cosine_similarity(
    question_embedding,
    document_embeddings
)[0]

top_indices = similarities.argsort()[-3:][::-1]

context = ""

for index in top_indices:
    context += documents[index] + " "

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

prompt = (
    "Answer the question using only the given context.\n\n"
    "Context: " + context +
    "\nQuestion: " + question +
    "\nAnswer:"
)

result = generator(
    prompt,
    max_new_tokens=100
)

print("\nQuestion:", question)
print("\nRetrieved Context:")
print(context)

print("\nAnswer:")
print(result[0]["generated_text"])
