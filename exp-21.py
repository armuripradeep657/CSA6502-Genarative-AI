from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

documents = [
    """
    Artificial intelligence is a field of computer science that focuses on
    creating systems capable of performing tasks that normally require human intelligence.
    """,
    """
    Machine learning is a subset of artificial intelligence.
    It allows computers to learn patterns from data and make predictions.
    """,
    """
    Deep learning is a subset of machine learning that uses neural networks
    with multiple layers to solve complex problems.
    """,
    """
    Natural language processing allows computers to understand,
    process and generate human language.
    """
]

text = "\n".join(documents)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)

question = input("\nEnter your question: ")

question_embedding = model.encode([question])

similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]

top_k = 3

top_indices = similarities.argsort()[-top_k:][::-1]

context = ""

print("\nRetrieved Chunks:")

for index in top_indices:
    print("\n", chunks[index])
    context += chunks[index] + " "

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

prompt = (
    "Answer the question using only the provided context. "
    "If the answer is not available in the context, say that the information "
    "is not available.\n\n"
    "Context: " + context +
    "\nQuestion: " + question +
    "\nAnswer:"
)

result = generator(
    prompt,
    max_new_tokens=100
)

print("\nFinal Answer:")
print(result[0]["generated_text"])
