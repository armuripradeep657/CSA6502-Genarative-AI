from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Artificial intelligence is used to develop intelligent computer systems.",
    "Machine learning allows computers to learn from historical data.",
    "Python is widely used for artificial intelligence and data science.",
    "Deep learning uses neural networks to solve complex problems.",
    "Cybersecurity protects systems and data from digital threats.",
    "Cloud computing provides services and storage over the internet.",
    "Natural language processing helps computers understand human language.",
    "Computer vision allows computers to analyze images and videos."
]

document_embeddings = model.encode(documents)
document_embeddings = document_embeddings.astype("float32")

dimension = document_embeddings.shape[1]

vector_database = faiss.IndexFlatL2(dimension)

vector_database.add(document_embeddings)

print("Total documents stored:", vector_database.ntotal)

query = input("Enter your question: ")

query_embedding = model.encode([query])
query_embedding = query_embedding.astype("float32")

k = 3

distances, indexes = vector_database.search(query_embedding, k)

print("\nTop", k, "Retrieved Documents")

for rank in range(k):
    index = indexes[0][rank]

    print("\nRank:", rank + 1)
    print("Document:", documents[index])
    print("Distance:", round(float(distances[0][rank]), 4))
