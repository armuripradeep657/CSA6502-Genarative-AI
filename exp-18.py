from sentence_transformers import SentenceTransformer
import faiss

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

documents = [
    "Machine learning allows computers to learn from data.",
    "Artificial intelligence enables machines to perform intelligent tasks.",
    "Python is a popular programming language for data science.",
    "Deep learning uses neural networks with multiple layers.",
    "Cybersecurity protects computer systems and networks.",
    "Natural language processing helps computers understand human language.",
    "Cloud computing provides computing resources through the internet.",
    "Computer vision allows computers to understand images and videos."
]

print("\nGenerating document embeddings...")

embeddings = model.encode(documents)
embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Documents stored in FAISS:", index.ntotal)

query = input("\nEnter your search query: ")

query_embedding = model.encode([query])
query_embedding = query_embedding.astype("float32")

k = 3

distances, indices = index.search(query_embedding, k)

print("\n===== SIMILAR DOCUMENTS =====")

for i in range(k):
    document_index = indices[0][i]

    print(f"\nRank {i + 1}")
    print("Document:", documents[document_index])
    print("Distance:", round(float(distances[0][i]), 4))

print("\nSearch completed successfully!")
