from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer("all-MiniLM-L6-v2")
documents = [
    "Python is a popular programming language used for artificial intelligence.",
    "Machine learning allows computers to learn from data.",
    "Deep learning uses artificial neural networks.",
    "Database systems store and manage large amounts of information.",
    "Cybersecurity protects systems and networks from attacks.",
    "Cloud computing provides computing services through the internet.",
    "Natural language processing helps computers understand human language.",
    "Computer vision allows machines to analyze images and videos."
]

document_embeddings = model.encode(documents)

query = input("Enter your search query: ")
query_embedding = model.encode([query])

similarity_scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]
results = list(zip(documents, similarity_scores))
results.sort(key=lambda x: x[1], reverse=True)
print("\nSemantic Search Results")
print("=" * 60)

for rank, (document, score) in enumerate(results, start=1):
    print(f"\nRank {rank}")
    print("Document:", document)
    print("Cosine Similarity:", round(score, 4))
