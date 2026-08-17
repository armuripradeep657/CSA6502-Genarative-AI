Python 3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> from sentence_transformers import SentenceTransformer
... from sklearn.metrics.pairwise import cosine_similarity
... from transformers import pipeline
... from langchain_text_splitters import RecursiveCharacterTextSplitter
... 
... documents = [
...     "Saveetha School of Engineering offers undergraduate and postgraduate engineering programs.",
...     "The Computer Science and Engineering department focuses on programming, artificial intelligence, data science and software development.",
...     "Students are required to attend classes regularly and complete their academic requirements.",
...     "The college provides laboratories for programming, networking, artificial intelligence and other technical subjects.",
...     "Artificial intelligence involves machine learning, deep learning, natural language processing and computer vision.",
...     "Machine learning enables computers to learn patterns from data and make predictions."
... ]
... 
... text = "\n".join(documents)
... 
... splitter = RecursiveCharacterTextSplitter(
...     chunk_size=150,
...     chunk_overlap=30
... )
... 
... chunks = splitter.split_text(text)
... 
... model = SentenceTransformer("all-MiniLM-L6-v2")
... 
... embeddings = model.encode(chunks)
... embeddings = embeddings.astype("float32")
... 
... question = input("Ask your college-related question: ")
... 
... question_embedding = model.encode([question])
question_embedding = question_embedding.astype("float32")

scores = cosine_similarity(
    question_embedding,
    embeddings
)[0]

top_k = 3

top_indices = scores.argsort()[-top_k:][::-1]

context = ""

for index in top_indices:
    context += chunks[index] + " "

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

prompt = (
    "You are a college information chatbot. "
    "Answer the question using only the provided information.\n\n"
    "Context: " + context +
    "\nQuestion: " + question +
    "\nAnswer:"
)

result = generator(
    prompt,
    max_new_tokens=100
)

print("\n===== COLLEGE CHATBOT =====")
print("\nQuestion:", question)
