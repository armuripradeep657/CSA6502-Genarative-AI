from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = [
    "Artificial intelligence is a field of computer science that enables machines to perform intelligent tasks.",
    "Machine learning is a part of artificial intelligence that allows computers to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand and process human language.",
    "Computer vision allows computers to analyze images and videos."
]

text = "\n".join(documents)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)

chunks = splitter.split_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

conversation_history = ""

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    question_embedding = model.encode([question])

    scores = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )[0]

    top_indices = scores.argsort()[-3:][::-1]

    context = ""

    for index in top_indices:
        context += chunks[index] + " "

    prompt = (
        "You are a helpful AI chatbot. "
        "Use the context and conversation history to answer the question.\n\n"
        "Context: " + context +
        "\nConversation History: " + conversation_history +
        "\nCurrent Question: " + question +
        "\nAnswer:"
    )

    result = generator(
        prompt,
        max_new_tokens=100
    )

    answer = result[0]["generated_text"]

    print("Chatbot:", answer)

    conversation_history += (
        "\nUser: " + question +
        "\nAssistant: " + answer
    )
