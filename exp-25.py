import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(
    page_title="Multi-Document AI Assistant",
    page_icon="📚"
)

st.title("Multi-Document AI Assistant")
st.write("Upload multiple PDF documents and ask questions about them.")

model = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

generator = load_generator()

uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    documents = []
    document_names = []

    for uploaded_file in uploaded_files:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        words = text.split()

        chunk_size = 150

        for i in range(0, len(words), chunk_size):

            chunk = " ".join(
                words[i:i + chunk_size]
            )

            if chunk.strip():
                documents.append(chunk)
                document_names.append(
                    uploaded_file.name
                )

    if documents:

        st.success(
            f"{len(uploaded_files)} documents loaded successfully."
        )

        embeddings = model.encode(documents)
        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        st.write(
            f"Total text chunks stored: {len(documents)}"
        )

        question = st.text_input(
            "Ask a question about your documents:"
        )

        if question:

            query_embedding = model.encode([question])
            query_embedding = np.asarray(
                query_embedding,
                dtype="float32"
            )

            k = min(3, len(documents))

            distances, indices = index.search(
                query_embedding,
                k
            )

            context = ""

            for position in range(k):

                document_index = indices[0][position]

                context += (
                    documents[document_index]
                    + "\n"
                )

            prompt = (
                "Answer the question using only "
                "the provided document context.\n\n"
                "Context:\n"
                + context
                + "\nQuestion: "
                + question
                + "\nAnswer:"
            )

            result = generator(
                prompt,
                max_new_tokens=150
            )

            st.subheader("Answer")

            st.write(
                result[0]["generated_text"]
            )

            st.subheader("Retrieved Documents")

            for position in range(k):

                document_index = indices[0][position]

                st.write(
                    document_names[document_index]
                )
