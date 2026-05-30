import streamlit as st
import pickle
import numpy as np
import faiss
import ollama

from sentence_transformers import SentenceTransformer

if "messages" not in st.session_state:
    st.session_state.messages = []

# PAGE CONFIG
st.set_page_config(
    page_title="AstroRAG",
    page_icon="🌌",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stChatMessage {
    padding: 12px;
    border-radius: 12px;
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:

    st.title("🌌 AstroRAG")

    st.markdown("""
    ### Scientific AI Assistant

    Powered by:

    - Ollama
    - FAISS
    - Sentence Transformers
    - Streamlit
    """)

    model_name = st.selectbox(
        "LLM Model",
        [
            "mistral",
            "phi3:mini"
        ]
    )

    st.metric(
        "Conversation Messages",
        len(st.session_state.messages)
    )

    top_k = st.slider(
        "Number of Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.caption("Developed by Rafaella Chaves")

# TITLE
st.title("🌌 AstroRAG")
st.subheader("Fast Radio Burst Scientific Assistant")
st.markdown("""
Ask questions about:

- Fast Radio Bursts (FRBs)
- Radio Astronomy
- CHIME
- Signal Detection
- Machine Learning for Astronomy

This assistant uses Retrieval-Augmented Generation (RAG)
to answer questions using scientific literature as context.
""")

# LOAD DATA
@st.cache_resource
def load_models():

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    texts = [chunk.page_content for chunk in chunks]

    embedding_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2'
    )

    embeddings = embedding_model.encode(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return chunks, texts, embedding_model, index


chunks, texts, embedding_model, index = load_models()

# RETRIEVAL
def retrieve_context(query, k=3):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        k
    )

    retrieved_chunks = []

    for distance, idx in zip(distances[0], indices[0]):

        similarity = 1 / (1 + float(distance))

        retrieved_chunks.append({
            "text": texts[idx],
            "chunk_id": int(idx),
            "similarity": similarity
        })

    context_text = "\n\n".join(
        [chunk["text"] for chunk in retrieved_chunks]
    )

    return context_text, retrieved_chunks

# HISTORY
def build_conversation_history():

    history = []

    # 6 last messages
    recent_messages = st.session_state.messages[-6:]

    for msg in recent_messages:

        history.append(
            {
                "role": msg["role"],
                "content": msg["content"]
            }
        )

    return history

# AGENT
def astroagent(query):

    context, retrieved_chunks = retrieve_context(
        query,
        k=top_k
    )

    conversation_history = build_conversation_history()

    prompt_text = f"""
You are an AI scientific assistant specialized in radio astronomy and Fast Radio Bursts (FRBs).

Use the scientific context below to answer the user's question.

Scientific Context:
{context}

Question:
{query}

Answer:
"""

    messages = conversation_history.copy()

    messages.append(
        {
            "role": "user",
            "content": prompt_text
        }
    )

    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True
    )

    return stream, retrieved_chunks


# DISPLAY CHAT HISTORY
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# USER INPUT
prompt = st.chat_input(
    "Ask something about FRBs..."
)


# RUN AGENT
if prompt:

    # USER MESSAGE
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # ASSISTANT MESSAGE
    with st.chat_message("assistant"):

        with st.spinner("AstroRAG is thinking..."):

            response_stream, retrieved_chunks = astroagent(prompt)

            response_placeholder = st.empty()

            full_response = ""

            for chunk in response_stream:

                content = chunk["message"]["content"]

                full_response += content

                response_placeholder.markdown(
                    full_response
                )

        # Retrieval Information
        st.caption(
            f"Retrieved {len(retrieved_chunks)} chunks"
        )

        with st.expander("Retrieved Context"):

            for chunk in retrieved_chunks:

                st.markdown(
                    f"""
**Chunk {chunk['chunk_id']}**

Similarity: {chunk['similarity']:.3f}
"""
                )

                st.write(
                    chunk["text"][:1000]
                )

                st.divider()

    # SAVE ASSISTANT RESPONSE
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )