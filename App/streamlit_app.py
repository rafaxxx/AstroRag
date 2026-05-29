import streamlit as st
import pickle
import numpy as np
import faiss
import ollama

from sentence_transformers import SentenceTransformer

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
    - Mistral
    - FAISS
    - Sentence Transformers
    - Streamlit

    ---
    """)

    st.success("Model: Mistral")

    top_k = st.slider(
        "Number of retrieved chunks",
        1,
        10,
        3
    )

    st.markdown("---")

    st.caption("Developed by Rafaella Chaves")

# TITLE
st.title("🌌 AstroRAG")
st.subheader("Fast Radio Burst Scientific Assistant")

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

    contexts = []

    for idx in indices[0]:
        contexts.append(texts[idx])

    return "\n\n".join(contexts)

# AGENT
def astroagent(query):

    context = retrieve_context(query, k=top_k)

    prompt = f"""
You are an AI scientific assistant specialized in radio astronomy and Fast Radio Bursts (FRBs).

Use the scientific context below to answer the user's question.

Scientific Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model='mistral',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    return response['message']['content']


# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# DISPLAY CHAT
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# USER INPUT
prompt = st.chat_input(
    "Ask something about FRBs..."
)

# RUN AGENT
if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("AstroRAG is thinking..."):

            response = astroagent(prompt)

            st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )