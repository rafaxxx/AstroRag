# AstroRAG — Scientific Retrieval-Augmented Generation for Fast Radio Burst Research

## Description

AstroRAG is a Retrieval-Augmented Generation (RAG) application developed for scientific question answering in the field of radio astronomy and Fast Radio Burst (FRB) research.

The system combines local Large Language Models (LLMs), semantic retrieval, vector similarity search, and scientific document processing to create a context-aware assistant capable of answering astronomy-related questions using scientific literature as knowledge source.

The project explores the application of Generative AI techniques to scientific workflows, focusing on domain-specific retrieval and local inference architectures.

---

## Motivation

Recent advances in Large Language Models have enabled new approaches for scientific knowledge retrieval and domain-specific question answering. However, general-purpose LLMs frequently lack specialized scientific context and may generate hallucinated or inaccurate responses when applied to technical domains such as astrophysics and radio astronomy.

This project investigates how Retrieval-Augmented Generation can improve scientific question answering by combining semantic search techniques with local LLM inference, allowing the model to retrieve contextual information directly from scientific documents before generating responses.

The project also explores local-first AI architectures using Ollama and open-source language models, avoiding dependency on external APIs while enabling reproducible experimentation in scientific environments.

---

## Features

* Retrieval-Augmented Generation (RAG)
* Local LLM inference with Ollama
* Semantic search using FAISS
* Scientific PDF ingestion
* Context-aware question answering
* Streamlit-based interactive interface
* Fast Radio Burst domain specialization
* Local-first Generative AI architecture

---

## Computational and AI Stack

| Category                | Technologies            |
| ----------------------- | ----------------------- |
| Programming Language    | Python                  |
| LLM Runtime             | Ollama                  |
| Language Model          | Mistral                 |
| Vector Database         | FAISS                   |
| Embedding Model         | Sentence Transformers   |
| Frameworks              | LangChain, Streamlit    |
| Development Environment | Jupyter Notebook, Conda |

---

## Methodology

```text id="b3mvcu"
Scientific PDFs
       ↓
Document Loading
       ↓
Text Chunking
       ↓
Embeddings Generation
       ↓
FAISS Vector Index
       ↓
Semantic Retrieval
       ↓
Prompt Augmentation
       ↓
Mistral via Ollama
       ↓
Generated Response
```

The methodology follows a standard Retrieval-Augmented Generation pipeline composed of document ingestion, semantic indexing, vector retrieval, contextual augmentation, and response generation.

Scientific documents are first processed and segmented into textual chunks. Embeddings are generated using Sentence Transformers and indexed with FAISS for efficient similarity search. During inference, user queries are embedded and matched against the vector database to retrieve semantically relevant context. The retrieved context is then incorporated into the prompt sent to the language model through Ollama.

The application interface was implemented using Streamlit to provide an interactive environment for scientific exploration and question answering.

---

## Project Structure

```bash id="w3m40f"
AstroRAG/
│
├── notebooks/
│   ├── notebook_1_data_loading.ipynb
│   ├── notebook_2_chunking.ipynb
│   ├── notebook_3_embeddings.ipynb
│   └── notebook_4_testing.ipynb
│
├── App/
│   └── streamlit_app.py
│
├── data/
│   └── papers/
│
├── assets/
│   └── app_demo.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Application Interface

<img width="1200" alt="AstroRAG Demo" src="assets/app_demo.png">

---

## Installation

### Clone the repository

```bash id="m1s1j3"
git clone https://github.com/YOUR_USERNAME/AstroRAG.git

cd AstroRAG
```

---

### Create the environment

```bash id="jlwm4f"
conda create -n astrorag310 python=3.10

conda activate astrorag310
```

---

### Install dependencies

```bash id="wr1bcw"
pip install \
streamlit \
ollama \
faiss-cpu \
sentence-transformers \
langchain \
langchain-community \
pypdf \
scikit-learn \
ipykernel
```

---

## Installing Ollama

Download Ollama:

https://ollama.com

Pull the Mistral model:

```bash id="hyjpv9"
ollama pull mistral
```

---

## Running the Application

Navigate to the application directory:

```bash id="2vmw8i"
cd App
```

Start Streamlit:

```bash id="7mwaswo"
streamlit run streamlit_app.py
```

The application will be available at:

```bash id="jlwm5f"
http://localhost:8501
```

---

## Example Questions

* What are Fast Radio Bursts?
* Explain dispersion measure in FRBs
* How does CHIME detect FRBs?
* What are the leading theories for FRB origins?
* Explain radio interferometry in astronomy

---

## License

This project is intended for educational, research, and portfolio purposes.
