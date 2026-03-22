# **WWII Local RAG — Churchill IA**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-app-red.svg?style=flat-square)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-vector--search-green.svg?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg?style=flat-square)](LICENSE)

A Retrieval-Augmented Generation (RAG) system designed for question answering on World War II. The system combines curated historical sources, vector search using FAISS, and both cloud-based and fully local language model inference.

## **Table of Contents**

* Background
* Install
* Usage
* Architecture
* Data
* Related Efforts
* Maintainers
* Contributing
* License

## **Background**

This project implements a complete RAG pipeline focused on historical question answering. The system retrieves relevant textual evidence from a curated corpus and generates responses strictly grounded in that evidence.

Two distinct versions of the system are provided:

* An **OpenAI-based pipeline**, using OpenAI embeddings and chat models.
* A **fully local pipeline**, using SentenceTransformers for embeddings and Ollama for language model inference.

The local pipeline enables offline execution and removes dependency on external APIs, while maintaining comparable functionality in retrieval and answer generation.

The Streamlit application included in this repository is built on top of the **local pipeline**, and integrates directly with a precomputed vector store hosted on Hugging Face. This allows the application to run without requiring local indexing or data preparation.

## **Install**

Clone the repository:

```bash
git clone https://github.com/mgonzalz/wwii-local-rag.git
cd wwii-local-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For local inference, Ollama must be installed and running:

```bash
ollama run llama3.1:8b
```

## **Usage**

The system is organized into ingestion, indexing, and querying components.

To build the FAISS index from local data:

```bash
python src/local/build_index_local.py # For local environment
python src/openai/build_index.py      # For OpenAI API Key
```

To run the OpenAI-based query pipeline:

```bash
python src/openai/ask_rag.py
```

To run the fully local query pipeline:

```bash
python src/local/ask_rag_ollama.py
```

To launch the Streamlit application:

```bash
streamlit run src/app/app.py
```

The application provides a conversational interface that retrieves relevant context, generates grounded answers, and displays associated sources.

## **Architecture**

The system follows a Retrieval-Augmented Generation architecture. A user query is encoded into a vector representation using an embedding model. This vector is used to perform similarity search over a FAISS index built from document chunks.

The most relevant chunks are selected and combined into a structured context block. This context is passed to a language model with strict instructions to ensure that the generated answer is based only on retrieved evidence.

Two configurations are supported:

* OpenAI: embeddings and generation via OpenAI APIs
* Local: embeddings via SentenceTransformers and generation via Ollama

## **Data**

The corpus consists of curated World War II sources, including Wikipedia articles in Spanish and structured book content. Documents are processed into chunks and stored with metadata for retrieval and citation.

Precomputed vector stores are hosted on Hugging Face: [WWII RAG Dataset](https://huggingface.co/datasets/mgonzalz/wwii-rag-data)

The dataset is structured as follows:

```bash
local/
  ├── index.faiss
  ├── docs.json

openai/
  ├── index.faiss
  ├── docs.json
```

The Streamlit application loads the vector store directly from this dataset, eliminating the need for local storage and improving reproducibility across environments.

## **Related Efforts**

This project builds upon established tools and methods in information retrieval and language models, including FAISS for similarity search, SentenceTransformers for embedding generation, and Ollama for local inference.

## **Contributing**

Contributions are welcome. Possible improvements include extending the dataset, refining retrieval strategies, improving prompt design, or adding evaluation frameworks.

## **License**

[MIT](LICENSE) © María González
