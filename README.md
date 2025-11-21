# 🕵️‍♂️ Corporate Intel GraphRAG (Free Edition)

A powerful **Graph Retrieval-Augmented Generation (GraphRAG)** application that combines **Knowledge Graphs** (structured data) with **Vector Search** (unstructured text) to provide deep insights.

Powered by:
- **Groq (Llama 3)**: Ultra-fast LLM inference.
- **Neo4j**: Graph database for structured relationships.
- **Pinecone**: Vector database for semantic search.
- **LangChain**: Framework for orchestration.
- **Streamlit**: Interactive UI.

---

## 🚀 Features
- **Hybrid Search**: Queries both the Knowledge Graph (Neo4j) and Vector Store (Pinecone).
- **Live Graph Visualization**: Interactive visualization of nodes and edges.
- **Free Tier Compatible**: Designed to run on free tiers of all services.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following API keys:

1.  **Groq API Key**: [Get it here](https://console.groq.com/keys)
2.  **Neo4j AuraDB (Free)**: [Create a free instance](https://neo4j.com/cloud/aura/)
    - Save your URI, Username, and Password.
3.  **Pinecone API Key**: [Get it here](https://app.pinecone.io/)

---

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone <your-repo-url>
    cd GraphRAG
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    - Copy the example file:
      ```bash
      cp .env.example .env
      ```
    - Open `.env` and paste your API keys.

---

## 🏃‍♂️ Usage

### 1. Ingest Data
First, you need to process your text data to extract entities (nodes) and relationships (edges) for the graph, and embeddings for the vector store.

1.  Open `ingest.py` and modify the `text` variable with your own data if desired.
2.  Run the ingestion script:
    ```bash
    python ingest.py
    ```
    *This will populate your Neo4j graph and Pinecone index.*

### 2. Run the App
Launch the Streamlit interface:
```bash
streamlit run app.py
```

---

## 💡 Example Queries
- "What is the relationship between Microsoft and OpenAI?"
- "Who is the CEO of SpaceX?"
- "What companies is Elon Musk connected to?"

---

## 📂 Project Structure
- `app.py`: Main Streamlit application.
- `ingest.py`: Script for processing text into Graph + Vector data.
- `requirements.txt`: Python dependencies.
- `.env`: API keys (Git ignored).

---

## ⚠️ Troubleshooting
- **ImportError: GraphCypherQAChain**: Ensure you are using the correct import path `from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain`.
- **Neo4j Connection Error**: Verify your URI and Password in `.env`.