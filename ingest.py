import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. Load Environment
load_dotenv()

# 2. Initialize Models
# GROQ: Using Llama 3.3 70B for high-quality extraction
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile"
)

# EMBEDDINGS: Running locally (Free, no API cost)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Setup Pinecone (One-time Index Creation)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "graph-rag-free"

if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384, # Matches all-MiniLM-L6-v2 dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    ) 
    time.sleep(10) # Wait for initialization

# 4. Dummy Data (Replace with your own text)
text = """
Yuvraj is the CEO of SpaceX and Tesla. 
SpaceX was founded in 2002 with the goal of reducing space transportation costs.
Tesla produces electric vehicles and clean energy storage.
In 2024, SpaceX launched the Starship rocket, heavily criticized by environmental groups.
"""
documents = [Document(page_content=text)]

# 5. GRAPH: Extract Nodes & Relationships using Groq
print("🚀 Extracting Graph Data with Groq...")
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Push to Neo4j
print(f"Pushing {len(graph_documents[0].nodes)} nodes and {len(graph_documents[0].relationships)} edges to Neo4j...")
graph = Neo4jGraph()
graph.add_graph_documents(graph_documents)

# 6. VECTOR: Store Embeddings in Pinecone
print("💾 Storing Vectors in Pinecone...")
vector_store = PineconeVectorStore.from_documents(
    documents, 
    embeddings, 
    index_name=index_name
)

print("✅ Ingestion Complete!")