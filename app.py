import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

# --- UPDATED IMPORTS FOR LANGCHAIN v0.2+ ---
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pyvis.network import Network

# 1. Setup
load_dotenv()
st.set_page_config(page_title="GraphRAG (Free)", layout="wide")

st.title("🕵️‍♂️ Corporate Intel GraphRAG (Free Edition)")
st.markdown("Powered by **Groq (Llama 3)**, **Neo4j**, and **HuggingFace**")

# 2. Initialize Resources
@st.cache_resource
def get_resources():
    # LLM
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Graph Connection
    graph = Neo4jGraph()
    
    # Vector Connection
    vector_store = PineconeVectorStore(
        index_name="graph-rag-free", 
        embedding=embeddings
    )
    
    return llm, graph, vector_store

try:
    llm, graph, vector_store = get_resources()
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# 3. Define Chains
# Graph Chain: Converts natural language -> Cypher SQL -> Graph Result
graph_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

# Vector Chain: Standard Semantic Search
vector_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# 4. The Hybrid Logic
def hybrid_query(question):
    # A. Ask Graph
    try:
        graph_response = graph_chain.invoke({"query": question})
        graph_context = graph_response.get('result', graph_response.get('output', 'No graph result'))
    except Exception as e:
        graph_context = f"Graph Error: {e}"

    # B. Ask Vectors
    try:
        vector_response = vector_chain.invoke({"query": question})
        vector_context = vector_response.get('result', vector_response.get('output', 'No vector result'))
    except Exception as e:
        vector_context = f"Vector Error: {e}"

    # C. Synthesize
    synthesis_prompt = f"""
    You are a helpful analyst. Answer the user's question using the provided contexts.
    
    --- KNOWLEDGE GRAPH CONTEXT (Structured Facts) ---
    {graph_context}
    
    --- DOCUMENT CONTEXT (Unstructured Text) ---
    {vector_context}
    
    Question: {question}
    Answer:
    """
    
    return llm.invoke(synthesis_prompt).content

# 5. UI Interface
col1, col2 = st.columns([1, 2])

with col1:
    query = st.text_input("Ask a question:", "What companies is Elon Musk connected to?")
    if st.button("Search"):
        with st.spinner("Traversing the Graph + Searching Vectors..."):
            answer = hybrid_query(query)
            st.success("Answer Generated")
            st.markdown(f"**Result:** {answer}")

with col2:
    st.subheader("🕸️ Live Knowledge Graph")
    if query:
        try:
            # Visualization Query
            viz_query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 20"
            data = graph.query(viz_query)
            
            net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white")
            
            if not data:
                st.warning("No data found. Please run ingest.py first.")
            else:
                for record in data:
                    src = record['n']['id']
                    dst = record['m']['id']
                    rel = record['r'][1]
                    
                    net.add_node(src, label=src, title=src, color="#00ff41")
                    net.add_node(dst, label=dst, title=dst, color="#00c2ff")
                    net.add_edge(src, dst, title=rel)

                # Save and display safely
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    path = tmp.name
                
                try:
                    net.save_graph(path)
                    with open(path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=400)
                finally:
                    # Cleanup temp file
                    if os.path.exists(path):
                        os.remove(path)
                        
        except Exception as e:
            st.error(f"Could not visualize graph: {e}")