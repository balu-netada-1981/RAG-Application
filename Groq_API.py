import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import time

# Load environment variables
load_dotenv()

# Safely get API keys from Streamlit secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

# Debugging: Check if API keys are loaded
if not os.environ["HUGGINGFACEHUB_API_TOKEN"]:
    st.warning("⚠️ Hugging Face API Key is missing!")

if not os.environ["GROQ_API_KEY"]:
    st.warning("⚠️ GROQ API Key is missing!")

# Initialize LLM (if API key is available)
if os.environ["GROQ_API_KEY"]:
    llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="gemma2-9b-it")
else:
    st.error("🚨 GROQ API Key is missing! Please check your Streamlit secrets.")
    st.stop()

# Define Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provided context only.
Provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")

# Function to create vector embeddings
def create_vector_embedding(url):
    if "vectors" not in st.session_state:
        try:
            st.session_state.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.loader = WebBaseLoader(url)
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=400)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)
            st.session_state.vectors_ready = True  # Mark vector database as ready
            st.success("✅ Vector Database Ready!")
        except Exception as e:
            st.error(f"⚠️ Error in document embedding: {str(e)}")

# Streamlit UI
st.title("🔍 RAG Query System")

# User inputs
url = st.text_input("🔗 Enter the document URL:")
user_prompt = st.text_input("📝 Enter your query")

# Button to create embeddings
if st.button("📂 Generate Document Embedding") and url:
    create_vector_embedding(url)

# Check if user input is given
if user_prompt:
    if "vectors" not in st.session_state or not st.session_state.get("vectors_ready", False):
        st.error("⚠️ Please generate the document embedding first by clicking the button above.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed_time = time.process_time() - start_time

        st.write(f"⏳ Response Time: {elapsed_time:.2f} seconds")
        st.write("### ✅ Answer:")
        st.write(response.get("answer", "No answer found."))

        # Display retrieved documents
        with st.expander("📑 Document Similarity Search Results"):
            if "context" in response:
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("---------------------------")
            else:
                st.write("No relevant documents found.")
