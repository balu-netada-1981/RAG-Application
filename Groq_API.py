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
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
st.secrets= os.getenv("GROQ_API_KEY", "")

# Ensure API key is available
groq_api_key = st.secrets
if not groq_api_key:
    st.error("GROQ API Key is missing. Please check your environment variables.")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Define Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only.
Provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
    """
)

# URL to scrape
#url = "https://www.langchain.com/"

# Function to create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = WebBaseLoader(url)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=400)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)
        st.session_state.vectors_ready = True  # Track that the vector database is ready

# Streamlit UI
st.title("RAG Query System")
url = st.text_input("Enter the document URL:",)
# User input
user_prompt = st.text_input("Enter your query")

# Button to create embeddings
if st.button("Document Embedding") and url and user_prompt:
    create_vector_embedding()
    st.success("Vector Database Ready!")

# Check if user input is given
if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please generate the document embedding first by clicking the button above.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed_time = time.process_time() - start_time

        st.write(f"Response Time: {elapsed_time:.2f} seconds")
        st.write("### Answer:")
        st.write(response.get("answer", "No answer found."))

        with st.expander("Document Similarity Search Results"):
            if "context" in response:
                for i, doc in enumerate(response["context"]):
                    # st.write(f"**Document {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---------------------------")
            else:
                st.write("No relevant documents found.")
