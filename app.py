import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI # Import Google's LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# --- App Title and Description ---
st.set_page_config(page_title="DocBot", page_icon="🤖")
st.title("🤖 DocBot: Your Document Assistant")

# --- Load the Knowledge Base ---
PERSIST_DIRECTORY = 'db'
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
retriever = db.as_retriever()

# --- Build the RAG Chain ---
template = """
Use the following context to answer the question. If you don't know, just say you don't know.
Context: {context}
Question: {question}
Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Instantiate Google's Gemini model
# Ensure your GOOGLE_API_KEY is set as an environment variable
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# --- User Interface ---
user_input = st.text_input("Ask a question about your documents:")

if user_input:
    # Before running, ensure the API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API key not set. Please set the environment variable.")
    else:
        with st.spinner("Searching..."):
            result = qa_chain.invoke({"query": user_input})
            st.subheader("Answer:")
            st.write(result["result"])

           
