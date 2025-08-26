import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

st.set_page_config(page_title="DocBot", page_icon="🤖")
st.title("🤖 DocBot: Your Document Assistant")

PERSIST_DIRECTORY = 'db'

# Use the same OpenAI embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
retriever = db.as_retriever()

# Build the rest of the RAG chain (this part is the same)
template = """
Use the following context to answer the question. If you don't know the answer, just say you don't know.
Context: {context}
Question: {question}
Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# User Interface
user_input = st.text_input("Ask a question about your documents:")

if user_input:
    with st.spinner("Searching..."):
        result = qa_chain.invoke({"query": user_input})
        st.subheader("Answer:")
        st.write(result["result"])