import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = 'db'
DATA_PATH = "data"

def main():
    # Make sure you have set your OPENAI_API_KEY as an environment variable
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    print("Starting the indexing process with OpenAI Embeddings...")
    
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()

    if not documents:
        print(f"No .txt documents found in '{DATA_PATH}'.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Use OpenAI's embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Rebuild the ChromaDB with the new embeddings
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print("✅ Success! Your knowledge base is ready.")

if __name__ == "__main__":
    main()