import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = 'db'
DATA_PATH = "data"

def main():
    print("Starting the indexing process...")

    # This loader is configured to find and load .txt files.
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="*.txt", 
        loader_cls=TextLoader,
        show_progress=True
    )
    documents = loader.load()

    if not documents:
        # This error will trigger if no .txt files are in the 'data' folder.
        print(f"No .txt documents found in the '{DATA_PATH}' directory. Please check the folder.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # The rest of the script remains the same.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    print("--------------------------------------------------")
    print(f"✅ Success! Your knowledge base is ready in the '{PERSIST_DIRECTORY}' directory.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()