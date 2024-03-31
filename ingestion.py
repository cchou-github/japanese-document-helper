from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def ingest_docs():
    loader = ReadTheDocsLoader("source_docs", custom_html_tag=("main", {"class": "main-contents"}), patterns="*")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Splitted into  {len(documents)} chunks")

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("source_docs", "https:/")
        new_url = new_url.replace(".html", "")
        doc.metadata.update({"source": new_url})
    
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')

    print(f"Going to add {len(documents)} to vectorstor")
    Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
