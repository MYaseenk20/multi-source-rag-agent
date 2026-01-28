from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

INDEX_DIR = "faiss_lcel_index"
def main():
    loader = WebBaseLoader("https://python.langchain.com/docs/expression_language/")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorStore = FAISS.from_documents(splitDocs, embedding=embedding)
    vectorStore.save_local(INDEX_DIR)
    print(f"Saved FAISS index to {INDEX_DIR} with {len(splitDocs)} chunks.")

if __name__ == "__main__":
    main()