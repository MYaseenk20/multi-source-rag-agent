from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "pdf-docs\HR-Policy-Revised-JUNE-2022.pdf"
index_name = "langchain-pdf-index"

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_pdf_faiss_index():
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(docs[0].metadata)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    print(f"Going to add {len(documents)} docs in Pinecone")

    PineconeVectorStore.from_documents(documents,embedding=embedding,index_name=index_name)
    print("*** Loading to vectorstore done ***")

if __name__ == '__main__':
    build_pdf_faiss_index()