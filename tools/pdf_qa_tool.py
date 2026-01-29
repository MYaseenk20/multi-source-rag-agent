from dotenv import load_dotenv
load_dotenv()
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langsmith import Client

index_name = "langchain-pdf-index"


def build_pdf_qa_chain(llm):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore(
        embedding=embedding,
        index_name=index_name
    )

    client = Client()
    prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat")

    combine_stuff_doc = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    rephrase_prompt = client.pull_prompt("langchain-ai/chat-langchain-rephrase")


    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        prompt=rephrase_prompt,
        retriever=vectorstore.as_retriever(),
    )

    create_retriever_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_stuff_doc,
    )

    return create_retriever_chain

# Build once (module-level singletons)
_llm = ChatOllama(model="llama3.1:latest", temperature=0.6)
_pdf_qa_chain = build_pdf_qa_chain(_llm)

@tool
def pdf_qa_tool(question: str) -> str:
    """
       Search company policy documents including leave policies, vacation policies,
       sick leave, PTO, employee benefits, HR policies, and other internal company documentation.

       Use this tool for questions about:
       - Leave policies (sick leave, vacation, PTO, parental leave)
       - Employee benefits and compensation
       - Company policies and procedures
       - HR-related questions
       - Internal company documentation

       Always use this tool FIRST before searching the web when asked about company or HR topics.
       Pass clear, self-contained questions with full context.
       """

    result = _pdf_qa_chain.invoke({"input": question,"chat_history":[]})
    answer = result.get("answer", "")
    ctx = result.get("context", [])

    sources = []
    for d in ctx:
        md = getattr(d, "metadata", {}) or {}
        src = md.get("source") or md.get("file_path") or "unknown"
        page = md.get("page")
        sources.append(f"- {src}" + (f" (page {page})" if page is not None else ""))

    sources_text = "\n".join(sources[:5]) if sources else "- (no sources returned)"
    return f"{answer}\n\nSources:\n{sources_text}"