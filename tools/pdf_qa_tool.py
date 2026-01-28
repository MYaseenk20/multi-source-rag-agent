from langchain_core.tools import tool


@tool
def pdf_qa_tool(question: str) -> str:
    # do retrieval + combine docs + answer
    return "answer"
