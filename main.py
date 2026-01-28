from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import create_retriever_tool
from langsmith import Client
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

from langchain_ollama import ChatOllama  # Changed import
INDEX_DIR = "faiss_lcel_index"


def load_retriever():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorStore = FAISS.load_local(INDEX_DIR, embeddings=embedding,allow_dangerous_deserialization=True)
    return vectorStore.as_retriever(search_kwargs={"k": 3})

def build_agent_executor():
    model = ChatOllama(
        model="llama3.1:latest",
        temperature=0.6,
    )


    client = Client()
    prompt = client.pull_prompt("hwchase17/react-chat")

    search = TavilySearch()

    retriever_tools = create_retriever_tool(
        load_retriever(),
        "lcel_search",
        "Use this tool when searching for information about Langchain Expression Language (LCEL)."
    )
    tools = [search, retriever_tools]

    # Use create_react_agent instead
    # agent = create_react_agent(
    #     llm=model,
    #     prompt=prompt,
    #     tools=tools
    # )
    #
    # return AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     verbose=False,                 # turn on only in dev
    #     handle_parsing_errors=True,
    #     return_intermediate_steps=False
    # )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are a helpful assistant that can search the web and retrieve information about LangChain Agent. Use the appropriate tools to answer user questions accurately."
    )

    return agent

def process_chat(agentExecutor, user_input, chat_history):
    # response = agentExecutor.invoke({
    #     "input": user_input,
    #     "chat_history": chat_history
    # })
    # return response["output"]

    messages = chat_history + [HumanMessage(content=user_input)]

    response = agentExecutor.invoke({
        "messages": messages
    })

    # Extract the final response
    return response["messages"][-1].content


if __name__ == '__main__':
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(build_agent_executor(), user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant:", response)